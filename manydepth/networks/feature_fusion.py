import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth_anything_v2.dinov2_layers.drop_path import DropPath
import numpy as np


class MultiFrameFeatureFusion(nn.Module):
    """
    Multi-frame attention block for fusing *current* and *lookup/previous*
    frame features on a flattened H×W grid.

    This module:
        - treats the current frame features as **queries** (`x1`)
        - when temporal fusion is on: attends only to **previous frame** (`x2`)
          and register tokens; when off: self-attends on the current frame
        - restricts attention to a configurable **spatial neighborhood**
        - supports per-scale **LoRA adapters** for lightweight fine-tuning
        - can append learnable **register tokens** that act as attention sinks.

    Args:
        input_dim: Channel dimension of each frame feature map.
        matching_height: Spatial height of the flattened feature grid (H).
        matching_width: Spatial width of the flattened feature grid (W).
        num_heads: Number of attention heads.
        dropout: Dropout probability applied to attention weights and FFN.
        drop_path: Stochastic depth probability for residual branches.
        neighborhood_size:
            - `(h, w)` tuple → rectangular neighborhood.
            - `int`          → square neighborhood, `h = w`.
            - `None`         → global attention (no spatial restriction).
        temporal_fusion: If `True`, keys/values come from previous frame only
            and register tokens are used; if `False`, self-attention on current frame.
        num_register_tokens: Number of extra learnable register tokens (used only when temporal_fusion is True).
        num_scales: Number of pyramid scales supported by separate LoRA heads.
        lora_rank: Rank of LoRA adapters. `0` or `None` disables LoRA.
        lora_alpha: LoRA scaling factor.
    """

    def __init__(
        self,
        input_dim: int,
        matching_height: int,
        matching_width: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        neighborhood_size=(3, 15),
        temporal_fusion: bool = True,
        num_register_tokens: int = 0,
                 num_scales: int = 4,
                 lora_rank: int = 32,
        lora_alpha: float = 4,
    ):
        super().__init__()

        # Allow tuple[int, int], int (square), or None for global attention.
        if not (
            neighborhood_size is None
            or isinstance(neighborhood_size, int)
            or isinstance(neighborhood_size, tuple)
        ):
            raise TypeError(
                f"neighborhood_size must be tuple[int, int] | int | None, "
                f"got {type(neighborhood_size)}"
            )
        self.input_dim = input_dim
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.num_heads = num_heads
        self.temporal_fusion = temporal_fusion
        self.neighborhood_size = neighborhood_size  # (height, width) for non-square neighborhoods, None for global attention
        self.drop_path = drop_path
        self.num_register_tokens = num_register_tokens
        self.num_scales = int(num_scales)

        # LoRA config (optional)
        self.lora_rank = int(lora_rank) if (lora_rank is not None and int(lora_rank) > 0) else None
        self.lora_alpha = float(lora_alpha) if (lora_alpha is not None) else None

        # ---------------- Attention projections ----------------
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Register tokens: learnable tokens that act as attention sinks
        # Initialize with small random values
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.normal(0, 1, (1, num_register_tokens, input_dim)))
        else:
            self.register_tokens = None
        
        # ---------------- LoRA adapters (per-scale, optional) ----------------
        # We keep base model weights shared. For each scale, maintain separate low-rank adapters
        # that add learned deltas before attention and for each FFN layer.
        if self.lora_rank is not None:
            # Define lightweight adapter module inline
            class _LoRAAdapter(nn.Module):
                def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
                    super().__init__()
                    self.down = nn.Linear(in_dim, rank, bias=False)
                    self.up = nn.Linear(rank, out_dim, bias=False)
                    self.scaling = (alpha / float(rank)) if (alpha is not None) else (1.0 / float(rank))

                    # Init following LoRA convention: down with kaiming, up with zeros so starts as no-op
                    nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5))
                    nn.init.zeros_(self.up.weight)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.up(self.down(x)) * self.scaling

            # Pre-attention LoRA (dim -> dim)
            self.lora_pre_attn = nn.ModuleList([
                _LoRAAdapter(input_dim, input_dim, self.lora_rank, self.lora_alpha)
                for _ in range(self.num_scales)
            ])

            # FFN per-layer LoRA
            hidden_dim = input_dim * 2
            self.lora_ffn1 = nn.ModuleList([
                _LoRAAdapter(input_dim, hidden_dim, self.lora_rank, self.lora_alpha)
                for _ in range(self.num_scales)
            ])
            self.lora_ffn2 = nn.ModuleList([
                _LoRAAdapter(hidden_dim, input_dim, self.lora_rank, self.lora_alpha)
                for _ in range(self.num_scales)
            ])
        else:
            self.lora_pre_attn = None
            self.lora_ffn1 = None
            self.lora_ffn2 = None

        # Pre-LN: normalize before attention and before feed-forward
        self.attn_norm = nn.LayerNorm(input_dim)
        self.ffn_norm = nn.LayerNorm(input_dim)
        # Explicit FFN layers to allow per-layer LoRA injection
        self.ffn_fc1 = nn.Linear(input_dim, input_dim * 2)
        self.ffn_act = nn.GELU()
        self.ffn_fc2 = nn.Linear(input_dim * 2, input_dim)
        self.ffn_dropout = nn.Dropout(dropout)
        
        # DropPath modules for stochastic depth
        self.drop_path_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Create attention mask for neighborhood constraint
        _mask = self.create_neighborhood_mask(
            self.matching_height,
            self.matching_width,
            self.neighborhood_size,
        )
        if _mask is not None:
            self.register_buffer("attn_mask", _mask, persistent=False)
        else:
            self.attn_mask = None

    def create_neighborhood_mask(self, height, width, neighborhood_size):
        """
        Create attention mask for rectangular neighborhood attention (single frame).
        This mask is later concatenated in forward() for temporal fusion.
        
        Args:
            height, width: spatial dimensions
            neighborhood_size: tuple (height_size, width_size) for rectangular neighborhood,
                              or single int for backward compatibility (creates square neighborhood)
            
        Returns:
            mask:
                Boolean tensor of shape ``[height*width, height*width]`` where
                ``True`` means *masked* (cannot attend) and ``False`` means
                the position is allowed.
        """
        if neighborhood_size is None:
            return None
            
        # Handle backward compatibility: if single int, create square neighborhood
        if isinstance(neighborhood_size, int):
            neighborhood_height = neighborhood_width = neighborhood_size
        else:
            neighborhood_height, neighborhood_width = neighborhood_size
            
        total_positions = height * width
        # Create mask for [query_positions, key_positions] for a single frame
        mask = torch.ones(total_positions, total_positions, dtype=torch.bool)
        
        half_height = neighborhood_height // 2
        half_width = neighborhood_width // 2
        
        for h in range(height):
            for w in range(width):
                query_idx = h * width + w
                
                # Define neighborhood bounds
                h_min = max(0, h - half_height)
                h_max = min(height, h + half_height + 1)
                w_min = max(0, w - half_width)  
                w_max = min(width, w + half_width + 1)
                
                # Mark positions in neighborhood as unmasked (False = can attend)
                for nh in range(h_min, h_max):
                    for nw in range(w_min, w_max):
                        neighbor_idx = nh * width + nw
                        mask[query_idx, neighbor_idx] = False        
        return mask

    def forward(self, input, scale_index: int = None):
        """
        Forward pass of the fusion block.

        Args:
            input:
                Concatenated current and lookup features of shape
                ``[B, N, 2 * input_dim]`` where
                ``N = matching_height * matching_width``.
                The first ``input_dim`` channels correspond to the **current**
                frame (`x1`), and the remaining ``input_dim`` channels to the
                **lookup / previous** frame (`x2`).
            scale_index:
                Optional index in ``[0, num_scales)`` used to select which
                LoRA adapters to apply. If ``None``, index ``0`` is used.

        Returns:
            Tensor of shape ``[B, N, input_dim]`` with updated current-frame
            features.
        """
        # Alias to avoid shadowing Python's built-in ``input`` in tooling.
        x_all = input
        x1 = x_all[:, :, : self.input_dim]
        x2 = x_all[:, :, self.input_dim :]
        
        b, n, _ = x1.shape
        
        # Verify spatial dimensions
        assert n == self.matching_height * self.matching_width, f"Expected N={self.matching_height * self.matching_width}, got N={n}"

        # Optionally apply scale-specific LoRA adapter before attention
        if self.lora_pre_attn is not None:
            idx = 0 if (scale_index is None) else int(scale_index)
            if not (0 <= idx < self.num_scales):
                raise ValueError(f"scale_index {idx} out of range [0,{self.num_scales-1}]")
            x1 = x1 + self.lora_pre_attn[idx](x1)

        # Pre-LN Attention
        if self.temporal_fusion:
            q = self.attn_norm(x1)
            # Only previous-frame keys/values (no current frame in key/value)
            k = self.attn_norm(x2)
            v = self.attn_norm(x2)
            attn_mask = self.attn_mask
        else:
            # Self-attention on current frame only
            q = self.attn_norm(x1)
            k = self.attn_norm(x1)
            v = self.attn_norm(x1)
            attn_mask = self.attn_mask

        # Register tokens only when temporal fusion is on
        if self.temporal_fusion and self.register_tokens is not None:
            register_tokens_expanded = self.register_tokens.expand(b, -1, -1)  # [B, num_reg, C]
            k = torch.cat([k, register_tokens_expanded], dim=1)  # [B, N+num_reg, C]
            v = torch.cat([v, register_tokens_expanded], dim=1)  # [B, N+num_reg, C]
            # Extend attention mask to allow attending to register tokens (unmasked)
            if attn_mask is not None:
                num_queries = attn_mask.shape[0]
                register_mask = torch.zeros(num_queries, self.num_register_tokens,
                                            dtype=attn_mask.dtype, device=attn_mask.device)
                attn_mask = torch.cat([attn_mask, register_mask], dim=1)

        # Convert boolean attn_mask (N, N_k) to additive mask [1,1,N,N_k]
        if attn_mask is not None:
            # Keep mask as bool for clarity: True = masked, False = allowed.
            attn_mask_bool = attn_mask.to(device=q.device, dtype=torch.bool)
            attn_mask_float = torch.zeros(
                1,
                1,
                attn_mask_bool.shape[0],
                attn_mask_bool.shape[1],
                device=q.device,
                dtype=q.dtype,
            )
            # Large negative where mask is True so those entries are removed
            # by the softmax.
            attn_mask_float = attn_mask_float.masked_fill(
                attn_mask_bool.unsqueeze(0).unsqueeze(1),
                float("-inf"),
            )
        else:
            attn_mask_float = None

        # ---------------- Custom multi-head attention ----------------
        head_dim = self.input_dim // self.num_heads
        if head_dim * self.num_heads != self.input_dim:
            raise ValueError("input_dim must be divisible by num_heads.")

        # Project to Q, K, V
        q_proj = self.q_proj(q)  # [B, N, C]
        k_proj = self.k_proj(k)  # [B, Nk, C]
        v_proj = self.v_proj(v)  # [B, Nk, C]

        # Reshape to [B, num_heads, N, head_dim]
        q_proj = q_proj.view(b, n, self.num_heads, head_dim).transpose(1, 2)
        k_proj = k_proj.view(b, k.shape[1], self.num_heads, head_dim).transpose(1, 2)
        v_proj = v_proj.view(b, k.shape[1], self.num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1))  # [B, H, N, Nk]
        attn_scores = attn_scores / math.sqrt(head_dim)

        # Add attention mask if present
        if attn_mask_float is not None:
            attn_scores = attn_scores + attn_mask_float

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v_proj)  # [B, H, N, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.input_dim)
        attn_output = self.out_proj(attn_output)

        # Residual connection
        x = x1 + self.drop_path_attn(attn_output)
        
        # Pre-LN Feed-forward with per-layer LoRA
        y = self.ffn_norm(x)
        y1 = self.ffn_fc1(y)
        if self.lora_ffn1 is not None:
            idx = 0 if (scale_index is None) else int(scale_index)
            if not (0 <= idx < self.num_scales):
                raise ValueError(f"scale_index {idx} out of range [0,{self.num_scales-1}]")
            y1 = y1 + self.lora_ffn1[idx](y)
        y1 = self.ffn_act(y1)
        y2 = self.ffn_fc2(y1)
        if self.lora_ffn2 is not None:
            idx = 0 if (scale_index is None) else int(scale_index)
            y2 = y2 + self.lora_ffn2[idx](y1)
        y2 = self.ffn_dropout(y2)
        x1_final = x + self.drop_path_ffn(y2)
        
        return x1_final


class CostVolumeFeatureFusion(nn.Module):
    """
    Cost volume based feature fusion using geometric warping.
    Stores debug tensors for TensorBoard logging.
    """

    def __init__(
        self,
        input_dim: int,
        num_depth_bins: int = 32,
        depth_min: float = 0.1,
        depth_max: float = 80.0,
        depth_bin_mode: str = "inv",
        dropout: float = 0.1,
        eps: float = 1e-6,
        chunk_size: int = 0,
        edge_border: int = 2,
        encoder_stride: int = 14
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_bin_mode = depth_bin_mode
        self.eps = eps
        self.chunk_size = chunk_size
        self.edge_border = edge_border
        self.encoder_stride = encoder_stride

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(input_dim + num_depth_bins + 1, input_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self._init_weights()
        self._uv1_cache = {}
        
        # Debug tensors
        self.debug_cost_volume = None
        self.debug_coverage = None
        self.debug_depth_bins = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, current_features, lookup_features, poses, intrinsics):
        """
        Args:
            current_features: [B, C, H, W]
            lookup_features: [B, N, C, H, W]
            poses: [B, N, 4, 4]
            intrinsics: [B, 3, 3] or [B, 4, 4]
        """
        B, C, H, W = current_features.shape
        
        K_feat = self._scale_intrinsics(intrinsics, H, W, current_features.dtype, current_features.device)
        depth_bins = self._make_depth_bins(current_features.device, current_features.dtype)
        
        cost_volume, coverage = self._build_cost_volume(current_features, lookup_features, poses, K_feat, depth_bins)
        
        # Store debug tensors
        self.debug_cost_volume = cost_volume.detach()
        self.debug_coverage = coverage.detach()
        self.debug_depth_bins = depth_bins.detach()
        
        coverage_clamped = coverage.clamp(0, 1).unsqueeze(1)
        fused = self.reduce_conv(torch.cat([current_features, cost_volume, coverage_clamped], dim=1))
        
        return fused + current_features

    def _scale_intrinsics(self, K_in, H, W, dtype, device):
        K = K_in[..., :3, :3].to(dtype=dtype, device=device).clone() if K_in.shape[-1] == 4 else K_in.to(dtype=dtype, device=device).clone()
        sx, sy = 1.0 / self.encoder_stride, 1.0 / self.encoder_stride
        K[..., 0, :] *= sx
        K[..., 1, :] *= sy
        return K

    def _make_depth_bins(self, device, dtype):
        if self.depth_bin_mode == "lin":
            return torch.linspace(self.depth_min, self.depth_max, self.num_depth_bins, device=device, dtype=dtype)
        elif self.depth_bin_mode == "log":
            return torch.logspace(np.log10(self.depth_min), np.log10(self.depth_max), self.num_depth_bins, device=device, dtype=dtype)
        else:  # inv
            inv = torch.linspace(1/self.depth_max, 1/self.depth_min, self.num_depth_bins, device=device, dtype=dtype)
            return 1.0 / inv

    def _get_uv1(self, H, W, device, dtype):
        key = (H, W, dtype, str(device))
        if key not in self._uv1_cache:
            y, x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                                  torch.arange(W, device=device, dtype=dtype), indexing='ij')
            self._uv1_cache[key] = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)
        return self._uv1_cache[key]

    def _build_cost_volume(self, curr_feat, lookup_feat, poses, K, depth_bins):
        B, C, H, W = curr_feat.shape
        N = lookup_feat.shape[1]
        D = self.num_depth_bins
        dev = curr_feat.device

        cv = torch.zeros(B, D, H, W, device=dev, dtype=curr_feat.dtype)
        counts = torch.zeros(B, D, H, W, device=dev, dtype=curr_feat.dtype)

        uv1 = self._get_uv1(H, W, dev, torch.float32)
        K_inv = torch.linalg.inv(K.float())

        for n in range(N):
            for d, depth in enumerate(depth_bins):
                rays = torch.bmm(K_inv, uv1.unsqueeze(0).expand(B, -1, -1)) * depth
                pts = torch.cat([rays.transpose(1, 2), torch.ones(B, H*W, 1, device=dev)], dim=-1)
                
                T = poses[:, n].float()
                pts_trans = torch.bmm(pts, T.transpose(-2, -1))[:, :, :3]
                
                proj = torch.bmm(pts_trans, K.float().transpose(-2, -1))
                z = proj[:, :, 2:3].clamp(min=self.eps)
                uv = proj[:, :, :2] / z
                
                grid = uv.view(B, H, W, 2)
                grid[..., 0] = grid[..., 0] / (W - 1) * 2 - 1
                grid[..., 1] = grid[..., 1] / (H - 1) * 2 - 1
                
                valid = (grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1) & (z.view(B, H, W) > self.eps)
                
                warped = F.grid_sample(lookup_feat[:, n], grid.to(curr_feat.dtype), 
                                       mode='bilinear', padding_mode='border', align_corners=True)
                
                cost = (warped - curr_feat).abs().mean(dim=1).clamp(0, 10)
                
                cv[:, d] += cost * valid.float()
                counts[:, d] += valid.float()

        cv = cv / counts.clamp(min=1)
        missing = counts == 0
        if missing.any():
            mean_cv = (cv * (counts > 0).float()).sum(1, keepdim=True) / counts.sum(1, keepdim=True).clamp(min=1)
            cv = torch.where(missing, mean_cv, cv)
        
        coverage = (counts > 0).float().sum(1) / D
        
        return cv, coverage
