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
        - optionally attends over both previous and current frame features
          (`x2`, `x1`) for temporal fusion
        - restricts attention to a configurable **spatial neighborhood**
        - adds a learnable **relative positional bias** per head
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
        temporal_fusion: If `True`, keys/values come from both previous and
            current frame; otherwise standard self-attention on current frame.
        num_register_tokens: Number of extra learnable register tokens.
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
        num_register_tokens: int = 8,
                 num_scales: int = 4,
                 lora_rank: int = 64,
        lora_alpha: float = 1,
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

        # ---------------- Attention projections (custom MHA with relative bias) ----------------
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # ---------------- Positional encodings (temporal only) ----------------
        # Temporal: two learnable embeddings for "lookup/previous" and "current" frame.
        # Broadcast over all spatial locations: [2, 1, 1, C] -> [B, N, C] via broadcasting.
        self.temporal_pos_encoding = nn.Parameter(
            torch.zeros(2, 1, 1, self.input_dim)
        )
        nn.init.normal_(self.temporal_pos_encoding, mean=0.0, std=0.02)
        
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
        #self.attn_norm_prev = nn.LayerNorm(input_dim)
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

        # ---------------- Relative positional bias (spatial, query-centric) ----------------
        # For each head, learn a bias as a function of relative (dh, dw) within the
        # neighborhood. Each query location treats itself as (0,0); neighbors are
        # indexed by their offset.

        if isinstance(self.neighborhood_size, int):
            neighborhood_height = neighborhood_width = self.neighborhood_size
        elif self.neighborhood_size is None:
            # Global attention: allow full height/width as neighborhood for bias
            neighborhood_height, neighborhood_width = self.matching_height, self.matching_width
        else:
            neighborhood_height, neighborhood_width = self.neighborhood_size

        # Clamp neighborhood to valid extents
        self.neighborhood_height = min(neighborhood_height, self.matching_height)
        self.neighborhood_width = min(neighborhood_width, self.matching_width)

        self._build_relative_position_indices(
            self.matching_height,
            self.matching_width,
            self.neighborhood_height,
            self.neighborhood_width,
        )

        # Learnable table of relative biases per head and per (dh, dw)
        num_rel_h = 2 * self.neighborhood_height - 1
        num_rel_w = 2 * self.neighborhood_width - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_heads, num_rel_h, num_rel_w)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    # ----------------------------------------------------------------------
    # Positional encoding helpers (sine-cosine for potential reuse)
    # ----------------------------------------------------------------------
    @staticmethod
    def _build_1d_sincos_position_embedding(length: int, dim: int) -> torch.Tensor:
        """Standard 1D sine–cosine positional encoding.

        Args:
            length: Sequence length (e.g. H or W).
            dim: Embedding dimension.

        Returns:
            Tensor of shape ``[length, dim]``.
        """
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(dim, 1))
        )  # [dim/2]
        pe = torch.zeros(length, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _build_2d_sincos_position_embedding(
        self,
        height: int,
        width: int,
        dim: int,
    ) -> torch.Tensor:
        """
        2D sine-cosine positional encoding over an HxW grid.

        Kept for potential reuse elsewhere, but MultiFrameFeatureFusion
        now uses a relative positional bias instead of absolute encodings.

        Returns:
            Tensor of shape ``[1, H*W, dim]``.
        """
        # Split channels between vertical and horizontal components
        dim_h = dim // 2
        dim_w = dim - dim_h

        pe_h = self._build_1d_sincos_position_embedding(height, dim_h)  # [H, dim_h]
        pe_w = self._build_1d_sincos_position_embedding(width, dim_w)   # [W, dim_w]

        # Combine to 2D grid
        # [H, 1, dim_h] + [1, W, dim_w] -> [H, W, dim]
        pe_h = pe_h[:, None, :].expand(height, width, dim_h)
        pe_w = pe_w[None, :, :].expand(height, width, dim_w)
        pe_2d = torch.cat([pe_h, pe_w], dim=-1)  # [H, W, dim]
        pe_2d = pe_2d.reshape(1, height * width, dim)  # [1, H*W, dim]
        return pe_2d

    # ----------------------------------------------------------------------
    # Relative positional bias helpers
    # ----------------------------------------------------------------------
    def _build_relative_position_indices(
        self,
        height: int,
        width: int,
        neighborhood_height: int,
        neighborhood_width: int,
    ) -> None:
        """
        Precompute relative index maps for all query/key spatial positions.

        For a query at (h_q, w_q) and key at (h_k, w_k), we compute:
            dh = h_k - h_q
            dw = w_k - w_q
        Then clamp dh, dw into [-neighborhood_height+1, neighborhood_height-1]
        and [-neighborhood_width+1, neighborhood_width-1], and map them to
        table indices in [0, 2*H-2] and [0, 2*W-2].

        These indices are shared across all heads and batches.
        """
        total_positions = height * width
        rel_pos_h = torch.zeros(total_positions, total_positions, dtype=torch.long)
        rel_pos_w = torch.zeros(total_positions, total_positions, dtype=torch.long)

        max_dh = neighborhood_height - 1
        max_dw = neighborhood_width - 1

        for h_q in range(height):
            for w_q in range(width):
                q_idx = h_q * width + w_q
                for h_k in range(height):
                    for w_k in range(width):
                        k_idx = h_k * width + w_k
                        dh = h_k - h_q
                        dw = w_k - w_q
                        # Clamp to the allowed neighborhood range for the bias
                        dh = max(-max_dh, min(max_dh, dh))
                        dw = max(-max_dw, min(max_dw, dw))
                        rel_pos_h[q_idx, k_idx] = dh + max_dh
                        rel_pos_w[q_idx, k_idx] = dw + max_dw

        self.register_buffer("rel_pos_h_idx", rel_pos_h, persistent=False)
        self.register_buffer("rel_pos_w_idx", rel_pos_w, persistent=False)

    def _get_relative_position_bias(
        self,
        num_keys_spatial: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build the relative positional bias tensor for the current attention.

        Args:
            num_keys_spatial:
                Number of spatial key positions **before** temporal duplication
                or register tokens. This is ``N`` when ``temporal_fusion`` is
                ``False``, and ``2N`` when it is ``True``.

        Returns:
            bias:
                Tensor of shape
                ``[1, num_heads, N_queries, num_keys_spatial]``.
        """
        # Base bias over the H*W spatial grid: [num_heads, N, N]
        bias_spatial = self.relative_position_bias_table[
            :, self.rel_pos_h_idx, self.rel_pos_w_idx
        ]  # [num_heads, N, N]

        # Queries are always exactly N spatial tokens (no registers as queries)
        # Keys may be N (no temporal fusion) or 2N (prev + curr).
        if num_keys_spatial == bias_spatial.shape[-1]:
            bias = bias_spatial
        elif num_keys_spatial == 2 * bias_spatial.shape[-1]:
            # Temporal fusion: duplicate bias for prev and current halves
            bias = torch.cat([bias_spatial, bias_spatial], dim=-1)
        else:
            raise ValueError(
                f"Unexpected num_keys_spatial={num_keys_spatial}, "
                f"expected N={bias_spatial.shape[-1]} or 2N."
            )

        # Add batch dimension and ensure correct device/dtype
        bias = bias.unsqueeze(0)  # [1, num_heads, N, num_keys_spatial]
        return bias.to(device=device, dtype=dtype)

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

        # Pre-LN Attention with temporal encodings
        if self.temporal_fusion:
            # Temporal embeddings: index 0 -> lookup/previous, 1 -> current
            t_prev = self.temporal_pos_encoding[0].to(dtype=x1.dtype, device=x1.device)
            t_curr = self.temporal_pos_encoding[1].to(dtype=x1.dtype, device=x1.device)

            q = self.attn_norm(x1 + t_curr)

            # Previous-frame keys/values
            k_prev = self.attn_norm(x2 + t_prev)
            v_prev = self.attn_norm(x2 + t_prev)

            # Current-frame keys/values (self-attention component)
            k_curr = self.attn_norm(x1 + t_curr)
            v_curr = self.attn_norm(x1 + t_curr)

            # Concatenate temporal and current features along sequence dim
            k = torch.cat((k_prev, k_curr), dim=1)
            v = torch.cat((v_prev, v_curr), dim=1)

            # Extend the spatial neighborhood mask to cover both temporal and
            # current features. Each query keeps the same neighborhood pattern
            # for both halves of the key sequence.
            attn_mask = (
                torch.cat((self.attn_mask, self.attn_mask), dim=1)
                if self.attn_mask is not None
                else None
            )
        else:
            q = self.attn_norm(x1)
            k = self.attn_norm(x1)
            v = self.attn_norm(x1)
            attn_mask = self.attn_mask

        # Concatenate register tokens to keys and values
        rel_pos_bias = self._get_relative_position_bias(
            num_keys_spatial=k.shape[1],
            device=q.device,
            dtype=q.dtype,
        )

        if self.register_tokens is not None:
            register_tokens_expanded = self.register_tokens.expand(b, -1, -1)  # [B, num_reg, C]
            k = torch.cat([k, register_tokens_expanded], dim=1)  # [B, N+num_reg, C]
            v = torch.cat([v, register_tokens_expanded], dim=1)  # [B, N+num_reg, C]
            
            # Extend attention mask to allow attending to register tokens (unmasked)
            if attn_mask is not None:
                # attn_mask is [N, N] or [N, 2*N], extend to [N, N+num_reg] or [N, 2*N+num_reg]
                # Register tokens should be unmasked (False = can attend)
                num_queries = attn_mask.shape[0]
                register_mask = torch.zeros(num_queries, self.num_register_tokens, 
                                           dtype=attn_mask.dtype, device=attn_mask.device)
                attn_mask = torch.cat([attn_mask, register_mask], dim=1)
            # Extend relative positional bias with zeros for register tokens
            zeros_reg_bias = torch.zeros(
                1,
                self.num_heads,
                n,
                self.num_register_tokens,
                dtype=rel_pos_bias.dtype,
                device=rel_pos_bias.device,
            )
            rel_pos_bias = torch.cat([rel_pos_bias, zeros_reg_bias], dim=-1)

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

        # ---------------- Custom multi-head attention with relative bias ----------------
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

        # Add relative positional bias (broadcast over batch)
        attn_scores = attn_scores + rel_pos_bias

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
    - Intrinsics passed at IMAGE scale ([B,3,3] or [B,4,4]); scaled to FEATURE scale inside.
    - Assumes encoder stride/patch_size = 14.
    - Poses are current -> lookup, shape [B,N,4,4].
    - Builds per-depth-bin cost volume and fuses with current features.
    - Stores debug tensors as attributes for TensorBoard logging:
        self.debug_cost_volume : [B,D,H,W]
        self.debug_coverage    : [B,H,W]
        self.debug_depth_bins  : [D]
    """

    def __init__(
        self,
        input_dim: int,
        num_depth_bins: int = 32,
        depth_min: float = 0.1,
        depth_max: float = 80.0,
        depth_bin_mode: str = "inv",  # "inv" | "log" | "lin"
        dropout: float = 0.1,
        eps: float = 1e-6,
        chunk_size: int = 0,
        edge_border: int = 2,
        encoder_stride: int = 14
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_depth_bins = int(num_depth_bins)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.depth_bin_mode = str(depth_bin_mode)
        self.eps = float(eps)
        self.chunk_size = int(chunk_size)
        self.edge_border = int(edge_border)
        self.encoder_stride = int(encoder_stride)

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(input_dim + self.num_depth_bins + 1, input_dim, kernel_size=3, padding=1),  # +1 for coverage channel
            nn.GELU(),
        )
        self._init_weights()

        # caches
        self._uv1_cache = {}

        # debug attrs (populated every forward)
        self.debug_cost_volume = None
        self.debug_coverage = None
        self.debug_depth_bins = None

    # -------------------------- Public API --------------------------

    def forward(
        self,
        current_features: torch.Tensor,  # [B, C, Hf, Wf]
        lookup_features: torch.Tensor,   # [B, N, C, Hf, Wf]
        poses: torch.Tensor,             # [B, N, 4, 4]
        intrinsics: torch.Tensor         # [B, 3, 3] or [B, 4, 4] at IMAGE scale
    ) -> torch.Tensor:
        if current_features.dim() != 4:
            raise ValueError(f"current_features must be [B,C,H,W], got {tuple(current_features.shape)}")
        if lookup_features.dim() != 5:
            raise ValueError(f"lookup_features must be [B,N,C,H,W], got {tuple(lookup_features.shape)}")
        if poses.dim() != 4 or poses.shape[-2:] != (4, 4):
            raise ValueError("poses must be [B,N,4,4]")
        if intrinsics.shape[-2:] not in ((3, 3), (4, 4)):
            raise ValueError("intrinsics must be [B,3,3] or [B,4,4] at IMAGE scale")

        B, C, Hf, Wf = current_features.shape
        _, N, C2, H2, W2 = lookup_features.shape
        if (C2, H2, W2) != (C, Hf, Wf):
            raise ValueError("lookup_features spatial/channel dims must match current_features")

        # 1) IMAGE->FEATURE intrinsics (3x3)
        K_feat = self._to_feature_intrinsics(intrinsics, Hf, Wf, current_features.dtype, current_features.device)

        # 2) Depth bins
        depth_bins = self._make_depth_bins(current_features.device, current_features.dtype)  # [D]

        # 3) Build cost volume
        cost_volume, coverage_ratio = self._build_cost_volume(
            current_features, lookup_features, poses, K_feat, depth_bins
        )  # [B,D,Hf,Wf], [B,Hf,Wf]

        # 4) Store debug tensors as attributes (detached to avoid autograd retention)
        self.debug_cost_volume = cost_volume.detach()
        self.debug_coverage = coverage_ratio.detach()
        self.debug_depth_bins = depth_bins.detach()

        # 5) Fuse and return features (no extras returned)
        # Concatenate cost volume with coverage as an additional channel
        # This allows the network to learn how to weight based on coverage
        # rather than incorrectly scaling costs (where low coverage would make high costs look good)
        coverage_ratio_clamped = torch.clamp(coverage_ratio, min=0.0, max=1.0).unsqueeze(1)
        fused = self.reduce_conv(torch.cat([current_features, cost_volume, coverage_ratio_clamped], dim=1))
        # Use a more stable residual connection with scaling
        return fused +  current_features

    # -------------------- Intrinsics scaling ------------------------

    def _to_feature_intrinsics(self, K_in: torch.Tensor, Hf: int, Wf: int,
                               dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if K_in.shape[-2:] == (4, 4):
            K3 = K_in[..., :3, :3].to(dtype=dtype, device=device).clone()
        else:
            K3 = K_in.to(dtype=dtype, device=device).clone()

        Hi = Hf * self.encoder_stride
        Wi = Wf * self.encoder_stride
        sx = Wf / float(Wi)
        sy = Hf / float(Hi)

        K3[..., 0, 0] *= sx  # fx
        K3[..., 1, 1] *= sy  # fy
        K3[..., 0, 2] *= sx  # cx
        K3[..., 1, 2] *= sy  # cy
        return K3

    # -------------------- Cost volume construction ------------------

    def _build_cost_volume(
        self,
        current_features: torch.Tensor,   # [B,C,H,W]
        lookup_features: torch.Tensor,    # [B,N,C,H,W]
        poses: torch.Tensor,              # [B,N,4,4]
        K_feat: torch.Tensor,             # [B,3,3]
        depth_bins: torch.Tensor          # [D]
    ):
        B, C, H, W = current_features.shape
        _, N, _, _, _ = lookup_features.shape
        D = int(depth_bins.numel())
        dev = current_features.device

        # Accumulators
        cv = torch.zeros(B, D, H, W, device=dev, dtype=current_features.dtype)
        counts = torch.zeros(B, D, H, W, device=dev, dtype=current_features.dtype)

        # Cached UV1 grid and K/K^-1 in fp32 for stability
        uv1 = self._get_uv1(H, W, dev, torch.float32)        # [3, HW]
        K3 = K_feat.to(dtype=torch.float32)
        K_inv = torch.linalg.inv(K3)                          # [B,3,3]

        # Expand across depth
        K_inv_exp = K_inv.unsqueeze(1).expand(B, D, 3, 3).reshape(B * D, 3, 3)
        K_exp = K3.unsqueeze(1).expand(B, D, 3, 3).reshape(B * D, 3, 3)

        uv1_exp = uv1.unsqueeze(0).unsqueeze(0).expand(B, D, 3, H * W).reshape(B * D, 3, H * W)
        depth_vals = depth_bins.to(device=dev, dtype=torch.float32).view(1, D, 1, 1)
        depth_scale = depth_vals.expand(B, D, 1, 1).reshape(B * D, 1, 1)

        # Rays in current cam for each depth
        rays = torch.bmm(K_inv_exp, uv1_exp) * depth_scale       # [B*D,3,HW]
        rays = rays.transpose(1, 2)                               # [B*D,HW,3]
        ones = torch.ones(rays.shape[0], rays.shape[1], 1, device=dev, dtype=torch.float32)
        points_h = torch.cat([rays, ones], dim=2)                 # [B*D,HW,4]

        for n in range(N):
            lookup = lookup_features[:, n]                         # [B,C,H,W]
            T_cl = poses[:, n].to(dtype=torch.float32)             # [B,4,4]
            T_exp = T_cl.unsqueeze(1).expand(-1, D, -1, -1).reshape(B * D, 4, 4)

            # Transform to lookup camera
            Xl_h = torch.bmm(points_h, T_exp.transpose(-2, -1))    # [B*D,HW,4]
            Xl = Xl_h[:, :, :3]                                    # [B*D,HW,3]
            Z = Xl[:, :, 2:3]                                      # [B*D,HW,1]
            z_valid = (Z > self.eps).to(torch.float32)

            # Project with K
            p_img = torch.bmm(Xl, K_exp.transpose(-2, -1))         # [B*D,HW,3]
            uv = p_img[:, :, :2] / torch.clamp(Z, min=self.eps)    # [B*D,HW,2]

            # To grid [-1,1] with align_corners=True
            grid = uv.reshape(B * D, H, W, 2)
            grid_x = grid[..., 0] / (W - 1) * 2.0 - 1.0
            grid_y = grid[..., 1] / (H - 1) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)           # [B*D,H,W,2]

            # In-bounds mask (avoid clamping)
            inb = (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)

            # Optional extra border erosion in pixel coords
            if self.edge_border > 0:
                x_pix = (grid_x + 1.0) * 0.5 * (W - 1)
                y_pix = (grid_y + 1.0) * 0.5 * (H - 1)
                b = float(self.edge_border)
                erode = (x_pix >= b) & (x_pix <= (W - 1 - b)) & (y_pix >= b) & (y_pix <= (H - 1 - b))
                inb = inb & erode

            # Expand features across depth and sample
            feats_exp = lookup.unsqueeze(1).expand(-1, D, -1, -1, -1).reshape(B * D, C, H, W)
            warped = self._grid_sample_chunked(
                feats_exp, grid.to(dtype=feats_exp.dtype), self.chunk_size
            )  # [B*D,C,H,W]

            # L1 cost per bin with numerical stability
            curr_exp = current_features.unsqueeze(2).expand(-1, -1, D, -1, -1).permute(0, 2, 1, 3, 4)
            curr_exp = curr_exp.reshape(B * D, C, H, W)
            # Clamp differences to prevent extreme values
            diffs = torch.abs(warped - curr_exp).mean(dim=1)       # [B*D,H,W]
            diffs = torch.clamp(diffs, min=0.0, max=10.0)  # Prevent extreme cost values

            valid = (inb & (z_valid.reshape(B * D, H, W) > 0)).to(diffs.dtype)

            diffs = (diffs * valid).reshape(B, D, H, W)
            valid = valid.reshape(B, D, H, W)
            cv = cv + diffs
            counts = counts + valid

        # Average over valid contributions with better numerical stability
        cv = cv / torch.clamp(counts, min=1.0)

        # Fill missing bins with a more stable approach
        missing = (counts == 0)
        if missing.any():
            # Use mean of valid bins instead of max to avoid extreme values
            valid_cv = cv * (counts > 0).float()
            mean_per_pix = valid_cv.sum(dim=1, keepdim=True) / torch.clamp(counts.sum(dim=1, keepdim=True), min=1.0)
            cv = torch.where(missing, mean_per_pix, cv)

        # Coverage ratio = fraction of bins observed
        observed = (counts > 0).to(cv.dtype).sum(dim=1)          # [B,H,W]
        coverage = torch.clamp(observed / float(self.num_depth_bins), min=1e-6, max=1.0)

        return cv, coverage

    # -------------------------- Utilities --------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization for better gradient flow with ReLU/GELU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_depth_bins(self, device, dtype):
        if self.depth_bin_mode == "lin":
            bins = torch.linspace(self.depth_min, self.depth_max, self.num_depth_bins, device=device, dtype=dtype)
        elif self.depth_bin_mode == "log":
            bins = torch.logspace(np.log10(self.depth_min), np.log10(self.depth_max), self.num_depth_bins, device=device, dtype=dtype)
        else:  # "inv" (default)
            inv = torch.linspace(1.0 / self.depth_max, 1.0 / self.depth_min, self.num_depth_bins, device=device, dtype=dtype)
            bins = 1.0 / inv
        return bins

    def _get_uv1(self, H, W, device, dtype):
        key = (H, W, dtype, str(device))  # Convert device to string for consistent hashing
        if key in self._uv1_cache:
            return self._uv1_cache[key]
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        ones = torch.ones_like(x)
        uv1 = torch.stack([x, y, ones], dim=0).reshape(3, H * W)  # [3, HW]
        self._uv1_cache[key] = uv1
        return uv1

    @staticmethod
    def _grid_sample_chunked(feats_exp, grid, chunk_size: int):
        if chunk_size is None or chunk_size <= 0:
            return F.grid_sample(
                feats_exp, grid,
                mode='bilinear', padding_mode='border', align_corners=True
            )
        outs = []
        for i in range(0, feats_exp.shape[0], chunk_size):
            outs.append(
                F.grid_sample(
                    feats_exp[i:i + chunk_size],
                    grid[i:i + chunk_size],
                    mode='bilinear', padding_mode='border', align_corners=True
                )
            )
        return torch.cat(outs, dim=0)
