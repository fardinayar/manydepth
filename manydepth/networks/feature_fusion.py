import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth_anything_v2.dinov2_layers.drop_path import DropPath
import numpy as np


class MultiFrameFeatureFusion(nn.Module):
    def __init__(self, input_dim, matching_height, matching_width, 
                 num_heads=4, dropout=0.1, drop_path=0.0,
                 neighborhood_size=(3, 15),  # (height, width) for non-square neighborhoods
                 temporal_fusion=True,
                 num_register_tokens=8):  # Add register tokens for attention sink
        super().__init__()
        assert isinstance(neighborhood_size, tuple)
        self.input_dim = input_dim
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.num_heads = num_heads
        self.temporal_fusion = temporal_fusion
        self.neighborhood_size = neighborhood_size  # (height, width) for non-square neighborhoods, None for global attention
        self.drop_path = drop_path
        self.num_register_tokens = num_register_tokens

        # Use PyTorch's built-in MultiheadAttention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        
        # Register tokens: learnable tokens that act as attention sinks
        # Initialize with small random values
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.normal(0, 1, (1, num_register_tokens, input_dim)))
        else:
            self.register_tokens = None
        

        
        # Pre-LN: normalize before attention and before feed-forward
        self.attn_norm = nn.LayerNorm(input_dim)
        #self.attn_norm_prev = nn.LayerNorm(input_dim)
        self.ffn_norm = nn.LayerNorm(input_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(dropout)
        )
        
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
            mask: [height*width, height*width] boolean mask where True means MASKED (not attend)
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

    def forward(self, input):
        """
        input: (B, N, input_dim*2) where N = matching_height * matching_width
        x1 and x2 are expected to be of shape (B, N, input_dim)
        Only x1 gets updated, attending to both x1 and x2
        """
        x1, x2 = input[:, :, :self.input_dim], input[:, :, self.input_dim:]
        b, n, _ = x1.shape

        
        # Verify spatial dimensions
        assert n == self.matching_height * self.matching_width, f"Expected N={self.matching_height * self.matching_width}, got N={n}"
        
        
        # Pre-LN Attention
        q = self.attn_norm(x1)
        if self.temporal_fusion:
            k_prev = self.attn_norm(x2)
            v_prev = self.attn_norm(x2)
            #k = self.attn_norm(x1)
            #v = self.attn_norm(x1)
            k = k_prev#torch.cat((k_prev, k), 1)
            v = v_prev#torch.cat((v_prev, v), 1)
            # attn_mask = (
            #     torch.cat((self.attn_mask, self.attn_mask), 1)
            #     if self.attn_mask is not None
            #     else None
            # )
            attn_mask = self.attn_mask
        else:
            k = self.attn_norm(x1)
            v = self.attn_norm(x1)
            attn_mask = self.attn_mask

        # Concatenate register tokens to keys and values
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

        attn_output, _ = self.multihead_attention(
            query=q,                  # [B, N, input_dim]
            key=k,          # [B, N+num_reg, input_dim] (or 2*N+num_reg with temporal)
            value=v,             # [B, N+num_reg, input_dim]
            attn_mask=attn_mask,      # [N, N+num_reg] or None
            need_weights=False
        )
        # Note: register tokens are automatically handled - they only affect attention computation
        # The output shape is [B, N, input_dim] (queries determine output length)
        x = x1 + self.drop_path_attn(attn_output)
        
        # Pre-LN Feed-forward
        x1_final = x + self.drop_path_ffn(self.feed_forward(self.ffn_norm(x)))
        
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
