import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth_anything_v2.dinov2_layers.drop_path import DropPath
import numpy as np


class MultiFrameFeatureFusion(nn.Module):
    def __init__(self, input_dim, matching_height, matching_width, 
                 num_heads=4, dropout=0.1, drop_path=0.0,
                 neighborhood_size=31,
                 temporal_fusion=True):
        super().__init__()
        self.input_dim = input_dim
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.num_heads = num_heads
        self.temporal_fusion = temporal_fusion
        self.neighborhood_size = neighborhood_size  # n for n×n neighborhood, None for global attention
        self.drop_path = drop_path

        # Use PyTorch's built-in MultiheadAttention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        

        
        # Pre-LN: normalize before attention and before feed-forward
        self.attn_norm = nn.LayerNorm(input_dim)
        self.attn_norm_prev = nn.LayerNorm(input_dim)
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
        
        
        
        #nn.init.constant_(self.feed_forward[2].weight, 0.02)
        #nn.init.constant_(self.feed_forward[2].bias, 0.02)
        #nn.init.constant_(self.multihead_attention.out_proj.weight, 0.02)
        #nn.init.constant_(self.multihead_attention.out_proj.bias, 0.02)
        

    def create_neighborhood_mask(self, height, width, neighborhood_size):
        """
        Create attention mask for n×n neighborhood attention.
        
        Args:
            height, width: spatial dimensions
            neighborhood_size: size of neighborhood (n for n×n)
            device: torch device
            
        Returns:
            mask: [height*width, 2*height*width] boolean mask where True means MASKED (not attend)
        """
        if neighborhood_size is None:
            return None
            
        total_positions = height * width
        # Create mask for [query_positions, key_positions] where key_positions = [x1_positions, x2_positions]
        mask = torch.ones(total_positions, total_positions, dtype=torch.bool)
        
        half_size = neighborhood_size // 2
        
        for h in range(height):
            for w in range(width):
                query_idx = h * width + w
                
                # Define neighborhood bounds
                h_min = max(0, h - half_size)
                h_max = min(height, h + half_size + 1)
                w_min = max(0, w - half_size)  
                w_max = min(width, w + half_size + 1)
                
                # Mark positions in neighborhood as unmasked (False) for both x1 and x2
                for nh in range(h_min, h_max):
                    for nw in range(w_min, w_max):
                        neighbor_idx = nh * width + nw
                        
                        # Unmask for x1 (first half of key/value)
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
            k_prev = self.attn_norm_prev(x2)
            v_prev = self.attn_norm_prev(x2)
            k = self.attn_norm(x1)
            v = self.attn_norm(x1)
            k = torch.cat((k_prev, k), 1)
            v = torch.cat((v_prev, v), 1)
            attn_mask = (
                torch.cat((self.attn_mask, self.attn_mask), 1)
                if self.attn_mask is not None
                else None
            )
        else:
            k = self.attn_norm(x1)
            v = self.attn_norm(x1)
            attn_mask = self.attn_mask


        attn_output, _ = self.multihead_attention(
            query=q,                  # [B, N, input_dim]
            key=k,          # [B, 2*N, input_dim]
            value=v,             # [B, 2*N, input_dim]
            attn_mask=attn_mask,      # [N, 2*N] or None
            need_weights=False
        )
        x = x1 + self.drop_path_attn(attn_output)
        
        # Pre-LN Feed-forward
        x1_final = x + self.drop_path_ffn(self.feed_forward(self.ffn_norm(x)))
        
        return x1_final


class CostVolumeFeatureFusion(nn.Module):
    """
    Cost Volume Feature Fusion module that uses poses from posenet to build cost volumes
    and fuse multi-frame features for depth estimation.
    """
    def __init__(self, input_dim, matching_height, matching_width, 
                 num_depth_bins=32, depth_min=0.1, depth_max=80.0,
                 dropout=0.1, eps=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.eps = eps  # Numerical stability parameter
        
        # Reduce convolution after cost volume: concat current features [B, C, H, W]
        # with cost volume [B, D, H, W], then reduce back to C channels
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(input_dim + num_depth_bins, input_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the cost volume and fusion layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Use Xavier init which generally works well with GELU activations
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def build_cost_volume(self, current_features, lookup_features, poses=None, intrinsics=None):
        """
        Build cost volume from current and lookup frame features using poses.
        
        Args:
            current_features: [B, C, H, W] current frame features
            lookup_features: [B, N, C, H, W] lookup frame features (N = num_matching_frames)
            poses: [B, N, 4, 4] relative poses from current to lookup frames (required)
            intrinsics: [B, 3, 3] camera intrinsics at feature resolution (required)
            
        Returns:
            cost_volume: [B, C, D, H, W] cost volume
        """
        # Input validation
        if current_features.dim() != 4:
            raise ValueError(f"current_features must be 4D tensor [B, C, H, W], got {current_features.dim()}D")
        if lookup_features.dim() != 5:
            raise ValueError(f"lookup_features must be 5D tensor [B, N, C, H, W], got {lookup_features.dim()}D")
        if poses is None:
            raise ValueError("poses are required for cost volume warping")
        if poses.dim() != 4:
            raise ValueError(f"poses must be 4D tensor [B, N, 4, 4], got {poses.dim()}D")
        if intrinsics is None:
            raise ValueError("intrinsics are required for cost volume warping")
        if intrinsics.dim() != 3:
            raise ValueError(f"intrinsics must be 3D tensor [B, 3, 3] or [B, 4, 4], got {intrinsics.dim()}D")
        
        B, C, H, W = current_features.shape
        _, N, _, _, _ = lookup_features.shape
        
        if N == 0:
            raise ValueError("lookup_features cannot have 0 frames")
        
        # Create depth bins
        depth_bins = torch.linspace(self.depth_min, self.depth_max, self.num_depth_bins, 
                                  device=current_features.device, dtype=torch.float32)
        
        # Initialize cost volume [B, D, H, W] and counts [B, D, H, W]
        cost_volume = torch.zeros(B, self.num_depth_bins, H, W,
                                  device=current_features.device, dtype=current_features.dtype)
        counts = torch.zeros(B, self.num_depth_bins, H, W,
                             device=current_features.device, dtype=current_features.dtype)
        
        # Build cost volume for each lookup frame
        for n in range(N):
            lookup_feat = lookup_features[:, n]  # [B, C, H, W]
            
            # Check if channel dimensions match
            if current_features.shape[1] != lookup_feat.shape[1]:
                raise ValueError(f"Channel dimension mismatch: current_features has {current_features.shape[1]} channels, "
                               f"but lookup_feat has {lookup_feat.shape[1]} channels")

            pose = poses[:, n]
            warped_features, grid, z_valid_mask = self._warp_features(
                lookup_feat, pose, depth_bins, H, W, intrinsics=intrinsics
            )

            # L1 cost between warped lookup and current features
            diffs = torch.abs(warped_features - current_features.unsqueeze(2)).mean(dim=1)  # [B, D, H, W]

            # Build edge mask using projected grid to ignore borders
            # grid is [B, D, H, W, 2] in [-1, 1]; convert to pixel coords
            grid_x = grid[..., 0]
            grid_y = grid[..., 1]
            x_vals = (grid_x / 2.0 + 0.5) * (W - 1)
            y_vals = (grid_y / 2.0 + 0.5) * (H - 1)
            edge_mask = (x_vals >= 2.0) * (x_vals <= (W - 2)) * (y_vals >= 2.0) * (y_vals <= (H - 2))
            edge_mask = edge_mask.float()

            # Validity mask from positive depth (z > eps)
            z_valid_mask = z_valid_mask.to(diffs.dtype)

            # Current image mask to ignore its borders (due to zero padding in backbones)
            current_mask = torch.zeros(B, 1, H, W, device=current_features.device, dtype=current_features.dtype)
            if H > 4 and W > 4:
                current_mask[:, :, 2:-2, 2:-2] = 1.0
            else:
                current_mask[:, :, :, :] = 1.0
            current_mask = current_mask.expand(-1, self.num_depth_bins, -1, -1)

            valid_mask = edge_mask * current_mask * z_valid_mask  # [B, D, H, W]

            # Apply mask and integrate
            diffs = diffs * valid_mask
            cost_volume = cost_volume + diffs
            counts = counts + (valid_mask > 0).float()
        
        # Average over observed frames per bin
        cost_volume = cost_volume / (counts + 1e-7)

        # Identify missing values (no observations)
        missing_mask = (counts == 0).float()

        # Replace missing with per-pixel max over depth bins
        max_per_pixel = cost_volume.max(dim=1, keepdim=True)[0]
        cost_volume = cost_volume * (1.0 - missing_mask) + max_per_pixel * missing_mask

        # Coverage ratio: fraction of depth bins observed at least once per pixel
        observed_bins = (counts > 0).float().sum(dim=1)  # [B, H, W]
        coverage_ratio = torch.clamp(observed_bins / float(self.num_depth_bins), 0.0, 1.0)

        return cost_volume, coverage_ratio
    
    def _warp_features(self, features, pose, depth_bins, H, W, intrinsics=None):
        """
        Warp features using pose and depth hypotheses with proper camera projection.
        
        Args:
            features: [B, C, H, W] input features
            pose: [B, 4, 4] relative pose matrix
            depth_bins: [D] depth values
            H, W: spatial dimensions
            intrinsics: [B, 3, 3] camera intrinsics at feature resolution (required)
            
        Returns:
            warped_features: [B, C, D, H, W] warped features
            grid_out: [B, D, H, W, 2] sampling grid in normalized coords
            z_valid_mask: [B, D, H, W] mask where projected depth z > eps
        """
        B, C, _, _ = features.shape
        D = len(depth_bins)
        device = features.device
        compute_dtype = torch.float32
        # Default intrinsics if not provided (assume identity intrinsics)
        if intrinsics is None:
            raise ValueError("Intrinsics are required for warping features")
        else:
            K_in = intrinsics.to(device=device, dtype=compute_dtype)
            # Accept [B, 4, 4] or [B, 3, 3]
            if K_in.shape[-1] == 4:
                K3 = K_in[:, :3, :3].contiguous()
            elif K_in.shape[-1] == 3:
                K3 = K_in
            else:
                raise ValueError(f"Unsupported intrinsics shape: {K_in.shape}")

        # Use higher precision for inversion, then cast back
        K_inv = torch.inverse(K3.to(compute_dtype))

        # Create pixel coordinate grid [H, W]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=compute_dtype),
            torch.arange(W, device=device, dtype=compute_dtype),
            indexing='ij'
        )

        # Homogeneous pixel coordinates [3, H, W] -> [3, HW]
        ones_hw = torch.ones_like(x_coords)
        uv1 = torch.stack([x_coords, y_coords, ones_hw], dim=0).reshape(3, H * W)

        # Prepare depth values [D]
        depth_values = depth_bins.to(device=device, dtype=compute_dtype)

        # Expand for batch and depth: K_inv [B*D, 3, 3], uv1 [B*D, 3, HW]
        K_inv_expanded = K_inv.unsqueeze(1).expand(B, len(depth_values), 3, 3).reshape(B * D, 3, 3)
        uv1_expanded = uv1.unsqueeze(0).unsqueeze(0).expand(B, D, 3, H * W).reshape(B * D, 3, H * W)

        # Rays in camera coordinates scaled by depth: [B*D, 3, HW]
        depth_scale = depth_values.view(1, D, 1, 1).expand(B, D, 1, 1).reshape(B * D, 1, 1)
        rays = torch.bmm(K_inv_expanded, uv1_expanded) * depth_scale

        # Convert to homogeneous 4D points [B*D, HW, 4]
        rays_xyz = rays.transpose(1, 2)  # [B*D, HW, 3]
        ones_col = torch.ones(rays_xyz.shape[0], rays_xyz.shape[1], 1, device=device, dtype=compute_dtype)
        points_homo = torch.cat([rays_xyz, ones_col], dim=2)

        # Expand poses for all depth bins [B*D, 4, 4]
        pose_expanded = pose.to(compute_dtype).unsqueeze(1).expand(-1, D, -1, -1).reshape(B * D, 4, 4)

        # Transform to lookup camera coordinates [B*D, HW, 4]
        transformed = torch.bmm(points_homo, pose_expanded.transpose(-2, -1))

        # Project with intrinsics: p_img = K * X', then normalize by z
        K_expanded = K3.unsqueeze(1).expand(B, D, 3, 3).reshape(B * D, 3, 3)
        xyz = transformed[:, :, :3]  # [B*D, HW, 3]
        p_img = torch.bmm(xyz, K_expanded.transpose(-2, -1))  # [B*D, HW, 3]

        z_raw = p_img[:, :, 2:3]
        z_valid = (z_raw > self.eps).to(compute_dtype)
        z = torch.clamp(z_raw, min=self.eps)
        uv = p_img[:, :, :2] / z  # [B*D, HW, 2]

        # Reshape to grid [B*D, H, W, 2]
        projected_coords = uv.reshape(B * D, H, W, 2)

        # Normalize to [-1, 1] for grid_sample with align_corners=True
        grid_x = projected_coords[:, :, :, 0] / (W - 1) * 2.0 - 1.0
        grid_y = projected_coords[:, :, :, 1] / (H - 1) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        # Clamp grid to valid range
        grid = torch.clamp(grid, min=-1.0, max=1.0)

        # Expand features for all depth bins [B*D, C, H, W]
        features_expanded = features.unsqueeze(1).expand(-1, D, -1, -1, -1).reshape(B * D, C, H, W)

        # Sample features
        warped_features = F.grid_sample(
            features_expanded, grid.to(features.dtype),
            mode='bilinear', padding_mode='border', align_corners=True
        )

        # Reshape back to [B, C, D, H, W]
        warped_features = warped_features.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)

        # Also return grid shaped [B, D, H, W, 2]
        grid_out = grid.reshape(B, D, H, W, 2)
        z_valid_out = z_valid.reshape(B, D, H, W)

        return warped_features, grid_out, z_valid_out
    
    def forward(self, current_features, lookup_features, poses=None, intrinsics=None):
        """
        Forward pass of cost volume feature fusion.
        
        Args:
            current_features: [B, C, H, W] current frame features
            lookup_features: [B, N, C, H, W] lookup frame features
            poses: [B, N, 4, 4] relative poses (required)
            intrinsics: [B, 3, 3] camera intrinsics at feature resolution (required)
            
        Returns:
            fused_features: [B, C, H, W] fused features
        """
        # Input validation
        if current_features.dim() != 4:
            raise ValueError(f"current_features must be 4D tensor [B, C, H, W], got {current_features.dim()}D")
        if lookup_features.dim() != 5:
            raise ValueError(f"lookup_features must be 5D tensor [B, N, C, H, W], got {lookup_features.dim()}D")
        
        # Build cost volume and coverage ratio
        cost_volume, coverage_ratio = self.build_cost_volume(current_features, lookup_features, poses, intrinsics)

        # Weight cost volume by coverage ratio
        cost_volume = cost_volume * coverage_ratio.unsqueeze(1)

        # Concatenate cost volume with current features along channel dim
        concatenated = torch.cat([current_features, cost_volume], dim=1)

        # Reduce back to input_dim channels
        fused_features = self.reduce_conv(concatenated)

        # Residual connection
        fused_features = fused_features + current_features

        return fused_features
