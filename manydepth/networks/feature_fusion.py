import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth_anything_v2.dinov2_layers.drop_path import DropPath


class MultiFrameFeatureFusion(nn.Module):
    def __init__(self, input_dim, matching_height, matching_width, 
                 num_heads=4, dropout=0.1, drop_path=0.0,
                 neighborhood_size=15,
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
            #attn_mask=attn_mask,      # [N, 2*N] or None
            need_weights=False
        )
        x = x1 + self.drop_path_attn(attn_output)
        
        # Pre-LN Feed-forward
        x1_final = x + self.drop_path_ffn(self.feed_forward(self.ffn_norm(x)))
        
        return x1_final
