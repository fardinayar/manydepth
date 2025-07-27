import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for spatial data"""
    def __init__(self, dim, max_height=512, max_width=512):
        super().__init__()
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        
        # Create frequency bases for height and width dimensions
        half_dim = dim // 4  # Split into 4 parts for 2D (h_cos, h_sin, w_cos, w_sin)
        
        # Frequency calculation similar to standard RoPE
        freqs = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        
        # Pre-compute position encodings for maximum dimensions
        h_pos = torch.arange(max_height).float().unsqueeze(1)  # [max_height, 1]
        w_pos = torch.arange(max_width).float().unsqueeze(1)   # [max_width, 1]
        
        h_freqs = h_pos * freqs.unsqueeze(0)  # [max_height, half_dim]
        w_freqs = w_pos * freqs.unsqueeze(0)  # [max_width, half_dim]
        
        # Register as buffers so they're moved to device with module
        self.register_buffer('h_cos', h_freqs.cos())
        self.register_buffer('h_sin', h_freqs.sin())
        self.register_buffer('w_cos', w_freqs.cos())
        self.register_buffer('w_sin', w_freqs.sin())
        
    def forward(self, x, height, width):
        """
        Apply 2D RoPE to input tensor
        x: [B, H*W, dim] - flattened spatial features
        height, width: spatial dimensions
        """
        batch_size, seq_len, dim = x.shape
        assert seq_len == height * width
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
        
        quarter_dim = dim // 4
        
        # Create position grids
        h_indices = torch.arange(height, device=x.device).repeat_interleave(width)  # [H*W]
        w_indices = torch.arange(width, device=x.device).repeat(height)             # [H*W]
        
        # Get position encodings (cast to tensor to satisfy linter)
        h_cos_tensor = torch.as_tensor(self.h_cos)
        h_sin_tensor = torch.as_tensor(self.h_sin)
        w_cos_tensor = torch.as_tensor(self.w_cos)
        w_sin_tensor = torch.as_tensor(self.w_sin)
        
        h_cos = h_cos_tensor[h_indices, :quarter_dim]  # [H*W, quarter_dim]
        h_sin = h_sin_tensor[h_indices, :quarter_dim]  # [H*W, quarter_dim]
        w_cos = w_cos_tensor[w_indices, :quarter_dim]  # [H*W, quarter_dim]
        w_sin = w_sin_tensor[w_indices, :quarter_dim]  # [H*W, quarter_dim]
        
        # Split input into 4 parts for 2D rotation
        x1 = x[..., :quarter_dim]                    # Height cos component
        x2 = x[..., quarter_dim:quarter_dim*2]       # Height sin component  
        x3 = x[..., quarter_dim*2:quarter_dim*3]     # Width cos component
        x4 = x[..., quarter_dim*3:]                  # Width sin component
        
        # Apply 2D rotations
        # Height rotation
        x1_rot = x1 * h_cos - x2 * h_sin
        x2_rot = x1 * h_sin + x2 * h_cos
        
        # Width rotation  
        x3_rot = x3 * w_cos - x4 * w_sin
        x4_rot = x3 * w_sin + x4 * w_cos
        
        # Concatenate rotated components
        return torch.cat([x1_rot, x2_rot, x3_rot, x4_rot], dim=-1)

class ViewEmbedding(nn.Module):
    """Learnable view embeddings to distinguish between different frames/views"""
    def __init__(self, embed_dim, num_views=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        
        # Learnable view embeddings
        self.view_embeddings = nn.Parameter(torch.randn(num_views, embed_dim) * 0.02)
        
    def forward(self, x, view_ids):
        """
        Add view embeddings to input features
        x: [B, N, embed_dim] - input features
        view_ids: [B] or int - view identifier for each batch item (0 for x1, 1 for x2)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        if isinstance(view_ids, int):
            # Single view ID for all batch items
            view_embed = self.view_embeddings[view_ids]  # [embed_dim]
            view_embed = view_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
            view_embed = view_embed.expand(batch_size, seq_len, -1)  # [B, N, embed_dim]
        else:
            # Different view IDs for each batch item
            view_embed = self.view_embeddings[view_ids]  # [B, embed_dim]
            view_embed = view_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [B, N, embed_dim]
        
        return x + view_embed

class MultiFrameFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, matching_height, matching_width, 
                 num_heads=4, dropout=0.2, use_rope=True, use_view_embedding=True, 
                 neighborhood_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_view_embedding = use_view_embedding
        self.neighborhood_size = neighborhood_size  # n for n×n neighborhood, None for global attention

        # Project inputs to output dimension
        self.input_proj_x1 = nn.Linear(input_dim, output_dim)
        self.input_proj_x2 = nn.Linear(input_dim, output_dim)
        
        # Use PyTorch's built-in MultiheadAttention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        
        # Optional 2D Rotary Position Embedding for spatial awareness
        if self.use_rope:
            if output_dim % 4 != 0:
                raise ValueError(f"output_dim ({output_dim}) must be divisible by 4 when using RoPE")
            self.rope_2d = RoPE2D(
                dim=output_dim,
                max_height=max(matching_height, 64),  # Set reasonable maximums
                max_width=max(matching_width, 64)
            )
        
        # Optional view embeddings to distinguish between frames
        if self.use_view_embedding:
            self.view_embedding = ViewEmbedding(embed_dim=output_dim, num_views=2)
        
        # Layer normalization and feed-forward
        self.layer_norm_1 = nn.LayerNorm(output_dim)
        self.layer_norm_2 = nn.LayerNorm(output_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout)
        )
        
        # Create attention mask for neighborhood constraint
        self.attn_mask = self.create_neighborhood_mask(
            self.matching_height, 
            self.matching_width, 
            self.neighborhood_size, 
        )

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
        mask = torch.ones(total_positions, 2 * total_positions, dtype=torch.bool)
        
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
                        # Unmask for x2 (second half of key/value)  
                        mask[query_idx, neighbor_idx + total_positions] = False
        
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
        
        # Project to output dimension
        x1_proj = self.input_proj_x1(x1)  # [B, N, output_dim]
        x2_proj = self.input_proj_x2(x2)  # [B, N, output_dim]
        
        # Apply view embeddings if enabled
        if self.use_view_embedding:
            x1_proj = self.view_embedding(x1_proj, view_ids=0)  # View 0 for first frame
            x2_proj = self.view_embedding(x2_proj, view_ids=1)  # View 1 for second frame
        
        # Apply 2D RoPE for spatial position encoding if enabled
        if self.use_rope:
            x1_rope = self.rope_2d(x1_proj, self.matching_height, self.matching_width)
            x2_rope = self.rope_2d(x2_proj, self.matching_height, self.matching_width)
        else:
            x1_rope = x1_proj
            x2_rope = x2_proj
        
        # Concatenate x1 and x2 as keys and values for cross-attention
        # x1 attends to both itself and x2
        keys_values = torch.cat([x1_rope, x2_rope], dim=1)  # [B, 2*N, output_dim]
        
        
        # Use PyTorch's MultiheadAttention
        # query: x1, key/value: [x1, x2]
        attn_output, _ = self.multihead_attention(
            query=x1_rope,           # [B, N, output_dim]
            key=keys_values,         # [B, 2*N, output_dim]  
            value=keys_values,       # [B, 2*N, output_dim]
            attn_mask=self.attn_mask.to(x1_rope.device),     # [N, 2*N] attention mask
            need_weights=False       # Don't return attention weights for efficiency
        )
        
        # Residual connection and layer norm
        x1_updated = self.layer_norm_1(attn_output + x1_proj)
        
        # Feed-forward network
        ff_output = self.feed_forward(x1_updated)
        x1_final = self.layer_norm_2(ff_output + x1_updated)
        
        return x1_final + x1

