import torch
import torch.nn as nn
import torch.nn.functional as F

def init_t_xy(end_x: int, end_y: int):
    """Initialize 2D position indices for axial RoPE"""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    """Compute axial complex exponentials for 2D RoPE using complex numbers"""
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def apply_axial_rope(x: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply axial RoPE using complex number operations for efficiency"""
    # Reshape to complex representation: [B, N, dim] -> [B, N, dim//2, 2] -> [B, N, dim//2]
    # Ensure tensor is contiguous for view_as_complex
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
    x_complex = torch.view_as_complex(x_reshaped)
    
    # Apply rotation via complex multiplication
    freqs_cis = freqs_cis.to(x.device)
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real representation
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)

class RoPE2D(nn.Module):
    """2D Axial Rotary Position Embedding using complex number operations"""
    def __init__(self, dim, theta: float = 100.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        assert dim % 4 == 0, "Dimension must be divisible by 4 for axial RoPE"
        
        # Cache for computed frequencies to avoid recomputation
        self._cached_freqs = {}
        
    def forward(self, x, height, width):
        """
        Apply axial 2D RoPE to input tensor
        x: [B, H*W, dim] - flattened spatial features
        height, width: spatial dimensions
        """
        batch_size, seq_len, dim = x.shape
        assert seq_len == height * width
        assert dim == self.dim, f"Input dim {dim} doesn't match expected {self.dim}"
        
        # Use cache for frequencies if available
        cache_key = (height, width, x.device)
        if cache_key not in self._cached_freqs:
            freqs_cis = compute_axial_cis(dim, width, height, self.theta)
            self._cached_freqs[cache_key] = freqs_cis
        else:
            freqs_cis = self._cached_freqs[cache_key]
        
        return apply_axial_rope(x, freqs_cis)

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
    def __init__(self, input_dim, matching_height, matching_width, 
                 num_heads=4, dropout=0.2, use_rope=False, use_view_embedding=False, 
                 neighborhood_size=5):
        super().__init__()
        self.input_dim = input_dim
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_view_embedding = use_view_embedding
        self.neighborhood_size = neighborhood_size  # n for n×n neighborhood, None for global attention

        # Use PyTorch's built-in MultiheadAttention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        
        # Optional 2D Rotary Position Embedding for spatial awareness
        if self.use_rope:
            if input_dim % 4 != 0:
                raise ValueError(f"input_dim ({input_dim}) must be divisible by 4 when using RoPE")
            self.rope_2d = RoPE2D(dim=input_dim)
        
        # Optional view embeddings to distinguish between frames
        if self.use_view_embedding:
            self.view_embedding = ViewEmbedding(embed_dim=input_dim, num_views=2)
        
        # Pre-LN: normalize before attention and before feed-forward
        self.attn_norm = nn.LayerNorm(input_dim)
        self.ffn_norm = nn.LayerNorm(input_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(dropout)
        )
        
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
        
        # Inputs to the block (no projection)
        x1_proj = x1  # [B, N, input_dim]
        x2_proj = x2  # [B, N, input_dim]
        
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
        
        # Pre-LN Attention
        q = self.attn_norm(x1_rope)
        k = self.attn_norm(x2_rope)
        v = self.attn_norm(x2_rope)
        attn_mask = self.attn_mask if hasattr(self, "attn_mask") else None
        attn_output, _ = self.multihead_attention(
            query=q,                  # [B, N, input_dim]
            key=k,          # [B, 2*N, input_dim]
            value=v,             # [B, 2*N, input_dim]
            attn_mask=attn_mask,      # [N, 2*N] or None
            need_weights=False
        )
        x = x1_proj + attn_output
        
        # Pre-LN Feed-forward
        x1_final = x + self.feed_forward(self.ffn_norm(x))
        
        return x1_final
