import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth_anything_v2.dinov2_layers.drop_path import DropPath


class _LoRAAdapter(nn.Module):
    """Low-rank adapter: out = up(down(x)) * (alpha / rank)."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / float(rank)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scaling


class MultiFrameFeatureFusion(nn.Module):
    """
    Cross-frame attention block: current-frame features are queries,
    lookup-frame features are keys and values.

    Per-scale LoRA adapters on all attention projections (Q, K, V, Out)
    and both FFN layers give each pyramid scale its own matching behaviour
    without duplicating the full weight set.

    Args:
        input_dim: Channel dimension of each frame's feature map.
        matching_height / matching_width: Spatial size of the flattened grid.
        num_heads: Multi-head attention heads.
        dropout: Dropout on attention weights and FFN.
        drop_path: Stochastic depth probability.
        neighborhood_size: (h, w) tuple, single int (square), or None (global).
        temporal_fusion: If True, K/V come from the lookup frame; if False,
            self-attention on the current frame only.
        num_register_tokens: Learnable register tokens appended to K/V.
        num_scales: Number of LoRA adapter sets (one per pyramid scale).
        lora_rank: LoRA rank; 0 / None disables LoRA.
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
        lora_alpha: float = 4.0,
        residual_scale_init: float = 0.01,
    ):
        super().__init__()

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.input_dim = input_dim
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.temporal_fusion = temporal_fusion
        self.num_register_tokens = num_register_tokens
        self.num_scales = int(num_scales)
        self.lora_rank = int(lora_rank) if (lora_rank and int(lora_rank) > 0) else None
        self.lora_alpha = float(lora_alpha) if lora_alpha else 1.0
        self.residual_scale_init = float(residual_scale_init)

        # Base attention projections (shared across all scales)
        self.q_proj   = nn.Linear(input_dim, input_dim)
        self.k_proj   = nn.Linear(input_dim, input_dim)
        self.v_proj   = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Register tokens (optional attention sinks)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.normal(0, 0.02, (1, num_register_tokens, input_dim))
            )
        else:
            self.register_tokens = None

        # Per-scale LoRA on Q, K, V, Out and both FFN layers
        if self.lora_rank is not None:
            r, a, d, h = self.lora_rank, self.lora_alpha, input_dim, input_dim * 2

            def _adapters(in_d, out_d):
                return nn.ModuleList(
                    [_LoRAAdapter(in_d, out_d, r, a) for _ in range(self.num_scales)]
                )

            self.lora_q    = _adapters(d, d)
            self.lora_k    = _adapters(d, d)
            self.lora_v    = _adapters(d, d)
            self.lora_out  = _adapters(d, d)
            self.lora_ffn1 = _adapters(d, h)
            self.lora_ffn2 = _adapters(h, d)
        else:
            self.lora_q = self.lora_k = self.lora_v = self.lora_out = None
            self.lora_ffn1 = self.lora_ffn2 = None

        # Pre-LN + FFN
        self.attn_norm   = nn.LayerNorm(input_dim)
        self.ffn_norm    = nn.LayerNorm(input_dim)
        self.ffn_fc1     = nn.Linear(input_dim, input_dim * 2)
        self.ffn_act     = nn.GELU()
        self.ffn_fc2     = nn.Linear(input_dim * 2, input_dim)
        self.ffn_dropout = nn.Dropout(dropout)

        self.drop_path_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_ffn  = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn_residual_scale = nn.Parameter(
            torch.full((self.num_scales,), self.residual_scale_init)
        )
        self.ffn_residual_scale = nn.Parameter(
            torch.full((self.num_scales,), self.residual_scale_init)
        )

        # Pre-computed neighbourhood mask [N, N], True = blocked
        _mask = self._make_neighborhood_mask(matching_height, matching_width, neighborhood_size)
        if _mask is not None:
            self.register_buffer("attn_mask", _mask, persistent=False)
        else:
            self.attn_mask = None

    # ------------------------------------------------------------------
    @staticmethod
    def _make_neighborhood_mask(height, width, neighborhood_size):
        """Vectorised rectangular neighbourhood mask. True = cannot attend."""
        if neighborhood_size is None:
            return None
        nh, nw = (
            (neighborhood_size, neighborhood_size)
            if isinstance(neighborhood_size, int)
            else neighborhood_size
        )
        hy = torch.arange(height)
        wx = torch.arange(width)
        # Row / col index for every spatial position
        qh = hy.unsqueeze(1).expand(height, width).reshape(-1)  # [N]
        qw = wx.unsqueeze(0).expand(height, width).reshape(-1)  # [N]
        dh = (qh.unsqueeze(1) - qh.unsqueeze(0)).abs()           # [N, N]
        dw = (qw.unsqueeze(1) - qw.unsqueeze(0)).abs()           # [N, N]
        return (dh > nh // 2) | (dw > nw // 2)                   # True = blocked

    def forward(self, input, scale_index: int = None):
        """
        Args:
            input: [B, N, 2*input_dim] -- concatenated current (x1) and lookup (x2)
                features.
            scale_index: Selects per-scale LoRA adapters (0-indexed).

        Returns:
            out [B, N, input_dim].
        """
        x1 = input[:, :, :self.input_dim]   # current frame  [B, N, C]
        x2 = input[:, :, self.input_dim:]   # lookup frame   [B, N, C]
        b, n, _ = x1.shape
        assert n == self.matching_height * self.matching_width, \
            f"Expected N={self.matching_height * self.matching_width}, got {n}"

        idx = 0 if scale_index is None else int(scale_index)

        # Pre-LN inputs
        q_in  = self.attn_norm(x1)
        kv_in = self.attn_norm(x2) if self.temporal_fusion else q_in

        # Append register tokens before K/V projection so they live in the same
        # projected space as lookup-frame keys and values.
        if self.temporal_fusion and self.register_tokens is not None:
            reg = self.attn_norm(self.register_tokens.expand(b, -1, -1))  # [B, R, C]
            kv_in = torch.cat([kv_in, reg], dim=1)                        # [B, N+R, C]

        # Projected Q, K, V with per-scale LoRA delta
        def _proj(base, lora, x):
            out = base(x)
            if lora is not None:
                out = out + lora[idx](x)
            return out

        q = _proj(self.q_proj, self.lora_q, q_in)    # [B, N,     C]
        k = _proj(self.k_proj, self.lora_k, kv_in)   # [B, N(+R), C]
        v = _proj(self.v_proj, self.lora_v, kv_in)   # [B, N(+R), C]

        nk = k.shape[1]  # N  or  N+R

        # Build boolean attention mask [N, nk], True = blocked.
        if self.attn_mask is not None:
            mask = self.attn_mask  # [N, N]
            if nk > n:
                # Allow attending freely to register tokens
                extra = mask.new_zeros(n, nk - n)
                mask = torch.cat([mask, extra], dim=1)  # [N, nk]
            mask = mask.unsqueeze(0).unsqueeze(1).to(q.device)
        else:
            mask = None

        # Reshape to multi-head layout
        hd = self.head_dim
        q = q.view(b, n,  self.num_heads, hd).transpose(1, 2)   # [B, H, N,  hd]
        k = k.view(b, nk, self.num_heads, hd).transpose(1, 2)   # [B, H, nk, hd]
        v = v.view(b, nk, self.num_heads, hd).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)  # [B, H, N, nk]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)   # [B, H, N, nk]
        attn_weights = self.attn_dropout(attn_weights)

        attn_out_raw = torch.matmul(attn_weights, v)                          # [B, H, N, hd]
        attn_out_raw = attn_out_raw.transpose(1, 2).contiguous().view(b, n, self.input_dim)

        # Output projection with per-scale LoRA
        attn_out = self.out_proj(attn_out_raw)
        if self.lora_out is not None:
            attn_out = attn_out + self.lora_out[idx](attn_out_raw)

        attn_scale = self.attn_residual_scale[idx].view(1, 1, 1)
        x = x1 + self.drop_path_attn(attn_out * attn_scale)

        # Pre-LN FFN with per-scale LoRA
        y = self.ffn_norm(x)
        y1 = self.ffn_fc1(y)
        if self.lora_ffn1 is not None:
            y1 = y1 + self.lora_ffn1[idx](y)
        y1 = self.ffn_act(y1)
        y2 = self.ffn_fc2(y1)
        if self.lora_ffn2 is not None:
            y2 = y2 + self.lora_ffn2[idx](y1)
        y2 = self.ffn_dropout(y2)
        ffn_scale = self.ffn_residual_scale[idx].view(1, 1, 1)
        out = x + self.drop_path_ffn(y2 * ffn_scale)

        return out
