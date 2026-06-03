import math
import warnings
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
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scaling


class MultiFrameFeatureFusion(nn.Module):
    """
    Cross-frame attention block: current-frame features are queries,
    lookup-frame features are keys and values.

    A shared lightweight temporal residual adapter. The heavy correspondence
    parameters are shared across DPT levels, while per-level LayerNorms, gates,
    and optional LoRA adapters give each level a small amount
    of specialization without duplicating full fusion blocks.

    Args:
        input_dim: Channel dimension of each frame's feature map.
        matching_height / matching_width: Spatial size of the flattened grid.
        num_heads: Multi-head attention heads.
        dropout: Dropout on attention weights.
        drop_path: Stochastic depth probability.
        neighborhood_size: (h, w) tuple, single int (square), or None (global).
        temporal_fusion: If True, K/V come from the lookup frame; if False,
            self-attention on the current frame only.
        num_register_tokens: Learnable register tokens appended to K/V.
        num_scales: Number of LoRA adapter sets (one per pyramid scale).
        qk_dim: Shared query/key projection dimension. If None, uses a small
            default based on input_dim while keeping values/output at input_dim.
        value_dim: Bottleneck value dimension used for the temporal residual.
            If None, uses a small default based on input_dim.
        attn_map_dim: Encoded local-attention-map feature dimension. Set to 0
            to disable attention-map conditioning.
        attn_map_hidden_dim: Hidden channels in the small convolutional
            attention-map encoder.
        lora_rank: LoRA rank; 0 / None disables LoRA.
        lora_alpha: LoRA scaling factor.
        separate_norms: If True, use separate LayerNorm affine parameters per scale.
        use_patch_gate: If True, learn a content-dependent scalar gate for
            every patch token.
        use_channel_gate: Optional fixed per-scale, per-channel residual gate.
            Disabled by default; the patch gate is the main confidence control.
        patch_gate_init: Initial patch-gate value in the open interval (0, 1).
        channel_gate_init: Initial channel-gate logit value.
    """

    def __init__(
        self,
        input_dim: int,
        matching_height: int,
        matching_width: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        neighborhood_size=(5, 15),
        temporal_fusion: bool = True,
        num_register_tokens: int = 0,
        num_scales: int = 4,
        qk_dim: int = None,
        value_dim: int = None,
        attn_map_dim: int = 64,
        attn_map_hidden_dim: int = 32,
        lora_rank: int = 32,
        lora_alpha: float = 4.0,
        separate_norms: bool = True,
        use_patch_gate: bool = True,
        use_channel_gate: bool = False,
        patch_gate_init: float = 0.01,
        channel_gate_init: float = 1,
    ):
        super().__init__()

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        if qk_dim is None:
            qk_dim = self._default_qk_dim(input_dim, num_heads)
        qk_dim = int(qk_dim)
        if value_dim is None:
            value_dim = self._default_value_dim(input_dim, num_heads)
        value_dim = int(value_dim)
        assert qk_dim > 0, "qk_dim must be positive"
        assert qk_dim % num_heads == 0, "qk_dim must be divisible by num_heads"
        assert value_dim > 0, "value_dim must be positive"
        assert value_dim % num_heads == 0, "value_dim must be divisible by num_heads"

        self.input_dim = input_dim
        self.qk_dim = qk_dim
        self.value_dim = value_dim
        self.attn_map_dim = int(attn_map_dim) if attn_map_dim else 0
        self.attn_map_hidden_dim = int(attn_map_hidden_dim)
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.num_heads = num_heads
        self.qk_head_dim = self.qk_dim // num_heads
        self.value_head_dim = self.value_dim // num_heads
        self.temporal_fusion = temporal_fusion
        self.num_register_tokens = num_register_tokens
        self.num_scales = int(num_scales)
        self.lora_rank = int(lora_rank) if (lora_rank and int(lora_rank) > 0) else None
        self.lora_alpha = float(lora_alpha) if lora_alpha else 1.0
        self.separate_norms = bool(separate_norms)
        self.use_patch_gate = bool(use_patch_gate)
        self.use_channel_gate = bool(use_channel_gate)
        self.patch_gate_init = float(patch_gate_init)
        self.channel_gate_init = float(channel_gate_init)
        if self.use_patch_gate and not 0.0 < self.patch_gate_init < 1.0:
            raise ValueError("patch_gate_init must be between 0 and 1")
        self.register_buffer("_patch_gate_version", torch.tensor(1, dtype=torch.int8))

        self.local_attention_size = self._normalize_neighborhood_size(neighborhood_size)
        if self.local_attention_size is None:
            self.attn_map_dim = 0

        # Base attention projections (shared across all scales). Q/K share the
        # same projection so matching happens in a single learned metric space.
        self.qk_proj  = nn.Linear(input_dim, self.qk_dim)
        self.v_down = nn.Linear(input_dim, self.value_dim)
        self.attn_dropout = nn.Dropout(dropout)

        if self.attn_map_dim > 0:
            nh, nw = self.local_attention_size
            self.local_attn_h = 2 * (nh // 2) + 1
            self.local_attn_w = 2 * (nw // 2) + 1
            attn_map_channels = self.num_heads * self.local_attn_h * self.local_attn_w
            self.attn_map_encoder = nn.Sequential(
                nn.Conv2d(attn_map_channels, self.attn_map_hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.attn_map_hidden_dim, self.attn_map_hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.attn_map_hidden_dim, self.attn_map_dim, kernel_size=1),
                nn.GELU(),
            )
            local_indices, local_valid = self._make_local_attention_index(
                matching_height, matching_width, self.local_attention_size
            )
            self.register_buffer("local_attn_indices", local_indices, persistent=False)
            self.register_buffer("local_attn_valid", local_valid, persistent=False)
            self.softargmax_dim = 0
            self.attn_map_flatten_proj = None
        else:
            self.local_attn_h = None
            self.local_attn_w = None
            self.attn_map_encoder = None
            self.softargmax_dim = 0
            self.attn_map_flatten_proj = None

        # Derived after attn_map_encoder so the dim always matches what the forward pass concatenates.
        self.residual_input_dim = self.value_dim + (
            (self.attn_map_dim + self.softargmax_dim) if self.attn_map_encoder is not None else 0
        )
        self.v_up = nn.Linear(self.residual_input_dim, input_dim)

        # Register tokens (optional attention sinks)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.normal(0, 0.02, (1, num_register_tokens, input_dim))
            )
        else:
            self.register_tokens = None

        # Per-level LoRA on the shared correspondence and residual projections.
        # The LoRA up matrices are zero-initialized, so every level starts from
        # the same shared temporal adapter and learns only a small delta.
        if self.lora_rank is not None:
            r, a, d = self.lora_rank, self.lora_alpha, input_dim

            def _adapters(in_d, out_d):
                return nn.ModuleList(
                    [_LoRAAdapter(in_d, out_d, r, a) for _ in range(self.num_scales)]
                )

            self.lora_qk   = _adapters(d, self.qk_dim)
            self.lora_v_down = _adapters(d, self.value_dim)
            self.lora_v_up = _adapters(self.residual_input_dim, d)
        else:
            self.lora_qk = self.lora_v_down = self.lora_v_up = None

        if self.use_patch_gate:
            gate_input_dim = input_dim * 3
            self.patch_gates = nn.ModuleList(
                [nn.Linear(gate_input_dim, 1) for _ in range(self.num_scales)]
            )
            for gate in self.patch_gates:
                nn.init.normal_(gate.weight, std=0.001)
                nn.init.constant_(
                    gate.bias,
                    0.001,
                )
        else:
            self.patch_gates = None
        self._last_patch_gate_activations = [None] * self.num_scales

        if self.use_channel_gate:
            self.channel_gates = nn.Parameter(
                torch.full((self.num_scales, input_dim), self.channel_gate_init)
            )
        else:
            self.channel_gates = None

        # Pre-LN attention. When enabled, each scale gets separate Q/KV
        # LayerNorm affine parameters while keeping the heavier projections shared.
        if self.separate_norms:
            self.attn_q_norms = nn.ModuleList(
                [nn.LayerNorm(input_dim) for _ in range(self.num_scales)]
            )
            self.attn_kv_norms = nn.ModuleList(
                [nn.LayerNorm(input_dim) for _ in range(self.num_scales)]
            )
            self.attn_norm = None
        else:
            self.attn_q_norms = None
            self.attn_kv_norms = None
            self.attn_norm = nn.LayerNorm(input_dim)

        self.drop_path_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Pre-computed neighbourhood mask [N, N], True = blocked
        _mask = self._make_neighborhood_mask(matching_height, matching_width, neighborhood_size)
        if _mask is not None:
            self.register_buffer("attn_mask", _mask, persistent=False)
        else:
            self.attn_mask = None

    # ------------------------------------------------------------------
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version_key = prefix + "_patch_gate_version"
        has_saved_patch_gates = any(
            key.startswith(prefix + "patch_gates.") for key in state_dict
        )
        if version_key not in state_dict and has_saved_patch_gates:
            state_dict[version_key] = self._patch_gate_version.new_tensor(0)
            warnings.warn(
                "Loading legacy fusion patch gates with their original unbounded "
                "behavior. Retrain the model to use bounded sigmoid gates.",
                stacklevel=2,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _default_qk_dim(input_dim, num_heads):
        target = 128 if input_dim <= 384 else 256
        qk_dim = min(input_dim, target)
        return max(num_heads, (qk_dim // num_heads) * num_heads)

    @staticmethod
    def _default_value_dim(input_dim, num_heads):
        target = 128 if input_dim <= 384 else 256
        value_dim = min(input_dim, target)
        return max(num_heads, (value_dim // num_heads) * num_heads)

    @staticmethod
    def _normalize_neighborhood_size(neighborhood_size):
        if neighborhood_size is None:
            return None
        if isinstance(neighborhood_size, int):
            return (int(neighborhood_size), int(neighborhood_size))
        return tuple(int(x) for x in neighborhood_size)

    @staticmethod
    def _make_neighborhood_mask(height, width, neighborhood_size):
        """Vectorised rectangular neighbourhood mask. True = cannot attend."""
        if neighborhood_size is None:
            return None
        nh, nw = MultiFrameFeatureFusion._normalize_neighborhood_size(neighborhood_size)
        hy = torch.arange(height)
        wx = torch.arange(width)
        # Row / col index for every spatial position
        qh = hy.unsqueeze(1).expand(height, width).reshape(-1)  # [N]
        qw = wx.unsqueeze(0).expand(height, width).reshape(-1)  # [N]
        dh = (qh.unsqueeze(1) - qh.unsqueeze(0)).abs()           # [N, N]
        dw = (qw.unsqueeze(1) - qw.unsqueeze(0)).abs()           # [N, N]
        return (dh > nh // 2) | (dw > nw // 2)                   # True = blocked

    @staticmethod
    def _make_local_attention_index(height, width, neighborhood_size):
        """Indices for the fixed-offset local attention map around each query."""
        nh, nw = MultiFrameFeatureFusion._normalize_neighborhood_size(neighborhood_size)
        y = torch.arange(height)
        x = torch.arange(width)
        qy = y.unsqueeze(1).expand(height, width).reshape(-1)
        qx = x.unsqueeze(0).expand(height, width).reshape(-1)

        offsets_y = torch.arange(-(nh // 2), nh // 2 + 1)
        offsets_x = torch.arange(-(nw // 2), nw // 2 + 1)
        oy = offsets_y.view(-1, 1).expand(-1, offsets_x.numel()).reshape(-1)
        ox = offsets_x.view(1, -1).expand(offsets_y.numel(), -1).reshape(-1)

        ty = qy.unsqueeze(1) + oy.unsqueeze(0)
        tx = qx.unsqueeze(1) + ox.unsqueeze(0)
        valid = (ty >= 0) & (ty < height) & (tx >= 0) & (tx < width)
        indices = ty.clamp(0, height - 1) * width + tx.clamp(0, width - 1)
        return indices.long(), valid

    def _q_norm(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.separate_norms:
            return self.attn_q_norms[idx](x)
        return self.attn_norm(x)

    def _kv_norm(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.separate_norms:
            return self.attn_kv_norms[idx](x)
        return self.attn_norm(x)

    def _channel_gate(self, idx: int) -> torch.Tensor:
        if self.channel_gates is None:
            return 1.0
        return (self.channel_gates[idx]).view(1, 1, self.input_dim)

    def _patch_gate(
        self,
        current: torch.Tensor,
        residual: torch.Tensor,
        idx: int,
    ) -> torch.Tensor:
        if self.patch_gates is None:
            return 1.0
        gate_input = torch.cat((current, residual, current - residual), dim=-1)
        gate = self.patch_gates[idx](gate_input)
        if self._patch_gate_version.item() < 1:
            activation = gate
        else:
            activation = (gate)
        self._last_patch_gate_activations[idx] = activation.detach()
        return activation

    def _residual_gate(
        self,
        current: torch.Tensor,
        residual: torch.Tensor,
        idx: int,
    ) -> torch.Tensor:
        return self._patch_gate(current, residual, idx) * self._channel_gate(idx)

    def _attention_map_feature(self, attn_weights: torch.Tensor, n: int) -> torch.Tensor:
        if self.attn_map_encoder is None:
            return None

        b, h, _, _ = attn_weights.shape
        spatial_attn = attn_weights[..., :n]
        gather_idx = self.local_attn_indices.view(1, 1, n, -1).expand(b, h, -1, -1)
        local_attn = spatial_attn.gather(dim=-1, index=gather_idx)
        local_attn = local_attn * self.local_attn_valid.view(1, 1, n, -1).to(dtype=local_attn.dtype)
        # local_attn: [B, H, N, local_h*local_w]

        # Cost-volume encoder: attention heads and local offsets are feature
        # channels, while the query-token grid is the convolutional spatial plane.
        cost_volume = local_attn.reshape(
            b, h, self.matching_height, self.matching_width, self.local_attn_h, self.local_attn_w
        )
        cost_volume = cost_volume.permute(0, 1, 4, 5, 2, 3).reshape(
            b,
            h * self.local_attn_h * self.local_attn_w,
            self.matching_height,
            self.matching_width,
        )
        encoded = self.attn_map_encoder(cost_volume)           # [B, attn_map_dim, matching_h, matching_w]
        encoded = encoded.flatten(2).transpose(1, 2).contiguous()

        return encoded                                         # [B, N, attn_map_dim]

    def forward(self, input, scale_index: int = None, residual_mask: torch.Tensor = None):
        """
        Args:
            input: [B, N, 2*input_dim] -- concatenated current (x1) and lookup (x2)
                features.
            scale_index: Selects per-scale LoRA adapters (0-indexed).
            residual_mask: Optional [B, 1, 1] mask for disabling temporal
                residuals when matching evidence is unavailable.

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
        q_in  = self._q_norm(x1, idx)
        kv_in = self._kv_norm(x2, idx) if self.temporal_fusion else q_in

        # Append register tokens before key projection; their values remain in
        # the same normalized feature space as lookup-frame values.
        if self.temporal_fusion and self.register_tokens is not None:
            reg = self._kv_norm(self.register_tokens.expand(b, -1, -1), idx)  # [B, R, C]
            kv_in = torch.cat([kv_in, reg], dim=1)                        # [B, N+R, C]

        # Projected Q/K plus a bottleneck value path with per-level LoRA deltas.
        def _proj(base, lora, x):
            out = base(x)
            if lora is not None:
                out = out + lora[idx](x)
            return out

        q = _proj(self.qk_proj, self.lora_qk, q_in)  # [B, N,     Dqk]
        k = _proj(self.qk_proj, self.lora_qk, kv_in) # [B, N(+R), Dqk]
        v = _proj(self.v_down, self.lora_v_down, kv_in)  # [B, N(+R), Dv]

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
        qk_hd = self.qk_head_dim
        v_hd = self.value_head_dim
        q = q.view(b, n,  self.num_heads, qk_hd).transpose(1, 2)   # [B, H, N,  qk_hd]
        k = k.view(b, nk, self.num_heads, qk_hd).transpose(1, 2)   # [B, H, nk, qk_hd]
        v = v.view(b, nk, self.num_heads, v_hd).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(qk_hd)  # [B, H, N, nk]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)   # [B, H, N, nk]
        attn_weights = self.attn_dropout(attn_weights)

        attn_map_feature = self._attention_map_feature(attn_weights, n)
        attn_out_raw = torch.matmul(attn_weights, v)                          # [B, H, N, v_hd]
        attn_out_raw = attn_out_raw.transpose(1, 2).contiguous().view(b, n, self.value_dim)
        if attn_map_feature is not None:
            attn_out_raw = torch.cat((attn_out_raw, attn_map_feature), dim=-1)

        # Project the low-rank temporal evidence back to the DPT feature space.
        attn_out = self.v_up(attn_out_raw)
        if self.lora_v_up is not None:
            attn_out = attn_out + self.lora_v_up[idx](attn_out_raw)

        residual_gate = self._residual_gate(q_in, attn_out, idx)
        if residual_mask is not None:
            if residual_mask.shape != (b, 1, 1):
                raise ValueError(
                    "residual_mask must have shape [B, 1, 1], got {}".format(
                        tuple(residual_mask.shape)
                    )
                )
            residual_gate = residual_gate * residual_mask.to(
                device=residual_gate.device,
                dtype=residual_gate.dtype,
            )
        return x1 + self.drop_path_attn(attn_out * residual_gate)
