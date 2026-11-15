# Semantics follow the Ripple-65B-Preview reference (Apache-2.0,
# ricursiveAI/Ripple-65B-Preview, modeling_ricursive.py).

"""Ripple modules and request-owned KV state in plain PyTorch.

State dict keys match the Hugging Face ``RicursiveForCausalLM`` checkpoint.
Nothing here imports ``transformers``, ``tokenizers`` or ``safetensors``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.models.ripple.config import (
    SLIDING_ATTENTION,
    RippleConfig,
    TensorParallelGeometry,
    attention_geometry,
)

AllReduce = Callable[[Tensor], Tensor]


class AttentionState:
    """One layer's retained keys and values for a single request.

    Full-attention layers keep every token. Sliding layers keep only the
    ``window - 1`` newest tokens, which is all a following query can still read.
    """

    def __init__(self, window: int | None) -> None:
        self.keep = None if window is None else window - 1
        self.keys: Tensor | None = None
        self.values: Tensor | None = None
        self.length = 0  # tokens seen, i.e. the next position

    @property
    def retained(self) -> int:
        return 0 if self.keys is None else self.keys.shape[-2]

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor, int]:
        """Add new tokens; return all keys/values to attend over and their start."""

        start = self.length - self.retained
        added = keys.shape[-2]
        if self.keys is not None:
            assert self.values is not None
            keys = torch.cat((self.keys, keys), dim=-2)
            values = torch.cat((self.values, values), dim=-2)
        self.length += added
        drop = 0 if self.keep is None else max(0, keys.shape[-2] - self.keep)
        self.keys, self.values = keys[..., drop:, :], values[..., drop:, :]
        return keys, values, start


class RippleState:
    """One request's ordered per-layer attention state."""

    def __init__(self, config: RippleConfig) -> None:
        self.layers = [
            AttentionState(config.sliding_window if kind == SLIDING_ATTENTION else None)
            for kind in config.layer_types
        ]

    @property
    def length(self) -> int:
        return self.layers[0].length

    @property
    def retained_tokens(self) -> tuple[int, ...]:
        return tuple(layer.retained for layer in self.layers)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        dtype = hidden.dtype
        normalized = hidden.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(dtype)


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(
        self,
        config: RippleConfig,
        layer_idx: int,
        geometry: TensorParallelGeometry,
        all_reduce: AllReduce | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.is_sliding = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.head_dim = config.head_dim
        self.num_heads = geometry.q_heads
        self.num_kv_heads = geometry.kv_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.all_reduce = all_reduce
        width = config.hidden_size
        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=False)
        if config.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1))
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        if config.use_gated_attention:
            self.g_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=False)

    def _rope(self, queries: Tensor, keys: Tensor, positions: Tensor):
        dim = self.head_dim
        exponents = torch.arange(0, dim, 2, device=positions.device).float() / dim
        inv_freq = 1.0 / self.config.rope_theta**exponents
        freqs = positions.float()[:, :, None] * inv_freq[None, None, :]
        angles = torch.cat((freqs, freqs), dim=-1)
        # The reference rounds cos/sin to the activation dtype, then rotates in fp32.
        cos = angles.cos().to(queries.dtype).float().unsqueeze(1)
        sin = angles.sin().to(queries.dtype).float().unsqueeze(1)

        def rotate(hidden: Tensor) -> Tensor:
            full = hidden.float()
            return (full * cos + _rotate_half(full) * sin).to(hidden.dtype)

        return rotate(queries), rotate(keys)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        states: Sequence[AttentionState],
        value_residual: Tensor | None,
    ) -> Tensor:
        batch, length, _ = hidden.shape
        values_in = hidden
        if value_residual is not None:
            blend = torch.sigmoid(self.value_res_lambda[0]).to(hidden.dtype)
            # lerp(a, b, w) = (1 - w) * a + w * b: the residual is the w = 0 end.
            values_in = torch.lerp(value_residual.to(hidden.dtype), hidden, blend)
        queries = self.q_proj(hidden).view(batch, length, -1, self.head_dim)
        keys = self.k_proj(hidden).view(batch, length, -1, self.head_dim)
        values = self.v_proj(values_in).view(batch, length, -1, self.head_dim)
        queries = self.q_norm(queries).transpose(1, 2)
        keys = self.k_norm(keys).transpose(1, 2)
        values = values.transpose(1, 2)
        if self.is_sliding:  # full-attention layers are NoPE
            queries, keys = self._rope(queries, keys, positions)

        key_rows, value_rows, starts = [], [], []
        for row, state in enumerate(states):
            row_keys, row_values, start = state.append(
                keys[row : row + 1], values[row : row + 1]
            )
            key_rows.append(row_keys)
            value_rows.append(row_values)
            starts.append(start)
        widths = [rows.shape[-2] for rows in key_rows]
        longest = max(widths)
        if min(widths) != longest:
            key_rows = [
                F.pad(rows, (0, 0, 0, longest - width))
                for rows, width in zip(key_rows, widths, strict=True)
            ]
            value_rows = [
                F.pad(rows, (0, 0, 0, longest - width))
                for rows, width in zip(value_rows, widths, strict=True)
            ]
        keys_all = torch.cat(key_rows).repeat_interleave(self.kv_groups, dim=1)
        values_all = torch.cat(value_rows).repeat_interleave(self.kv_groups, dim=1)

        # Absolute-position mask: causal, inside the window, and not padding.
        device = hidden.device
        key_index = torch.arange(longest, device=device)
        key_positions = torch.tensor(starts, device=device)[:, None] + key_index
        query_positions = positions[:, :, None]
        allowed = key_index[None, :] < torch.tensor(widths, device=device)[:, None]
        allowed = allowed[:, None, :] & (key_positions[:, None, :] <= query_positions)
        if self.is_sliding:
            window = self.config.sliding_window
            allowed &= key_positions[:, None, :] >= query_positions - (window - 1)
        mask = None if bool(allowed.all()) else allowed[:, None]
        output = F.scaled_dot_product_attention(
            queries, keys_all, values_all, attn_mask=mask, dropout_p=0.0
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        if self.config.use_gated_attention:
            gate = self.g_proj(hidden)
            if self.config.gate_full_precision:
                gate = gate.float()
            output = output * torch.sigmoid(gate).to(output.dtype)
        output = self.o_proj(output)
        # Row-parallel partial sums must be reduced before the post-attention norm.
        return self.all_reduce(output) if self.all_reduce is not None else output


class MLP(nn.Module):
    def __init__(
        self,
        config: RippleConfig,
        geometry: TensorParallelGeometry,
        all_reduce: AllReduce | None,
    ) -> None:
        super().__init__()
        self.clamp = config.ffn_clamp_threshold
        self.all_reduce = all_reduce
        size = geometry.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, size, bias=False)
        self.down_proj = nn.Linear(size, config.hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, up = self.gate_proj(hidden), self.up_proj(hidden)
        if self.clamp:
            gate = gate.clamp(max=self.clamp)
            up = up.clamp(min=-self.clamp, max=self.clamp)
        output = self.down_proj(F.silu(gate) * up)
        return self.all_reduce(output) if self.all_reduce is not None else output


class DecoderLayer(nn.Module):
    """PeriNorm block: the post norms see the reduced sublayer outputs."""

    def __init__(
        self,
        config: RippleConfig,
        layer_idx: int,
        geometry: TensorParallelGeometry,
        all_reduce: AllReduce | None,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx, geometry, all_reduce)
        self.mlp = MLP(config, geometry, all_reduce)
        width, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RMSNorm(width, eps)
        self.post_attention_layernorm = RMSNorm(width, eps)
        self.pre_feedforward_layernorm = RMSNorm(width, eps)
        self.post_feedforward_layernorm = RMSNorm(width, eps)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        states: Sequence[AttentionState],
        value_residual: Tensor | None,
    ) -> Tensor:
        attended = self.self_attn(
            self.input_layernorm(hidden), positions, states, value_residual
        )
        hidden = hidden + self.post_attention_layernorm(attended)
        fed = self.mlp(self.pre_feedforward_layernorm(hidden))
        return hidden + self.post_feedforward_layernorm(fed)


class RippleModel(nn.Module):
    def __init__(
        self,
        config: RippleConfig,
        geometry: TensorParallelGeometry,
        all_reduce: AllReduce | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, index, geometry, all_reduce)
            for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, tokens: Tensor, positions: Tensor, states: Sequence[RippleState]
    ) -> Tensor:
        hidden = self.embed_tokens(tokens)
        if self.config.embed_scale is not None:
            hidden = hidden * self.config.embed_scale
        # The value residual is the scaled embedding, seeded before layer 0.
        value_residual = hidden if self.config.use_valres else None
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                positions,
                [state.layers[index] for state in states],
                value_residual,
            )
        return self.norm(hidden)


class RippleForCausalLM(nn.Module):
    def __init__(
        self,
        config: RippleConfig,
        *,
        geometry: TensorParallelGeometry | None = None,
        all_reduce: AllReduce | None = None,
    ) -> None:
        super().__init__()
        if geometry is None:
            from resin.backend.cuda.parallel import TensorParallelLayout

            geometry = attention_geometry(config, TensorParallelLayout())
        if geometry.size > 1 and all_reduce is None:
            raise ValueError("tensor-parallel Ripple requires an all_reduce")
        self.config = config
        self.geometry = geometry
        self.model = RippleModel(config, geometry, all_reduce)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> RippleState:
        return RippleState(self.config)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(hidden)

    def forward(
        self,
        tokens: Tensor,
        state: RippleState | None = None,
        positions: Tensor | None = None,
        *,
        logits_to_keep: int | None = None,
        return_hidden: bool = False,
    ) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("tokens must contain exactly one sequence")
        state = self.make_cache() if state is None else state
        if positions is None:
            positions = torch.arange(
                state.length, state.length + tokens.shape[1], device=tokens.device
            )[None]
        return self._run(tokens, [state], positions, logits_to_keep, return_hidden)

    def forward_batch(
        self,
        tokens: Tensor,
        states: Sequence[RippleState],
        positions: Tensor | None = None,
    ) -> Tensor:
        """Decode one token for each independently cached sequence."""

        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError("batched decode requires one token per sequence")
        if tokens.shape[0] != len(states) or not states:
            raise ValueError("batched decode requires one state per sequence")
        if positions is None:
            positions = torch.tensor(
                [[state.length] for state in states], device=tokens.device
            )
        return self._run(tokens, list(states), positions, None, False)

    def _run(self, tokens, states, positions, logits_to_keep, return_hidden) -> Tensor:
        if positions.shape != tokens.shape:
            raise ValueError("positions must have the same shape as tokens")
        hidden = self.model(tokens, positions, states)
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            hidden = hidden[:, -logits_to_keep:]
        return hidden if return_hidden else self.logits(hidden)


def shard_state_dict(
    state: dict[str, Tensor], config: RippleConfig, geometry: TensorParallelGeometry
) -> dict[str, Tensor]:
    """Slice a full state dict into one tensor-parallel rank's tensors.

    Column-parallel ``q/k/v/g`` and ``gate/up`` take output rows; row-parallel
    ``o`` and ``down`` take input columns; everything else is replicated.
    """

    d = config.head_dim
    q_start = geometry.q_head_start
    q_slice = slice(q_start * d, (q_start + geometry.q_heads) * d)
    kv_start = geometry.kv_head_start
    kv_slice = slice(kv_start * d, (kv_start + geometry.kv_heads) * d)
    ffn_start = geometry.intermediate_start
    ffn_slice = slice(ffn_start, ffn_start + geometry.intermediate_size)
    rows = {
        "q_proj": q_slice,
        "g_proj": q_slice,
        "k_proj": kv_slice,
        "v_proj": kv_slice,
        "gate_proj": ffn_slice,
        "up_proj": ffn_slice,
    }
    columns = {"o_proj": q_slice, "down_proj": ffn_slice}
    sharded = {}
    for name, tensor in state.items():
        owner = name.rsplit(".", 2)[-2] if name.endswith(".weight") else None
        if owner in rows:
            tensor = tensor[rows[owner]]
        elif owner in columns:
            tensor = tensor[:, columns[owner]]
        sharded[name] = tensor.clone()
    return sharded


__all__ = [
    "AttentionState",
    "RMSNorm",
    "RippleForCausalLM",
    "RippleModel",
    "RippleState",
    "shard_state_dict",
]
