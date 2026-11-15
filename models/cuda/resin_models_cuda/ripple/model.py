# Semantics follow the Ripple-65B-Preview reference (Apache-2.0,
# ricursiveAI/Ripple-65B-Preview, modeling_ricursive.py).

"""Checkpoint-shaped BF16 Ripple modules for flat-token CUDA execution.

State-dict keys match the Hugging Face ``RicursiveForCausalLM`` checkpoint.
Hidden states are flat ``[tokens, width]`` tensors. The attention modules never
touch a KV cache: the caller supplies an ``attend(query, key, value)`` callback
that appends the new keys and values and returns the attended heads. This keeps
the numerical contract, tensor parallelism, and packing runnable (and testable)
on CPU, while the CUDA runner supplies FlashInfer paged attention.

Nothing here imports ``flashinfer``, ``triton`` or any CUDA kernel package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.models.ripple.config import SLIDING_ATTENTION, RippleConfig

# ``attend(layer, query, key, value)`` takes ``[tokens, heads, head_dim]`` query
# and ``[tokens, kv_heads, head_dim]`` key/value (RoPE already applied to
# sliding layers) and returns ``[tokens, heads, head_dim]`` (or flattened).
Attend = Callable[[int, Tensor, Tensor, Tensor], Tensor]
LayerAttend = Callable[[Tensor, Tensor, Tensor], Tensor]


class RippleRMSNorm(nn.Module):
    """``weight * (x.float() * rsqrt(mean(x^2) + eps)).to(dtype)``.

    This is the reference's exact rounding: the normalized value is rounded to
    the activation dtype *before* the learned scale. FlashInfer's and Quack's
    fused norms scale in fp32 and round once, which differs by up to one BF16
    ulp per element, so Ripple keeps this explicit, graph-safe formulation.
    """

    def __init__(
        self, width: int, eps: float, *, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width, dtype=dtype))

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


class RippleRotaryEmbedding(nn.Module):
    """NeoX ``rotate_half`` cos/sin from flat positions and a fixed ``inv_freq``."""

    inv_freq: Tensor | None

    def __init__(self, config: RippleConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.register_buffer("inv_freq", None, persistent=False)

    def materialize(self, device: torch.device | str) -> None:
        exponents = torch.arange(0, self.head_dim, 2).float() / self.head_dim
        self.inv_freq = (1.0 / self.theta**exponents).to(device)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Return ``[tokens, 1, head_dim]`` fp32 cos/sin rounded to ``dtype``."""

        if self.inv_freq is None:
            raise RuntimeError("rotary buffers must be materialized before inference")
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        angles = torch.cat((freqs, freqs), dim=-1)
        # The reference rounds cos/sin to the activation dtype, then rotates in fp32.
        return (
            angles.cos().to(dtype).float().unsqueeze(1),
            angles.sin().to(dtype).float().unsqueeze(1),
        )


def apply_rotary(hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    full = hidden.float()
    return (full * cos + _rotate_half(full) * sin).to(hidden.dtype)


def _packed(projections: Sequence[nn.Linear]) -> nn.Linear:
    weight = torch.cat([projection.weight for projection in projections])
    packed = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device=weight.device,
        dtype=weight.dtype,
    )
    packed.weight = nn.Parameter(weight, requires_grad=False)
    return packed


class RippleCudaAttention(nn.Module):
    """Gated, value-residual attention with per-head Q/K norms.

    ``num_heads`` and ``num_kv_heads`` are rank-local after tensor-parallel
    sharding. Full-attention layers are NoPE; only sliding layers apply RoPE.
    """

    def __init__(
        self,
        config: RippleConfig,
        layer_idx: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        width = config.hidden_size

        def linear(inputs: int, outputs: int) -> nn.Linear:
            return nn.Linear(inputs, outputs, bias=False, dtype=dtype)

        self.q_proj = linear(width, self.num_heads * self.head_dim)
        self.k_proj = linear(width, self.num_kv_heads * self.head_dim)
        self.v_proj = linear(width, self.num_kv_heads * self.head_dim)
        # Tensor parallelism wraps this in a TensorParallelSum.
        self.o_proj: nn.Module = linear(self.num_heads * self.head_dim, width)
        if config.use_gated_attention:
            self.g_proj = linear(width, self.num_heads * self.head_dim)
        if config.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.q_norm = RippleRMSNorm(self.head_dim, config.rms_norm_eps, dtype=dtype)
        self.k_norm = RippleRMSNorm(self.head_dim, config.rms_norm_eps, dtype=dtype)
        self.qgk_proj: nn.Linear | None = None

    def pack_projections(self) -> None:
        """Fuse the projections that read the same normalized hidden state.

        Call after tensor-parallel sharding. With value residuals ``v_proj``
        reads the blended ``v_in`` instead, so only ``q``, ``g`` and ``k`` pack.
        """

        if self.qgk_proj is not None:
            raise RuntimeError("attention projections were already packed")
        names = ["q_proj"]
        if self.config.use_gated_attention:
            names.append("g_proj")
        names.append("k_proj")
        if not self.config.use_valres:
            names.append("v_proj")
        self.qgk_proj = _packed([getattr(self, name) for name in names])
        for name in names:
            delattr(self, name)

    def forward(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        value_residual: Tensor | None,
        attend: LayerAttend,
    ) -> Tensor:
        """``hidden`` is the input-normalized ``[tokens, width]`` state.

        Returns the o_proj output, already summed across tensor-parallel ranks
        (the sum is part of ``o_proj``), before ``post_attention_layernorm``.
        """

        config = self.config
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        gate = value = None
        if self.qgk_proj is not None:
            sizes = [q_width]
            if config.use_gated_attention:
                sizes.append(q_width)
            sizes.append(kv_width)
            if not config.use_valres:
                sizes.append(kv_width)
            parts = list(self.qgk_proj(hidden).split(sizes, dim=-1))
            query = parts.pop(0)
            if config.use_gated_attention:
                gate = parts.pop(0)
            key = parts.pop(0)
            value = parts.pop(0) if parts else None
        else:
            query, key = self.q_proj(hidden), self.k_proj(hidden)
            if config.use_gated_attention:
                gate = self.g_proj(hidden)
        if value is None:
            values_in = hidden
            if value_residual is not None:
                # Graph-safe: the blend stays a device tensor (no .item()).
                blend = torch.sigmoid(self.value_res_lambda[0]).to(hidden.dtype)
                values_in = torch.lerp(value_residual.to(hidden.dtype), hidden, blend)
            value = self.v_proj(values_in)

        query = self.q_norm(query.reshape(-1, self.num_heads, self.head_dim))
        key = self.k_norm(key.reshape(-1, self.num_kv_heads, self.head_dim))
        value = value.reshape(-1, self.num_kv_heads, self.head_dim)
        if self.is_sliding:  # full-attention layers are NoPE
            query = apply_rotary(query, cos, sin)
            key = apply_rotary(key, cos, sin)

        attended = attend(query, key, value).reshape(len(hidden), -1)
        if gate is not None:
            if config.gate_full_precision:
                gate = gate.float()
            attended = attended * torch.sigmoid(gate).to(attended.dtype)
        return self.o_proj(attended)


class RippleCudaMLP(nn.Module):
    """SwiGLU with Ripple's ``ffn_clamp_threshold``; ``down_proj`` reduces TP."""

    def __init__(self, config: RippleConfig, *, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.clamp = config.ffn_clamp_threshold
        self.intermediate_size = config.intermediate_size
        width, size = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(width, size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(width, size, bias=False, dtype=dtype)
        self.down_proj: nn.Module = nn.Linear(size, width, bias=False, dtype=dtype)
        self.gate_up_proj: nn.Linear | None = None

    def pack_projections(self) -> None:
        """Fuse gate and up after tensor-parallel sharding."""

        if self.gate_up_proj is not None:
            raise RuntimeError("MLP projections were already packed")
        self.gate_up_proj = _packed([self.gate_proj, self.up_proj])
        del self.gate_proj
        del self.up_proj

    def forward(self, hidden: Tensor) -> Tensor:
        if self.gate_up_proj is not None:
            gate, up = self.gate_up_proj(hidden).split(self.intermediate_size, dim=-1)
        else:
            gate, up = self.gate_proj(hidden), self.up_proj(hidden)
        if self.clamp is not None:
            gate = gate.clamp(max=self.clamp)
            up = up.clamp(min=-self.clamp, max=self.clamp)
        return self.down_proj(F.silu(gate) * up)


class RippleCudaDecoderLayer(nn.Module):
    """PeriNorm block: the post norms see the already-reduced sublayer outputs."""

    def __init__(
        self,
        config: RippleConfig,
        layer_idx: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.self_attn = RippleCudaAttention(config, layer_idx, dtype=dtype)
        self.mlp = RippleCudaMLP(config, dtype=dtype)
        width, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RippleRMSNorm(width, eps, dtype=dtype)
        self.post_attention_layernorm = RippleRMSNorm(width, eps, dtype=dtype)
        self.pre_feedforward_layernorm = RippleRMSNorm(width, eps, dtype=dtype)
        self.post_feedforward_layernorm = RippleRMSNorm(width, eps, dtype=dtype)

    def attention_block(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        value_residual: Tensor | None,
        attend: LayerAttend,
    ) -> Tensor:
        attended = self.self_attn(
            self.input_layernorm(hidden), cos, sin, value_residual, attend
        )
        return hidden + self.post_attention_layernorm(attended)

    def feed_forward(self, hidden: Tensor) -> Tensor:
        fed = self.mlp(self.pre_feedforward_layernorm(hidden))
        return hidden + self.post_feedforward_layernorm(fed)


class RippleCudaModel(nn.Module):
    scale_tensor: Tensor | None

    def __init__(
        self, config: RippleConfig, *, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=dtype
        )
        self.layers = cast(
            Sequence[RippleCudaDecoderLayer],
            nn.ModuleList(
                RippleCudaDecoderLayer(config, index, dtype=dtype)
                for index in range(config.num_hidden_layers)
            ),
        )
        self.norm = RippleRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype)
        self.rotary_emb = RippleRotaryEmbedding(config)
        # fp32 so ``embed.float() * scale`` rounds exactly like ``embed * float``.
        self.register_buffer("scale_tensor", None, persistent=False)

    def materialize_buffers(self, device: torch.device | str) -> None:
        self.rotary_emb.materialize(device)
        if self.config.embed_scale is not None:
            self.scale_tensor = torch.tensor(
                self.config.embed_scale, dtype=torch.float32, device=device
            )

    def embed(self, token_ids: Tensor) -> Tensor:
        """Scaled embeddings: the scale precedes the value-residual seed."""

        hidden = self.embed_tokens(token_ids)
        if self.config.embed_scale is None:
            return hidden
        if self.scale_tensor is None:
            raise RuntimeError("embedding scale must be materialized before inference")
        # A tensor (not a host scalar) keeps CUDA graph capture allocation-free.
        return (hidden.float() * self.scale_tensor).to(hidden.dtype)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        attend: Attend,
        feed_forward: Sequence[Callable[[Tensor], Tensor]] | None = None,
    ) -> Tensor:
        """Run every layer over flat ``hidden`` and return the final-normed state.

        ``feed_forward[i]`` may replace layer ``i``'s feed-forward block, for
        example with a captured CUDA graph region.
        """

        cos, sin = self.rotary_emb(positions, hidden.dtype)
        value_residual = hidden if self.config.use_valres else None
        for index, layer in enumerate(self.layers):
            hidden = layer.attention_block(
                hidden, cos, sin, value_residual, partial(attend, index)
            )
            fed = layer.feed_forward if feed_forward is None else feed_forward[index]
            hidden = fed(hidden)
        return self.norm(hidden)


class RippleCudaForCausalLM(nn.Module):
    def __init__(
        self, config: RippleConfig, *, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.config = config
        self.model = RippleCudaModel(config, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype
        )

    def materialize_buffers(self, device: torch.device | str) -> None:
        self.model.materialize_buffers(device)

    def pack_projections(self) -> None:
        """Pack after tensor-parallel sharding; buffers must be materialized."""

        for layer in self.model.layers:
            layer.self_attn.pack_projections()
            layer.mlp.pack_projections()

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        return self.model.embed(token_ids)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(hidden)

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor,
        attend: Attend,
        feed_forward: Sequence[Callable[[Tensor], Tensor]] | None = None,
    ) -> Tensor:
        """Final-normed hidden states for flat ``token_ids``/``positions``."""

        return self.model(self.embed_tokens(token_ids), positions, attend, feed_forward)


__all__ = [
    "Attend",
    "RippleCudaAttention",
    "RippleCudaDecoderLayer",
    "RippleCudaForCausalLM",
    "RippleCudaMLP",
    "RippleCudaModel",
    "RippleRMSNorm",
    "RippleRotaryEmbedding",
    "apply_rotary",
]
