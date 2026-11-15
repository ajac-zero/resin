"""Tensor-parallel lowering for Ripple attention and MLP modules."""

from __future__ import annotations

from typing import cast

from torch import nn

from resin.backend.cuda.parallel import TensorParallelContext, TensorShard
from resin.backend.loading import prepare_model
from resin.models.ripple.config import attention_geometry
from resin_models_cuda.dense_parallel import TensorParallelSum, shard_linear
from resin_models_cuda.ripple.model import RippleCudaAttention, RippleCudaMLP


def shard_attention(
    module: RippleCudaAttention, parallel: TensorParallelContext
) -> RippleCudaAttention:
    """Partition loaded heads; ``o_proj`` sums its partials before the post norm.

    Query and gate heads follow the rank's query slice, key and value heads its
    KV slice (replicated when ranks outnumber KV heads). Run before
    ``pack_projections``.
    """

    if not parallel.enabled:
        return module
    if module.qgk_proj is not None:
        raise ValueError("shard Ripple attention before packing its projections")
    geometry = attention_geometry(module.config, parallel.layout)
    d = module.head_dim
    query = TensorShard(
        geometry.q_head_start * d, (geometry.q_head_start + geometry.q_heads) * d
    )
    kv = TensorShard(
        geometry.kv_head_start * d, (geometry.kv_head_start + geometry.kv_heads) * d
    )
    projections = [("q_proj", query), ("k_proj", kv), ("v_proj", kv)]
    if hasattr(module, "g_proj"):
        projections.append(("g_proj", query))
    for name, shard in projections:
        setattr(
            module,
            name,
            shard_linear(getattr(module, name), shard, output=True, rank=parallel.rank),
        )
    module.o_proj = TensorParallelSum(
        shard_linear(
            cast(nn.Linear, module.o_proj), query, output=False, rank=parallel.rank
        ),
        parallel,
    )
    module.num_heads = geometry.q_heads
    module.num_kv_heads = geometry.kv_heads
    return module


def shard_mlp(module: RippleCudaMLP, parallel: TensorParallelContext) -> RippleCudaMLP:
    """Shard the intermediate axis; ``down_proj`` sums before the post norm.

    The clamp is elementwise on the local gate/up slice, so it commutes with
    sharding. Run before ``pack_projections``.
    """

    if not parallel.enabled:
        return module
    if module.gate_up_proj is not None:
        raise ValueError("shard the Ripple MLP before packing its projections")
    geometry = attention_geometry(module.config, parallel.layout)
    shard = TensorShard(
        geometry.intermediate_start,
        geometry.intermediate_start + geometry.intermediate_size,
    )
    for name in ("gate_proj", "up_proj"):
        setattr(
            module,
            name,
            shard_linear(getattr(module, name), shard, output=True, rank=parallel.rank),
        )
    module.down_proj = TensorParallelSum(
        shard_linear(
            cast(nn.Linear, module.down_proj), shard, output=False, rank=parallel.rank
        ),
        parallel,
    )
    module.intermediate_size = geometry.intermediate_size
    return module


def prepare_ripple_tp(
    model: nn.Module,
    *,
    attention_parallel: TensorParallelContext | None = None,
    tensor_parallel: TensorParallelContext | None = None,
) -> nn.Module:
    """Shard every attention and MLP module of a loaded model in place."""

    def lower(module: nn.Module) -> nn.Module:
        if attention_parallel is not None and isinstance(module, RippleCudaAttention):
            return shard_attention(module, attention_parallel)
        if tensor_parallel is not None and isinstance(module, RippleCudaMLP):
            return shard_mlp(module, tensor_parallel)
        return module

    return prepare_model(model, lower)


__all__ = ["prepare_ripple_tp", "shard_attention", "shard_mlp"]
