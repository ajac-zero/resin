"""Backend-neutral configuration for Ripple (``ricursive``) checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from resin.backend.cuda.parallel import TensorParallelLayout

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

# Cache group order shared by CacheLayout and RippleBatchPlan.
FULL_CACHE_GROUP = 0
WINDOW_CACHE_GROUP = 1


@dataclass(frozen=True, slots=True)
class RippleConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    sliding_window: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    layer_types: tuple[str, ...]
    use_gated_attention: bool
    gate_full_precision: bool
    use_valres: bool
    ffn_clamp_threshold: float | None
    embed_scale: float | None
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        dimensions = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "sliding_window": self.sliding_window,
            "max_position_embeddings": self.max_position_embeddings,
        }
        invalid = [name for name, value in dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"Ripple dimensions must be positive: {invalid}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        unsupported = set(self.layer_types) - {SLIDING_ATTENTION, FULL_ATTENTION}
        if unsupported:
            raise ValueError(f"unsupported Ripple layer types: {sorted(unsupported)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.ffn_clamp_threshold is not None and self.ffn_clamp_threshold <= 0:
            raise ValueError("ffn_clamp_threshold must be positive or None")
        if self.embed_scale is not None and self.embed_scale <= 0:
            raise ValueError("embed_scale must be positive or None")
        if self.tie_word_embeddings:
            raise ValueError("Ripple requires untied input/output embeddings")
        if self.gate_full_precision and not self.use_gated_attention:
            raise ValueError("gate_full_precision requires use_gated_attention")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the vocabulary")
        if not self.eos_token_ids:
            raise ValueError("eos_token_ids must not be empty")
        if not all(0 <= token < self.vocab_size for token in self.eos_token_ids):
            raise ValueError("eos_token_ids must be inside the vocabulary")

    @property
    def local_layer_count(self) -> int:
        return self.layer_types.count(SLIDING_ATTENTION)

    @property
    def global_layer_count(self) -> int:
        return self.layer_types.count(FULL_ATTENTION)

    @property
    def history_blocks_per_parent(self) -> int:
        """Pack full-attention blocks into a slab the size of one window block.

        Every layer stores the same KV width, so a window block spans
        ``local_layer_count`` layers and a full-attention block spans
        ``global_layer_count``: 64 / 16 = 4 on Ripple-65B, 8 / 2 on the tiny model.
        """

        if not self.global_layer_count or not self.local_layer_count:
            raise ValueError("Ripple requires both sliding and full-attention layers")
        if self.local_layer_count % self.global_layer_count:
            raise ValueError(
                "local/global layer counts do not share an integral parent"
            )
        return self.local_layer_count // self.global_layer_count

    def layer_cache_group(self, layer: int) -> int:
        """Cache group (``FULL_CACHE_GROUP`` or ``WINDOW_CACHE_GROUP``) of a layer."""

        return (
            FULL_CACHE_GROUP
            if self.layer_types[layer] == FULL_ATTENTION
            else WINDOW_CACHE_GROUP
        )

    @property
    def layer_cache_slots(self) -> tuple[tuple[int, int], ...]:
        """``(cache_group, index_within_group)`` for every layer, in order."""

        counts = [0, 0]
        slots = []
        for layer in range(self.num_hidden_layers):
            group = self.layer_cache_group(layer)
            slots.append((group, counts[group]))
            counts[group] += 1
        return tuple(slots)

    def same_shape_reduced(self, num_layers: int = 5) -> RippleConfig:
        """Keep the first ``num_layers`` layers with the same cache packing.

        Five layers of the real model are one ``[S, S, S, S, G]`` period, so
        ``history_blocks_per_parent`` stays 4.
        """

        if not 0 < num_layers <= self.num_hidden_layers:
            raise ValueError("num_layers must be in [1, num_hidden_layers]")
        reduced = replace(
            self,
            num_hidden_layers=num_layers,
            layer_types=self.layer_types[:num_layers],
        )
        if reduced.history_blocks_per_parent != self.history_blocks_per_parent:
            raise ValueError(
                f"{num_layers} layers change history_blocks_per_parent from "
                f"{self.history_blocks_per_parent} to "
                f"{reduced.history_blocks_per_parent}"
            )
        return reduced

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> RippleConfig:
        if config.get("model_type") != "ricursive":
            raise ValueError(
                f"expected a Ripple checkpoint, got {config.get('model_type')!r}"
            )
        layers = config["num_hidden_layers"]
        layer_types = config.get("layer_types")
        if layer_types is None:
            layer_types = [
                FULL_ATTENTION if i % 5 == 4 else SLIDING_ATTENTION
                for i in range(layers)
            ]
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("Ripple requires the silu activation")
        eos = config.get("eos_token_id")
        eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
        threshold = config.get("ffn_clamp_threshold")
        embed_scale = config.get("embed_scale")
        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=layers,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            sliding_window=config["sliding_window"],
            max_position_embeddings=config["max_position_embeddings"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            rope_theta=config.get("rope_theta", 10_000.0),
            layer_types=tuple(layer_types),
            use_gated_attention=config.get("use_gated_attention", True),
            gate_full_precision=config.get("gate_full_precision", True),
            use_valres=config.get("use_valres", False),
            # Like the reference model, a zero threshold disables clamping.
            ffn_clamp_threshold=float(threshold) if threshold else None,
            embed_scale=None if embed_scale is None else float(embed_scale),
            pad_token_id=config["pad_token_id"],
            eos_token_ids=eos_ids,
            tie_word_embeddings=config.get("tie_word_embeddings", False),
        )


@dataclass(frozen=True, slots=True)
class TensorParallelGeometry:
    """One rank's contiguous head and FFN slices of a Ripple layer."""

    size: int
    rank: int
    q_heads: int
    q_head_start: int
    kv_heads: int
    kv_head_start: int
    kv_replicas: int
    intermediate_size: int
    intermediate_start: int


def attention_geometry(
    config: RippleConfig, layout: TensorParallelLayout
) -> TensorParallelGeometry:
    """Shard query heads and the FFN evenly; replicate KV heads past ``kv`` ranks.

    Raises ``ValueError`` when the layout cannot shard the model cleanly.
    """

    q = layout.even_shard(config.num_attention_heads, name="num_attention_heads")
    ffn = layout.even_shard(config.intermediate_size, name="intermediate_size")
    kv_heads = config.num_key_value_heads
    if kv_heads % layout.size == 0:
        kv = layout.even_shard(kv_heads, name="num_key_value_heads")
        local_kv, kv_start, replicas = kv.size, kv.start, 1
    elif layout.size % kv_heads == 0:
        replicas = layout.size // kv_heads
        local_kv, kv_start = 1, layout.rank // replicas
        if (config.num_attention_heads // kv_heads) % q.size:
            raise ValueError("a replicated KV head must serve whole query shards")
    else:
        raise ValueError(
            f"num_key_value_heads={kv_heads} cannot shard across TP={layout.size}"
        )
    return TensorParallelGeometry(
        size=layout.size,
        rank=layout.rank,
        q_heads=q.size,
        q_head_start=q.start,
        kv_heads=local_kv,
        kv_head_start=kv_start,
        kv_replicas=replicas,
        intermediate_size=ffn.size,
        intermediate_start=ffn.start,
    )


__all__ = [
    "FULL_ATTENTION",
    "FULL_CACHE_GROUP",
    "RippleConfig",
    "SLIDING_ATTENTION",
    "TensorParallelGeometry",
    "WINDOW_CACHE_GROUP",
    "attention_geometry",
]
