"""A tiny Ripple config that mirrors every architectural key of the real one.

Only dimensions are scaled down. The 4 sliding : 1 full hybrid period, the 10:1
query:KV head ratio, gating, value residuals, clamping, norms, RoPE and
``embed_scale = sqrt(hidden_size)`` are all preserved.
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

import torch

from ripple.reference.verify import REFERENCE_DIR, verify_reference_files

TINY_OVERRIDES: dict[str, Any] = {
    "vocab_size": 256,
    "hidden_size": 320,
    "intermediate_size": 1280,  # 4x hidden; divisible by TP=1,2,4,8
    "num_hidden_layers": 10,  # two 4S:1G hybrid periods
    "num_attention_heads": 20,  # 10:1 query:KV ratio, as in the real model
    "num_key_value_heads": 2,
    "head_dim": 16,
    "sliding_window": 8,  # length 20..28 crosses 2x and 3x the window
    "max_position_embeddings": 256,
    "bos_token_id": 250,
    "eos_token_id": [254, 250],
    "pad_token_id": 255,
}
# Keys that legitimately differ between the real and tiny configs.
EXPECTED_DIFF_KEYS = frozenset(
    {*TINY_OVERRIDES, "layer_types", "hybrid_layer_pattern", "embed_scale"}
)
_IGNORED_KEYS = frozenset({"transformers_version", "dtype", "torch_dtype"})
_SERIALIZATION_ONLY = (
    "architectures",
    "auto_map",
    "model_type",
    "transformers_version",
)


def real_config_dict() -> dict[str, Any]:
    verify_reference_files()
    return json.loads((REFERENCE_DIR / "config.json").read_text())


def tiny_layer_types(num_layers: int) -> list[str]:
    return [
        "full_attention" if i % 5 == 4 else "sliding_attention"
        for i in range(num_layers)
    ]


def tiny_config_dict() -> dict[str, Any]:
    config = real_config_dict()
    config.update(TINY_OVERRIDES)
    layers = config["num_hidden_layers"]
    config["layer_types"] = tiny_layer_types(layers)
    config["hybrid_layer_pattern"] = [
        int(t == "sliding_attention") for t in config["layer_types"]
    ]
    config["embed_scale"] = math.sqrt(config["hidden_size"])
    return config


def build_tiny_config():
    from ripple.reference.configuration_ricursive import RicursiveConfig

    raw = tiny_config_dict()
    for key in _SERIALIZATION_ONLY:
        raw.pop(key, None)
    raw.pop("hybrid_layer_pattern")
    return RicursiveConfig(**raw)


def real_config_diff(tiny_cfg) -> dict[str, tuple[Any, Any]]:
    """Return ``{key: (real, tiny)}`` for every real-config key that differs.

    Raises ``AssertionError`` if anything outside ``EXPECTED_DIFF_KEYS`` differs.
    """

    real = real_config_dict()
    tiny = tiny_cfg.to_dict()
    diff = {}
    for key, real_value in real.items():
        if key in _IGNORED_KEYS:
            continue
        tiny_value = tiny.get(key)
        if isinstance(real_value, list):
            tiny_value = list(tiny_value) if tiny_value is not None else None
        if tiny_value != real_value:
            diff[key] = (real_value, tiny_value)
    unexpected = set(diff) - EXPECTED_DIFF_KEYS
    assert not unexpected, f"unexpected architectural drift: {sorted(unexpected)}"
    return diff


def build_tiny_hf_model(
    seed: int = 0, dtype: torch.dtype = torch.float32, *, clamp_stress: bool = False
):
    """Seeded tiny ``RicursiveForCausalLM`` with non-trivial values everywhere.

    * ``value_res_lambda`` spans negative and positive values across layers
      (the HF default of ``0`` would make every blend ``sigmoid(0) = 0.5``).
    * Norm weights are non-unit.
    * ``clamp_stress`` scales ``gate_proj``/``up_proj`` so activations exceed
      ``ffn_clamp_threshold`` and the clamp is really exercised.
    """

    from ripple.reference.modeling_ricursive import RicursiveForCausalLM

    config = build_tiny_config()
    torch.manual_seed(seed)
    model = RicursiveForCausalLM(config)
    generator = torch.Generator().manual_seed(seed)
    layers = config.num_hidden_layers
    lambdas = torch.linspace(-1.5, 1.5, layers)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("value_res_lambda"):
                layer = int(name.split(".")[2])
                parameter.copy_(lambdas[layer])
                continue
            if parameter.ndim == 1:  # RMSNorm weights
                values = 1.0 + 0.2 * torch.randn(parameter.shape, generator=generator)
            else:
                fan_in = parameter.shape[-1]
                std = 1.0 if name.endswith("embed_tokens.weight") else fan_in**-0.5
                if (
                    name.endswith(("gate_proj.weight", "up_proj.weight"))
                    and clamp_stress
                ):
                    std = 1.5
                elif name.endswith(("q_proj.weight", "k_proj.weight")):
                    std *= 2.0
                values = std * torch.randn(parameter.shape, generator=generator)
            parameter.copy_(values)
    cast(Any, model).to(dtype=dtype)  # ty mis-types PreTrainedModel.to
    model.eval()
    return model
