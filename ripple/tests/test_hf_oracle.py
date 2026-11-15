"""CPU tests of the vendored HF Ripple model: cache, window and mask semantics."""

from __future__ import annotations

import math

import pytest
import torch
from transformers import DynamicCache

from ripple.reference.verify import REFERENCE_DIR, verify_reference_files
from ripple.tiny_config import (
    build_tiny_config,
    build_tiny_hf_model,
    real_config_diff,
    real_config_dict,
)

WINDOW = 8
FP32 = {"rtol": 1e-5, "atol": 1e-5}
# bf16 keeps ~3 significant digits and different kernels/summation orders give
# differences up to ~3e-2 (observed) on logits of magnitude ~4; 0.1 is the documented
# budget for prefill-vs-cached comparisons.
BF16 = {"rtol": 5e-2, "atol": 0.1}
DTYPES = [(torch.float32, FP32), (torch.bfloat16, BF16)]


def _ids(length: int, seed: int = 0, batch: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 250, (batch, length), generator=generator)


def _cache(model) -> DynamicCache:
    return DynamicCache(config=model.config)


def _logits(model, ids, cache=None):
    with torch.no_grad():
        return model(ids, past_key_values=cache, use_cache=cache is not None).logits


def _decode_all(model, ids):
    cache = _cache(model)
    return torch.cat([_logits(model, ids[:, i : i + 1], cache) for i in range(ids.shape[1])], 1)


@pytest.fixture(scope="module", params=DTYPES, ids=["fp32", "bf16"])
def model_and_tol(request):
    dtype, tol = request.param
    return build_tiny_hf_model(seed=1, dtype=dtype), tol


def test_reference_files_verify(tmp_path) -> None:
    assert set(verify_reference_files()) == {
        "config.json",
        "configuration_ricursive.py",
        "modeling_ricursive.py",
    }
    for name in ("config.json", "SHA256SUMS", "configuration_ricursive.py", "modeling_ricursive.py"):
        (tmp_path / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    (tmp_path / "modeling_ricursive.py").write_bytes(
        (REFERENCE_DIR / "modeling_ricursive.py").read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="modeling_ricursive.py"):
        verify_reference_files(tmp_path)


def test_tiny_config_mirrors_real_architecture() -> None:
    config = build_tiny_config()
    real = real_config_dict()
    diff = real_config_diff(config)
    assert set(diff) == {
        "bos_token_id", "embed_scale", "eos_token_id", "head_dim", "hidden_size",
        "hybrid_layer_pattern", "intermediate_size", "layer_types",
        "max_position_embeddings", "num_attention_heads", "num_hidden_layers",
        "num_key_value_heads", "pad_token_id", "sliding_window", "vocab_size",
    }  # fmt: skip
    assert config.layer_types == ["sliding_attention"] * 4 + ["full_attention"] + [
        "sliding_attention"
    ] * 4 + ["full_attention"]
    assert config.is_hybrid_swa and config.hybrid_layer_pattern == [1, 1, 1, 1, 0] * 2
    assert real["num_attention_heads"] // real["num_key_value_heads"] == 10
    assert config.num_attention_heads // config.num_key_value_heads == 10
    assert real["embed_scale"] == pytest.approx(math.sqrt(real["hidden_size"]))
    assert config.embed_scale == pytest.approx(math.sqrt(config.hidden_size))
    for key in (
        "use_gated_attention", "gate_full_precision", "use_valres",
        "ffn_clamp_threshold", "rms_norm_eps", "rope_theta", "tie_word_embeddings",
    ):  # fmt: skip
        assert getattr(config, key) == real[key], key
    for token in (config.bos_token_id, config.pad_token_id, *config.eos_token_id):
        assert 0 <= token < config.vocab_size
    assert 2 * WINDOW < 21 < 28 and 3 * WINDOW < 28


def test_model_instantiates_with_finite_logits(model_and_tol) -> None:
    model, _ = model_and_tol
    logits = _logits(model, _ids(21))
    assert logits.shape == (1, 21, 256)
    assert torch.isfinite(logits.float()).all()
    values = [layer.self_attn.value_res_lambda.item() for layer in model.model.layers]
    assert min(values) < 0 < max(values)


@pytest.mark.parametrize("length", [21, 28])
def test_prefill_matches_token_by_token_decode(model_and_tol, length) -> None:
    model, tol = model_and_tol
    ids = _ids(length)
    torch.testing.assert_close(_logits(model, ids), _decode_all(model, ids), **tol)


@pytest.mark.parametrize("chunks", [(5, 9, 7), (8, 8, 5), (13, 1, 7), (1, 20)])
def test_chunked_prefill_matches_one_shot(model_and_tol, chunks) -> None:
    model, tol = model_and_tol
    ids = _ids(sum(chunks), seed=3)
    cache = _cache(model)
    pieces, start = [], 0
    for size in chunks:
        pieces.append(_logits(model, ids[:, start : start + size], cache))
        start += size
    torch.testing.assert_close(torch.cat(pieces, 1), _logits(model, ids), **tol)


@pytest.mark.parametrize("prefix", [3, 9, 17])
def test_verify_block_on_warm_cache_matches_sequential_decode(model_and_tol, prefix) -> None:
    model, tol = model_and_tol
    ids = _ids(prefix + 5, seed=4)
    block_cache, step_cache = _cache(model), _cache(model)
    _logits(model, ids[:, :prefix], block_cache)
    _logits(model, ids[:, :prefix], step_cache)
    block = _logits(model, ids[:, prefix:], block_cache)
    steps = torch.cat(
        [_logits(model, ids[:, i : i + 1], step_cache) for i in range(prefix, prefix + 5)], 1
    )
    torch.testing.assert_close(block, steps, **tol)
    # The block's last row must also equal the full-sequence last row.
    torch.testing.assert_close(block[:, -1], _logits(model, ids)[:, -1], **tol)


def test_sliding_cache_evicts_but_global_cache_grows() -> None:
    model = build_tiny_hf_model(seed=1)
    ids = _ids(28)
    cache = _cache(model)
    _logits(model, ids, cache)
    for layer, kind in zip(cache.layers, model.config.layer_types, strict=True):
        held = layer.keys.shape[2]
        if kind == "full_attention":
            assert held == 28
        else:
            assert held < 28 and held <= WINDOW


def test_value_residual_lambda_changes_outputs_with_documented_blend() -> None:
    model = build_tiny_hf_model(seed=2)
    ids = _ids(12)
    base = _logits(model, ids)
    with torch.no_grad():
        model.model.layers[3].self_attn.value_res_lambda.add_(2.0)
    assert not torch.allclose(base, _logits(model, ids), atol=1e-3)

    # torch.lerp(residual, hidden, w) == (1 - w) * residual + w * hidden, w = sigmoid(lambda).
    residual, hidden = torch.randn(4, 6), torch.randn(4, 6)
    weight = torch.sigmoid(torch.tensor(0.7))
    torch.testing.assert_close(
        torch.lerp(residual, hidden, weight), (1 - weight) * residual + weight * hidden
    )


def test_gate_projection_changes_outputs() -> None:
    model = build_tiny_hf_model(seed=2)
    ids = _ids(12)
    base = _logits(model, ids)
    with torch.no_grad():
        model.model.layers[0].self_attn.g_proj.weight.mul_(-1.0)
    assert not torch.allclose(base, _logits(model, ids), atol=1e-3)


def test_ffn_clamp_bounds_gate_and_up_activations() -> None:
    model = build_tiny_hf_model(seed=2, clamp_stress=True)
    seen: dict[str, tuple[float, float]] = {}
    hooks = []
    for name in ("gate_proj", "up_proj"):
        module = getattr(model.model.layers[0].mlp, name)
        hooks.append(
            module.register_forward_hook(
                lambda _m, _i, out, key=name: seen.__setitem__(
                    key, (out.min().item(), out.max().item())
                )
            )
        )
    mlp = model.model.layers[0].mlp
    try:
        _logits(model, _ids(12))
    finally:
        for hook in hooks:
            hook.remove()
    threshold = model.config.ffn_clamp_threshold
    assert threshold == 20.0
    # Raw projections exceed the threshold, so the clamp is genuinely active...
    assert seen["gate_proj"][1] > threshold and seen["up_proj"][0] < -threshold
    assert seen["up_proj"][1] > threshold
    # ...and the MLP equals the explicitly clamped formula, not the unclamped one.
    x = torch.randn(1, 5, model.config.hidden_size)
    with torch.no_grad():
        gate, up = mlp.gate_proj(x), mlp.up_proj(x)
        expected = mlp.down_proj(
            torch.nn.functional.silu(gate.clamp(max=threshold)) * up.clamp(-threshold, threshold)
        )
        unclamped = mlp.down_proj(torch.nn.functional.silu(gate) * up)
        actual = mlp(x)
    assert gate.max() > threshold and up.abs().max() > threshold
    torch.testing.assert_close(actual, expected, **FP32)
    assert not torch.allclose(actual, unclamped, atol=1e-3)


def test_generate_matches_manual_incremental_loop() -> None:
    model = build_tiny_hf_model(seed=5)
    prompt = _ids(21, seed=6)
    steps = 10
    with torch.no_grad():
        generated = model.generate(
            prompt, max_new_tokens=steps, do_sample=False, eos_token_id=None, pad_token_id=255
        )
    cache = _cache(model)
    logits = _logits(model, prompt, cache)
    manual = []
    for _ in range(steps):
        token = logits[:, -1].argmax(-1, keepdim=True)
        manual.append(token)
        logits = _logits(model, token, cache)
    assert generated[:, prompt.shape[1] :].tolist() == torch.cat(manual, 1).tolist()
