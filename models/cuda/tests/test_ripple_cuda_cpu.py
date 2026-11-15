"""Ripple's CUDA modules, loader, TP sharding and runner glue, all on CPU.

The vendored Hugging Face ``RicursiveForCausalLM`` is the oracle. FlashInfer is
replaced by a reference paged-attention stand-in that follows FlashInfer's
page-table contract, so the runner's plans, KV arena and Engine integration are
checked end to end without a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from resin import Engine, RunnerConfig, SamplingParams, SchedulerConfig
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.resources import CachePoolBinding
from resin.models.ripple.config import RippleConfig, attention_geometry
from resin.models.ripple.plan import ripple_cache_layout
from resin_models_cuda.ripple import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    RippleCudaForCausalLM,
    RippleCudaRunner,
    load_ripple_cuda_model,
    plan_ripple_kv_capacity,
)
from resin_models_cuda.ripple import checkpoint as ripple_checkpoint
from resin_models_cuda.ripple.cuda import _RippleKvArena
from resin_models_cuda.ripple.model import RippleRMSNorm
from resin_models_cuda.ripple.parallel import shard_attention, shard_mlp
from ripple.tiny_config import (
    build_tiny_hf_model,
    real_config_dict,
    tiny_config_dict,
    tiny_layer_types,
)

WINDOW = 8
FP32: dict[str, Any] = {"rtol": 1e-4, "atol": 1e-4}
# Same documented budget as ripple/tests/test_hf_oracle.py.
BF16: dict[str, Any] = {"rtol": 5e-2, "atol": 0.1}
DTYPES = [(torch.float32, FP32), (torch.bfloat16, BF16)]


def _config() -> RippleConfig:
    return RippleConfig.from_dict(tiny_config_dict())


def _ids(length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 250, (length,), generator=generator)


def _hf_logits(hf, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return hf(ids[None]).logits[0]


def _cuda_model(hf, *, pack: bool = False, config: RippleConfig | None = None):
    dtype = next(hf.parameters()).dtype
    model = RippleCudaForCausalLM(config or _config(), dtype=dtype)
    model.load_state_dict(hf.state_dict(), strict=True)
    model.materialize_buffers("cpu")
    if pack:
        model.pack_projections()
    return model.eval()


def dense_attend(config: RippleConfig):
    """Whole-prompt causal/window attention over flat tokens."""

    def attend(layer, query, key, value):
        sliding = config.layer_types[layer] == "sliding_attention"
        repeat = query.shape[1] // key.shape[1]
        count = len(query)
        rows = torch.arange(count)[:, None]
        columns = torch.arange(count)[None, :]
        allowed = columns <= rows
        if sliding:
            allowed &= columns >= rows - (config.sliding_window - 1)
        heads = [x.repeat_interleave(repeat, dim=1) for x in (key, value)]
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1)[None],
            heads[0].transpose(0, 1)[None],
            heads[1].transpose(0, 1)[None],
            attn_mask=allowed,
        )
        return output[0].transpose(0, 1)

    return attend


def _forward(model, ids, attend=None):
    positions = torch.arange(len(ids))
    with torch.inference_mode():
        hidden = model(ids, positions, attend or dense_attend(model.config))
        return model.logits(hidden)


class TestImport:
    def test_package_imports_without_kernel_packages(self) -> None:
        script = """
import sys
import resin_models_cuda.ripple as ripple
assert ripple.DEFAULT_REVISION
for name in ('flashinfer', 'triton', 'quack', 'cutlass', 'transformers'):
    assert name not in sys.modules, name
"""
        root = Path(__file__).resolve().parents[3]
        env = {**os.environ, "PYTHONPATH": f"{root}{os.pathsep}{root / 'models/cuda'}"}
        subprocess.run([sys.executable, "-c", script], check=True, cwd=root, env=env)

    def test_defaults_pin_the_reference_checkpoint(self) -> None:
        assert DEFAULT_MODEL == "ricursiveAI/Ripple-65B-Preview"
        assert DEFAULT_REVISION == "b172323ffd63e62de9ce1ba3830dfffe493c7671"

    def test_state_dict_keys_match_hf(self) -> None:
        hf = build_tiny_hf_model(seed=1)
        assert set(_cuda_model(hf).state_dict()) == set(hf.state_dict())


@pytest.fixture(scope="module", params=DTYPES, ids=["fp32", "bf16"])
def pair(request):
    dtype, tolerance = request.param
    return build_tiny_hf_model(seed=1, dtype=dtype), tolerance


class TestHfParity:
    @pytest.mark.parametrize("pack", [False, True])
    @pytest.mark.parametrize("length", [5, 21])
    def test_logits_match_across_sliding_and_global_layers(
        self, pair, pack, length
    ) -> None:
        hf, tolerance = pair
        model = _cuda_model(hf, pack=pack)
        ids = _ids(length)
        torch.testing.assert_close(
            _forward(model, ids), _hf_logits(hf, ids), **tolerance
        )

    def test_value_projection_stays_separate_from_qgk_when_valres(self) -> None:
        model = _cuda_model(build_tiny_hf_model(seed=1), pack=True)
        config = model.config
        width = config.head_dim
        for layer in model.model.layers:
            attention = layer.self_attn
            assert config.use_valres and hasattr(attention, "v_proj")
            assert not hasattr(attention, "q_proj")
            assert attention.qgk_proj is not None
            expected = (
                2 * config.num_attention_heads + config.num_key_value_heads
            ) * width
            assert attention.qgk_proj.out_features == expected
            assert layer.mlp.gate_up_proj is not None
            assert not hasattr(layer.mlp, "gate_proj")

    def test_value_residual_and_gating_matter(self) -> None:
        hf = build_tiny_hf_model(seed=1)
        model = _cuda_model(hf)
        ids = _ids(12)
        expected = _hf_logits(hf, ids)
        torch.testing.assert_close(_forward(model, ids), expected, **FP32)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.value_res_lambda.neg_()  # blend -> 1 - blend
        assert not torch.allclose(_forward(model, ids), expected, atol=1e-2)

    def test_clamp_is_active_and_matches(self) -> None:
        hf = build_tiny_hf_model(seed=3, clamp_stress=True)
        model = _cuda_model(hf, pack=True)
        ids = _ids(14)
        expected = _hf_logits(hf, ids)
        torch.testing.assert_close(_forward(model, ids), expected, **FP32)
        object.__setattr__(model.config, "ffn_clamp_threshold", None)
        for layer in model.model.layers:
            layer.mlp.clamp = None
        assert not torch.allclose(_forward(model, ids), expected, atol=1e-2)

    def test_embedding_scale_precedes_value_residual_seed_and_matches_hf(self) -> None:
        hf = build_tiny_hf_model(seed=1, dtype=torch.bfloat16)
        model = _cuda_model(hf)
        ids = _ids(32)
        with torch.no_grad():
            expected = hf.model.embed_tokens(ids[None]) * model.config.embed_scale
        torch.testing.assert_close(model.embed_tokens(ids), expected[0], rtol=0, atol=0)
        assert model.model.scale_tensor is not None
        assert model.model.scale_tensor.dtype == torch.float32

    def test_missing_buffers_are_rejected_not_silently_wrong(self) -> None:
        model = RippleCudaForCausalLM(_config(), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="materialized"):
            model.embed_tokens(_ids(3))


class TestRMSNormRounding:
    def test_matches_reference_exactly_and_single_rounding_stays_within_one_ulp(
        self,
    ) -> None:
        from ripple.reference.modeling_ricursive import RicursiveRMSNorm

        generator = torch.Generator().manual_seed(0)
        x = 3 * torch.randn(64, 320, generator=generator, dtype=torch.bfloat16)
        reference = RicursiveRMSNorm(320, eps=1e-6)
        mine = RippleRMSNorm(320, 1e-6)
        weight = 1 + 0.2 * torch.randn(320, generator=generator)
        with torch.no_grad():
            reference.weight.copy_(weight)
            mine.weight.copy_(weight)
        reference.bfloat16()
        expected, actual = reference(x), mine(x)
        assert torch.equal(actual, expected)

        # A fused kernel scaling in fp32 and rounding once (FlashInfer/Quack).
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + 1e-6
        )
        fused = (mine.weight.float() * normalized).to(torch.bfloat16)
        relative = (
            fused.float() - expected.float()
        ).abs() / expected.float().abs().clamp(min=1e-3)
        assert relative.max() <= 2**-7 + 2**-8  # about one BF16 ulp, never more
        assert not torch.equal(fused, expected)  # so the explicit form matters


def _save_checkpoint(directory, hf, *, shard_at: int = 5):
    """Embeddings/head/norm plus layers < shard_at, then the remaining layers."""

    state = {k: v.contiguous() for k, v in hf.state_dict().items()}
    late = {
        k: v
        for k, v in state.items()
        if k.startswith("model.layers.") and int(k.split(".")[2]) >= shard_at
    }
    early = {k: v for k, v in state.items() if k not in late}
    save_file(early, directory / "model-00001-of-00002.safetensors")
    save_file(late, directory / "model-00002-of-00002.safetensors")
    weight_map = {k: "model-00001-of-00002.safetensors" for k in early}
    weight_map |= {k: "model-00002-of-00002.safetensors" for k in late}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (directory / "config.json").write_text(json.dumps(tiny_config_dict()))


def _reduced_hf(hf, layers: int):
    from ripple.reference.configuration_ricursive import RicursiveConfig
    from ripple.reference.modeling_ricursive import RicursiveForCausalLM

    raw = tiny_config_dict()
    for key in ("architectures", "auto_map", "model_type", "transformers_version"):
        raw.pop(key)
    raw.pop("hybrid_layer_pattern")
    raw.update(num_hidden_layers=layers, layer_types=tiny_layer_types(10)[:layers])
    reduced = RicursiveForCausalLM(RicursiveConfig(**raw)).eval()
    reduced.load_state_dict(
        {k: v for k, v in hf.state_dict().items() if k in reduced.state_dict()}
    )
    return reduced


class _Group:
    """A CPU stand-in for a tensor-parallel group: threads meet to sum partials."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.barrier = threading.Barrier(size)
        self.parts: list[torch.Tensor | None] = [None] * size
        self.local = threading.local()

    def context(self, rank: int) -> TensorParallelContext:
        return cast(TensorParallelContext, _Context(self, rank))


class _Context:
    def __init__(self, group: _Group, rank: int) -> None:
        self.group, self.rank, self.size = group, rank, group.size
        self.layout = TensorParallelLayout(rank, group.size)
        self.enabled = group.size > 1

    def sum(self, tensor):
        group = self.group
        group.parts[self.rank] = tensor.clone()
        group.barrier.wait()
        total = sum(part for part in group.parts if part is not None)
        group.barrier.wait()
        tensor.copy_(total)
        return tensor


class TestCheckpoint:
    def test_five_layer_load_opens_only_the_first_shard(self, tmp_path) -> None:
        hf = build_tiny_hf_model(seed=2)
        _save_checkpoint(tmp_path, hf)
        (tmp_path / "model-00002-of-00002.safetensors").unlink()
        with pytest.raises(FileNotFoundError, match="00002"):
            load_ripple_cuda_model(tmp_path, device="cpu")
        model, config, raw = load_ripple_cuda_model(
            tmp_path, device="cpu", num_layers=5
        )
        assert config.num_hidden_layers == 5
        assert config.layer_types == ("sliding_attention",) * 4 + ("full_attention",)
        assert config.history_blocks_per_parent == 4
        assert raw["num_hidden_layers"] == 10
        assert len(model.model.layers) == 5
        assert model.model.layers[0].self_attn.qgk_proj is not None
        assert not model.training
        ids = _ids(21)
        reduced = _reduced_hf(hf, 5).bfloat16()
        torch.testing.assert_close(
            _forward(model, ids), _hf_logits(reduced, ids), **BF16
        )

    def test_missing_and_unexpected_tensors_are_rejected(self, tmp_path) -> None:
        _save_checkpoint(tmp_path, build_tiny_hf_model(seed=2))
        index_path = tmp_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"]["lm_head.weight"]
        index_path.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="lm_head.weight"):
            load_ripple_cuda_model(tmp_path, device="cpu", num_layers=5)
        index["weight_map"]["lm_head.weight"] = "model-00001-of-00002.safetensors"
        index["weight_map"]["model.layers.0.bogus.weight"] = index["weight_map"][
            "lm_head.weight"
        ]
        index_path.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="bogus"):
            load_ripple_cuda_model(tmp_path, device="cpu", num_layers=5)

    def test_tp_shards_on_cpu_before_the_only_device_transfer(
        self, tmp_path, monkeypatch
    ) -> None:
        hf = build_tiny_hf_model(seed=2)
        _save_checkpoint(tmp_path, hf)
        config = _config()
        full_q = config.num_attention_heads * config.head_dim
        transferred = []

        def transfer(module, device):
            assert torch.device(device).type == "cuda"
            assert all(t.device.type == "cpu" for t in module.parameters())
            assert all(t.device.type == "cpu" for t in module.buffers())
            for layer in module.model.layers:
                attention, mlp = layer.self_attn, layer.mlp
                assert attention.num_heads == config.num_attention_heads // 2
                assert attention.num_kv_heads == config.num_key_value_heads // 2
                for name in ("q_proj", "g_proj"):
                    assert getattr(attention, name).weight.shape[0] == full_q // 2
                assert attention.k_proj.weight.shape[0] == config.head_dim
                assert attention.v_proj.weight.shape[0] == config.head_dim
                assert cast(Any, attention.o_proj).local.weight.shape[1] == full_q // 2
                half = config.intermediate_size // 2
                assert mlp.gate_proj.weight.shape[0] == half
                assert mlp.up_proj.weight.shape[0] == half
                assert cast(Any, mlp.down_proj).local.weight.shape[1] == half
                assert attention.qgk_proj is None  # packing follows sharding
            transferred.append(True)
            return module

        def premature_allocation(*_args, **_kwargs):
            raise AssertionError("the full checkpoint must not be allocated on CUDA")

        monkeypatch.setattr(RippleCudaForCausalLM, "to", transfer)
        monkeypatch.setattr(RippleCudaForCausalLM, "to_empty", premature_allocation)
        group = _Group(2)
        context = group.context(1)
        model = load_ripple_cuda_model(
            tmp_path, device="cuda", tensor_parallel=context, attention_parallel=context
        )[0]
        assert transferred == [True]
        # Packing then fuses the rank-local shards.
        attention = model.model.layers[0].self_attn
        assert attention.qgk_proj is not None
        assert attention.qgk_proj.out_features == 2 * (full_q // 2) + config.head_dim
        gate_up = model.model.layers[0].mlp.gate_up_proj
        assert gate_up is not None
        assert gate_up.out_features == (config.intermediate_size)
        assert all(t.device.type == "cpu" for t in model.buffers())

    def test_resolve_pins_the_default_revision(self, monkeypatch) -> None:
        calls = []

        def download(repo, revision):
            calls.append((repo, revision))
            return tmp

        tmp = "path"
        monkeypatch.setattr(ripple_checkpoint, "_download_model", download)
        ripple_checkpoint.resolve_model_path(DEFAULT_MODEL)
        assert calls == [(DEFAULT_MODEL, DEFAULT_REVISION)]


def _tp_models(hf, size: int, *, pack: bool):
    full = _cuda_model(hf)
    group = _Group(size)
    ranks = []
    for rank in range(size):
        context = group.context(rank)
        local = _cuda_model(hf)
        for layer in local.model.layers:
            shard_attention(layer.self_attn, context)
            shard_mlp(layer.mlp, context)
        if pack:
            local.pack_projections()
        ranks.append(local)
    return full, ranks


class TestTensorParallel:
    @pytest.mark.parametrize("size", [2, 4])
    @pytest.mark.parametrize("pack", [False, True])
    def test_sharded_model_matches_unsharded(self, size, pack) -> None:
        hf = build_tiny_hf_model(seed=4, clamp_stress=True)
        full, ranks = _tp_models(hf, size, pack=pack)
        ids = _ids(24, seed=9)
        with ThreadPoolExecutor(size) as pool:
            outputs = list(pool.map(lambda model: _forward(model, ids), ranks))
        expected = _forward(full, ids)
        torch.testing.assert_close(expected, _hf_logits(hf, ids), **FP32)
        for output in outputs:
            torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    def test_attention_and_mlp_sums_match_before_post_norms(self) -> None:
        hf = build_tiny_hf_model(seed=4, clamp_stress=True)
        full, ranks = _tp_models(hf, 2, pack=True)
        ids = _ids(10, seed=5)
        config = full.config

        def run(model):
            with torch.no_grad():
                embedded = model.embed_tokens(ids)
                cos, sin = model.model.rotary_emb(
                    torch.arange(len(ids)), embedded.dtype
                )
                layer = model.model.layers[0]
                attended = layer.self_attn(
                    layer.input_layernorm(embedded),
                    cos,
                    sin,
                    embedded,
                    lambda q, k, v: dense_attend(config)(0, q, k, v),
                )
                fed = layer.mlp(layer.pre_feedforward_layernorm(embedded))
                return attended, fed

        expected = run(full)
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, ranks))
        for attended, fed in results:
            torch.testing.assert_close(attended, expected[0], rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(fed, expected[1], rtol=1e-3, atol=1e-3)
        # The clamp really bites: unclamped gate activations exceed the threshold.
        assert full.config.ffn_clamp_threshold == 20.0
        with torch.no_grad():
            layer = full.model.layers[0]
            raw = layer.mlp.gate_proj(
                layer.pre_feedforward_layernorm(full.embed_tokens(ids))
            )
        assert raw.max() > 20.0

    def test_sharding_after_packing_is_rejected(self) -> None:
        model = _cuda_model(build_tiny_hf_model(seed=1), pack=True)
        context = _Group(2).context(0)
        layer = model.model.layers[0]
        with pytest.raises(ValueError, match="before packing"):
            shard_attention(layer.self_attn, context)
        with pytest.raises(ValueError, match="before packing"):
            shard_mlp(layer.mlp, context)

    def test_unsharded_context_is_a_no_op(self) -> None:
        model = _cuda_model(build_tiny_hf_model(seed=1))
        layer = model.model.layers[0]
        context = _Group(1).context(0)
        assert shard_attention(layer.self_attn, context) is layer.self_attn
        assert shard_mlp(layer.mlp, context) is layer.mlp

    def test_pack_twice_is_rejected(self) -> None:
        model = _cuda_model(build_tiny_hf_model(seed=1), pack=True)
        with pytest.raises(RuntimeError, match="already packed"):
            model.model.layers[0].self_attn.pack_projections()
        with pytest.raises(RuntimeError, match="already packed"):
            model.model.layers[0].mlp.pack_projections()


class TestCachePlanning:
    @pytest.mark.parametrize(
        ("layers", "tp", "parent_bytes"),
        [
            (5, 1, 2 * 4 * 64 * 8 * 128 * 2),
            (5, 2, 2 * 4 * 64 * 4 * 128 * 2),
            (80, 1, 2 * 64 * 64 * 8 * 128 * 2),
            (80, 2, 2 * 64 * 64 * 4 * 128 * 2),
        ],
    )
    def test_real_geometry_parent_bytes(self, layers, tp, parent_bytes) -> None:
        config = RippleConfig.from_dict(real_config_dict()).same_shape_reduced(layers)
        total = 80 * 1024**3
        plan = plan_ripple_kv_capacity(
            total, total - 9 * 1024**3, 0.95, config=config, tensor_parallel_size=tp
        )
        assert plan.bytes_per_parent == parent_bytes
        assert config.history_blocks_per_parent == 4
        # 5 layers: 1 MiB/parent (TP1), 80 layers: 16 MiB/parent.
        assert plan.allocation_bytes == (plan.parent_capacity + 1) * parent_bytes
        assert plan.allocation_bytes <= plan.cache_budget_bytes + parent_bytes

    def test_invalid_tensor_parallelism_is_rejected(self) -> None:
        config = RippleConfig.from_dict(real_config_dict())
        with pytest.raises(ValueError, match="TP=3"):
            plan_ripple_kv_capacity(
                10**11, 10**10, 0.9, config=config, tensor_parallel_size=3
            )

    def test_arena_parent_and_child_views_share_one_slab(self) -> None:
        config = _config()
        packing = config.history_blocks_per_parent
        assert (config.local_layer_count, config.global_layer_count) == (8, 2)
        assert packing == 4
        arena = _RippleKvArena(
            "cpu", config, 3, kv_heads=2, page_size=4, dtype=torch.float32
        )
        assert arena.local_keys.shape == (8, 4, 4, 2, 16)
        assert arena.global_keys.shape == (2, 16, 4, 2, 16)
        assert arena.global_values.shape == arena.global_keys.shape
        for tensor in (arena.local_keys, arena.global_values):
            assert tensor.shape[-3:] == (4, 2, 16)
        # Every parent is a whole number of full-attention pages.
        storage = arena.storage
        arena.storage.zero_()
        arena.global_keys[1, 4 * 2 + 3] = 7.0  # parent 2, slot 3
        parents = [p for p in range(4) if storage[p].abs().sum() > 0]
        assert parents == [2]
        arena.storage.zero_()
        arena.local_values[5, 1] = 3.0  # window page == parent 1
        assert [p for p in range(4) if storage[p].abs().sum() > 0] == [1]
        # A parent's four history slots tile its bytes without overlap.
        covered = []
        for slot in range(4):
            arena.storage.zero_()
            arena.global_keys[:, 4 + slot] = 1.0
            arena.global_values[:, 4 + slot] = 2.0
            covered.append(arena.storage[1].reshape(-1) != 0)
            assert arena.storage[0].abs().sum() == 0
            assert arena.storage[2].abs().sum() == 0
        stacked = torch.stack(covered)
        assert stacked.sum(0).max() == 1 and stacked.all(0).sum() == 0
        assert stacked.any(0).all()

    def test_cache_storage_buffers_alias_live_rows(self) -> None:
        config = _config()
        arena = _RippleKvArena(
            "cpu", config, 4, kv_heads=2, page_size=4, dtype=torch.float32
        )
        arena.storage.zero_()
        (full,) = arena.cache_storage_buffers([(0, [(2, 1)])], prefix_length=3)
        assert set(full) == {f"{n}/{i}" for n in ("keys", "values") for i in range(2)}
        arena.global_keys[1, 2 * 4 + 1, :3] = 5.0
        assert full["keys/1"].view(torch.float32).eq(5.0).all()
        assert full["keys/1"].numel() == 3 * 2 * 16 * 4
        full["values/0"].view(torch.float32).fill_(9.0)
        assert arena.global_values[0, 2 * 4 + 1, :3].eq(9.0).all()
        assert arena.global_values[0, 2 * 4 + 1, 3:].eq(0.0).all()
        # Window block 1 of a 12-token prefix holds positions 4..7; the read
        # window (8 tokens) starts at 4, so every row is live. With a 16-token
        # prefix it starts at 8 and nothing in the block is.
        arena.local_keys[2, 3] = 4.0
        (live,) = arena.cache_storage_buffers(
            [(0, []), (1, [(3, 0)])], prefix_length=12
        )
        assert set(live) == {f"{n}/{i}" for n in ("keys", "values") for i in range(8)}
        assert live["keys/2"].numel() == 4 * 2 * 16 * 4
        assert live["keys/2"].view(torch.float32).eq(4.0).all()
        (expired,) = arena.cache_storage_buffers(
            [(0, []), (1, [(3, 0)])], prefix_length=16
        )
        assert expired == {}


class _FakeAttention:
    """FlashInfer's paged-attention page-table contract in plain torch."""

    def __init__(self, decode: bool, config: RippleConfig) -> None:
        self.decode, self.config = decode, config
        self.planned: dict[str, Any] = {}

    def plan(self, *args, **kwargs) -> None:
        assert kwargs["pos_encoding_mode"] == "NONE"
        # Ripple uses the standard scale; Gemma's 1.0 would be silently wrong.
        assert kwargs["sm_scale"] == pytest.approx(self.config.head_dim**-0.5)
        names = ["kv_indptr", "indices", "last", "q_heads", "kv_heads", "head_dim"]
        names = (["qo_indptr"] if not self.decode else []) + names + ["page_size"]
        self.planned = dict(zip(names, args, strict=True)) | kwargs

    def run(self, query: torch.Tensor, cache) -> torch.Tensor:
        plan = self.planned
        keys, values = cache
        page = plan["page_size"]
        if self.decode:
            query_indptr = torch.arange(len(plan["kv_indptr"]))
        else:
            query_indptr = plan["qo_indptr"]
        window = plan["window_left"]
        outputs = []
        for row in range(len(plan["kv_indptr"]) - 1):
            start, stop = plan["kv_indptr"][row : row + 2].tolist()
            pages = plan["indices"][start:stop].long()
            length = (len(pages) - 1) * page + int(plan["last"][row])
            k = keys[pages].reshape(-1, *keys.shape[2:])[:length]
            v = values[pages].reshape(-1, *values.shape[2:])[:length]
            q = query[query_indptr[row] : query_indptr[row + 1]]
            repeat = q.shape[1] // k.shape[1]
            k, v = (x.repeat_interleave(repeat, dim=1) for x in (k, v))
            positions = length - len(q) + torch.arange(len(q))[:, None]
            columns = torch.arange(length)[None, :]
            allowed = columns <= positions
            if window >= 0:
                allowed &= columns >= positions - window
            outputs.append(
                F.scaled_dot_product_attention(
                    q.transpose(0, 1)[None],
                    k.transpose(0, 1)[None],
                    v.transpose(0, 1)[None],
                    attn_mask=allowed,
                    scale=plan["sm_scale"],
                )[0].transpose(0, 1)
            )
        return torch.cat(outputs)


def _cpu_runner(
    hf, *, page_size: int = 4, chunk: int = 8, capacity: int = 64
) -> RippleCudaRunner:
    """A runner whose CUDA-only parts are replaced; all scheduling glue is real."""

    model = _cuda_model(hf, pack=True)
    config = model.config
    runner = RippleCudaRunner.__new__(RippleCudaRunner)
    runner.device = torch.device("cpu")
    runner.config, runner._model = config, model
    runner.geometry = attention_geometry(config, TensorParallelLayout())
    runner.page_size, runner.prefill_chunk_size = page_size, chunk
    runner.sliding_window = config.sliding_window
    runner.cuda_graph_prefill = runner.cuda_graph_decode = "disabled"
    runner._prefill_graph_active = False
    runner._metrics = None
    runner.logits_observer = None
    runner.cache_parent_capacity = capacity
    runner.cache_layout = ripple_cache_layout(
        config,
        parent_capacity=capacity,
        block_tokens=page_size,
        prefill_chunk_size=chunk,
    )
    runner.runner_config = RunnerConfig(
        max_batch_tokens=chunk,
        prefill_chunk_size=chunk,
        continuation_prefill=True,
        cache_layout=runner.cache_layout,
    )
    runner._cache_binding = CachePoolBinding(runner.cache_layout, resource="Ripple KV")
    runner._cache = _RippleKvArena(
        "cpu",
        config,
        capacity,
        kv_heads=runner.geometry.kv_heads,
        page_size=page_size,
        dtype=next(model.parameters()).dtype,
    )
    runner._window_prefill = _FakeAttention(False, config)
    runner._full_prefill = _FakeAttention(False, config)
    runner._window_decode = _FakeAttention(True, config)
    runner._full_decode = _FakeAttention(True, config)
    runner._prefill_feed_forward = tuple(
        layer.feed_forward for layer in model.model.layers
    )
    setattr(runner, "_cuda_failure", nullcontext())
    setattr(runner, "_check_cuda_graphs", lambda *_: None)
    setattr(runner, "_graph_forward", lambda *_: None)
    setattr(runner, "_sample_target", lambda logits, plan, step: logits.argmax(-1))
    return runner


class _Observer:
    def __init__(self) -> None:
        self.rows: list[tuple[list[int], list[int], torch.Tensor]] = []

    def __call__(self, plan, logits) -> None:
        lengths = np.diff(plan.token_indptr)
        last = [
            int(c + n - 1) for c, n in zip(plan.context_lengths, lengths, strict=True)
        ]
        self.rows.append(([int(s) for s in plan.sequence_ids], last, logits.clone()))


def _generate(runner, prompts, steps, chunk):
    observer = _Observer()
    runner.logits_observer = observer
    engine = Engine(
        runner,
        scheduler_config=SchedulerConfig(prefill_chunk_size=chunk),
        enable_metrics=False,
    )
    requests = [
        engine.submit(
            list(prompt),
            sampling_params=SamplingParams(max_output_tokens=steps, temperature=0),
        )
        for prompt in prompts
    ]
    while engine.has_work():
        engine.step()
    return [request.take_result() for request in requests], observer, engine


class TestRunnerAgainstHf:
    def _check(self, hf, prompts, steps, *, chunk=8, page_size=4):
        runner = _cpu_runner(hf, chunk=chunk, page_size=page_size)
        results, observer, engine = _generate(runner, prompts, steps, chunk)
        lengths = {i: len(p) for i, p in enumerate(prompts)}
        positions_seen = set()
        sequence_ids = sorted({s for ids, _, _ in observer.rows for s in ids})
        for ids, last, logits in observer.rows:
            for row, (sequence, position) in enumerate(zip(ids, last, strict=True)):
                index = sequence_ids.index(sequence)
                prompt = prompts[index]
                generated = results[index]
                if position < lengths[index] - 1 or position - lengths[index] >= steps:
                    continue
                context = torch.tensor(
                    list(prompt) + list(generated[: position - lengths[index] + 1])
                )
                expected = _hf_logits(hf, context)[-1]
                torch.testing.assert_close(logits[row], expected, **FP32)
                positions_seen.add((index, position))
        for index, prompt in enumerate(prompts):
            # Greedy tokens equal the HF argmax at every step.
            context = torch.tensor(list(prompt))
            for token in results[index]:
                assert token == int(_hf_logits(hf, context)[-1].argmax())
                context = torch.cat((context, torch.tensor([token])))
            assert {p for i, p in positions_seen if i == index} == set(
                range(len(prompt) - 1, len(prompt) - 1 + steps)
            )
        assert engine.cache_free_parent_count == runner.cache_parent_capacity

    def test_prefill_and_decode_past_three_windows(self) -> None:
        self._check(build_tiny_hf_model(seed=5), [_ids(21, seed=6).tolist()], 8)

    @pytest.mark.parametrize("chunk", [1, 5, 8])
    def test_chunked_prefill_continues_over_window_blocks(self, chunk) -> None:
        self._check(
            build_tiny_hf_model(seed=5),
            [_ids(26, seed=7).tolist()],
            4,
            chunk=chunk,
            page_size=4,
        )

    def test_window_edge_prompts(self) -> None:
        hf = build_tiny_hf_model(seed=5)
        for length in (WINDOW - 1, WINDOW, WINDOW + 1, 2 * WINDOW + 4):
            self._check(hf, [_ids(length, seed=length).tolist()], 3)

    def test_batched_decode_with_different_lengths(self) -> None:
        self._check(
            build_tiny_hf_model(seed=5),
            [
                _ids(5, seed=11).tolist(),
                _ids(21, seed=12).tolist(),
                _ids(9, seed=13).tolist(),
            ],
            6,
        )

    def test_page_size_not_dividing_window(self) -> None:
        self._check(
            build_tiny_hf_model(seed=5), [_ids(23, seed=14).tolist()], 5, page_size=3
        )

    def test_softmax_scale_is_the_standard_head_dim_scale(self) -> None:
        hf = build_tiny_hf_model(seed=5)
        runner = _cpu_runner(hf)
        assert RippleCudaRunner._sm_scale.fget(runner) == pytest.approx(16**-0.5)


class TestGraphInputs:
    def test_graph_page_inputs_pad_with_parent_zero(self) -> None:
        from resin.models.gemma4.plan import PagedAttentionPlan

        runner = _cpu_runner(build_tiny_hf_model(seed=5), page_size=4)
        plan = PagedAttentionPlan(
            qo_indptr=np.array([0, 1, 2], dtype=np.int32),
            kv_indptr=np.array([0, 2, 5], dtype=np.int32),
            page_indices=np.array([7, 8, 3, 4, 5], dtype=np.int32),
            last_page_lengths=np.array([2, 4], dtype=np.int32),
            append_slots=np.array([8 * 4 + 1, 5 * 4 + 3], dtype=np.int64),
            first_token_positions=(0, 0),
            kv_lengths=(6, 16),
        )
        inputs: dict[str, Any] = runner._graph_page_inputs(
            "full", plan, active_rows=2, graph_rows=4, width=3
        )
        assert inputs["full_kv_indptr"].tolist() == [0, 2, 5, 6, 7]
        assert inputs["full_page_indices"].tolist() == [
            7,
            8,
            3,
            4,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert inputs["full_last_page_lengths"].tolist() == [2, 4, 1, 1]
        assert inputs["full_append_pages"].tolist() == [8, 5, 0, 0]
        assert inputs["full_append_offsets"].tolist() == [1, 3, 2, 3]
        from resin.backend.cuda.failures import GraphInputError

        with pytest.raises(GraphInputError):
            runner._graph_page_inputs(
                "full", plan, active_rows=2, graph_rows=4, width=2
            )
