"""Resin's pure-PyTorch Ripple against the vendored Hugging Face oracle (CPU)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from ripple.tiny_config import (
    build_tiny_hf_model,
    tiny_config_dict,
    tiny_layer_types,
)

WINDOW = 8
FP32: dict[str, Any] = {"rtol": 1e-5, "atol": 1e-5}
# Same documented budget as ripple/tests/test_hf_oracle.py.
BF16: dict[str, Any] = {"rtol": 5e-2, "atol": 0.1}
DTYPES = [(torch.float32, FP32), (torch.bfloat16, BF16)]


def _config():
    from resin_models_torch.ripple import RippleConfig

    return RippleConfig.from_dict(tiny_config_dict())


def _resin_from(hf, config=None):
    from resin_models_torch.ripple.model import RippleForCausalLM

    model = RippleForCausalLM(config or _config())
    model.load_state_dict(hf.state_dict(), strict=True)
    return model.to(next(hf.parameters()).dtype).eval()


def _ids(length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 250, (1, length), generator=generator)


def _hf_logits(hf, ids) -> torch.Tensor:
    with torch.no_grad():
        return hf(ids).logits


@pytest.fixture(scope="module", params=DTYPES, ids=["fp32", "bf16"])
def pair(request):
    dtype, tol = request.param
    hf = build_tiny_hf_model(seed=1, dtype=dtype)
    return hf, _resin_from(hf), tol


class TestRippleTorchImport:
    def test_import_loads_torch_but_not_checkpoint_stacks(self) -> None:
        script = """
import sys
sys.path.insert(0, 'models/torch')
import resin_models_torch.ripple
assert 'torch' in sys.modules
for name in ('transformers', 'tokenizers', 'huggingface_hub', 'safetensors'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_state_dict_keys_match_hf(self) -> None:
        hf = build_tiny_hf_model(seed=1)
        assert list(_resin_from(hf).state_dict()) == list(hf.state_dict())


class TestHfParity:
    def test_one_shot_prefill_past_two_windows(self, pair) -> None:
        hf, model, tol = pair
        ids = _ids(21)
        with torch.inference_mode():
            actual = model(ids)
        torch.testing.assert_close(actual, _hf_logits(hf, ids), **tol)

    def test_token_by_token_decode_past_three_windows(self, pair) -> None:
        hf, model, tol = pair
        ids = _ids(28)
        state = model.make_cache()
        with torch.inference_mode():
            rows = [model(ids[:, i : i + 1], state) for i in range(28)]
        torch.testing.assert_close(torch.cat(rows, 1), _hf_logits(hf, ids), **tol)
        # Sliding layers stay bounded; full layers keep everything.
        kinds = model.config.layer_types
        for kind, retained in zip(kinds, state.retained_tokens, strict=True):
            assert retained == (28 if kind == "full_attention" else WINDOW - 1)

    @pytest.mark.parametrize("chunks", [(5, 9, 7), (8, 8, 5), (13, 1, 7), (1, 20)])
    def test_chunked_prefill_and_continuation(self, pair, chunks) -> None:
        hf, model, tol = pair
        ids = _ids(sum(chunks), seed=3)
        state, rows, start = model.make_cache(), [], 0
        with torch.inference_mode():
            for size in chunks:
                rows.append(model(ids[:, start : start + size], state))
                start += size
        torch.testing.assert_close(torch.cat(rows, 1), _hf_logits(hf, ids), **tol)

    @pytest.mark.parametrize("prefix", [3, 14, 20])
    def test_verify_block_on_warm_cache(self, pair, prefix) -> None:
        hf, model, tol = pair
        ids = _ids(prefix + 5, seed=4)
        state = model.make_cache()
        with torch.inference_mode():
            model(ids[:, :prefix], state)
            block = model(ids[:, prefix:], state)
        torch.testing.assert_close(block, _hf_logits(hf, ids)[:, prefix:], **tol)

    def test_batched_decode_with_different_prefix_lengths(self, pair) -> None:
        hf, model, tol = pair
        short, long = _ids(5, seed=7), _ids(21, seed=8)
        states = [model.make_cache(), model.make_cache()]
        with torch.inference_mode():
            model(short[:, :-1], states[0])
            model(long[:, :-1], states[1])
            tokens = torch.cat((short[:, -1:], long[:, -1:]))
            batched = model.forward_batch(tokens, states)
        expected = torch.cat(
            [_hf_logits(hf, short)[:, -1:], _hf_logits(hf, long)[:, -1:]]
        )
        torch.testing.assert_close(batched, expected, **tol)

    def test_clamp_stress_weights_still_match(self) -> None:
        hf = build_tiny_hf_model(seed=3, clamp_stress=True)
        model = _resin_from(hf)
        ids = _ids(21)
        with torch.inference_mode():
            torch.testing.assert_close(model(ids), _hf_logits(hf, ids), **FP32)
        # The clamp must change the answer, so agreement above is meaningful.
        threshold = model.config.ffn_clamp_threshold
        assert threshold is not None
        object.__setattr__(model.model.config, "ffn_clamp_threshold", None)
        for layer in model.model.layers:
            layer.mlp.clamp = None
        with torch.inference_mode():
            assert not torch.allclose(model(ids), _hf_logits(hf, ids), atol=1e-2)

    def test_value_residual_direction_matters(self) -> None:
        hf = build_tiny_hf_model(seed=1)
        model = _resin_from(hf)
        ids = _ids(12)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.value_res_lambda.neg_()  # blend -> 1 - blend
        with torch.inference_mode():
            assert not torch.allclose(model(ids), _hf_logits(hf, ids), atol=1e-2)


def _save_checkpoint(directory, hf) -> None:
    """Two shards: embeddings/head/norm plus layers 0-4, then layers 5-9."""

    state = {k: v.contiguous() for k, v in hf.state_dict().items()}
    late = {
        k: v
        for k, v in state.items()
        if k.startswith("model.layers.") and int(k.split(".")[2]) >= 5
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


class TestSafetensorsLoading:
    def test_sharded_round_trip_matches_hf(self, tmp_path) -> None:
        from resin_models_torch.ripple import load_model

        hf = build_tiny_hf_model(seed=2)
        _save_checkpoint(tmp_path, hf)
        model, config = load_model(tmp_path, device="cpu")
        assert config.num_hidden_layers == 10
        ids = _ids(21)
        with torch.inference_mode():
            torch.testing.assert_close(model(ids), _hf_logits(hf, ids), **FP32)

    def test_five_layer_prefix_does_not_need_unused_shards(self, tmp_path) -> None:
        from resin_models_torch.ripple import load_model
        from ripple.reference.configuration_ricursive import RicursiveConfig
        from ripple.reference.modeling_ricursive import RicursiveForCausalLM

        hf = build_tiny_hf_model(seed=2)
        _save_checkpoint(tmp_path, hf)
        (tmp_path / "model-00002-of-00002.safetensors").unlink()
        raw = tiny_config_dict()
        for key in ("architectures", "auto_map", "model_type", "transformers_version"):
            raw.pop(key)
        raw.pop("hybrid_layer_pattern")
        raw.update(num_hidden_layers=5, layer_types=tiny_layer_types(10)[:5])
        reduced_hf = RicursiveForCausalLM(RicursiveConfig(**raw)).eval()
        reduced_hf.load_state_dict(
            {k: v for k, v in hf.state_dict().items() if k in reduced_hf.state_dict()}
        )

        with pytest.raises(FileNotFoundError, match="00002"):
            load_model(tmp_path, device="cpu")
        model, config = load_model(tmp_path, device="cpu", num_layers=5)
        assert config.num_hidden_layers == 5
        assert config.layer_types == ("sliding_attention",) * 4 + ("full_attention",)
        assert len(model.model.layers) == 5
        ids = _ids(21)
        with torch.inference_mode():
            torch.testing.assert_close(model(ids), _hf_logits(reduced_hf, ids), **FP32)

    def test_prefix_load_rejects_missing_tensor(self, tmp_path) -> None:
        from resin_models_torch.ripple import load_model

        _save_checkpoint(tmp_path, build_tiny_hf_model(seed=2))
        index_path = tmp_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"]["lm_head.weight"]
        index_path.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="lm_head.weight"):
            load_model(tmp_path, device="cpu", num_layers=5)


class _ThreadedAllReduce:
    """Sums rank-local partial results in lockstep, like a CPU process group."""

    def __init__(self, size: int) -> None:
        self._barrier = threading.Barrier(size)
        self._parts: list[torch.Tensor | None] = [None] * size

    def for_rank(self, rank: int):
        def reduce(partial: torch.Tensor) -> torch.Tensor:
            self._parts[rank] = partial
            self._barrier.wait()
            total = torch.zeros_like(partial)
            for part in self._parts:
                assert part is not None
                total = total + part
            self._barrier.wait()
            return total

        return reduce


class TestTensorParallel:
    @pytest.mark.parametrize("size", [2, 4])
    def test_sharded_forward_matches_unsharded(self, size) -> None:
        from resin.backend.cuda.parallel import TensorParallelLayout
        from resin.models.ripple.config import attention_geometry
        from resin_models_torch.ripple.model import RippleForCausalLM, shard_state_dict

        hf = build_tiny_hf_model(seed=4)
        full = _resin_from(hf)
        config = full.config
        reducer = _ThreadedAllReduce(size)
        ranks = []
        for rank in range(size):
            geometry = attention_geometry(config, TensorParallelLayout(rank, size))
            shard = RippleForCausalLM(
                config, geometry=geometry, all_reduce=reducer.for_rank(rank)
            )
            shard.load_state_dict(shard_state_dict(full.state_dict(), config, geometry))
            ranks.append(shard.eval())
        # The KV-head shape is local: replicated heads when ranks outnumber them.
        k_rows = (
            ranks[0].state_dict()["model.layers.0.self_attn.k_proj.weight"].shape[0]
        )
        assert k_rows == config.head_dim * (config.num_key_value_heads // size or 1)

        ids = _ids(24, seed=9)

        def run(model):
            state = model.make_cache()
            with torch.inference_mode():
                prefill = model(ids[:, :17], state)
                block = model(ids[:, 17:], state)
            return torch.cat((prefill, block), 1)

        with ThreadPoolExecutor(size) as pool:
            outputs = list(pool.map(run, ranks))
        expected = run(full)
        torch.testing.assert_close(expected, _hf_logits(hf, ids), **FP32)
        for output in outputs:
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_reduction_happens_before_post_norms(self) -> None:
        """Skipping the all-reduce must break equivalence: the post norms are nonlinear."""

        from resin.backend.cuda.parallel import TensorParallelLayout
        from resin.models.ripple.config import attention_geometry
        from resin_models_torch.ripple.model import RippleForCausalLM, shard_state_dict

        full = _resin_from(build_tiny_hf_model(seed=4))
        geometry = attention_geometry(full.config, TensorParallelLayout(0, 2))
        local = RippleForCausalLM(
            full.config, geometry=geometry, all_reduce=lambda t: t
        )
        local.load_state_dict(
            shard_state_dict(full.state_dict(), full.config, geometry)
        )
        ids = _ids(10)
        with torch.inference_mode():
            assert not torch.allclose(local.eval()(ids), full(ids), atol=1e-2)


class TestEngine:
    def _runner(self, hf, **kwargs):
        from resin_models_torch.ripple import RippleTorchRunner

        return RippleTorchRunner(_resin_from(hf), **kwargs)

    def _layout(self, config, chunk):
        from resin.models.ripple.plan import ripple_cache_layout

        return ripple_cache_layout(
            config, parent_capacity=64, block_tokens=4, prefill_chunk_size=chunk
        )

    def test_greedy_generation_scores_and_hybrid_blocks(self) -> None:
        from resin import BatchPhase, Engine, SamplingParams, SchedulerConfig

        hf = build_tiny_hf_model(seed=5)
        config = _config()
        runner = self._runner(
            hf, cache_layout=self._layout(config, 8), prefill_chunk_size=8
        )
        engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=8))
        tokens = _ids(21, seed=6)[0].tolist()
        steps = 8
        request = engine.submit(
            tokens,
            sampling_params=SamplingParams(
                max_output_tokens=steps, temperature=0, top_logprobs=3
            ),
        )
        for _ in range(steps):
            expected = _hf_logits(hf, torch.tensor([tokens]))[0, -1].log_softmax(-1)
            events = []
            while not events:  # chunked prefill emits nothing until the last chunk
                events = engine.step()
            (event,) = events
            assert event.token_id == expected.argmax().item()
            assert event.logprobs is not None
            assert event.logprobs.logprob == pytest.approx(
                expected[event.token_id].item(), abs=1e-4
            )
            assert [t for t, _ in event.logprobs.top_logprobs] == expected.topk(
                3
            ).indices.tolist()
            tokens.append(event.token_id)
        assert request.take_result() == tokens[-steps:]
        assert runner.live_cache_count == 0
        assert engine.cache_free_parent_count == 64

        # Block tables: full history always starts at block 0; the window group
        # evicts old blocks once positions pass the window.
        decode = [plan for plan in runner.plans if plan.phase == BatchPhase.DECODE]
        assert decode and all(
            plan.full.first_token_positions == (0,) for plan in decode
        )
        starts = [plan.window.first_token_positions[0] for plan in decode]
        assert starts[0] > 0 and starts == sorted(starts) and starts[-1] > starts[0]
        for plan in decode:
            position = int(plan.positions[0])
            assert plan.window.kv_lengths[0] - 1 <= position
            assert plan.full.kv_lengths[0] == position + 1
            # Window pages cover at least the window, but not the whole history.
            assert plan.window.kv_lengths[0] >= WINDOW
            assert plan.window.kv_lengths[0] < position + 1

    @pytest.mark.parametrize("chunk", [1, 5, 8])
    def test_prompt_scores_match_full_context(self, chunk) -> None:
        from resin import Engine, SamplingParams, SchedulerConfig

        hf = build_tiny_hf_model(seed=5)
        runner = self._runner(hf)
        tokens = _ids(21, seed=2)[0].tolist()
        distributions = _hf_logits(hf, torch.tensor([tokens]))[0, :-1].log_softmax(-1)
        expected = distributions[range(len(tokens) - 1), tokens[1:]]
        engine = Engine(
            runner, scheduler_config=SchedulerConfig(prefill_chunk_size=chunk)
        )
        request = engine.submit(
            tokens,
            sampling_params=SamplingParams(max_output_tokens=0, prompt_top_logprobs=2),
        )
        while engine.has_work():
            assert engine.step() == []
        scores = request.prompt_logprobs
        assert scores is not None and scores[0] is None
        assert [s.logprob for s in scores[1:] if s is not None] == pytest.approx(
            expected.tolist(), abs=1e-4
        )
        assert runner.forward_calls == -(-len(tokens) // chunk)
        assert runner.live_cache_count == 0

    def test_batched_decode_of_two_requests_matches_independent_runs(self) -> None:
        from resin import Engine, SamplingParams

        hf = build_tiny_hf_model(seed=5)
        runner = self._runner(hf)
        engine = Engine(runner)
        prompts = [_ids(4, seed=11)[0].tolist(), _ids(21, seed=12)[0].tolist()]
        requests = [
            engine.submit(
                list(prompt),
                sampling_params=SamplingParams(max_output_tokens=6, temperature=0),
            )
            for prompt in prompts
        ]
        while engine.has_work():
            engine.step()
        for prompt, request in zip(prompts, requests, strict=True):
            sequence = list(prompt)
            for _ in range(6):
                logits = _hf_logits(hf, torch.tensor([sequence]))[0, -1]
                sequence.append(int(logits.argmax()))
            assert request.take_result() == sequence[len(prompt) :]
        assert runner.live_cache_count == 0

    def test_runner_rejects_non_contiguous_positions(self) -> None:
        from resin import BatchPlan

        runner = self._runner(build_tiny_hf_model(seed=5))
        with pytest.raises(ValueError, match="non-contiguous positions for request 7"):
            runner.forward(
                BatchPlan(
                    sequence_ids=[7],
                    context_lengths=[3],
                    token_ids=[1, 2],
                    token_indptr=[0, 2],
                )
            )
