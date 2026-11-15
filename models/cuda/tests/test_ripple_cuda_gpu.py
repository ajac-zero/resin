"""Ripple CUDA runner on real GPUs against the vendored HF oracle.

Skipped without an SM80+ CUDA device. Models are tiny random-weight Ripples with
``sliding_window=128`` and 64-token pages, so 256+ token sequences cross two
windows and exercise window-page eviction. BF16 budgets match
``ripple/tests/test_hf_oracle.py`` with a slightly larger atol for GPU kernels.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from resin import Engine, Parallelism, SamplingParams, SchedulerConfig
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups
from ripple import tiny_config
from ripple.tiny_config import build_tiny_hf_model, tiny_config_dict

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 0),
    reason="Ripple CUDA tests require an SM80+ GPU",
)

WINDOW = 128
BF16: dict[str, Any] = {"rtol": 5e-2, "atol": 0.25}
# 64-token chunks keep only 64 + 127 window tokens per request, so 300-token
# prompts evict window pages during prefill and decode.
CHUNK = 64
GRAPH_SHAPES = [(1, 8), (4, 8)]


def _ids(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 250, (length,), generator=generator).tolist()


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    """A bf16 HF model on the GPU and its checkpoint directory."""

    overrides = {"sliding_window": WINDOW, "max_position_embeddings": 1024}
    with pytest.MonkeyPatch.context() as patch:
        for key, value in overrides.items():
            patch.setitem(tiny_config.TINY_OVERRIDES, key, value)
        hf = build_tiny_hf_model(seed=1, dtype=torch.bfloat16)
        raw = tiny_config_dict()
    directory = tmp_path_factory.mktemp("ripple")
    state = {k: v.contiguous() for k, v in hf.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(raw))
    return hf.to("cuda:0"), directory


def _hf_logits(hf, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return hf(torch.tensor([ids], device="cuda:0")).logits[0, -1].float().cpu()


class Recorder:
    """Records ``(sequence, position, logits)`` for every sampled last token."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, int, torch.Tensor]] = []
        self.request_ids: list[int] = []

    def __call__(self, plan, logits) -> None:
        for row, sequence in enumerate(plan.sequence_ids):
            end = int(plan.token_indptr[row + 1] - plan.token_indptr[row])
            position = int(plan.context_lengths[row]) + end - 1
            self.rows.append((int(sequence), position, logits[row].float().cpu()))


def _runner(path, *, graphs: bool, device="cuda:0", **options):
    from resin_models_cuda.ripple import RippleCudaRunner

    runner = RippleCudaRunner(
        path,
        device=device,
        page_size=64,
        prefill_chunk_size=options.pop("chunk", CHUNK),
        cache_parent_capacity=64,
        cuda_graph_decode="full" if graphs else "disabled",
        cuda_graph_prefill="breakable" if graphs else "disabled",
        cuda_graph_decode_shapes=GRAPH_SHAPES if graphs else None,
        **options,
    )
    if graphs:
        runner.capture_cuda_graphs()
    return runner


def _generate(runner, prompts, steps, *, chunk=CHUNK, engine_factory=None, **sampling):
    recorder = Recorder()
    runner.logits_observer = recorder
    options: dict[str, Any] = dict(
        scheduler_config=SchedulerConfig(prefill_chunk_size=chunk),
        enable_overlap=False,
        enable_metrics=False,
    )
    engine = (engine_factory or Engine)(runner, **options)
    requests = [
        engine.submit(
            prompt,
            sampling_params=SamplingParams(
                max_output_tokens=steps, temperature=0, **sampling
            ),
        )
        for prompt in prompts
    ]
    while engine.has_work():
        engine.step()
    recorder.request_ids = [int(request.request_id) for request in requests]
    results = [request.take_result() for request in requests]
    return results, recorder, engine


def _step_logits(recorder, prompts, steps):
    """``{(prompt index, step): logits}``; step k predicts position prompt_len + k."""

    table = {}
    for sequence, position, logits in recorder.rows:
        index = recorder.request_ids.index(sequence)
        step = position - (len(prompts[index]) - 1)
        if 0 <= step < steps:
            table[index, step] = logits
    assert len(table) == len(prompts) * steps
    return table


def _assert_matches_hf(hf, prompts, results, table, steps):
    for index, prompt in enumerate(prompts):
        context = list(prompt)
        for step in range(steps):
            expected = _hf_logits(hf, context)
            torch.testing.assert_close(table[index, step], expected, **BF16)
            # Greedy choices agree except at a near-tie the BF16 budget allows.
            top = expected.topk(2).values
            if top[0] - top[1] > 0.5:
                assert results[index][step] == int(expected.argmax())
            context.append(results[index][step])


class TestSingleGpu:
    @pytest.mark.parametrize("length", [64, 300])
    def test_prefill_and_decode_past_two_windows(self, oracle, length) -> None:
        hf, path = oracle
        prompts = [_ids(length, seed=length)]
        runner = _runner(path, graphs=False)
        results, recorder, _ = _generate(runner, prompts, 24)
        _assert_matches_hf(
            hf, prompts, results, _step_logits(recorder, prompts, 24), 24
        )

    def test_chunked_prefill_across_window_blocks(self, oracle) -> None:
        hf, path = oracle
        prompts = [_ids(300, seed=3)]
        runner = _runner(path, graphs=False, chunk=64)
        results, recorder, _ = _generate(runner, prompts, 8, chunk=64)
        _assert_matches_hf(hf, prompts, results, _step_logits(recorder, prompts, 8), 8)

    def test_batched_multi_sequence_decode(self, oracle) -> None:
        hf, path = oracle
        prompts = [_ids(37, seed=4), _ids(129, seed=5), _ids(260, seed=6)]
        runner = _runner(path, graphs=False)
        results, recorder, _ = _generate(runner, prompts, 10)
        _assert_matches_hf(
            hf, prompts, results, _step_logits(recorder, prompts, 10), 10
        )

    def test_full_decode_graph_matches_eager(self, oracle) -> None:
        _, path = oracle
        prompts = [_ids(37, seed=4), _ids(129, seed=5), _ids(300, seed=6)]
        eager, eager_recorder, _ = _generate(_runner(path, graphs=False), prompts, 12)
        graph_runner = _runner(path, graphs=True)
        graphed, graph_recorder, _ = _generate(graph_runner, prompts, 12)
        counts = graph_runner.cuda_graph_counts["decode"]
        assert counts["replay_count"] > 0
        eager_table = _step_logits(eager_recorder, prompts, 12)
        graph_table = _step_logits(graph_recorder, prompts, 12)
        for key, logits in eager_table.items():
            torch.testing.assert_close(graph_table[key], logits, rtol=2e-2, atol=0.1)
        for tokens, other in zip(eager, graphed, strict=True):
            assert tokens == other

    def test_engine_logprobs_match_hf(self, oracle) -> None:
        pytest.importorskip("cutlass", reason="CuTeDSL scoring needs cutlass-dsl")
        pytest.importorskip("quack", reason="CuTeDSL scoring needs quack-kernels")
        hf, path = oracle
        prompts = [_ids(64, seed=8)]
        runner = _runner(path, graphs=False)
        engine = Engine(runner, enable_overlap=False, enable_metrics=False)
        request = engine.submit(
            prompts[0],
            sampling_params=SamplingParams(
                max_output_tokens=4, temperature=0, top_logprobs=3
            ),
        )
        context = list(prompts[0])
        for _ in range(4):
            expected = _hf_logits(hf, context).log_softmax(-1)
            events = []
            while not events:
                events = engine.step()
            (event,) = events
            assert event.logprobs is not None
            assert event.logprobs.logprob == pytest.approx(
                expected[event.token_id].item(), abs=0.1
            )
            context.append(event.token_id)
        assert request.take_result() == context[-4:]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tp_worker(rank: int, port: int, path: str, output: str, prompts, steps) -> None:
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
    )
    groups = ParallelGroups.create(Parallelism(world_size=2, attn_tp=2))
    runner = _runner(
        Path(path), graphs=False, device=f"cuda:{rank}", parallel_groups=groups
    )
    expected_heads = runner.config.num_attention_heads // 2
    assert runner.geometry.q_heads == expected_heads
    worker = DistributedWorker(runner, groups)
    if groups.is_driver:
        results, recorder, engine = _generate(
            runner,
            prompts,
            steps,
            engine_factory=lambda _runner, **options: DistributedEngine(
                worker, **options
            ),
        )
        engine.close()
        torch.save(
            {
                "results": results,
                "rows": recorder.rows,
                "request_ids": recorder.request_ids,
                "model_bytes": runner.model_memory_bytes,
            },
            output,
        )
    else:
        worker.serve()
    groups.close()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_two_gpu_nccl_tp_matches_single_gpu_and_halves_weights(
    oracle, tmp_path
) -> None:
    _, path = oracle
    prompts = [_ids(300, seed=11), _ids(40, seed=12)]
    steps = 8
    single = _runner(path, graphs=False)
    expected, recorder, _ = _generate(single, prompts, steps)
    expected_table = _step_logits(recorder, prompts, steps)
    single_bytes = single.model_memory_bytes
    del single
    torch.cuda.empty_cache()

    output = tmp_path / "tp.pt"
    mp.spawn(
        _tp_worker,
        args=(_free_port(), str(path), str(output), prompts, steps),
        nprocs=2,
        join=True,
    )
    saved = torch.load(output, weights_only=False)
    recorder = type(
        "R", (), {"rows": saved["rows"], "request_ids": saved["request_ids"]}
    )
    table = _step_logits(recorder, prompts, steps)
    for key, logits in expected_table.items():
        torch.testing.assert_close(table[key], logits, rtol=2e-2, atol=0.1)
    assert saved["results"] == expected
    # Attention and MLP weights halve; embeddings, norms and the LM head replicate.
    assert saved["model_bytes"] < 0.75 * single_bytes
