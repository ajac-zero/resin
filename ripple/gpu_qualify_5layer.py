#!/usr/bin/env python
"""Qualify the Ripple SM80 CUDA runner on GPUBox with the real 5-layer checkpoint.

One self-contained, offline run (no downloads) that

1. records the environment and hardware,
2. builds the vendored Hugging Face ``RicursiveForCausalLM`` oracle on ``cuda:0``
   with only the first ``--num-layers`` layers (5 = one ``[S, S, S, S, G]``
   period at full width: ~8.3 GB BF16) and records golden BF16 logits and greedy
   trajectories,
3. runs Resin's ``RippleCudaRunner`` on one GPU (``TP=1``), eager and with CUDA
   graphs, and on two GPUs over real NCCL (``TP=2``), through ``Engine`` /
   ``DistributedEngine``,
4. compares every step's logits against the oracle (and TP=2 / graph against TP=1
   and eager), applies pass gates and writes a JSON report.

Example (GPUBox, Standard_NC96ads_A100_v4)::

    python ripple/gpu_qualify_5layer.py \\
        --checkpoint-dir /data/ripple/hf/hub/models--ricursiveAI--Ripple-65B-Preview/snapshots/b172323ffd63e62de9ce1ba3830dfffe493c7671 \\
        --num-layers 5 --mode all --output-json /data/ripple/qualify_5layer.json

``--dry-run`` validates arguments, the checkpoint index and the case table on a
machine without a GPU.

Logit comparison: step ``k`` of a case is the logits that predict the ``k``-th
generated token. Resin generates greedily; logits are compared up to and
including the first step where its token differs from the oracle's, because the
context is still identical there. A divergence is a benign near-tie only when
the oracle's top-2 logit gap at that step is within ``--tie-margin``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.metadata
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Generator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "models" / "cuda"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

DEFAULT_CHECKPOINT = (
    "/data/ripple/hf/hub/models--ricursiveAI--Ripple-65B-Preview/snapshots/"
    "b172323ffd63e62de9ce1ba3830dfffe493c7671"
)
SCHEMA_VERSION = 1
# Vendored reference config keys that are not RicursiveConfig constructor input.
_SERIALIZATION_ONLY = (
    "architectures",
    "auto_map",
    "model_type",
    "transformers_version",
    "hybrid_layer_pattern",
)


# --------------------------------------------------------------------------- #
# Cases, tolerances and prompts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    """Prompts of the given lengths, each generating ``steps`` greedy tokens."""

    name: str
    lengths: tuple[int, ...]
    steps: int
    batched: bool = False  # submit all prompts together (decoded as one batch)


def default_cases(sliding_window: int = 1024) -> tuple[Case, ...]:
    """Cases A, B and C for a model with the given sliding window.

    B spans the window boundary (``w-1``, ``w``, ``w+1``) and ``2w+52`` (> 2
    windows: window pages are evicted), decoding 32 steps past it. C decodes a
    short, a just-over-window and a long request together.
    """

    w = sliding_window
    return (
        Case("A", (64,), 16),
        Case("B", (w - 1, w, w + 1, 2 * w + 52), 32),
        Case("C", (37, w + 1, w + 576), 16, batched=True),
    )


def make_prompt(length: int, seed: int, vocab_size: int) -> list[int]:
    """Deterministic synthetic token ids (same on every machine)."""

    generator = torch.Generator().manual_seed(seed)
    high = min(vocab_size, 50_000)
    low = min(1000, vocab_size // 4)  # skip the lowest ids; tiny vocabularies shrink it
    return torch.randint(low, high, (length,), generator=generator).tolist()


def case_prompts(case: Case, vocab_size: int) -> list[list[int]]:
    seed_base = {"A": 1000, "B": 2000, "C": 3000}.get(case.name, 9000)
    return [
        make_prompt(length, seed_base + length, vocab_size) for length in case.lengths
    ]


@dataclass(frozen=True)
class Tolerances:
    min_cosine: float = 0.9999
    max_top5_logprob_err: float = 0.05
    max_abs_err: float = 0.25
    # Resin vs Resin (eager vs graph, TP=2 vs TP=1): only reduction order differs.
    cross_min_cosine: float = 0.9999
    cross_max_abs_err: float = 0.1
    # An oracle top-2 logit gap at or below this makes a token flip a benign tie.
    tie_margin: float = 0.25


# --------------------------------------------------------------------------- #
# Metrics (pure torch; unit-tested on CPU)
# --------------------------------------------------------------------------- #

REL_FLOOR = 1.0  # relative error is |a - e| / max(|e|, REL_FLOOR)


def compare_logits(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Error metrics of ``[steps, vocab]`` logits against a reference."""

    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError(
            f"logit shapes differ: {tuple(actual.shape)} vs {tuple(expected.shape)}"
        )
    a, e = actual.double(), expected.double()
    diff = (a - e).abs()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=-1)
    top5 = expected.topk(5, dim=-1).indices
    actual_logprob = a.log_softmax(-1).gather(-1, top5)
    expected_logprob = e.log_softmax(-1).gather(-1, top5)
    return {
        "steps_compared": int(len(actual)),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "max_rel_err": float((diff / e.abs().clamp(min=REL_FLOOR)).max()),
        "cosine_similarity": float(cosine.min()),
        "top1_match_rate": float((a.argmax(-1) == e.argmax(-1)).double().mean()),
        "max_top5_logprob_abs_err": float(
            (actual_logprob - expected_logprob).abs().max()
        ),
    }


def trajectory_match_length(actual: Sequence[int], expected: Sequence[int]) -> int:
    """Length of the common prefix of two token sequences."""

    for index, (left, right) in enumerate(zip(actual, expected, strict=False)):
        if left != right:
            return index
    return min(len(actual), len(expected))


def top2_gaps(logits: torch.Tensor) -> list[float]:
    top = logits.topk(2, dim=-1).values
    return (top[:, 0] - top[:, 1]).tolist()


def compared_steps(match_length: int, steps: int) -> int:
    """Steps whose context is identical: through the first divergent step."""

    return steps if match_length >= steps else match_length + 1


def evaluate(
    actual_tokens: Sequence[int],
    actual_logits: torch.Tensor,
    expected_tokens: Sequence[int],
    expected_logits: torch.Tensor,
    *,
    tolerance: Tolerances,
    cross: bool = False,
) -> dict[str, Any]:
    """Compare one generation with a reference and apply the pass gates."""

    steps = len(expected_tokens)
    match = trajectory_match_length(actual_tokens, expected_tokens)
    count = compared_steps(match, steps)
    metrics: dict[str, Any] = compare_logits(
        actual_logits[:count], expected_logits[:count]
    )
    metrics["trajectory_match_length"] = match
    metrics["steps"] = steps
    reasons: list[str] = []
    min_cosine = tolerance.cross_min_cosine if cross else tolerance.min_cosine
    max_abs = tolerance.cross_max_abs_err if cross else tolerance.max_abs_err
    if not metrics["cosine_similarity"] > min_cosine:
        reasons.append(f"cosine {metrics['cosine_similarity']:.6f} <= {min_cosine}")
    if not metrics["max_abs_err"] < max_abs:
        reasons.append(f"max_abs_err {metrics['max_abs_err']:.4f} >= {max_abs}")
    if not cross and not (
        metrics["max_top5_logprob_abs_err"] < tolerance.max_top5_logprob_err
    ):
        reasons.append(
            f"top5 logprob err {metrics['max_top5_logprob_abs_err']:.4f} >= "
            f"{tolerance.max_top5_logprob_err}"
        )
    if match < steps:
        gap = top2_gaps(expected_logits[match : match + 1])[0]
        metrics["divergence_reference_gap"] = gap
        if gap > tolerance.tie_margin:
            reasons.append(
                f"greedy trajectory diverged at step {match} (reference gap {gap:.3f})"
            )
        else:
            metrics["benign_tie"] = True
    metrics["passed"] = not reasons
    metrics["failures"] = reasons
    return metrics


# --------------------------------------------------------------------------- #
# Environment inventory
# --------------------------------------------------------------------------- #


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {"sha": git("rev-parse", "HEAD"), "dirty": bool(status)}


def _driver_version() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return output[0] if output else None


def collect_environment() -> dict[str, Any]:
    gpus = []
    nccl = None
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "vram_bytes": properties.total_memory,
                }
            )
        with contextlib.suppress(Exception):
            from torch.cuda import nccl as torch_nccl

            nccl = ".".join(map(str, torch_nccl.version()))
    return {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": _driver_version(),
        "nccl": nccl,
        "flashinfer": _version("flashinfer-python") or _version("flashinfer"),
        "triton": _version("triton"),
        "transformers": _version("transformers"),
        "safetensors": _version("safetensors"),
        "resin_git": _git_sha(),
        "gpus": gpus,
        "NCCL_GRAPH_FILE": os.environ.get("NCCL_GRAPH_FILE"),
    }


# --------------------------------------------------------------------------- #
# Hugging Face oracle
# --------------------------------------------------------------------------- #


def _needed_shards(
    checkpoint_dir: Path, wanted: Sequence[str]
) -> dict[Path, list[str]]:
    """Group the wanted tensors by shard, never touching shards they do not need."""

    index = checkpoint_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        from safetensors import safe_open

        single = checkpoint_dir / "model.safetensors"
        with safe_open(str(single), framework="pt") as handle:
            weight_map = dict.fromkeys(handle.keys(), single.name)
    if missing := [name for name in wanted if name not in weight_map]:
        raise ValueError(f"checkpoint is missing tensors: {missing[:8]}")
    by_file: dict[Path, list[str]] = {}
    for name in wanted:
        by_file.setdefault(checkpoint_dir / weight_map[name], []).append(name)
    for path in by_file:
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint shard: {path}")
    return by_file


def load_hf_oracle(
    checkpoint_dir: Path,
    num_layers: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    """Vendored ``RicursiveForCausalLM`` with the first ``num_layers`` layers.

    Built on ``meta``; only the needed tensors are read from the shards that
    hold them, straight into ``device`` allocations.
    """

    from safetensors import safe_open

    from ripple.reference.configuration_ricursive import RicursiveConfig
    from ripple.reference.modeling_ricursive import RicursiveForCausalLM
    from ripple.reference.verify import verify_reference_files

    verify_reference_files()
    raw = json.loads((checkpoint_dir / "config.json").read_text())
    layer_types = list(raw["layer_types"][:num_layers])
    for key in _SERIALIZATION_ONLY:
        raw.pop(key, None)
    raw.update(num_hidden_layers=num_layers, layer_types=layer_types)
    config = RicursiveConfig(**raw)
    with torch.device("meta"):
        model = RicursiveForCausalLM(config)
    model = cast(Any, model).to(dtype).to_empty(device=device)
    parameters = dict(model.named_parameters())
    for path, names in _needed_shards(checkpoint_dir, list(parameters)).items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if tuple(tensor.shape) != tuple(parameters[name].shape):
                    raise ValueError(f"shape mismatch for {name}")
                with torch.no_grad():
                    parameters[name].copy_(tensor)
    return model.eval()


@torch.no_grad()
def hf_greedy(
    model, prompt: Sequence[int], steps: int, device: str | torch.device
) -> tuple[list[int], torch.Tensor]:
    """Greedy tokens and the fp32 logits that predicted each (``[steps, vocab]``)."""

    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    ids = torch.tensor([list(prompt)], device=device)
    output = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    rows, tokens = [], []
    logits = output.logits[0, -1].float()
    for step in range(steps):
        rows.append(logits.cpu())
        token = int(logits.argmax())
        tokens.append(token)
        if step + 1 < steps:
            output = model(
                torch.tensor([[token]], device=device),
                past_key_values=cache,
                use_cache=True,
            )
            logits = output.logits[0, -1].float()
    return tokens, torch.stack(rows)


def case_key(case: Case, length: int) -> str:
    return f"{case.name}[{length}]"


def build_golden(
    model,
    cases: Sequence[Case],
    vocab_size: int,
    device: str | torch.device,
    log: Callable[[str], None],
) -> dict[str, dict[str, Any]]:
    golden: dict[str, dict[str, Any]] = {}
    for case in cases:
        for length, prompt in zip(
            case.lengths, case_prompts(case, vocab_size), strict=True
        ):
            started = time.perf_counter()
            tokens, logits = hf_greedy(model, prompt, case.steps, device)
            key = case_key(case, length)
            golden[key] = {
                "tokens": tokens,
                "logits": logits,
                "seconds": time.perf_counter() - started,
            }
            log(f"  oracle {key}: {time.perf_counter() - started:.1f}s")
    return golden


# --------------------------------------------------------------------------- #
# Driving a Resin engine
# --------------------------------------------------------------------------- #


class Recorder:
    """Logits observer: every sampled last-token row, with its sequence position."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, int, torch.Tensor]] = []

    def __call__(self, plan: Any, logits: torch.Tensor) -> None:
        for row, sequence in enumerate(plan.sequence_ids):
            count = int(plan.token_indptr[row + 1] - plan.token_indptr[row])
            position = int(plan.context_lengths[row]) + count - 1
            self.rows.append((int(sequence), position, logits[row].float().cpu()))


def step_logits(
    recorder: Recorder, request_ids: Sequence[int], lengths: Sequence[int], steps: int
) -> list[torch.Tensor]:
    """Per request, ``[steps, vocab]`` logits; step ``k`` predicts token ``k``."""

    out: list[dict[int, torch.Tensor]] = [{} for _ in request_ids]
    index = {int(request): i for i, request in enumerate(request_ids)}
    for sequence, position, logits in recorder.rows:
        slot = index.get(sequence)
        if slot is None:
            continue
        step = position - (lengths[slot] - 1)
        if 0 <= step < steps:
            out[slot][step] = logits
    for slot, rows in enumerate(out):
        if sorted(rows) != list(range(steps)):
            raise RuntimeError(
                f"request {slot}: recorded steps {sorted(rows)}, expected 0..{steps - 1}"
            )
    return [torch.stack([rows[step] for step in range(steps)]) for rows in out]


def run_case(
    engine: Any,
    set_observer: Callable[[Recorder | None], None],
    case: Case,
    vocab_size: int,
) -> dict[str, dict[str, Any]]:
    """Run one case on an engine; return ``{case_key: {tokens, logits, seconds}}``."""

    from resin import SamplingParams

    prompts = case_prompts(case, vocab_size)
    groups = (
        [list(range(len(prompts)))]
        if case.batched
        else [[index] for index in range(len(prompts))]
    )
    results: dict[str, dict[str, Any]] = {}
    for group in groups:
        recorder = Recorder()
        set_observer(recorder)
        started = time.perf_counter()
        requests = [
            engine.submit(
                prompts[index],
                sampling_params=SamplingParams(
                    max_output_tokens=case.steps, temperature=0.0
                ),
            )
            for index in group
        ]
        while engine.has_work():
            engine.step()
        seconds = time.perf_counter() - started
        set_observer(None)
        lengths = [len(prompts[index]) for index in group]
        logits = step_logits(
            recorder, [request.request_id for request in requests], lengths, case.steps
        )
        for index, request, rows in zip(group, requests, logits, strict=True):
            results[case_key(case, len(prompts[index]))] = {
                "tokens": request.take_result(),
                "logits": rows,
                "seconds": seconds,
            }
    return results


def run_cases(
    engine: Any,
    set_observer: Callable[[Recorder | None], None],
    cases: Sequence[Case],
    vocab_size: int,
    log: Callable[[str], None],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        results.update(run_case(engine, set_observer, case, vocab_size))
        log(f"  case {case.name}: {', '.join(map(str, case.lengths))} tokens done")
    return results


# --------------------------------------------------------------------------- #
# Resin runners
# --------------------------------------------------------------------------- #


def graph_shapes(cases: Sequence[Case], page_size: int) -> list[tuple[int, int]]:
    """A small decode-graph catalog covering the cases (batch 1, 2, 4)."""

    longest = max(max(case.lengths) + case.steps for case in cases)
    width = -(-longest // page_size) + 2
    return [(batch, width) for batch in (1, 2, 4)]


def build_runner(
    args: argparse.Namespace,
    cases: Sequence[Case],
    *,
    graphs: bool,
    device: str,
    parallel_groups: Any = None,
):
    from resin_models_cuda.ripple import RippleCudaRunner

    runner = RippleCudaRunner(
        Path(args.checkpoint_dir),
        device=device,
        num_layers=args.num_layers,
        prefill_chunk_size=args.prefill_chunk_size,
        cache_parent_capacity=args.cache_parent_capacity,
        cuda_graph_decode="full" if graphs else "disabled",
        cuda_graph_prefill="breakable" if graphs else "disabled",
        cuda_graph_decode_shapes=(
            graph_shapes(cases, args.page_size) if graphs else None
        ),
        page_size=args.page_size,
        parallel_groups=parallel_groups,
    )
    if graphs:
        runner.capture_cuda_graphs()
    return runner


def layer_projection_bytes(model: torch.nn.Module) -> int:
    """Bytes of the attention and MLP weights, which tensor parallelism shards."""

    total = 0
    for layer in cast(Any, model).model.layers:
        for module in (layer.self_attn, layer.mlp):
            for name, parameter in module.named_parameters():
                if "norm" not in name and not name.endswith("value_res_lambda"):
                    total += parameter.numel() * parameter.element_size()
    return total


def runner_footprint(runner: Any) -> dict[str, Any]:
    plan = runner.cache_capacity_plan
    return {
        "model_bytes": runner.model_memory_bytes,
        "layer_projection_bytes": layer_projection_bytes(runner.model),
        "load_peak_allocated_bytes": runner.model_load_peak_bytes,
        "kv_bytes_per_parent": plan.bytes_per_parent,
        "kv_parent_capacity": runner.cache_parent_capacity,
        "kv_allocation_bytes": plan.allocation_bytes,
        "q_heads_local": runner.geometry.q_heads,
        "kv_heads_local": runner.geometry.kv_heads,
        "allocated_bytes": torch.cuda.memory_allocated(runner.device),
    }


def engine_options(args: argparse.Namespace) -> dict[str, Any]:
    from resin import SchedulerConfig

    return {
        "max_running_requests": 8,
        "scheduler_config": SchedulerConfig(prefill_chunk_size=args.prefill_chunk_size),
        "enable_overlap": False,
        "enable_metrics": False,
    }


def graph_report(runner: Any) -> dict[str, Any]:
    with contextlib.suppress(Exception):
        return json.loads(json.dumps(runner.cuda_graph_counts, default=str))
    return {}


def run_tp1(
    args: argparse.Namespace,
    cases: Sequence[Case],
    vocab_size: int,
    log: Callable[[str], None],
    *,
    graphs: bool,
) -> dict[str, Any]:
    from resin import Engine

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    runner = build_runner(args, cases, graphs=graphs, device="cuda:0")
    load_seconds = time.perf_counter() - started
    log(f"  runner ready in {load_seconds:.1f}s")

    def observe(recorder: Recorder | None, target: Any = runner) -> None:
        target.logits_observer = recorder

    engine = Engine(runner, **engine_options(args))
    try:
        results = run_cases(engine, observe, cases, vocab_size, log)
    finally:
        engine.close()
    report = {
        "load_seconds": load_seconds,
        "footprint": runner_footprint(runner),
        "graph_counts": graph_report(runner),
        "results": results,
    }
    if graphs:
        runner.close_cuda_graphs()
    del engine, runner
    torch.cuda.empty_cache()
    return report


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def tp2_worker(rank: int, port: int, config: dict[str, Any]) -> None:
    """One TP=2 rank: NCCL init, runner per mode, ``DistributedEngine`` on rank 0."""

    import torch.distributed as dist

    from resin import Parallelism
    from resin.backend.distributed import DistributedEngine, DistributedWorker
    from resin.backend.topology import ParallelGroups

    args = argparse.Namespace(**config["args"])
    cases = [
        Case(c["name"], tuple(c["lengths"]), c["steps"], c["batched"])
        for c in config["cases"]
    ]
    out_dir = Path(config["out_dir"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
    )
    groups = ParallelGroups.create(Parallelism(world_size=2, attn_tp=2))
    summary: dict[str, Any] = {"rank": rank, "modes": {}}
    log = print if rank == 0 else (lambda _message: None)
    try:
        for mode in config["modes"]:
            graphs = mode == "graph"
            entry: dict[str, Any] = {}
            summary["modes"][mode] = entry
            try:
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(rank)
                runner = build_runner(
                    args,
                    cases,
                    graphs=graphs,
                    device=f"cuda:{rank}",
                    parallel_groups=groups,
                )
                entry["load_seconds"] = time.perf_counter() - started
                entry["footprint"] = runner_footprint(runner)
                worker = DistributedWorker(runner, groups)
                if groups.is_driver:
                    engine = DistributedEngine(worker, **engine_options(args))

                    def observe(recorder, runner=runner):
                        runner.logits_observer = recorder

                    try:
                        results = run_cases(
                            engine, observe, cases, config["vocab_size"], log
                        )
                    finally:
                        engine.close()
                    entry["graph_counts"] = graph_report(runner)
                    torch.save(results, out_dir / f"tp2_{mode}_results.pt")
                else:
                    worker.serve()
                entry["status"] = "ok"
                if graphs:
                    runner.close_cuda_graphs()
                del worker, runner
                torch.cuda.empty_cache()
            except Exception:
                entry["status"] = "error"
                entry["error"] = traceback.format_exc()
                print(
                    f"[rank {rank}] TP=2 {mode} failed:\n{entry['error']}", flush=True
                )
                break  # peers may be out of step; do not issue further collectives
    finally:
        (out_dir / f"tp2_rank{rank}.json").write_text(json.dumps(summary, default=str))
        with contextlib.suppress(Exception):
            groups.close()
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def run_tp2(
    args: argparse.Namespace,
    cases: Sequence[Case],
    vocab_size: int,
    log: Callable[[str], None],
    modes: Sequence[str],
    timeout: float,
) -> dict[str, Any]:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory(prefix="ripple-tp2-") as directory:
        config = {
            "args": vars(args),
            "cases": [asdict(case) for case in cases],
            "modes": list(modes),
            "vocab_size": vocab_size,
            "out_dir": directory,
        }
        context = mp.spawn(
            tp2_worker,
            args=(_free_port(), config),
            nprocs=2,
            join=False,
        )
        assert context is not None
        deadline = time.monotonic() + timeout
        finished = False
        error: str | None = None
        try:
            while not finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = f"TP=2 run exceeded {timeout:.0f}s"
                    break
                finished = context.join(timeout=min(5.0, remaining))
        except Exception:
            error = traceback.format_exc()
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=30)
        report: dict[str, Any] = {"error": error, "ranks": {}, "modes": {}}
        for rank in (0, 1):
            path = Path(directory, f"tp2_rank{rank}.json")
            if path.is_file():
                report["ranks"][rank] = json.loads(path.read_text())
        for mode in modes:
            path = Path(directory, f"tp2_{mode}_results.pt")
            if path.is_file():
                report["modes"][mode] = torch.load(path, weights_only=False)
        log(f"  TP=2 finished: {sorted(report['modes'])}")
        return report


# --------------------------------------------------------------------------- #
# Orchestration and report
# --------------------------------------------------------------------------- #


class StageTimeout(Exception):
    pass


@contextlib.contextmanager
def stage_timeout(seconds: float, name: str) -> Generator[None]:
    def expired(_signum: int, _frame: Any) -> None:
        raise StageTimeout(f"{name} exceeded {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def compare_runs(
    label: str,
    actual: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    tolerance: Tolerances,
    *,
    reference_name: str,
    cross: bool,
) -> list[dict[str, Any]]:
    rows = []
    for key, expected in reference.items():
        entry: dict[str, Any] = {
            "stage": label,
            "reference": reference_name,
            "case": key,
        }
        if key not in actual:
            entry.update(passed=False, failures=["missing result"])
        else:
            entry.update(
                evaluate(
                    actual[key]["tokens"],
                    actual[key]["logits"],
                    expected["tokens"],
                    expected["logits"],
                    tolerance=tolerance,
                    cross=cross,
                )
            )
        rows.append(entry)
    return rows


def memory_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """TP=2 halves the sharded weights and the KV bytes per parent."""

    checks = []
    tp1 = report.get("tp1", {}).get("eager", {}).get("footprint")
    ranks = report.get("tp2", {}).get("ranks", {})
    for rank, summary in ranks.items():
        footprint = summary.get("modes", {}).get("eager", {}).get("footprint")
        if not tp1 or not footprint:
            continue
        expected = tp1["model_bytes"] - tp1["layer_projection_bytes"] // 2
        checks.append(
            {
                "check": f"rank {rank} model bytes == TP1 - sharded/2",
                "tp1": tp1["model_bytes"],
                "tp2": footprint["model_bytes"],
                "expected_tp2": expected,
                "passed": abs(footprint["model_bytes"] - expected)
                <= 0.005 * tp1["model_bytes"],
            }
        )
        checks.append(
            {
                "check": f"rank {rank} KV bytes/parent == TP1 / 2",
                "tp1": tp1["kv_bytes_per_parent"],
                "tp2": footprint["kv_bytes_per_parent"],
                "passed": footprint["kv_bytes_per_parent"] * 2
                == tp1["kv_bytes_per_parent"],
            }
        )
    return checks


def strip_tensors(value: Any) -> Any:
    """JSON-safe copy of a report: logits tensors are dropped."""

    if isinstance(value, dict):
        return {
            key: strip_tensors(item)
            for key, item in value.items()
            if not isinstance(item, torch.Tensor)
        }
    if isinstance(value, list | tuple):
        return [strip_tensors(item) for item in value]
    return value


def format_summary(comparisons: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'stage':<12}{'vs':<12}{'case':<12}{'max_abs':>9}{'cosine':>10}"
        f"{'top5_lp':>9}{'top1':>6}{'match':>8}  result"
    )
    lines = [header, "-" * len(header)]
    for row in comparisons:
        if "max_abs_err" not in row:
            lines.append(
                f"{row['stage']:<12}{row['reference']:<12}{row['case']:<12}"
                f"{'':>9}{'':>10}{'':>9}{'':>6}{'':>8}  FAIL {row.get('failures')}"
            )
            continue
        lines.append(
            f"{row['stage']:<12}{row['reference']:<12}{row['case']:<12}"
            f"{row['max_abs_err']:>9.4f}{row['cosine_similarity']:>10.6f}"
            f"{row['max_top5_logprob_abs_err']:>9.4f}{row['top1_match_rate']:>6.2f}"
            f"{row['trajectory_match_length']:>4}/{row['steps']:<3}"
            f"  {'PASS' if row['passed'] else 'FAIL ' + '; '.join(row['failures'])}"
            f"{' (tie)' if row.get('benign_tie') else ''}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT,
        help="cached HF snapshot directory (no downloads are ever attempted)",
    )
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument(
        "--mode", choices=("all", "tp1", "tp2"), default="all", help="which stages"
    )
    parser.add_argument("--output-json", default="ripple_qualify_5layer.json")
    parser.add_argument(
        "--skip-graphs",
        action="store_true",
        help="run eager decode only (no CUDA graph capture)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate arguments, checkpoint files and cases; no GPU work",
    )
    parser.add_argument(
        "--oracle-cache",
        help="torch file to load golden HF logits from (or save them to)",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--cache-parent-capacity", type=int, default=1024)
    parser.add_argument(
        "--nccl-graph-file", help="sets NCCL_GRAPH_FILE for the TP=2 workers"
    )
    timeouts = parser.add_argument_group("timeouts (seconds)")
    timeouts.add_argument("--timeout-oracle", type=float, default=900)
    timeouts.add_argument("--timeout-tp1", type=float, default=1200)
    timeouts.add_argument("--timeout-tp2", type=float, default=1500)
    gates = parser.add_argument_group("pass gates")
    defaults = Tolerances()
    gates.add_argument("--min-cosine", type=float, default=defaults.min_cosine)
    gates.add_argument(
        "--max-top5-logprob-err", type=float, default=defaults.max_top5_logprob_err
    )
    gates.add_argument("--max-abs-err", type=float, default=defaults.max_abs_err)
    gates.add_argument(
        "--cross-min-cosine", type=float, default=defaults.cross_min_cosine
    )
    gates.add_argument(
        "--cross-max-abs-err", type=float, default=defaults.cross_max_abs_err
    )
    gates.add_argument("--tie-margin", type=float, default=defaults.tie_margin)
    args = parser.parse_args(argv)
    if args.num_layers <= 0:
        parser.error("--num-layers must be positive")
    for name in ("prefill_chunk_size", "page_size", "cache_parent_capacity"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def tolerances(args: argparse.Namespace) -> Tolerances:
    return Tolerances(
        min_cosine=args.min_cosine,
        max_top5_logprob_err=args.max_top5_logprob_err,
        max_abs_err=args.max_abs_err,
        cross_min_cosine=args.cross_min_cosine,
        cross_max_abs_err=args.cross_max_abs_err,
        tie_margin=args.tie_margin,
    )


def read_checkpoint_config(checkpoint_dir: Path) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    return json.loads(config_path.read_text())


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    """Everything that can be checked without a GPU or reading a weight."""

    checkpoint_dir = Path(args.checkpoint_dir)
    raw = read_checkpoint_config(checkpoint_dir)
    from resin.models.ripple.config import RippleConfig

    config = RippleConfig.from_dict(raw).same_shape_reduced(args.num_layers)
    cases = default_cases(config.sliding_window)
    from resin_models_cuda.ripple import RippleCudaForCausalLM

    with torch.device("meta"):
        wanted = list(RippleCudaForCausalLM(config).state_dict())
    shards = _needed_shards(checkpoint_dir, wanted)
    for case in cases:
        for prompt in case_prompts(case, config.vocab_size):
            assert all(0 <= token < config.vocab_size for token in prompt)
    return {
        "layer_types": list(config.layer_types),
        "history_blocks_per_parent": config.history_blocks_per_parent,
        "shards": sorted(path.name for path in shards),
        "tensors": len(wanted),
        "cases": [asdict(case) for case in cases],
        "tolerances": asdict(tolerances(args)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - started:7.1f}s] {message}", flush=True)

    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "args": vars(args),
        "tolerances": asdict(tolerances(args)),
        "environment": collect_environment(),
        "errors": [],
    }
    output = Path(args.output_json)

    def write(status: str) -> None:
        report["status"] = status
        report["elapsed_seconds"] = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(strip_tensors(report), indent=2, default=str))

    if args.nccl_graph_file:
        if not Path(args.nccl_graph_file).is_file():
            print(f"NCCL graph file not found: {args.nccl_graph_file}", file=sys.stderr)
            return 2
        os.environ["NCCL_GRAPH_FILE"] = args.nccl_graph_file
        report["environment"]["NCCL_GRAPH_FILE"] = args.nccl_graph_file

    if args.dry_run:
        try:
            report["dry_run"] = dry_run(args)
        except Exception:
            report["errors"].append(traceback.format_exc())
            write("dry-run-failed")
            print(report["errors"][-1], file=sys.stderr)
            return 1
        write("dry-run")
        log(f"dry run ok; wrote {output}")
        print(json.dumps(report["dry_run"], indent=2))
        return 0

    if not torch.cuda.is_available():
        print("CUDA is required (use --dry-run on CPU)", file=sys.stderr)
        return 2
    wanted_gpus = 2 if args.mode in ("all", "tp2") else 1
    if torch.cuda.device_count() < wanted_gpus:
        print(f"--mode {args.mode} needs {wanted_gpus} GPU(s)", file=sys.stderr)
        return 2

    checkpoint_dir = Path(args.checkpoint_dir)
    raw = read_checkpoint_config(checkpoint_dir)
    from resin.models.ripple.config import RippleConfig

    config = RippleConfig.from_dict(raw).same_shape_reduced(args.num_layers)
    cases = default_cases(config.sliding_window)
    tolerance = tolerances(args)
    comparisons: list[dict[str, Any]] = []
    report["comparisons"] = comparisons
    log(f"Ripple {args.num_layers} layers {config.layer_types}")

    # 1. HF oracle on cuda:0.
    golden: dict[str, dict[str, Any]] | None = None
    cache_path = Path(args.oracle_cache) if args.oracle_cache else None
    if cache_path is not None and cache_path.is_file():
        golden = torch.load(cache_path, weights_only=False)
        log(f"loaded golden logits from {cache_path}")
    else:
        try:
            with stage_timeout(args.timeout_oracle, "HF oracle"):
                log("loading HF oracle")
                torch.cuda.reset_peak_memory_stats(0)
                loaded = time.perf_counter()
                model = load_hf_oracle(checkpoint_dir, args.num_layers, "cuda:0")
                report["oracle"] = {
                    "load_seconds": time.perf_counter() - loaded,
                    "gpu_allocated_bytes": torch.cuda.memory_allocated(0),
                }
                golden = build_golden(model, cases, config.vocab_size, "cuda:0", log)
                report["oracle"]["gpu_peak_bytes"] = torch.cuda.max_memory_allocated(0)
                del model
                torch.cuda.empty_cache()
                if cache_path is not None:
                    torch.save(golden, cache_path)
        except Exception:
            report["errors"].append("oracle: " + traceback.format_exc())
            log(f"oracle FAILED\n{report['errors'][-1]}")
        finally:
            torch.cuda.empty_cache()
    if golden is None:
        write("fail")
        return 1
    report["oracle"] = {
        **report.get("oracle", {}),
        "cases": {
            key: {"tokens": value["tokens"], "seconds": value.get("seconds")}
            for key, value in golden.items()
        },
    }
    write("running")

    # 2. Resin TP=1: eager, then CUDA-graph decode.
    tp1: dict[str, dict[str, Any]] = {}
    report["tp1"] = {}
    if args.mode in ("all", "tp1"):
        for mode in ("eager",) + (() if args.skip_graphs else ("graph",)):
            try:
                with stage_timeout(args.timeout_tp1, f"TP=1 {mode}"):
                    log(f"TP=1 {mode}")
                    run = run_tp1(
                        args, cases, config.vocab_size, log, graphs=mode == "graph"
                    )
                tp1[mode] = run["results"]
                report["tp1"][mode] = {k: v for k, v in run.items() if k != "results"}
                comparisons += compare_runs(
                    f"tp1_{mode}", run["results"], golden, tolerance,
                    reference_name="hf", cross=False,
                )  # fmt: skip
            except Exception:
                report["errors"].append(f"tp1 {mode}: " + traceback.format_exc())
                log(f"TP=1 {mode} FAILED\n{report['errors'][-1]}")
                torch.cuda.empty_cache()
            write("running")
        if "eager" in tp1 and "graph" in tp1:
            comparisons += compare_runs(
                "tp1_graph", tp1["graph"], tp1["eager"], tolerance,
                reference_name="tp1_eager", cross=True,
            )  # fmt: skip

    # 3. Resin TP=2 over real NCCL.
    report["tp2"] = {}
    if args.mode in ("all", "tp2"):
        modes = ("eager",) + (() if args.skip_graphs else ("graph",))
        log("TP=2 (NCCL)")
        tp2 = run_tp2(args, cases, config.vocab_size, log, modes, args.timeout_tp2)
        report["tp2"] = strip_tensors(tp2)
        if tp2["error"]:
            report["errors"].append("tp2: " + tp2["error"])
        for rank, summary in tp2["ranks"].items():
            for mode, entry in summary["modes"].items():
                if entry.get("status") == "error":
                    report["errors"].append(f"tp2 rank {rank} {mode}: {entry['error']}")
        for mode in modes:
            results = tp2["modes"].get(mode)
            if results is None:
                report["errors"].append(f"tp2 {mode}: no results")
                continue
            comparisons += compare_runs(
                f"tp2_{mode}", results, golden, tolerance,
                reference_name="hf", cross=False,
            )  # fmt: skip
            if "eager" in tp1:
                comparisons += compare_runs(
                    f"tp2_{mode}", results, tp1["eager"], tolerance,
                    reference_name="tp1_eager", cross=True,
                )  # fmt: skip
        report["memory_checks"] = memory_checks(report)

    # 4. Verdict.
    failed = [row for row in comparisons if not row["passed"]]
    failed += [row for row in report.get("memory_checks", []) if not row["passed"]]
    passed = not failed and not report["errors"] and bool(comparisons)
    write("pass" if passed else "fail")
    print()
    print(format_summary(comparisons))
    for row in report.get("memory_checks", []):
        print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['check']}: {row}")
    for error in report["errors"]:
        print(f"ERROR {error.splitlines()[0]}")
    log(f"{'PASS' if passed else 'FAIL'}; report: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
