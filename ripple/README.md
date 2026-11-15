# Ripple-65B CPU oracle harness

`ripple/` holds a standalone Hugging Face oracle for Ripple-65B-Preview
(`ricursive`). It is kept apart from Resin's runtime modules, which must never
import `transformers`. `ripple/reference/` is the pinned, byte-identical
upstream source (Apache-2.0, revision `b172323f…`; digests in `SHA256SUMS`).
Nothing is downloaded at test time.

## CPU environment

`./.agents/setup` creates `~/.cache/resin-cpu-venv` (symlinked at `~/resin-venv`)
from `ripple/requirements-cpu.txt` (`torch==2.13.0+cpu`, no `nvidia-*` wheels),
fetches Cargo dependencies and installs `resin` and `resin-models-torch` in
editable mode. A marker file makes reruns a no-op. After adding files, rebuild
the editable installs with:

```sh
PYO3_PYTHON="$HOME/resin-venv/bin/python" uv pip install --python "$HOME/resin-venv/bin/python" \
  --no-deps --no-build-isolation -e . -e models/torch
```

## Commands

```sh
PY="$HOME/resin-venv/bin/python"
"$PY" -m ripple.reference.verify                               # reference digests
"$PY" -m pytest -q ripple/tests                                # HF oracle (tiny model)
"$PY" -m pytest -q models/torch/tests tests/test_ripple_config_plan.py  # Resin runner, config, plans
"$HOME/resin-venv/bin/ruff" check .
"$HOME/resin-venv/bin/ty" check
PYO3_PYTHON="$PY" cargo clippy --locked --workspace --all-targets -- -D warnings
```

## What is covered

* `ripple/tiny_config.py`: a 10-layer tiny config that mirrors every real
  architectural key; `real_config_diff` lists the only keys that differ.
* `ripple/tests`: prefill vs token-by-token decode, chunked prefill and
  multi-token verify blocks past the sliding window, ablations, clamping and
  `generate()` parity, in fp32 and bf16.
* `resin/models/ripple`: backend-neutral `RippleConfig`, tensor-parallel
  geometry, hybrid full/window `CacheLayout` and `RippleBatchPlan`.
* `models/torch/resin_models_torch/ripple`: a pure-PyTorch model with the HF
  state-dict keys, a safetensors loader (including a first-N-layers load that
  never opens unused shards) and `RippleTorchRunner` for the Rust `Engine`.

## Reference notes

* The value residual is `lerp(embedding, normed_hidden, sigmoid(value_res_lambda))`,
  so the sigmoid weights the *current* layer input and `0` gives an even blend.
* Window masks are end-anchored on absolute positions: a query at `p` reads keys
  `[p - (window - 1), p]`.
* bf16 prefill vs cached decode differs by up to ~3e-2 on logits of magnitude ~4;
  tests allow `atol=0.1, rtol=5e-2`. fp32 uses `1e-5`.
* The tiny config has 20 query heads, so it shards at TP=1, 2 and 4 (KV heads
  replicated at TP=4) but not TP=8. The real model shards at TP=1, 2, 4, 8 and 16.

## SM80 CUDA runner and GPUBox qualification

`models/cuda/resin_models_cuda/ripple` is the BF16 Ampere+ CUDA runner
(`RippleCudaRunner`): FlashInfer paged attention with the standard
`1/sqrt(head_dim)` scale, a shared full/window KV arena (4 history pages per
window page on the 80-layer and 5-layer models), TP via `attn_tp`/`expert_tp`,
eager or full-CUDA-graph decode and breakable prefill FFN graphs. `num_layers=5`
runs one `[S, S, S, S, G]` period at full width (~8.3 GB) through the same code.
It imports without CUDA, `flashinfer` or `triton`; `models/cuda` is installed
editable by `.agents/setup` (`--no-deps`, no CUDA wheels).

* Norms are explicit torch ops matching the reference rounding
  (`weight * rms(x).to(bf16)`); FlashInfer/Quack fused norms round once and can
  differ by one bf16 ulp.
* Scoring: `logprobs`/`top_logprobs`/`prompt_logprobs` use the CuTeDSL kernels
  (`nvidia-cutlass-dsl`, `quack-kernels`, tested on SM80+); greedy/temperature
  sampling and plain decode do not. `RippleCudaRunner.logits_observer` exposes the
  raw last-token logits without scoring.
* CPU tests: `models/cuda/tests/test_ripple_cuda_cpu.py` (HF parity, packing, CPU-first
  TP sharding, 2-shard checkpoint, arena geometry, and the whole runner through
  `Engine` against a torch stand-in for FlashInfer). GPU tests:
  `models/cuda/tests/test_ripple_cuda_gpu.py` (skip without SM80+; TP=2 needs 2 GPUs).

`ripple/gpu_qualify_5layer.py` is the single offline GPUBox harness: HF 5-layer oracle on
`cuda:0`, Resin TP=1 (eager and CUDA graphs) and TP=2 over NCCL, per-step logit
metrics against the oracle, pass gates and a JSON report. Try it without a GPU:

```sh
"$PY" ripple/gpu_qualify_5layer.py --help
"$PY" ripple/gpu_qualify_5layer.py --dry-run --checkpoint-dir "$SNAPSHOT"
```

## Non-claims

No GPU, CUDA, Modal or Azure run has happened: the CUDA runner's FlashInfer calls,
CUDA graphs and NCCL TP are unexecuted until the GPUBox run, no performance or
memory claims are made, no real 65B checkpoint was loaded, and CPU TP checks use
an in-process simulated all-reduce rather than a real process group.
