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

## Non-claims

CPU only: no GPU, CUDA, Modal or Azure runs, no performance or memory claims,
no real 65B checkpoint was loaded, and TP is checked with an in-process
simulated all-reduce rather than a real process group.
