# Verified kernel evaluator — local scaffold

Standalone experiment, outside Resin imports/builds. No dependencies, model weights,
GPU allocations or changes to serving code.

```
python3 -m unittest discover -s experiments/kernel_eval/tests -v
python3 experiments/kernel_eval/check_contract.py
```

The manifest pins the real Gemma Edge activation/multiply boundary. Large-producer
cuBLASLt fused linear/GELU is excluded because its intermediate-rounding semantics
differ. Shapes are workload probes, not a production traffic distribution.

`Candidate.launch` is a conceptual caller-owned output/stream interface, NOT an
implemented Rust ABI. The host, not the candidate, will allocate/fill fresh buffers,
check results and time execution. CPU contract validation earns no performance
score and cannot produce a GPU-qualified report.

CPU scalar specification, mutation tests and bounded fixture subprocesses are now
implemented. Next qualify
PyTorch/Triton/Rust on B200, calibrate and freeze numerical thresholds, and implement
actual GPU adapter. Do not equate an idealized scalar tanh formula with exact
PyTorch GPU results before calibration.

Security: hashes identify artifacts, not trustworthiness. This local scaffold is
not isolation for malicious native code. Final hidden tests and trusted evaluator
must run outside agent-writable state; process isolation alone is not a sandbox.
No hidden fixtures or GPU measurements exist yet. Manifest validation currently
covers schema version, operation, dtype and positive shapes, not full JSON Schema.

## Scalar numerical harness

`numerics.py` adds nearest-even binary32→BF16 conversion and a scalar tanh-GELU
specification with explicit intermediate BF16 rounding. Python tanh/arithmetic
are not the CUDA/PyTorch implementation: exact comparisons here test the harness,
not GPU equivalence. NaNs match by class; signed zeros compare equivalent.

Unit tests now reject zero outputs, wrong activation, dropped/unwritten tails,
skipped intermediate rounding, mutated inputs and stale output on changed-input
replay. Fixtures are deterministic/public; these are not held-out security tests.
The in-process checker catches Python exceptions but cannot stop hangs, native
crashes or malicious code. Bounded subprocess execution is now available below.

## Bounded local commands

`process.run` now supplies Linux wall/CPU time, address-space and per-output-file
size limits, disables core dumps and kills the child's process group on exit.
Tests exercise success, abort, timeout, output flooding and allocation failure.
It is a single-threaded harness utility (`preexec_fn`), not a general service API.
GPU processes need different memory limits; this profile must not wrap CUDA as-is.

This does NOT restrict network, credentials or filesystem access, and descendants
that deliberately escape the process group are not contained. Run only our own
fixtures here. No arbitrary agent-produced native code on Pim Computer. The built-in fixture JSON protocol is now wired below; a genuine remote isolation
boundary is still required before untrusted execution.


## Integrated local milestone

```
python3 experiments/kernel_eval/evaluate.py reference
python3 experiments/kernel_eval/evaluate.py skip_round  # fails with exit 1
```

The parent generates six small deterministic cases, runs the built-in worker in
a bounded subprocess and independently checks its JSON output. Reports include
manifest, worker, evaluator and oracle hashes. Known failure fixtures exercise
incorrect results, hangs, crashes and malformed responses. Only our own fixtures
can execute locally. Input immutability is tested in-process; returned JSON alone
cannot prove a worker did not mutate its private inputs. This is an integrated
harness self-test, not a complete arbitrary-candidate evaluator.

## Prepared Modal calibration (not executed)

`modal_probe.py` defines one B200 container, no retries, a 180-second function
timeout and explicit CLI spending acknowledgment. Definition import was checked
using the installed Modal 1.5.5 tool environment; this does not validate remote
image availability, package installation or CUDA execution. The image uses the
CUDA tag from Resin's deployment image and pinned PyTorch 2.13.0, not digest locks.

After separate spending approval, the pilot reports sampled scalar-vs-PyTorch
errors and CUDA-event baseline timings for six shapes. It does not download model
weights or load arbitrary candidates. A function timeout is NOT a dollar budget
or a bound on image build costs. Establish account budget/total run limit before
launch. Rust kernel implementation, existing Triton comparison, sanitizer and
changed-input CUDA graph replay remain subsequent GPU gates. No speedup or GPU
qualification may be inferred from this scaffold.

## First B200 calibration executed (2026-11-20 UTC)

`results/b200-calibration.json` records the first successful GPU run:
NVIDIA B200, capability 10.0, torch 2.13.0+cu130 (runtime CUDA 13.0 inside
CUDA 13.3.1 devel base). All six shapes' 257-element samples matched the scalar
spec exactly. This is sampled evidence, not exhaustive numerical qualification.
Unfused eager PyTorch baseline batch medians were 6.37–6.48 us/call. These are
launch/allocation-inclusive microbenchmarks, not pure kernel or model latency.

Run: https://modal.com/apps/ajac-zero/main/ap-bMLio3K1Q4YCQ1nxxUbAGx
The preceding attempt finished computation but failed local deserialization:
TorchVersion is a torch-defined str subclass. Converting version metadata to a
plain str fixed it without installing PyTorch locally. Both ephemeral apps were
confirmed stopped with zero tasks. Billing summary is workspace-wide and delayed;
it is not an exact per-run cost receipt. No model weights or persistent volumes
were created. $5 total qualification authorization remains the phase limit.

## Standalone CUDA Rust numerical milestone

`rust/` is an isolated Cargo workspace using locked cuTile 0.4.0 and half BF16.
The actual Rust GPU kernel loads/stores BF16, computes tanh-GELU in FP32 and
explicitly rounds its intermediate to BF16 before multiplying. Modal builds it
on CPU and executes it on B200. Host JSON transport is for correctness only:
it includes copies and process startup and MUST NOT be used for kernel timing.

`results/b200-rust-numerics.json` records zero mismatches against BOTH PyTorch
and the existing Resin Triton kernel across 1, 7, 257, 6144 and 12288 elements
(18,697 total). These are deterministic finite random probes, not exhaustive
coverage, sanitizer qualification, graph-replay qualification or full model parity.

Successful run: https://modal.com/apps/ajac-zero/main/ap-pt9ffobVci4SAgIEVErGjk
Build troubleshooting: explicit tile constants required instead of scalar
arithmetic; remote module import must not evaluate local ROOT.parents paths.
Startup failures were observed to retry despite function retries=0, so an outer
wall timeout and explicit app stop remain necessary. All phase apps confirmed
stopped with zero tasks after this milestone. Delayed workspace metered ephemeral
cost grew from $2.18 before Rust work to $2.23 on the final check; this is NOT a
complete settled per-run charge. No performance speedup claimed yet.

Next gate: persistent owned buffers + CUDA event/graph timing inside Rust, paired
with PyTorch/Triton timing in the same B200 run; changed-input replay, nonfinite
and tail probes, Compute Sanitizer. The Rust kernel is not wired into Resin's
runner and is not agent-editable code executing under an immutable trusted judge.

## Preliminary graph timing and memory check

`results/b200-graph-timing.json` records five event-timed replays of a graph
containing 100 calls on stable owned buffers, after JIT/capture/warmup. Copies,
allocation and process startup are excluded. PyTorch, Triton and Rust run on the
same B200, but Rust owns a separate process/context. Fixed execution order and
separate contexts mean these are preliminary observations, not a frozen fair
benchmark or serving speedup. Tiny 1/7/257-element cases are tail/launch probes,
not actual model widths. No tuning sweep or held-out evaluation yet.

Median us per call (PyTorch / Triton / Rust):
- 6144 elements: 2.106 / 1.614 / 1.467
- 12288 elements: 2.440 / 1.774 / 1.484

All five shapes still have zero numerical mismatches against both comparators.
Compute Sanitizer memcheck on the 257-element Rust case returned exit0 and
ERROR SUMMARY: 0 errors. This is one tail case, not racecheck/initcheck/soak
qualification. Sanitized timings in its raw output are invalid for comparison.

Run: https://modal.com/apps/ajac-zero/main/ap-RKTpZlwSkStsyVkAzya24t
All phase apps confirmed stopped with zero tasks. Delayed workspace ephemeral
meter is $2.26 (previous $2.23); exact phase charges remain unsettled. Existing
$5 total phase authorization applies. No production integration or main merge.

## Expanded shapes and changed-input replay

`results/b200-expanded-replay.json` extends probes to 43,008, 202,752, 405,504
and 792,576 elements. After timing, an independent GPU kernel updates the ORIGINAL
gate allocation with BF16(gate+0.5); replay of the SAME captured GELU graph matches
a freshly computed PyTorch reference. All nine cases have zero initial mismatch
against both comparators and zero changed-input replay mismatch. This proves the
probe consumes changed device inputs rather than assuming graph capture freezes
values. It does not test arbitrary new allocations or concurrent stream ownership.

The broader matrix rejects the universal-speedup hypothesis: median Triton/Rust
us per call were 1.954/1.770 at 202752, 2.084/2.096 at 405504 and 2.384/3.000 at
792576. The largest case is ~26% slower in Rust. Preserve these losses. Shape-aware
selection may be useful; do not replace Triton globally. These remain fixed-order,
separate-context preliminary measurements, not statistically controlled results.

Run: https://modal.com/apps/ajac-zero/main/ap-jnUSJibre4c5zqwllB9rxq
Rust build exposed a useful ownership constraint: Tensor::to_host_vec consumes
its Tensor. Output is now Arc-owned for multiple host reads around replay. Two
failed build attempts are retained in local audit logs; they allocated no GPU
function. Delayed workspace compute meter reached $2.35, not a settled phase bill.

## BF16-domain and bounded safety probes

`modal run experiments/kernel_eval/modal_rust.py --approve-spend --adversarial`
exercises every one of the 65,536 BF16 gate encodings (including signed zeros,
subnormals, infinities and NaNs) against eight up patterns: +1, -1, +0, largest
finite, smallest subnormal, +infinity, quiet NaN and a permutation of all encodings.
This is **524,288 operand pairs, not the exhaustive two-operand Cartesian product**.
All outputs match PyTorch and existing Resin Triton; changed-input graph outputs
also match freshly computed PyTorch. The independent parent `bits.py` verdict
requires identical bits, including zero/infinity signs, ignoring only NaN payload
and sign. Five CPU tests cover malformed responses and intentional wrong outputs.
No ULP tolerance is introduced. This probe policy is not a universal GPU oracle.

Bit-array input/output avoids JSON NaN/Infinity/null ambiguities and preserves
subnormals and signed zeros. Normal finite-number transport remains available.
Results: `results/b200-bf16-domain.json`; final run:
https://modal.com/apps/ajac-zero/main/ap-d65UWCa9o8u4Q0XLTFS2rL

`--safety` sends 32 requests through one persistent Rust process, varying tails
at 1/7/255/256/257/511/512/513 elements with seeded random BF16 encodings.
All initial and changed-input outputs match PyTorch by numerical equality with
NaN-class equivalence (unlike the domain probe, this older safety comparison
ignores signed-zero differences). Independent sanitizer executions on a 257-tail
case report memcheck 0 errors, racecheck 0 hazards/errors/warnings, initcheck
0 errors. This is not a long-duration soak, leak proof, uninitialized-output
write-coverage proof (output starts zeroed), cross-stream race proof or
hostile-code isolation. Reusing the same process still allocates new tensors
per request; it is not a persistent shared-buffer integration.
Results: `results/b200-safety.json`; run:
https://modal.com/apps/ajac-zero/main/ap-bKFI4Q5KL3CkUVIAm1RPEz

The candidate and judge still share a container/filesystem. Keep
`gpu_qualified=false`: these are recorded probes, not final secure acceptance.
No serving-code change, dispatch replacement or engine speedup follows from them.

## Same-process bridge, frozen dispatch and held-out replication

The owned-buffer subprocess benchmark is superseded for timing by
`modal_bridge.py`, which builds a Rust cdylib from the SAME `kernels.rs` used by
the standalone probe. Rust borrows PyTorch GPU pointers and its non-default
stream; no device copies or subprocess/context switch occur in the Rust path.
The owner keeps all buffers, the stream and the native handle alive through
capture/replay, synchronizes, drops graphs, and destroys the handle in that order.
This is a controlled FFI prototype, NOT a safe public API: raw pointer callers
must uphold shape/device/nonaliasing/liveness obligations. Handles must not be
used concurrently or destroyed while captured work can still replay. Only device
0 and contiguous BF16 arrays up to 1,048,576 elements are supported. No model
runner is modified. API obligations were read directly in cuda-core 0.4.0
runtime.rs and cutile 0.4.0 tensor.rs/device operation sources.

Benchmarks prewarm all paths, capture 100 calls per graph and randomize path
order in each of 15 rounds. Same GPU/process/context/stream/input/output buffers;
CUDA event measurement only. Output is poisoned before changed-input replay;
input mutation and both output guards are checked. Aligned final runs use a
16-byte aligned output slice with eight-element guards on both sides. Separate
`*-misaligned-stress.json` artifacts preserve earlier two-byte-offset experiments;
do not silently combine them. All three Rust block sizes 256/512/1024 pass
correctness, guards and changed-input replay in every recorded study.

The frozen experimental dispatch (chosen before held-out measurements) is:
- n <= 12,288: Rust block 256
- n <= 405,504: Rust block 512
- larger: retain Triton

**This rule is an analysis selection, not installed serving dispatch.** It isn't
promoted to all shapes/hardware. Fixed tuning shapes and seven disjoint held-out
shapes were tested, with two separate held-out GPU runs/random seeds. Bootstrap
paired-round intervals are reproducible with `python experiments/kernel_eval/analyze_shared.py`.
They describe within-run repetition, NOT population-level confidence across GPUs,
clock states or production request mixes. Graphs repeatedly use cache-resident
buffers; streaming/cache-cold and serving timings are not measured. PyTorch is
only a sanity comparator and includes a copy into the common output; Triton is
the optimized comparison used for all dispatch conclusions.

Aligned tuning (seed 817): Triton/Rust256 at 6144 = 1.637/1.476 us and at
12288 = 1.639/1.497 us. Frozen Rust512 held-out wins replicate: at 24576,
1.858/1.544 and 1.813/1.526 us; at 190464, 1.983/1.744 and 1.957/1.731 us;
at 393216, 2.090/1.972 and 2.078/1.955 us. At 780288 retain Triton; do NOT
count the fallback's zero regression as a Rust performance win. Block 256 still
loses badly at large sizes; block tuning helps but does not universally win.

Runs: tuning ap-uVMxy2C6Ar7TgeQg6XXpJB; held-out ap-dwRNAXour05IMhOHxYpnTl
and ap-AjutlJSp3baOlZavpk3Oor. Results `b200-shared-buffer-*.json`.
One initial startup failed because a remote module imported a local-only module;
that app was explicitly stopped and the import guarded with `modal.is_local()`.
Its native build succeeded; the failed run is not a performance sample.

**Decision:** stop spending optimization effort on this activation for now.
The gain is real within this experiment, but small in absolute terms. Applying
the aligned one-row savings to 15 width-6144 and 20 width-12288 layers yields
only ~5.3 microseconds of kernel-time savings per decode step, CONDITIONAL on
those standalone savings transferring unchanged. This is arithmetic, not an
end-to-end prediction or bound (model cache/overlap/launch paths differ).
Prioritize whole-model profiling and a trusted evaluator boundary before chasing
more activation microseconds. Keep this operation as the evaluator regression
fixture, not as evidence that a specialized engine beats tuned serving engines.
The cdylib shares the judge address space and must never be presented as hostile
candidate isolation. Secure acceptance still requires a separately reviewed
artifact/launcher/isolation design, with trusted expected outputs unavailable to
candidate host code and measured timing not supplied by a writable candidate.

## Pinned whole-model reference and matched CUDA-graph decode

`modal_model_profile.py --approve-spend` now downloads the REAL pinned checkpoint
in a CPU image-build step, loads BF16 Gemma4ForConditionalGeneration on one B200,
and runs text-only whole-model inference. The public HF pinned inventory exposes
one `model.safetensors` of 10,246,621,918 bytes (not gated); the GPU allocated peak
in the measured reference was 10,413,630,976 bytes. No weights are added to git.
The default Resin gemma4-e2b preset is Ampere/A100 and sets architecture 8.0;
we did NOT silently treat it as a qualified B200 deployment. Its CUDA package
also pins torch 2.15.0 while this independent reference uses 2.13.0. A separately
qualified Resin B200 image is still needed.

The final report is `results/b200-gemma-hf-reference.json`, run
https://modal.com/apps/ajac-zero/main/ap-vJAVMmkjuTZEUq2DkVUyOC . It contains
256/1024-token synthetic repeated-text prefill, greedy dynamic-cache decode,
and a matched static-cache eager-vs-graph decode experiment. The latter has
independent caches initialized from the SAME 256-token prefix, same static-cache
code path and weights, alternating measurement order, explicit mutable position
and mask buffers, and 16 changing teacher-forced token inputs. All 16 captured
logit tensors match their eager counterparts EXACTLY (max abs error zero and
same argmax); not merely a matching generated token string.

Median static-cache eager / graph decode was 33.469 / 7.906 ms (~4.23x) on that
run; the graph path uses unchanged HF kernels. This demonstrates that eager HF
is a profoundly weak optimization comparator for this workload. It is NOT a
Resin or CUDA Rust kernel win, and is NOT evidence of beating vLLM/SGLang.
Earlier independent graph bring-up was ~8.633 ms but was compared against a
different dynamic-cache path; that unmatched comparison is not used for the
4.23x figure. Timing includes CUDA stream gaps; the measured wall time likewise
includes per-step synchronization. No serving scheduler, request concurrency,
long-context window rollover, SLO or throughput qualification is implied.

Four-step eager profiler decomposition attributes ~24.6% of operator-associated
self-device time to mm, ~22.8% to copy_, ~14.2% to mul, ~9.6% to pow. The report
uses CPU operator self-device attribution ONLY: summing those and CUDA kernel
rows would double count. An early profiling attempt did double count those;
its aggregate is discarded. The observed cast/norm/elementwise overhead is a
hypothesis for investigation, not a prediction of wins over Resin: direct source
inspection shows Resin ALREADY has CUDA graph decode, norm/residual and norm/RoPE
fusions (`edge_cuda.py`, `edge_kernels.py`, `edge_model.py`).

**Next decision:** establish the actual B200 Resin/tuned-engine baseline and
profile THAT, before selecting another kernel target. The activation fixture
and this HF reference are useful qualification tools, but neither is the north
star performance baseline. Kernel isolation remains a separate unresolved gate.
Image loading initially failed for missing accelerate; the bounded retry adds
accelerate==1.12.0 after the cached checkpoint layer. All runs have explicit
function/CLI deadlines and no persistent server/volume was provisioned.

Primary sources checked for this step:
- pinned HF API inventory: https://huggingface.co/api/models/google/gemma-4-E2B-it/revision/3e22461f65e89153144f8adb70e3b8c2cc9845a7?blobs=true
- model code: https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py
- cache API: https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/cache_utils.py
- Resin files above and `deploy/models.py`, at source revision pinned in task.json.

## Resin Gemma E2B on B200: first experimental engine baseline

`modal_resin_baseline.py` builds a minimal model-specific image (torch 2.15.0+cu132,
the version Resin's CUDA package pins; flashinfer 0.7.1; quack/cutlass-dsl pins),
installs Resin's Rust extension plus model packages from the pinned fork source,
downloads the pinned Gemma target and `google/gemma-4-E2B-it-assistant` (rev
2d874ef7...), and runs Resin's OWN `run_benchmark` harness on one B200
(capability 10.0). Result: `results/b200-resin-baseline.json`, run
https://modal.com/apps/ajac-zero/main/ap-TEsEkJtrGIjbuTgXGQjKde .

Experimental overrides (NOT a released B200 preset): gpu=B200 instead of the
preset's A100, max batch 1, context 2048, prefill chunk 1024, memory fraction 0.15,
synthetic deterministic prompt source, numa off. Preset arch flags were replaced
with 10.0a. Full Resin CUDA-graph catalog capture took 124 s (cold, untimed).

Measured (single run, single repeat, batch 1, 256-token context):
- Decode, draft tokens 3 with assistant: median 4.117 ms per verify step over 16
  steps; 56 tokens committed in 0.0674 s (~831 tokens/s); 15 graph replays, ZERO
  eager fallbacks; measured acceptance 0.833 on synthetic deterministic prompts.
  **Acceptance is workload dependent; this is not a chat/summarization quality or
  speedup number.**
- Prefill 256 tokens: median 17.3 ms over 3 cohorts with the assistant loaded
  (draft-free runs measured 7.7-8.8 ms; see below). Only 3 samples.

Context for the earlier HF static-cache graph reference (7.9 ms/token): that is a
token-at-a-time greedy reference; Resin commits ~3.5 tokens per 4.1 ms step here
(~1.2 ms per committed token). These are DIFFERENT workloads/algorithms (speculative
decode vs not), different batch semantics and token sources, so the ratio is not a
like-for-like speedup. The like-for-like draft-free run is below.

**Draft-free decode defect: reproduced, FIXED on our fork, verified on B200.**
With `draft_tokens=0`, E2B decode failed in `edge_cuda.py::_finish_enqueue`
(`IndexError: too many indices for tensor of dimension 1`) because the
non-speculative sampler returns `[rows]` while acceptance needs an empty draft block
plus one bonus column. Upstream Gemma E2B/E4B tests only used
`speculative_draft_tokens=3`. Fix: new `DeviceAcceptance.from_single_samples`
(accepts `[rows]` or `[rows,1]`, rejects other shapes) used by the Edge runner; CPU
regression test `tests/test_cuda_completion.py::test_single_sample_acceptance_for_non_speculative_decode`
passes (13 passed / 4 GPU-only skipped, Modal CPU run ap-hffnoxyzsvM1sqA9f9OAl6).
This is a fork change, not yet reported or offered upstream.

Like-for-like draft-free Resin on B200 (same experimental overrides, batch 1,
256-token context; two separate runs of the final code path, results
`results/b200-resin-draft0.json`, final run ap-yCODlAKCSeSz9LNfGSIyX6):
- Decode: **2.961 ms/token median**, range 2.955-2.970 ms over 16 steps, 16 graph
  replays, 0 eager fallbacks. The pre-refactor run measured 2.979 ms.
- Prefill 256 tokens: 8.085 ms median (n=3). An earlier identical-config run showed
  17.5 ms; treat prefill as noisy until measured with more cohorts.
- Versus the HF static-cache CUDA-graph reference (7.906 ms/token, same model,
  same batch, similar 256-token context): Resin is ~2.7x faster per token. The
  comparison is greedy single-token decode on both sides, but prompts differ
  (repeated English text vs Resin's synthetic deterministic token ids) and the
  harnesses differ (HF includes explicit position/mask updates). Directionally
  reliable; not a controlled benchmark and not a comparison to vLLM/SGLang.
- Arithmetic hint, not a measurement: ~2+ B parameters read per decoded token in
  BF16 is ~4-5 GB; at ~7 TB/s achievable HBM bandwidth that is a ~0.7 ms floor, so
  2.96 ms is several times above bandwidth-bound. Per-kernel profiling is needed
  to tell launch/latency/attention/head costs apart before choosing a target.

Process notes: three setup failures preceded success and were all CPU/image or
Python-path problems, not GPU measurements: (1) a build killed by my tool abort,
stopped explicitly; (2) the source tree shadowed the installed package because
`scripts/bench.py` prepends its repo root, so the image now keeps only `deploy/`
and `scripts/` in `/opt/aux` and deletes the source tree; (3) the draft-free decode
failure fixed below. No persistent volumes or servers; all apps stopped, zero tasks.

## Correction and first in-engine gain: wire the fused GELU*up kernel

**Correction to the earlier activation-tuning conclusion.** A profile of Resin's real
draft-free B200 decode (`modal_resin_baseline.py --trace`, results
`b200-resin-decode-trace.json`, run ap-jQQkgL7DOWs6QH2BNca6Bv) shows that in this
fork revision the E2B forward never called `gelu_tanh_mul`: both the MLP and the
per-layer-input gate ran unfused `F.gelu(...) * ...` (70 GELU + 70 multiply kernels
per token, ~250 us of profiled kernel time). The Triton kernel was exported and
unit-tested but unused. My earlier "~5.3 us/step" estimate compared Rust to Triton
and therefore understated the opportunity: the relevant baseline is the unfused
eager path that actually ran. (The standalone Rust/Triton kernel measurements
themselves stand; the engine-level inference from them was wrong.)

**Change (fork, `edge_model.py`):** a guarded `_gelu_tanh_mul` helper uses the
existing fused Triton kernel for the MLP and PLE gate when inputs are matching
contiguous BF16 CUDA tensors, and falls back to the original eager expression
otherwise. The producer-layer fused linear+GELU path (>= 8192 rows) is unchanged.
The kernel retains Gemma's BF16 rounding of the GELU output (my B200 domain probe
found zero exact mismatches versus PyTorch over all 65,536 BF16 gate encodings
and eight up patterns). CPU model/completion unit tests: 27 passed, 5 GPU-only
skipped (ap-w4V3bdqey4dsIgWbCp4fAK).

**Whole-engine result (same overrides, batch 1, 256-token context, draft 0):**
- Decode median **2.961 -> 2.886 ms/token (-2.5%, ~75 us)**; every one of the 16
  measured steps is faster than every baseline step (range 2.876-2.900 vs
  2.955-2.970 ms). Kernels per token 795 -> 725 (exactly 70 fewer); the fused kernel
  costs 104.6 us/token in the profile versus ~250 us for the 140 eager kernels.
- The first 22 greedy generated token IDs are identical to the unfused baseline
  (the traced run generated 25; compared on the common prefix). That is weak
  end-to-end evidence; the stronger correctness evidence is the exact kernel probe.
- Prefill unchanged within noise (8.085 -> 8.31 ms, n=3).
- Single run each; baseline and fused sweeps were not interleaved. Profile times
  differ from unprofiled wall time (profiled kernel sum ~3.05 ms vs 2.886 ms wall).
  Saved kernel time (~146 us) exceeds saved wall time (~75 us): removed kernels
  were partly hidden by launch gaps. Not a statistically controlled result.

**Where the remaining decode time goes** (fused-run profile, per token, 725 kernels,
GPU ~90% busy over the traced span; percentages are of ~3.05 ms profiled kernel time):
- GEMMs (nvjet/cublasLt incl. split-K reductions): ~1.66 ms (54%)
- Attention (FlashInfer): ~0.54 ms (18%) - ~35 launches at ~15 us each, for a 256-token context
- Norm/residual/RoPE (Quack RMSNorm, `_norm_residual`, QK norm+RoPE): ~0.51 ms (17%);
  `_norm_residual` alone is 105 launches
- Fused GELU*up 0.10 ms; KV `index_put` and other eager ops ~0.16 ms; copies 0.04 ms
Bandwidth arithmetic (not a measurement): the GEMM bucket alone is ~2.5x the
~0.65 ms weight-read floor for ~4.4 GB at ~7 TB/s, so M=1 GEMM efficiency and
split-K behavior are the largest candidate; attention and the many small
norm/residual launches are the next.

## M=1 projection and fused gate/up/GELU kernels (decode latency)

**Why:** the fused-GELU profile put 54% of per-token kernel time in GEMMs. A
separate L2-cold study (`modal_gemv.py`, `modal_gemv2.py`; results
`b200-m1-gemm-study.json`, `b200-m1-kernel-sweep.json`; weights rotate through
>4x the 132.6 MB L2 so repeated calls cannot hit cache) found that cuBLAS is already
efficient on the big MLP matrices (4-5.6 TB/s) but latency-bound on small M=1 ones.
The model already packs Q/K/V into one `qkv_proj` GEMM (my first estimate treated
K and V as separate launches and overstated that bucket).

**Kernels (fork):** `_linear_m1` (FP32 accumulate, one BF16 rounding of the result,
no FMA contraction) and `_gate_up_gelu_m1` (keeps both BF16 rounds: gate and up
outputs are rounded before GELU, the GELU output is rounded before the multiply,
matching the eager expression). Per-shape configs are pinned from the sweep and
only shapes that beat cuBLAS in isolation are enabled: o_proj (1536x2048 -> x2.00,
1536x4096 -> x1.18), PLE gate (256x1536 -> x1.51), PLE projection (1536x256 -> x1.20),
fused gate+up+GELU (6144 -> x1.32, 12288 -> x1.18). QKV, Q and down-projection
stay on cuBLAS (no win). Guards require exactly one row of contiguous BF16 CUDA
data, no bias and inference mode; anything else falls back to the original module.
Batch>1 decode, speculative verification (M=4) and prefill are NOT affected and NOT
optimized by this change.

**Correctness evidence:**
- Kernel level: fp64-oracle error is the same as cuBLAS's in every swept shape
  (max and mean error ratio 1.00, sampled weight copies); 7 new B200 pytest cases
  pass (fp64 oracle, scale 1/30/0.001, exact-cancellation case, changed-input CUDA
  graph replay, and fallback outside the contract); Modal run ap-ODrTIzyE44AtcUt5mxNaPt.
  Reduction order differs from cuBLAS, so results are NOT bit-identical to the old path.
- Engine level, real prompts (WildChat; see `prepare_wildchat_prompts.py`; this
  workload is code/paste heavy because Resin's packing picks the longest user turns):
  6 prompts at 128-1024 tokens (4 distinct greedy continuations, so ~84 compared
  steps), 22 greedy tokens each. Fused-GELU-only vs +M=1 kernels: **all token IDs
  identical**; on equal-history steps the top-1 logit differs by at most 0.25
  (median 0-0.125), i.e. one or two BF16 steps at these magnitudes. Results
  `b200-resin-real-{base,m1}-draft0.json`, runs ap-pBJK5KPru2QJ3Fwh3924r5 and
  ap-mBqjnjdpQVO8nPTboexOot.
- Warning from the synthetic random-token workload: there the same two variants
  diverged greedy tokens at position 13 and some top-1 logits differed by up to
  ~4 with identical history (`b200-resin-diag-*.json`). The baseline's own margin
  at the flip was only 0.094 (a near-tie), and random-token prompts put the model
  far from its training distribution, so I do not treat it as evidence of an error;
  but it shows the kernels are not numerically invariant to cuBLAS on
  out-of-distribution inputs. Not yet checked: baseline-vs-baseline run-to-run
  determinism, longer generations, other prompt types, or a high-precision oracle.

**Speed (draft 0, batch 1, synthetic prompts, 256 context; no diagnostic hook):**
decode median 2.961 -> 2.886 (fused GELU) -> **2.582 ms/token** (all M=1 kernels);
-12.8% from the unmodified upstream path, -10.5% from the GELU step; every one
of 16 steps in range 2.575-2.708 ms. Kernels per token 795 -> 725 -> 648.
Single run each, not interleaved; synthetic prompts. Real-prompt A/B timings
(`b200-resin-real-*.json`) are INVALID in absolute terms because the diagnostic
hook synchronizes and writes every step (decode ~3.2-3.9 ms with it); only their
paired difference (~0.25-0.27 ms at every context, consistent with the synthetic
result) is informative. Prefill timings in the same runs are noise-level
(7.7-17.5 ms across identical configurations; n=3): a bimodal prefill result
has appeared in three separate runs and is unexplained.

## Split-KV paged decode attention, clean interleaved A/B, and vLLM baseline

**Attention kernel (fork):** the profile showed FlashInfer's q=1 decode launching a
single CTA per layer (8 query heads share one KV head), ~10 us per layer in an L2-cold
isolated study (`modal_resin_baseline.py --attn ...`, results `b200-paged-decode-study.json`;
35 distinct synthetic KV layers rotated per graph, one request, FlashInfer decode wrapper
built exactly as Resin builds it). `paged_decode_q1` (`edge_kernels.py`,
`edge_kernels_triton.py`) splits the visible KV range over 16 CTAs and merges the partial
softmax states in a second tiny kernel. Page tables, last-page lengths and the sliding
window are read from the same device buffers FlashInfer reads, so it replays in the
existing decode CUDA graph. fp64-oracle max error was equal to or below FlashInfer's at
every swept length (e.g. 0.0074-0.0085 vs 0.0074-0.0150).
Isolated speed vs FlashInfer: local layers (head width 256, window) x1.7-2.0
(~10.2 -> 5-6 us); global layers (width 512) x1.3-1.65 up to ~1K tokens, tie at 2K,
**slower at 4K (x0.72)**. Therefore global layers use the kernel only when the graph's
page capacity is <= 1,024 tokens; everything else (prefill, verification, long global
history, host plans) runs FlashInfer unchanged. Only the draft-free single-query decode
path is covered.
Tests: 6 new B200 pytest cases pass (fp64 oracle vs FlashInfer error over lengths
1/63/64/65/300/511/512/513/1500, window and full history, widths 256/512; CUDA graph
replayed after changing lengths and page tables in place; a zero-page padded row stays
zero; unsupported inputs rejected), Modal run ap-otC4vXBflGHdPkYdUpSazB.

**Clean A/B (fresh engine per run, one container, two interleaved rounds, decode-only
graphs, WildChat-derived prompts, 48 timed steps, no diagnostic hook)**, results
`b200-resin-ab-interleaved.json`, run ap-WPCVIhGCFQCFsoMkdqJxCd. Median decode ms/token,
mean of the two rounds (rounds agree within ~3 us except where noted):

| config | ctx 256 | ctx 1024 |
|---|---:|---:|
| upstream path | 3.078 | 3.495 |
| + fused GELU*up | 2.969 (-3.5%) | 3.389 (-3.0%) |
| + M=1 projection/gate-up kernels | 2.692 (-12.6%) | 3.112 (-11.0%) |
| + split-KV attention | **2.516 (-18.3%)** | **2.826 (-19.1%)** |

All 54 generated tokens are identical across all four configurations, both prompts and
both rounds (graph replays 48/48, zero eager fallbacks). Caveats: two prompts only
(real WildChat text, code/paste heavy), batch 1, greedy, draft 0, one B200 container;
one run showed a 0.9 ms min/3.8 ms max spread in a single window (round 2 fusedgelu
ctx1024 median unaffected). Logit-level comparison of the full configuration against
upstream on long generations is still outstanding.

**vLLM 0.34.0 baseline** (`modal_engines.py`; same pinned checkpoint and the same six
real prompts as token IDs, bf16, greedy, prefix caching off, default CUDA graphs,
`max_model_len` 2048, one B200; `b200-vllm-baseline.json`, run
ap-f2B0iaRPmFtC2Pp7pVL1NW): decode **2.85-2.95 ms/token** (ctx 128-1024), derived by
differencing end-to-end `generate` times at 8 and 72 output tokens (5 alternating
repeats each), so it includes vLLM's scheduler/host overhead. Its first 22 greedy
tokens equal Resin's on all six prompts.
Comparison (different measurement methods - vLLM end-to-end wall differencing vs Resin's
harness step interval - so treat as approximate): at ctx 256 vLLM 2.85 vs Resin upstream
3.08 (Resin 8% slower), optimized Resin 2.52 (12% faster than vLLM); at ctx 1024 vLLM
2.95 vs upstream 3.50 (Resin 18% slower) vs optimized 2.83 (4% faster). Resin's lead
over vLLM at 1024 is small enough that method differences could erase it. vLLM runs
here were default-configured; its speculative-decoding (MTP) mode and tuning flags
were not explored, and TTFT, batch>1, throughput and SGLang were not compared in this
step.
