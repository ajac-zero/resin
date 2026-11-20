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
