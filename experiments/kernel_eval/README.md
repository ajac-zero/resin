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
