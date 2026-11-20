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

Next: independent CPU numerical oracle with explicit BF16 rounding, adversarial
fixtures/mutants, subprocess failure handling and bounded memory. Then qualify
PyTorch/Triton/Rust on B200, calibrate and freeze numerical thresholds, and implement
actual GPU adapter. Do not equate an idealized scalar tanh formula with exact
PyTorch GPU results before calibration.

Security: hashes identify artifacts, not trustworthiness. This local scaffold is
not isolation for malicious native code. Final hidden tests and trusted evaluator
must run outside agent-writable state; process isolation alone is not a sandbox.
No hidden fixtures or GPU measurements exist yet. Manifest validation currently
covers schema version, operation, dtype and positive shapes, not full JSON Schema.
