"""Dependency-free protocol scaffold; not a CUDA qualification or sandbox."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TensorSpec:
    rows: int
    width: int
    dtype: str = "bfloat16"
    contiguous: bool = True

    def validate(self) -> None:
        if type(self.rows) is not int or type(self.width) is not int:
            raise ValueError("dimensions must be integers")
        if self.rows <= 0 or self.width <= 0:
            raise ValueError("dimensions must be positive")
        if self.dtype != "bfloat16" or self.contiguous is not True:
            raise ValueError("only contiguous BF16 matrices are supported")


class Candidate(Protocol):
    """Future GPU adapter: caller owns buffers and stream; no retained inputs.

    Writes every output element, leaves inputs unchanged, enqueues on the supplied
    stream. Timing, synchronization and correctness decisions belong to evaluator.
    Opaque objects here deliberately do not invent an unverified CUDA Rust ABI.
    """

    def launch(self, gate: object, up: object, output: object,
               spec: TensorSpec, stream: object) -> None: ...


def load_task(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    task = json.loads(raw)
    if task.get("schema_version") != 1:
        raise ValueError("unsupported task schema")
    if task.get("operation") != "gelu_tanh_mul":
        raise ValueError("unsupported operation")
    if task.get("dtype") != "bfloat16":
        raise ValueError("unsupported dtype")
    for key in ("rows", "widths"):
        values = task.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
        if any(type(x) is not int or x <= 0 for x in values):
            raise ValueError(f"invalid {key}")
    return task, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Report:
    task_id: str
    task_sha256: str
    candidate_sha256: str | None
    backend: str
    status: str
    checks: tuple[str, ...]
    gpu_qualified: bool = False
    performance: None = None

    def to_json(self) -> str:
        if self.backend != "cpu" or self.gpu_qualified or self.performance is not None:
            raise ValueError("scaffold cannot certify GPU execution or performance")
        if self.status not in {"contract_validated", "failed"}:
            raise ValueError("invalid scaffold status")
        return json.dumps(asdict(self), indent=2, allow_nan=False)
