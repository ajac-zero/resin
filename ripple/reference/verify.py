"""Verify the vendored reference files against ``SHA256SUMS``."""

from __future__ import annotations

import hashlib
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent
REFERENCE_FILES = ("config.json", "configuration_ricursive.py", "modeling_ricursive.py")


def read_sha256sums(directory: Path = REFERENCE_DIR) -> dict[str, str]:
    sums = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.strip().lstrip("*")] = digest
    return sums


def verify_reference_files(directory: Path = REFERENCE_DIR) -> dict[str, str]:
    """Return the verified digests; raise ``ValueError`` on any mismatch."""

    expected = read_sha256sums(directory)
    problems = []
    for name in REFERENCE_FILES:
        if name not in expected:
            problems.append(f"{name}: missing from SHA256SUMS")
            continue
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: file is missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected[name]:
            problems.append(f"{name}: sha256 {actual} != {expected[name]}")
    if problems:
        raise ValueError(
            "reference files do not match SHA256SUMS: " + "; ".join(problems)
        )
    return {name: expected[name] for name in REFERENCE_FILES}


if __name__ == "__main__":
    for file_name, file_digest in verify_reference_files().items():
        print(f"OK  {file_digest}  {file_name}")
