"""Linux bounded command runner for accidental failures, NOT hostile-code isolation."""
import math
import os
import resource
import signal
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessResult:
    status: str
    returncode: int
    stdout: bytes
    stderr: bytes


def run(command, *, timeout=2.0, memory_mb=256, output_bytes=65536):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    if type(memory_mb) is not int or memory_mb < 32:
        raise ValueError('invalid memory limit')
    if type(output_bytes) is not int or output_bytes <= 0:
        raise ValueError('invalid output limit')
    # preexec_fn is appropriate only in this single-threaded CLI/test harness.
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb*1024**2,)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(timeout)),)*2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes,)*2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                             start_new_session=True, preexec_fn=limits)
        timed_out = False
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Clean up descendants even when the direct child exits first.
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
        oversized = out.seek(0, 2) >= output_bytes or err.seek(0, 2) >= output_bytes
        out.seek(0)
        err.seek(0)
        status = ('timeout' if timed_out else 'output_limit' if oversized else
                  'ok' if p.returncode == 0 else 'failed')
        return ProcessResult(status, p.returncode, out.read(output_bytes),
                             err.read(output_bytes))
