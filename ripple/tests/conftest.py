from __future__ import annotations

import signal

import pytest

TEST_TIMEOUT_SECONDS = 120


@pytest.fixture(autouse=True)
def _per_test_timeout():
    def expired(signum, frame):
        raise TimeoutError(f"test exceeded {TEST_TIMEOUT_SECONDS}s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(TEST_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
