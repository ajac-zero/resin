"""Scalar BF16 specification for harness testing, not a GPU bitwise oracle."""
import math
import struct


def bf16(x):
    """Round binary32 to BF16, nearest-even; preserve nonfinite class."""
    try:
        bits = struct.unpack('<I', struct.pack('<f', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)
    if bits & 0x7f800000 == 0x7f800000:
        if bits & 0x007fffff:
            bits |= 0x00400000  # quiet NaN, never round into infinity
        return struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def gelu(x):
    if math.isnan(x):
        return x
    if math.isinf(x):
        return x if x > 0 else -0.0
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + .044715 * x**3)))


def reference(gate, up):
    if len(gate) != len(up):
        raise ValueError('input lengths differ')
    return [bf16(bf16(gelu(bf16(g))) * bf16(u)) for g, u in zip(gate, up)]


def equal(a, b):
    # NaNs match by class; signed zero is deliberately equivalent in episode 1.
    return (math.isnan(a) and math.isnan(b)) or a == b


def check(candidate, gate, up):
    """Exact scalar-spec comparison. Not calibrated GPU acceptance tolerance."""
    expected = reference(gate, up)
    g, u = list(gate), list(up)
    try:
        actual = candidate(g, u)
        if len(actual) != len(expected):
            return False
        if not all(equal(a, b) for a, b in zip(g, gate)):
            return False
        if not all(equal(a, b) for a, b in zip(u, up)):
            return False
        return all(equal(a, b) for a, b in zip(actual, expected))
    except Exception:
        return False
