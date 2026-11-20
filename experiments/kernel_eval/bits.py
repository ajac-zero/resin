"""Parent-side BF16 probe verdict. Not a universal oracle or isolation boundary.

Exact BF16 bits required, including signed zero and infinity sign. NaN payloads
are ignored, but only another NaN is equivalent. No numerical tolerance.
"""


def validate(bits):
    if not isinstance(bits, list) or not bits:
        raise ValueError('expected a nonempty BF16 bit array')
    if any(type(x) is not int or not 0 <= x <= 65535 for x in bits):
        raise ValueError('invalid BF16 encoding')


def is_nan(x):
    return x & 0x7f80 == 0x7f80 and x & 0x007f != 0


def compare(got, expected):
    validate(got)
    validate(expected)
    if len(got) != len(expected):
        raise ValueError('wrong output length')
    bad = [i for i, (g, e) in enumerate(zip(got, expected))
           if g != e and not (is_nan(g) and is_nan(e))]
    return {'mismatches': len(bad), 'indices': bad[:16]}
