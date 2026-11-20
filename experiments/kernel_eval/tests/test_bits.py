import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bits import compare


class BitJudgeTests(unittest.TestCase):
    def test_all_encodings_identity(self):
        all_bits = list(range(65536))
        self.assertEqual(compare(all_bits, all_bits)['mismatches'], 0)

    def test_nan_payload_equivalence_only(self):
        self.assertEqual(compare([0x7fc0, 0xffc1], [0xffff, 0x7f81])['mismatches'], 0)
        self.assertEqual(compare([0x7fc0], [0x7f80])['mismatches'], 1)
        self.assertEqual(compare([0x7fc0], [0x3f80])['mismatches'], 1)

    def test_strict_sign_and_rounding(self):
        self.assertEqual(compare([0, 0x7f80, 0x3f80], [0x8000, 0xff80, 0x3f81])['mismatches'], 3)

    def test_malformed_is_not_numeric_pass(self):
        for bad in (None, [], [True], [-1], [65536], [1.0], ['1'], [1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare(bad, [1])

    def test_tail_and_stale_output_rejected(self):
        expected = [0x3f80] * 257
        got = expected[:-1] + [0]
        self.assertEqual(compare(got, expected)['indices'], [256])
        self.assertEqual(compare([0x3f00] * 257, expected)['mismatches'], 257)
