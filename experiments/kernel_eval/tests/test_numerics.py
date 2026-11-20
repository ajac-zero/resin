import math
import random
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numerics import bf16, check, gelu, reference


class NumericalTests(unittest.TestCase):
    def test_rounding_ties_and_specials(self):
        self.assertEqual(bf16(1 + 1/256), 1)
        self.assertEqual(bf16(1 + 3/256), 1 + 2/128)
        self.assertEqual(bf16(2**-133), 2**-133)
        self.assertEqual(bf16(2**-134), 0)
        self.assertLess(math.copysign(1, bf16(-0.0)), 0)
        self.assertTrue(math.isnan(bf16(float('nan'))))
        self.assertEqual(bf16(float('inf')), float('inf'))
        self.assertEqual(bf16(1e100), float('inf'))

    def test_reference_known_values(self):
        self.assertEqual(reference([0, 1, -1], [2, 1, 1]),
                         [0, .83984375, -.1591796875])
        with self.assertRaises(ValueError):
            reference([1], [])

    def test_mutants_rejected(self):
        rng = random.Random(719)
        gate = [bf16(rng.uniform(-4, 4)) for _ in range(257)]
        up = [bf16(rng.uniform(-3, 3)) for _ in gate]
        def mutate(g, u):
            out = reference(g, u)
            g[0] = 999
            return out
        def incomplete(g, u):
            return reference(g, u)[:-1] + [float('nan')]
        stale = reference(gate, up)
        mutants = {
            'zeros': lambda g, u: [0.] * len(g),
            'wrong_activation': lambda g, u: [bf16(max(0, x)*y) for x,y in zip(g,u)],
            'dropped_tail': lambda g, u: reference(g,u)[:-1],
            'unwritten_tail': incomplete,
            'skipped_bf16_intermediate': lambda g, u: [bf16(gelu(x)*y) for x,y in zip(g,u)],
            'input_mutation': mutate,
            'stale_output': lambda g, u: stale,
        }
        # Changed-input replay catches a candidate that caches the first result.
        changed = [bf16(x + .5) for x in gate]
        self.assertTrue(check(reference, changed, up))
        for name, fn in mutants.items():
            with self.subTest(mutant=name):
                self.assertFalse(check(fn, changed, up))

    def test_nonfinite_mismatch_and_exception(self):
        self.assertFalse(check(lambda g,u: [float('nan')], [1], [1]))
        self.assertFalse(check(lambda g,u: [float('inf')], [1], [1]))
        def crash(g,u):
            raise RuntimeError('intentional')
        self.assertFalse(check(crash, [1], [1]))
