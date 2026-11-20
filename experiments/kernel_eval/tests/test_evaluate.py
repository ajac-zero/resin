import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate


class EndToEndTests(unittest.TestCase):
    def test_fixture_results(self):
        expected = {'reference': 'scalar_spec_pass', 'zero': 'incorrect',
                    'skip_round': 'incorrect', 'tail': 'incorrect', 'stale': 'incorrect',
                    'hang': 'timeout', 'crash': 'failed', 'malformed': 'protocol_error'}
        for fixture, status in expected.items():
            with self.subTest(fixture=fixture):
                report = evaluate(fixture, timeout=.5)
                self.assertEqual(report['status'], status)
                self.assertFalse(report['gpu_qualified'])
                self.assertIsNone(report['performance'])

    def test_arbitrary_candidate_refused(self):
        with self.assertRaises(ValueError):
            evaluate('/tmp/unknown.py')
