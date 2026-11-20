import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_shared import dispatch, interval, analyze

class SharedAnalysisTests(unittest.TestCase):
    def test_frozen_boundaries(self):
        self.assertEqual([dispatch(n) for n in (1,12288,12289,405504,405505)],
                         ['rust_256','rust_256','rust_512','rust_512','triton'])

    def test_interval_reproducible(self):
        self.assertEqual(interval([1,2,3]),interval([1,2,3]))
        self.assertEqual(interval([5]*15),[5,5])

    def test_fallback_is_not_claimed_as_a_rust_win(self):
        report={'results':[{'elements':792576,'us_per_call':{'triton':[2]*15,'rust_512':[3]*15}}]}
        result=analyze(report)[0]
        self.assertEqual(result['paired_median_latency_reduction_percent'],0)
        self.assertEqual(result['selected'],'triton')
