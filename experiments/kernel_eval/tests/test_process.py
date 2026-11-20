import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from process import run


class ProcessTests(unittest.TestCase):
    def execute(self, source, **kwargs):
        return run([sys.executable, '-I', '-c', source], **kwargs)

    def test_success(self):
        r = self.execute('print("ok")')
        self.assertEqual(r.status, 'ok')
        self.assertEqual(r.stdout, b'ok\n')

    def test_crash(self):
        self.assertEqual(self.execute('import os; os.abort()').status, 'failed')

    def test_timeout(self):
        self.assertEqual(self.execute('import time; time.sleep(30)', timeout=.15).status,
                         'timeout')

    def test_output_cap(self):
        r = self.execute('import os; os.write(1, b"x"*1000000)', output_bytes=1024)
        self.assertEqual(r.status, 'output_limit')
        self.assertLessEqual(len(r.stdout), 1024)

    def test_memory_limit(self):
        self.assertEqual(self.execute('x=bytearray(512*1024*1024)', memory_mb=64).status,
                         'failed')

    def test_validation(self):
        for options in [{'timeout': 0}, {'timeout': float('nan')}, {'memory_mb': 0},
                        {'output_bytes': 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.execute('', **options)
