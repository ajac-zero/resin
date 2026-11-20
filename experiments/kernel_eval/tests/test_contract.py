import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import Report, TensorSpec, load_task


class ContractTests(unittest.TestCase):
    def test_valid_spec(self):
        TensorSpec(1, 6144).validate()

    def test_invalid_specs(self):
        for spec in [TensorSpec(0, 1), TensorSpec(-1, 1), TensorSpec(True, 1),
                     TensorSpec(1, 2.5), TensorSpec(1, 1, "float32"),
                     TensorSpec(1, 1, contiguous=False)]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                spec.validate()

    def test_manifest_and_hash(self):
        p = Path(__file__).resolve().parents[1] / "task.json"
        task, digest = load_task(p)
        self.assertEqual(len(digest), 64)
        self.assertEqual(task["widths"], [6144, 12288])
        with tempfile.TemporaryDirectory() as d:
            q = Path(d) / "task.json"
            q.write_bytes(p.read_bytes() + b"\n")
            self.assertNotEqual(load_task(q)[1], digest)
            task["rows"] = [True]
            q.write_text(json.dumps(task))
            with self.assertRaises(ValueError):
                load_task(q)

    def test_report_not_gpu_claim(self):
        r = Report("task", "abc", None, "cpu", "contract_validated", ())
        self.assertFalse(json.loads(r.to_json())["gpu_qualified"])
        with self.assertRaises(ValueError):
            Report("task", "abc", None, "cpu", "contract_validated", (), True).to_json()


if __name__ == "__main__":
    unittest.main()
