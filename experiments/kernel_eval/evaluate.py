"""Integrated CPU harness self-test; no arbitrary local candidates or GPU claims."""
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from numerics import bf16, equal, reference
from process import run
from contract import load_task

ROOT = Path(__file__).resolve().parent
FIXTURES = ('reference', 'zero', 'skip_round', 'tail', 'stale', 'hang', 'crash', 'malformed')


def evaluate(fixture, timeout=2):
    if fixture not in FIXTURES:
        raise ValueError('only built-in fixtures allowed')
    task, digest = load_task(ROOT / 'task.json')
    rng = random.Random(719)
    cases = []
    for n in (1, 7, 257):
        gate = [bf16(rng.uniform(-4, 4)) for _ in range(n)]
        up = [bf16(rng.uniform(-3, 3)) for _ in range(n)]
        cases.extend([{'gate': gate, 'up': up},
                      {'gate': [bf16(x+.5) for x in gate], 'up': up}])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'inputs.json'
        path.write_text(json.dumps({'cases': cases}))
        result = run([sys.executable, '-I', str(ROOT / 'fixture_worker.py'),
                      fixture, str(path)], timeout=timeout)
    status = result.status
    if status == 'ok':
        try:
            payload = json.loads(result.stdout)
            outputs = payload['outputs']
            if type(outputs) is not list or len(outputs) != len(cases):
                raise ValueError('case count mismatch')
            passed = True
            for case, actual in zip(cases, outputs):
                expected = reference(case['gate'], case['up'])
                if type(actual) is not list or len(actual) != len(expected):
                    passed = False
                    break
                if not all(type(a) in (int, float) and equal(a, b)
                           for a, b in zip(actual, expected)):
                    passed = False
                    break
            status = 'scalar_spec_pass' if passed else 'incorrect'
        except (ValueError, KeyError, TypeError):
            status = 'protocol_error'
    return {'task_id': task['task_id'], 'task_sha256': digest,
            'fixture': fixture, 'worker_sha256': hashlib.sha256(
                (ROOT/'fixture_worker.py').read_bytes()).hexdigest(),
            'evaluator_sha256': hashlib.sha256((ROOT/'evaluate.py').read_bytes()).hexdigest(),
            'oracle_sha256': hashlib.sha256((ROOT/'numerics.py').read_bytes()).hexdigest(),
            'backend': 'cpu', 'status': status, 'returncode': result.returncode,
            'cases': len(cases), 'gpu_qualified': False, 'performance': None}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fixture', choices=FIXTURES)
    args = parser.parse_args()
    report = evaluate(args.fixture)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['status'] == 'scalar_spec_pass' else 1)
