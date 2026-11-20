"""Prepared GPU calibration only. Import is local; running incurs Modal charges.

After explicit budget approval:
  modal run experiments/kernel_eval/modal_probe.py --approve-spend
No weights, volumes, secrets, serving endpoint or arbitrary candidate imports.
"""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App('resin-kernel-eval-calibration')
# Version tags are pinned but not digest-locked; capture actual runtime versions.
image = (modal.Image.from_registry('nvidia/cuda:13.3.1-devel-ubuntu22.04', add_python='3.12')
         .pip_install('torch==2.13.0')
         .add_local_file(ROOT/'task.json', '/opt/eval/task.json')
         .add_local_file(ROOT/'numerics.py', '/opt/eval/numerics.py'))


@app.function(image=image, gpu='B200', timeout=180, retries=0,
              max_containers=1, scaledown_window=2)
def probe():
    import sys
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, '/opt/eval')
    from numerics import reference
    task = json.loads(Path('/opt/eval/task.json').read_text())
    name = torch.cuda.get_device_name(0)
    if 'B200' not in name:
        raise RuntimeError(f'Unexpected GPU: {name}')
    torch.manual_seed(719)
    results = []
    # Small pilot; largest shapes and sanitizer/graph tests are later gates.
    for width in task['widths']:
        for rows in (1, 7, 33):
            g = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)
            u = torch.randn_like(g)
            def baseline():
                return F.gelu(g, approximate='tanh') * u
            out = baseline()
            # Compare a bounded sample against the scalar spec, report only.
            gg, uu = g.flatten()[:257].tolist(), u.flatten()[:257].tolist()
            expected = torch.tensor(reference(gg, uu), dtype=torch.float32)
            observed = out.flatten()[:257].float().cpu()
            abs_error = (expected-observed).abs()
            for _ in range(20):
                baseline()
            torch.cuda.synchronize()
            times = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    baseline()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end)*1000/100)
            results.append({'rows': rows, 'width': width,
                            'sample_max_abs_error': abs_error.max().item(),
                            'sample_mismatch_count': (abs_error != 0).sum().item(),
                            'baseline_us_per_call_batches': times})
    return {'status': 'calibration_only', 'gpu_qualified': False,
            'candidate': None, 'gpu': name, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'capability': torch.cuda.get_device_capability(),
            'results': results}


@app.local_entrypoint()
def main(approve_spend: bool = False):
    if not approve_spend:
        raise RuntimeError('GPU run requires explicit spending approval; pass --approve-spend only after approval')
    print(json.dumps(probe.remote(), indent=2))
