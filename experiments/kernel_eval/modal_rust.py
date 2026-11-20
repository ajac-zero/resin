"""Bounded standalone cuTile Rust qualification; no Resin zero-copy integration."""
import json
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent
app = modal.App('resin-rust-gelu-qualification')
image = (modal.Image.from_registry('nvidia/cuda:13.3.1-devel-ubuntu22.04', add_python='3.12')
 .apt_install('build-essential', 'curl', 'pkg-config', 'libclang-dev')
 .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1")
 .env({'PATH': '/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'CUDA_TOOLKIT_PATH': '/usr/local/cuda'})
 .pip_install('torch==2.13.0')
 .add_local_dir(ROOT/'rust', '/opt/rust', copy=True, ignore=['target'])
 .run_commands('cd /opt/rust && timeout 240 cargo build --locked --release')
 .add_local_file((ROOT.parents[1]/'models/cuda/resin_models_cuda/gemma4/edge_kernels_triton.py') if modal.is_local() else Path('/opt/comparator.py'), '/opt/comparator.py'))

@app.function(image=image, gpu='B200', timeout=120, retries=0, max_containers=1, scaledown_window=2)
def qualify():
    import subprocess
    import torch
    import torch.nn.functional as F
    import importlib.util
    spec=importlib.util.spec_from_file_location("comparator", "/opt/comparator.py")
    comparator=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    torch.manual_seed(719)
    reports=[]
    for n in (1, 7, 257, 6144, 12288, 43008, 202752, 405504, 792576):
        g=torch.randn(n,dtype=torch.bfloat16,device='cuda')
        u=torch.randn_like(g)
        request=json.dumps({'gate': g.tolist(), 'up': u.tolist()})+'\n'
        r=subprocess.run(['/opt/rust/target/release/resin-gelu-probe'],input=request,
                         text=True,capture_output=True,timeout=20)
        if r.returncode:
            return {'status':'rust_failed','stderr':r.stderr[-12000:]}
        rust_report=json.loads(r.stdout)
        out=torch.tensor(rust_report['output'],dtype=torch.bfloat16,device='cuda')
        ref=F.gelu(g, approximate='tanh')*u
        tri=torch.empty_like(g)
        comparator.launch_gelu_tanh_mul(g,u,tri)
        torch.cuda.synchronize()
        changed=torch.tensor(rust_report['changed_output'],dtype=torch.bfloat16,device='cuda')
        changed_ref=F.gelu((g.float()+0.5).bfloat16(), approximate='tanh')*u
        error=(out.float()-ref.float()).abs()
        timings={}
        for label, fn in [('pytorch', lambda: F.gelu(g, approximate='tanh')*u),
                          ('triton', lambda: comparator.launch_gelu_tanh_mul(g,u,tri))]:
            for _ in range(10): fn()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100): fn()
            graph.replay()
            torch.cuda.synchronize()
            measured=[]
            for _ in range(5):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(); graph.replay(); b.record(); b.synchronize()
                measured.append(a.elapsed_time(b)*1000/100)
            timings[label]=measured
        timings['rust']=rust_report['graph_us_per_call']
        reports.append({'elements':n,'mismatch_count':(out!=ref).sum().item(),
                        'max_abs_error':error.max().item(),
                        'triton_mismatch_count':(out!=tri).sum().item(),
                        'changed_replay_mismatch_count':(changed!=changed_ref).sum().item(), 'graph_us_per_call':timings})
    import shutil
    sanitizer={'status':'unavailable'}
    if shutil.which('compute-sanitizer'):
        request=json.dumps({'gate':[1.0]*257, 'up':[0.5]*257})+'\n'
        checked=subprocess.run(['compute-sanitizer','--tool','memcheck','--error-exitcode','99',
                                '/opt/rust/target/release/resin-gelu-probe'],
                               input=request,text=True,capture_output=True,timeout=45)
        sanitizer={'status':'passed' if checked.returncode==0 else 'failed',
                   'returncode':checked.returncode,'stdout':checked.stdout[-2000:],
                   'stderr':checked.stderr[-4000:]}
    return {'numerical_gate_passed': all(r['mismatch_count']==0 and r['triton_mismatch_count']==0 and r['changed_replay_mismatch_count']==0 for r in reports), 'sanitizer':sanitizer,'status':'numerical_probe_only' , 'gpu':torch.cuda.get_device_name(),
            'results':reports,'gpu_qualified':False, 'performance':None}

@app.local_entrypoint()
def main(approve_spend: bool=False):
    if not approve_spend:
        raise RuntimeError('Explicit spending approval required')
    print(json.dumps(qualify.remote(),indent=2))
