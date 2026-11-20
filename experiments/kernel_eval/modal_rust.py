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
 .add_local_file((ROOT.parents[1]/'models/cuda/resin_models_cuda/gemma4/edge_kernels_triton.py') if modal.is_local() else Path('/opt/comparator.py'), '/opt/comparator.py')
 .add_local_file(ROOT/'bits.py', '/opt/bitjudge.py'))

@app.function(image=image, gpu='B200', timeout=120, retries=0, max_containers=1, scaledown_window=2)
def qualify(adversarial: bool=False, safety: bool=False):
    if safety:
        return safety_probe()
    if adversarial:
        return adversarial_probe()
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
def main(approve_spend: bool=False, adversarial: bool=False, safety: bool=False):
    if not approve_spend:
        raise RuntimeError('Explicit spending approval required')
    print(json.dumps(qualify.remote(adversarial, safety),indent=2))


def adversarial_probe():
    """Exhaust BF16 gate encodings, NOT the Cartesian product of both operands."""
    import subprocess
    import torch
    import torch.nn.functional as F
    import importlib.util
    spec=importlib.util.spec_from_file_location("comparator", "/opt/comparator.py")
    comparator=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    def tensor(bits):
        return torch.tensor(bits,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    def bitlist(t):
        return (t.view(torch.int16).int() & 65535).tolist()
    judge_spec=importlib.util.spec_from_file_location("bitjudge", "/opt/bitjudge.py")
    judge=importlib.util.module_from_spec(judge_spec)
    judge_spec.loader.exec_module(judge)
    gates=list(range(65536))
    g=tensor(gates)
    results=[]
    for label, ups in [("one",[0x3f80]*65536),("minus_one",[0xbf80]*65536),
                       ("zero",[0]*65536),("max_finite",[0x7f7f]*65536),
                       ("min_subnormal",[1]*65536),("infinity",[0x7f80]*65536),
                       ("nan",[0x7fc0]*65536),
                       ("permuted_bits",[(i*40503+719)%65536 for i in gates])]:
        u=tensor(ups)
        r=subprocess.run(['/opt/rust/target/release/resin-gelu-probe'],
                         input=json.dumps({'gate_bits':gates,'up_bits':ups})+'\n',
                         text=True,capture_output=True,timeout=25)
        if r.returncode: raise RuntimeError(r.stderr[-4000:])
        report=json.loads(r.stdout)
        out=tensor(report['output_bits'])
        changed=tensor(report['changed_bits'])
        ref=F.gelu(g,approximate='tanh')*u
        tri=torch.empty_like(g)
        comparator.launch_gelu_tanh_mul(g,u,tri)
        changed_ref=F.gelu((g.float()+0.5).bfloat16(),approximate='tanh')*u
        comparisons={}
        for name, got, expected in [('pytorch',out,ref),('triton',out,tri),('changed_replay',changed,changed_ref)]:
            equal=(got==expected)|(torch.isnan(got)&torch.isnan(expected))
            bad=(~equal).nonzero().flatten()
            examples=bad[:16].tolist()
            gb,eb=bitlist(got),bitlist(expected)
            verdict=judge.compare(gb,eb)
            examples=verdict['indices']
            comparisons[name]={'mismatches':verdict['mismatches'],
                'zero_sign_differences':int(((got==0)&(expected==0)&(torch.signbit(got)!=torch.signbit(expected))).sum().item()),
                'examples':[{'index':i,'gate_bits':gates[i],'up_bits':ups[i],
                             'got_bits':gb[i],'expected_bits':eb[i]} for i in examples]}
        results.append({'up_pattern':label,'elements':65536,'comparisons':comparisons})
    return {'status':'adversarial_probe_only','gpu_qualified':False,
            'nan_policy':'exact BF16 bits including signed zeros; ignore only NaN payload/sign',
            'gate_domain':'all 65536 BF16 encodings, eight up patterns; not all operand pairs',
            'gpu':torch.cuda.get_device_name(),'results':results,
            'numerical_gate_passed':all(c['mismatches']==0 for r in results for c in r['comparisons'].values())}


def safety_probe():
    import subprocess, random, shutil
    import torch
    import torch.nn.functional as F
    rng=random.Random(91213)
    cases=[]
    for i in range(32):
        n=(1,7,255,256,257,511,512,513)[i%8]
        cases.append({'gate_bits':[rng.randrange(65536) for _ in range(n)],
                      'up_bits':[rng.randrange(65536) for _ in range(n)]})
    binary='/opt/rust/target/release/resin-gelu-probe'
    r=subprocess.run([binary],input=''.join(json.dumps(c)+'\n' for c in cases),
                     text=True,capture_output=True,timeout=55)
    if r.returncode: raise RuntimeError(r.stderr[-4000:])
    outputs=[json.loads(line) for line in r.stdout.splitlines()]
    if len(outputs)!=len(cases): raise RuntimeError('wrong response count')
    def tensor(bits):
        return torch.tensor(bits,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    results=[]
    for c,o in zip(cases,outputs):
        g,u=tensor(c['gate_bits']),tensor(c['up_bits'])
        ref=F.gelu(g,approximate='tanh')*u
        changed_ref=F.gelu((g.float()+0.5).bfloat16(),approximate='tanh')*u
        counts=[]
        for key,expected in [('output_bits',ref),('changed_bits',changed_ref)]:
            got=tensor(o[key])
            counts.append(int((~((got==expected)|(torch.isnan(got)&torch.isnan(expected)))).sum().item()))
        results.append({'elements':g.numel(),'mismatches':counts})
    checks=[]
    if shutil.which('compute-sanitizer'):
        request=json.dumps(cases[4])+'\n'
        for tool in ('memcheck','racecheck','initcheck'):
            r=subprocess.run(['compute-sanitizer','--tool',tool,'--error-exitcode','99',binary],
                             input=request,text=True,capture_output=True,timeout=35)
            checks.append({'tool':tool,'returncode':r.returncode,
                           'stdout_tail':r.stdout[-1200:],'stderr_tail':r.stderr[-4000:]})
    return {'status':'bounded_safety_probe_only','gpu_qualified':False,
            'persistent_process_requests':len(results),'results':results,'sanitizers':checks,
            'passed':all(all(n==0 for n in r['mismatches']) for r in results) and len(checks)==3 and all(c['returncode']==0 and ('0 hazards displayed (0 errors, 0 warnings)' if c['tool']=='racecheck' else 'ERROR SUMMARY: 0 errors') in c['stdout_tail']+c['stderr_tail'] for c in checks),
            'limitations':'32 requests, not a long soak or memory-leak proof; racecheck checks supported shared-memory hazards, not arbitrary stream races; initcheck observes this zero-initialized-output workload'}
