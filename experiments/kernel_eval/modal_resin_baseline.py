"""Bounded B200 bring-up; model-specific minimal dependency image, not full distro."""
import json
import modal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/resin')
if modal.is_local():
    from modal_model_profile import image as reference_image
    image=(reference_image
        .run_commands('timeout 240 python -c "from huggingface_hub import snapshot_download; snapshot_download(\'google/gemma-4-E2B-it-assistant\',revision=\'2d874ef7d29f9a30599a1e4b3c1cbc9595f005df\',allow_patterns=[\'config.json\',\'model.safetensors\'],local_dir=\'/opt/assistant\')"')
        .apt_install('build-essential','curl','pkg-config')
        .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1")
        .env({'PATH':'/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
              'TORCH_CUDA_ARCH_LIST':'10.0a','FLASHINFER_CUDA_ARCH_LIST':'10.0a',
              'CUTE_DSL_ARCH':'sm_100a','MAX_JOBS':'4'})
        .pip_install('torch==2.15.0','flashinfer-python==0.7.1','nvidia-cutlass-dsl==4.8.1',
                     'nvidia-cutlass-dsl-libs-base==4.8.1','apache-tvm-ffi==0.1.12','quack-kernels==0.6.5','blake3','pillow','pytest>=8,<9','hypothesis>=6.140,<7')
        .add_local_dir(ROOT,'/opt/resin',copy=True,ignore=['.git','.venv','target','**/__pycache__','experiments'])
        .run_commands('cd /opt/resin && timeout 600 python -m pip install --no-deps . ./models/torch ./models/cuda && mkdir -p /opt/aux && cp -r deploy scripts tests /opt/aux/ && mkdir -p /opt/aux/models/cuda && cp -r models/cuda/tests /opt/aux/models/cuda/ && rm -rf /opt/resin && python -c "import resin._resin,resin;print(resin.__file__)"'))
else:
    image=modal.Image.debian_slim()
app=modal.App('resin-b200-baseline-qualification')

@app.function(image=image,gpu='B200',cpu=4,memory=16384,timeout=240,retries=0,max_containers=1,scaledown_window=2)
def qualify(draft_tokens: int=3, trace: bool=False):
    import sys,os
    sys.path.append('/opt/aux')
    import torch
    from dataclasses import replace
    from deploy.models import MODEL_SPECS,prepare_runtime
    from scripts.bench import run_benchmark
    if torch.cuda.get_device_capability()!=(10,0):raise RuntimeError('B200 capability mismatch')
    spec=MODEL_SPECS['gemma4-e2b']
    MODEL_SPECS['gemma4-e2b']=replace(spec,gpu='B200',context_length=2048,
                                   max_batch_size=1,prefill_chunk_size=1024,draft_tokens=draft_tokens)
    prepare_runtime(MODEL_SPECS['gemma4-e2b'])
    records=run_benchmark('gemma4-e2b','/opt/model',
        [{'phase':'prefill','context_length':256,'batch_size':1},
         {'phase':'decode','context_length':256,'batch_size':1}],
        draft_tokens=draft_tokens,assistant_checkpoint='/opt/assistant' if draft_tokens else None,warmup_steps=4,measure_steps=16,repeats=1,
        prefill_warmup_cohorts=1,prefill_measure_cohorts=3,
        prefill_chunk_size=1024,gpu_memory_utilization=0.15,prompt_source='synthetic',
        retain_outputs=True,numa='off',overlap=True,trace_dir='/tmp/resin-trace' if trace else None)
    summary=trace_summary('/tmp/resin-trace') if trace else None
    return {'trace_summary':summary,'status':'experimental_resin_b200_baseline','torch':str(torch.__version__),
            'gpu':torch.cuda.get_device_name(),'records':records,
            'limitations':'Minimal model-specific dependency image; experimental B200 overrides, not released preset; synthetic batch1 workload; serving baseline not yet independently qualified.'}

def trace_summary(directory):
    import gzip,glob,json,collections
    out={}
    for path in sorted(glob.glob(directory+'/*.json.gz')):
        events=json.load(gzip.open(path,'rt')).get('traceEvents',[])
        kernels=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
        launches=[e for e in events if e.get('ph')=='X' and 'cudaGraphLaunch' in e.get('name','')]
        by=collections.defaultdict(lambda:[0.0,0])
        for e in kernels:
            by[e['name']][0]+=e['dur'];by[e['name']][1]+=1
        iv=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
        busy=0.0;cur=None
        for a,b in iv:
            if cur is None:cur=[a,b]
            elif a<=cur[1]:cur[1]=max(cur[1],b)
            else:busy+=cur[1]-cur[0];cur=[a,b]
        if cur:busy+=cur[1]-cur[0]
        span=(iv[-1][1]-iv[0][0]) if iv else 0
        n=max(1,len(launches))
        top=sorted(by.items(),key=lambda kv:kv[1][0],reverse=True)[:30]
        out[path.split('/')[-1]]={'graph_launches':len(launches),'kernel_events':len(kernels),
          'gpu_busy_us_total':busy,'gpu_span_us':span,'busy_over_span':busy/span if span else None,
          'sum_kernel_us_per_launch':sum(v[0] for v in by.values())/n,
          'kernel_count_per_launch':len(kernels)/n,
          'top_kernels':[{'name':k[:140],'us_per_launch':v[0]/n,'count_per_launch':v[1]/n} for k,v in top]}
    return out

@app.function(image=image,cpu=2,memory=8192,timeout=300,retries=0,max_containers=1)
def unit(path: str='tests/test_cuda_completion.py models/cuda/tests/test_gemma4_edge_model.py'):
    import subprocess
    r=subprocess.run(['python','-m','pytest',*path.split(),'-q','-x','-p','no:cacheprovider'],cwd='/opt/aux',capture_output=True,text=True)
    return {'returncode':r.returncode,'stdout':r.stdout[-6000:],'stderr':r.stderr[-3000:]}

@app.local_entrypoint()
def main(approve_spend:bool=False,draft_tokens:int=3,unit_only:bool=False,trace:bool=False):
    if unit_only:
        print(json.dumps(unit.remote(),indent=2));return
    if not approve_spend:raise RuntimeError('Explicit spending approval required')
    print(json.dumps(qualify.remote(draft_tokens,trace),indent=2))
