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
import os as _os
if modal.is_local() and _os.environ.get('RESIN_PROMPT_ARTIFACT'):
    image=image.add_local_file(_os.environ['RESIN_PROMPT_ARTIFACT'],'/opt/prompts.zip')
app=modal.App('resin-b200-baseline-qualification')

@app.function(image=image,gpu='B200',cpu=4,memory=16384,timeout=240,retries=0,max_containers=1,scaledown_window=2)
def qualify(draft_tokens: int=3, trace: bool=False, diag: str='', real: bool=False):
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
    if diag:
        import resin_models_cuda.gemma4.edge_cuda as edge_cuda_module
        source_path=edge_cuda_module.__file__
        with open(source_path,'a') as handle:
            handle.write('''

import os as _os
if _os.environ.get('RESIN_DIAG_LOGITS'):
    import json as _json
    import torch as _torch
    if _os.environ.get('RESIN_DIAG_DISABLE_M1'):
        from resin_models_cuda.gemma4 import edge_kernels as _ek
        _ek._M1_LINEAR.clear()
        _ek._M1_GATE_UP.clear()
    _original_run = Gemma4EdgeCudaRunner._run_cuda_graph
    def _diag_run(self, phase, shape, inputs):
        result = _original_run(self, phase, shape, inputs)
        if phase == 'decode':
            _torch.cuda.synchronize()
            values, ids = _torch.topk(self._graph_logits[shape][0].float(), 5)
            with open(_os.environ['RESIN_DIAG_LOGITS'], 'a') as _file:
                _file.write(_json.dumps({'values': values.tolist(), 'ids': ids.tolist()}) + chr(10))
        return result
    Gemma4EdgeCudaRunner._run_cuda_graph = _diag_run
''')
        os.environ['RESIN_DIAG_LOGITS']='/tmp/resin-diag-logits.jsonl'
        if diag=='base':os.environ['RESIN_DIAG_DISABLE_M1']='1'
    if real:
        contexts=(128,256,384,512,768,1024)
        cases=[{'phase':'prefill','context_length':c,'batch_size':1} for c in (256,1024)]+[{'phase':'decode','context_length':c,'batch_size':1} for c in contexts]
        prompt_options=dict(prompt_source='saved',prompt_corpus='/opt/prompts.zip')
    else:
        cases=[{'phase':'prefill','context_length':256,'batch_size':1},{'phase':'decode','context_length':256,'batch_size':1}]
        prompt_options=dict(prompt_source='synthetic')
    records=run_benchmark('gemma4-e2b','/opt/model',cases,
        draft_tokens=draft_tokens,assistant_checkpoint='/opt/assistant' if draft_tokens else None,warmup_steps=4,measure_steps=16,repeats=1,
        prefill_warmup_cohorts=1,prefill_measure_cohorts=3,
        prefill_chunk_size=1024,gpu_memory_utilization=0.15,
        retain_outputs=True,numa='off',overlap=True,trace_dir='/tmp/resin-trace' if trace else None,**prompt_options)
    summary=trace_summary('/tmp/resin-trace') if trace else None
    diag_lines=open('/tmp/resin-diag-logits.jsonl').read().splitlines() if diag else None
    return {'diag_logits':diag_lines,'trace_summary':summary,'status':'experimental_resin_b200_baseline','torch':str(torch.__version__),
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
        split=None
        ordered=sorted(kernels,key=lambda e:e['ts']);launch_count=max(1,len(launches));per=len(ordered)//launch_count
        if per:
            chunk=ordered[:per];bound=next((i for i,e in enumerate(chunk) if 'o1536' in e['name']),None)
            if bound:
                def part(es):
                    names=collections.defaultdict(lambda:[0.0,0])
                    for e in es:names[e['name']][0]+=e['dur'];names[e['name']][1]+=1
                    top=sorted(names.items(),key=lambda kv:kv[1][0],reverse=True)[:14]
                    return {'kernels':len(es),'sum_us':sum(e['dur'] for e in es),'wall_us':(es[-1]['ts']+es[-1]['dur']-es[0]['ts']) if es else 0,
                            'top':[{'name':k[:110],'us':v[0],'count':v[1]} for k,v in top]}
                split={'first_target_kernel_index':bound,'draft_phase':part(chunk[:bound]),'verify_and_rest':part(chunk[bound:])}
        out[path.split('/')[-1]]={'phase_split_first_launch':split,'graph_launches':len(launches),'kernel_events':len(kernels),
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

@app.function(image=image,gpu='B200',cpu=4,memory=16384,timeout=420,retries=0,max_containers=1,scaledown_window=2)
def gpu_unit(path: str='models/cuda/tests/test_gemma4_edge_kernels_cuda.py', select: str='m1'):
    import subprocess
    cmd=['python','-m','pytest',*path.split(),'-q','-p','no:cacheprovider']
    if select:cmd+=['-k',select]
    r=subprocess.run(cmd,cwd='/opt/aux',capture_output=True,text=True,timeout=380)
    return {'returncode':r.returncode,'stdout':r.stdout[-8000:],'stderr':r.stderr[-3000:]}


OVERLAY = '''

import os as _os
_mode = _os.environ.get('RESIN_EDGE_MODE')
if _mode:
    import torch.nn.functional as _F
    from resin_models_cuda.gemma4 import edge_kernels as _ek, edge_model as _em
    if _mode in ('upstream', 'fusedgelu'):
        _ek._M1_LINEAR.clear()
        _ek._M1_GATE_UP.clear()
    if _mode in ('upstream', 'fusedgelu', 'm1', 'paged', 'nodraft'):
        _ek._ROWS_LINEAR.clear()
    if _mode in ('upstream', 'fusedgelu', 'm1'):
        _ek._PAGED_DECODE.clear()
    if _mode in ('upstream', 'fusedgelu', 'm1', 'paged'):
        kv_append = lambda *args: False
    if _mode in ('upstream', 'fusedgelu', 'm1', 'paged', 'nodraft'):
        from resin_models_cuda.gemma4 import edge_assistant as _ea
        _ea.Gemma4EdgeAssistantForCausalLM.fused_attention_plans = False
    if _mode == 'upstream':
        _em._gelu_tanh_mul = lambda g, u: _F.gelu(g, approximate='tanh') * u
if _os.environ.get('RESIN_DIAG_LOGITS'):
    import json as _json
    import torch as _torch
    _original_run = Gemma4EdgeCudaRunner._run_cuda_graph
    def _diag_run(self, phase, shape, inputs):
        result = _original_run(self, phase, shape, inputs)
        if phase == 'decode':
            _torch.cuda.synchronize()
            values, ids = _torch.topk(self._graph_logits[shape][0].float(), 5)
            with open(_os.environ['RESIN_DIAG_LOGITS'], 'a') as _file:
                _file.write(_json.dumps({'values': values.tolist(), 'ids': ids.tolist()}) + chr(10))
        return result
    Gemma4EdgeCudaRunner._run_cuda_graph = _diag_run
'''

@app.function(image=image,gpu='B200',cpu=4,memory=16384,timeout=900,retries=0,max_containers=1,scaledown_window=2)
def ab(modes: list, contexts: list, steps: int=48, log_logits: bool=False, draft_tokens: int=0, trace: bool=False, phases: bool=False):
    """Several configurations in ONE container, each run in a fresh engine, decode-only graphs."""
    import sys,os,time,json as _j
    sys.path.append('/opt/aux')
    import torch
    from dataclasses import replace
    from deploy.models import MODEL_SPECS,prepare_runtime
    from scripts.bench import run_benchmark
    import resin_models_cuda.gemma4.edge_cuda as edge_cuda_module
    if torch.cuda.get_device_capability()!=(10,0):raise RuntimeError('B200 capability mismatch')
    spec=MODEL_SPECS['gemma4-e2b']
    MODEL_SPECS['gemma4-e2b']=replace(spec,gpu='B200',context_length=2048,max_batch_size=1,prefill_chunk_size=1024,draft_tokens=draft_tokens)
    prepare_runtime(MODEL_SPECS['gemma4-e2b'])
    with open(edge_cuda_module.__file__,'a') as handle:handle.write(OVERLAY)
    results=[]
    for index,mode in enumerate(modes):
        os.environ['RESIN_EDGE_MODE']=mode
        log=f'/tmp/logits-{index}.jsonl'
        if log_logits:os.environ['RESIN_DIAG_LOGITS']=log
        started=time.time()
        cases=[{'phase':'decode','context_length':c,'batch_size':1} for c in contexts]
        records=run_benchmark('gemma4-e2b','/opt/model',cases,draft_tokens=draft_tokens,assistant_checkpoint='/opt/assistant' if draft_tokens else None,warmup_steps=4,measure_steps=steps,repeats=1,
            prefill_chunk_size=1024,prefill_graph='disabled',gpu_memory_utilization=0.15,retain_outputs=True,numa='off',overlap=True,trace_dir=f'/tmp/trace-{index}' if trace else None,profile_graph_phases=phases,
            prompt_source='saved',prompt_corpus='/opt/prompts.zip')
        entry={'index':index,'mode':mode,'wall_seconds':time.time()-started,
               'records':[{'context':r['context_length'],'step_seconds':r['step_seconds'],'tokens':r['local_output_token_ids'][0],
                           'graph_counts':r['graph_counts'],'window_seconds':r.get('window_seconds'),'committed_tokens':r.get('committed_tokens'),'accept_rate':r.get('measured_accept_rate'),'phases':{k:v for k,v in r.items() if 'phase' in k and k not in ('phase',)}} for r in records]}
        if log_logits:entry['logits']=open(log).read().splitlines()
        if trace:entry['trace_summary']=trace_summary(f'/tmp/trace-{index}')
        results.append(entry)
    return {'status':'resin_ab_sequence','gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'runs':results,
            'method':'fresh engine per run in one container; order as given; decode graphs only; prompts WildChat-derived artifact'}

@app.function(image=image,gpu='B200',cpu=4,memory=16384,timeout=600,retries=0,max_containers=1,scaledown_window=2)
def attn_study(contexts: list, layers: int=35, seed: int=7, qlen: int=1):
    """Paged q=1 attention: Triton split-KV kernel vs FlashInfer decode wrapper, L2-cold layer rotation."""
    import itertools,statistics,torch
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
    from resin_models_cuda.gemma4.edge_kernels import paged_decode_q1
    torch.manual_seed(seed)
    PAGE=64;HEADS=8;out=[]
    def timed(fn,calls):
        for _ in range(3):fn(0)
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(calls):fn(i)
        g.replay();torch.cuda.synchronize();v=[]
        for _ in range(9):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();v.append(a.elapsed_time(b)*1000/calls)
        return statistics.median(v)
    for D,window in ((256,511),(512,-1)):
        for L in contexts:
            pages=(L+PAGE-1)//PAGE;total=pages+3
            caches=[]
            for _ in range(layers):
                k=torch.randn(total,PAGE,1,D,dtype=torch.bfloat16,device='cuda');v=torch.randn_like(k)
                caches.append((k,v))
            perm=torch.randperm(total,dtype=torch.int32,device='cuda')[:pages].contiguous()
            indptr=torch.tensor([0,pages],dtype=torch.int32,device='cuda');last=torch.tensor([L-(pages-1)*PAGE],dtype=torch.int32,device='cuda')
            q=torch.randn(qlen,HEADS,D,dtype=torch.bfloat16,device='cuda')
            ws=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
            ip,ix,ll=indptr.clone(),torch.zeros(pages+8,dtype=torch.int32,device='cuda'),last.clone()
            ix[:pages]=perm
            if qlen==1:
                wrap=BatchDecodeWithPagedKVCacheWrapper(ws,'NHD',use_cuda_graph=True,paged_kv_indptr_buffer=ip,paged_kv_indices_buffer=ix,paged_kv_last_page_len_buffer=ll,backend='fa2',use_tensor_cores=True)
                wrap.plan(indptr,perm,last,HEADS,1,D,PAGE,window_left=window,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,o_data_type=torch.bfloat16,sm_scale=1.0,pos_encoding_mode='NONE')
            else:
                qo=torch.tensor([0,qlen],dtype=torch.int32,device='cuda');qob=qo.clone()
                wrap=BatchPrefillWithPagedKVCacheWrapper(ws,'NHD',use_cuda_graph=True,backend='auto',qo_indptr_buf=qob,paged_kv_indptr_buf=ip,paged_kv_indices_buf=ix,paged_kv_last_page_len_buf=ll)
                wrap.plan(qo,indptr,perm,last,HEADS,1,D,PAGE,window_left=window,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,o_data_type=torch.bfloat16,sm_scale=1.0,pos_encoding_mode='NONE',causal=True,disable_split_kv=True)
            fi=timed(lambda i:wrap.run(q,caches[i%layers]),layers)
            # fp64 oracle on layer 0
            def oracle(kv):
                k=kv[0][perm.long()].reshape(-1,D)[:L].double();v=kv[1][perm.long()].reshape(-1,D)[:L].double()
                outs=[]
                for j in range(qlen):
                    end=L-(qlen-1-j);lo=max(0,end-1-window) if window>=0 else 0
                    sc=q[j].double()@k[lo:end].T;outs.append(torch.softmax(sc,dim=-1)@v[lo:end])
                return torch.stack(outs)
            ref=oracle(caches[0]);fiout=wrap.run(q,caches[0]).double();fi_err=(fiout-ref).abs().max().item()
            best=None;tried=[]
            for splits,block,warps,stages in itertools.product((1,2,4,8,16),(16,32,64),(4,8),(2,3)):
                cfg=(splits,block,warps,stages)
                try:
                    f=lambda i,cfg=cfg:paged_decode_q1(q,caches[i%layers][0],caches[i%layers][1],indptr,perm,last,window_left=window,page_size=PAGE,query_len=qlen,config=cfg)
                    got=f(0).double();err=(got-ref).abs().max().item()
                    t=timed(f,layers)
                except Exception as e:
                    continue
                tried.append((t,err,cfg))
            tried.sort(key=lambda x:x[0])
            out.append({'D':D,'window_left':window,'L':L,'flashinfer_us':fi,'flashinfer_err':fi_err,
                        'best':{'us':tried[0][0],'err':tried[0][1],'cfg':tried[0][2]},'top5':[{'us':t,'err':e,'cfg':c} for t,e,c in tried[:5]],'speedup':fi/tried[0][0]})
            del caches
    return {'status':'paged_decode_study','gpu':torch.cuda.get_device_name(),'results':out,
            'method':'35 distinct synthetic KV layers rotated per graph; one request; fp64 oracle on layer 0; FlashInfer decode wrapper (use_tensor_cores, fa2, cuda-graph buffers) as Resin builds it'}

@app.local_entrypoint()
def main(approve_spend:bool=False,draft_tokens:int=3,unit_only:bool=False,trace:bool=False,gpu_tests:str='',diag:str='',real:bool=False,ab_modes:str='',ab_contexts:str='256,1024',ab_steps:int=48,ab_logits:bool=False,attn:str='',ab_draft:int=0,ab_trace:bool=False,ab_phases:bool=False,attn_qlen:int=1):
    if attn:
        print(json.dumps(attn_study.remote([int(x) for x in attn.split(',')],qlen=attn_qlen),indent=2));return
    if ab_modes:
        print(json.dumps(ab.remote(ab_modes.split(','),[int(x) for x in ab_contexts.split(',')],ab_steps,ab_logits,ab_draft,ab_trace,ab_phases),indent=2));return
    if gpu_tests:
        print(json.dumps(gpu_unit.remote(select=gpu_tests if gpu_tests!='all' else ''),indent=2));return
    if unit_only:
        print(json.dumps(unit.remote(),indent=2));return
    if not approve_spend:raise RuntimeError('Explicit spending approval required')
    print(json.dumps(qualify.remote(draft_tokens,trace,diag,real),indent=2))
