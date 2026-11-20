"""vLLM / SGLang batch-1 decode baselines for pinned Gemma E2B on B200 (same real prompts)."""
import json
import modal
from pathlib import Path
HERE=Path(__file__).resolve().parent if modal.is_local() else Path('/opt')
app=modal.App('resin-engine-baselines')
ASSISTANT_DL=("timeout 240 python -c \"from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it-assistant',revision='2d874ef7d29f9a30599a1e4b3c1cbc9595f005df',allow_patterns=['config.json','model.safetensors'],local_dir='/opt/assistant')\"")
MODEL_DL=('timeout 300 python -c "from huggingface_hub import snapshot_download; '
          'snapshot_download(\'google/gemma-4-E2B-it\',revision=\'3e22461f65e89153144f8adb70e3b8c2cc9845a7\','
          'allow_patterns=[\'*.json\',\'*.safetensors\',\'tokenizer*\',\'*.jinja\'],local_dir=\'/opt/model\')"')
def base_image():
    return (modal.Image.from_registry('nvidia/cuda:13.3.1-devel-ubuntu22.04',add_python='3.12')
            .apt_install('build-essential','git','curl','ninja-build').pip_install('huggingface_hub')
            .env({'CUDA_HOME':'/usr/local/cuda','PATH':'/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin','TORCH_CUDA_ARCH_LIST':'10.0a','FLASHINFER_CUDA_ARCH_LIST':'10.0a'}))
import os
ONLY=os.environ.get('ENGINE_ONLY','')
if modal.is_local():
    vllm_image=(base_image().pip_install('vllm==0.34.0').run_commands(MODEL_DL).run_commands(ASSISTANT_DL)
                .add_local_file(HERE/'prompts_wildchat.json','/opt/prompts.json')) if ONLY in ('','vllm') else modal.Image.debian_slim()
    sglang_image=(base_image().pip_install('sglang==0.5.23').run_commands(MODEL_DL).run_commands(ASSISTANT_DL)
                  .add_local_file(HERE/'prompts_wildchat.json','/opt/prompts.json')) if ONLY in ('','sglang') else modal.Image.debian_slim()
else:
    vllm_image=sglang_image=modal.Image.debian_slim()

def _measure(generate, prompts, resin, label):
    import time,statistics
    out={'engine':label,'cases':[]}
    short,long=8,72
    for ctx,ids in prompts.items():
        for _ in range(2):generate(ids,short)
        times={short:[],long:[]}
        for rep in range(5):
            for n in (short,long) if rep%2==0 else (long,short):
                t=time.perf_counter();tokens=generate(ids,n);times[n].append(time.perf_counter()-t)
        tokens=generate(ids,24)
        exp=resin.get(ctx,[])
        k=min(len(tokens),len(exp));first=next((i for i in range(k) if tokens[i]!=exp[i]),None)
        per=(statistics.median(times[long])-statistics.median(times[short]))/(long-short)*1000
        out['cases'].append({'context':int(ctx),'decode_ms_per_token':per,'short_ms':1000*statistics.median(times[short]),
            'long_ms':1000*statistics.median(times[long]),'short_all_ms':[1000*x for x in times[short]],'long_all_ms':[1000*x for x in times[long]],
            'tokens24':tokens,'compared_with_resin':k,'first_diff_vs_resin':first})
    return out

@app.function(image=vllm_image,gpu='B200',cpu=8,memory=32768,timeout=900,retries=0,max_containers=1,scaledown_window=2)
def vllm_bench():
    import json,vllm
    data=json.load(open('/opt/prompts.json'))
    from vllm import LLM,SamplingParams
    llm=LLM(model='/opt/model',dtype='bfloat16',max_model_len=2048,gpu_memory_utilization=0.35,
            enable_prefix_caching=False,seed=0,limit_mm_per_prompt={'image':0,'audio':0,'video':0})
    def generate(ids,n):
        r=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=n,ignore_eos=True),use_tqdm=False)
        return list(r[0].outputs[0].token_ids)
    res=_measure(generate,data['prompts'],data['resin_greedy'],'vllm')
    res['version']=vllm.__version__;return res

@app.function(image=sglang_image,gpu='B200',cpu=8,memory=32768,timeout=900,retries=0,max_containers=1,scaledown_window=2)
def sglang_bench():
    import json,sglang
    data=json.load(open('/opt/prompts.json'))
    engine=sglang.Engine(model_path='/opt/model',dtype='bfloat16',context_length=2048,mem_fraction_static=0.35,
                         disable_radix_cache=True,random_seed=0,log_level='error')
    def generate(ids,n):
        r=engine.generate(input_ids=ids,sampling_params={'temperature':0,'max_new_tokens':n,'ignore_eos':True})
        return list(r['output_ids'])
    res=_measure(generate,data['prompts'],data['resin_greedy'],'sglang')
    res['version']=sglang.__version__;engine.shutdown();return res

@app.function(image=vllm_image,gpu='B200',cpu=8,memory=32768,timeout=900,retries=0,max_containers=1,scaledown_window=2)
def vllm_spec(speculative_tokens: int=3):
    import json,vllm
    data=json.load(open('/opt/prompts.json'))
    from vllm import LLM,SamplingParams
    llm=LLM(model='/opt/model',dtype='bfloat16',max_model_len=2048,gpu_memory_utilization=0.35,
            enable_prefix_caching=False,seed=0,limit_mm_per_prompt={'image':0,'audio':0,'video':0},
            speculative_config={'model':'/opt/assistant','num_speculative_tokens':speculative_tokens})
    def generate(ids,n):
        r=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=n,ignore_eos=True),use_tqdm=False)
        return list(r[0].outputs[0].token_ids)
    res=_measure(generate,data['prompts'],data['resin_greedy'],f'vllm-mtp{speculative_tokens}')
    res['version']=vllm.__version__;res['speculative_tokens']=speculative_tokens;return res

@app.function(image=sglang_image,gpu='B200',cpu=8,memory=32768,timeout=900,retries=0,max_containers=1,scaledown_window=2)
def sglang_spec():
    import json,sglang
    data=json.load(open('/opt/prompts.json'))
    engine=sglang.Engine(model_path='/opt/model',dtype='bfloat16',context_length=2048,mem_fraction_static=0.6,
                         disable_radix_cache=True,random_seed=0,log_level='error',
                         speculative_algorithm='NEXTN',speculative_draft_model_path='/opt/assistant',
                         speculative_num_steps=5,speculative_num_draft_tokens=6,speculative_eagle_topk=1)
    def generate(ids,n):
        r=engine.generate(input_ids=ids,sampling_params={'temperature':0,'max_new_tokens':n,'ignore_eos':True})
        return list(r['output_ids'])
    res=_measure(generate,data['prompts'],data['resin_greedy'],'sglang-nextn-5-6')
    res['version']=sglang.__version__;engine.shutdown();return res

@app.local_entrypoint()
def main(approve_spend:bool=False,engine:str='vllm'):
    if not approve_spend:raise RuntimeError('Explicit spending approval required')
    fn={'vllm':vllm_bench,'sglang':sglang_bench,'vllm_spec':vllm_spec,'sglang_spec':sglang_spec}[engine]
    print(json.dumps(fn.remote(),indent=2))
