"""Bounded pinned HF whole-model reference, NOT a tuned-engine benchmark."""
import json
import modal
app=modal.App('resin-gemma-reference-profile')
MODEL='google/gemma-4-E2B-it'
REV='3e22461f65e89153144f8adb70e3b8c2cc9845a7'
image=(modal.Image.from_registry('nvidia/cuda:13.3.1-devel-ubuntu22.04',add_python='3.12')
       .pip_install('torch==2.13.0','transformers==5.14.1','safetensors','huggingface_hub')
       .run_commands('timeout 180 python -c "from huggingface_hub import snapshot_download; snapshot_download(\'google/gemma-4-E2B-it\',revision=\'3e22461f65e89153144f8adb70e3b8c2cc9845a7\',allow_patterns=[\'*.json\',\'*.safetensors\',\'tokenizer*\',\'*.jinja\'],local_dir=\'/opt/model\')"').pip_install('accelerate==1.12.0'))

@app.function(image=image,gpu='B200',memory=16384,timeout=180,retries=0,max_containers=1,scaledown_window=2)
def profile():
    import time, statistics
    import torch,transformers
    from transformers import Gemma4ForConditionalGeneration,AutoTokenizer
    torch.manual_seed(8821)
    start=time.monotonic()
    model=Gemma4ForConditionalGeneration.from_pretrained('/opt/model',dtype=torch.bfloat16,
                 device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained('/opt/model',local_files_only=True)
    torch.cuda.synchronize()
    loaded=time.monotonic()-start
    base=tokenizer('Explain the tradeoffs between latency, throughput and memory when serving language models. ',return_tensors='pt').input_ids.cuda()
    def inputs(n):return base.repeat(1,(n+base.shape[1]-1)//base.shape[1])[:,:n].contiguous()
    results=[]
    def timing(fn):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();wall=time.perf_counter()
        a.record();out=fn();b.record();b.synchronize()
        return out,{'cuda_stream_ms':a.elapsed_time(b),'synchronized_wall_ms':(time.perf_counter()-wall)*1000}
    with torch.inference_mode():
        for n in (256,1024):
            x=inputs(n)
            for _ in range(2): model(input_ids=x,use_cache=True,logits_to_keep=1)
            measures=[]
            for _ in range(5):
                out,t=timing(lambda:model(input_ids=x,use_cache=True,logits_to_keep=1));measures.append(t)
            results.append({'phase':'prefill','batch':1,'tokens':n,'measurements':measures,
                            'logits_finite':bool(torch.isfinite(out.logits).all().item())})
        out=model(input_ids=inputs(256),use_cache=True,logits_to_keep=1)
        cache=out.past_key_values;token=out.logits[:,-1].argmax(-1,keepdim=True)
        decode=[]
        for i in range(20):
            out,t=timing(lambda:model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1))
            cache=out.past_key_values;token=out.logits[:,-1].argmax(-1,keepdim=True)
            if i>=4:decode.append(t)
        results.append({'phase':'decode','batch':1,'initial_context':256,'warmup_steps':4,'measurements':decode,
                        'logits_finite':bool(torch.isfinite(out.logits).all().item())})
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(4):
                out=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=out.past_key_values;token=out.logits[:,-1].argmax(-1,keepdim=True)
            torch.cuda.synchronize()
        ops=[]
        for e in prof.key_averages():
            us=getattr(e,'self_device_time_total',0)
            if us>0 and e.device_type == torch.autograd.DeviceType.CPU:ops.append({'op':e.key,'self_device_us':us,'calls':e.count})
        ops.sort(key=lambda x:x['self_device_us'],reverse=True)
    graph_probe=graph_decode_probe(model,inputs)
    return {'graph_decode_probe':graph_probe,'status':'hf_reference_only','model':MODEL,'revision':REV,'gpu':torch.cuda.get_device_name(),
            'torch':str(torch.__version__),'transformers':transformers.__version__,'attention':'sdpa',
            'load_seconds':loaded,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'results':results,'decode_profile_top_ops':ops[:25],
            'profile_total_self_device_us':sum(x['self_device_us'] for x in ops),
            'limitations':'Synthetic repeated text, batch1 only; HF eager reference, not Resin/vLLM/SGLang. Stream events include launch gaps. No SLO/goodput or quantized measurements. Profile timings separate from unprofiled timings.'}

@app.local_entrypoint()
def main(approve_spend:bool=False):
    if not approve_spend:raise RuntimeError('Explicit spending approval required')
    print(json.dumps(profile.remote(),indent=2))


def graph_decode_probe(model,inputs):
    """Teacher-forced, independent caches. Compare each changing-input step."""
    import torch,time
    from transformers import StaticCache
    with torch.inference_mode():
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            a=StaticCache(config=model.config,max_cache_len=512)
            b=StaticCache(config=model.config,max_cache_len=512)
            token=inputs(1).clone()
            pos=torch.tensor([[256]],device='cuda')
            mask=torch.arange(512,device='cuda').view(1,1,1,-1)<=pos.view(1,1,1,1)
            masks={key:mask for key in ('sliding_attention','full_attention')}
            def step(cache):
                return model(input_ids=token,position_ids=pos,attention_mask=masks,
                             past_key_values=cache,use_cache=True,logits_to_keep=1).logits
            model(input_ids=inputs(256),past_key_values=a,use_cache=True,logits_to_keep=1)
            step(a);stream.synchronize()
            a.reset()
            model(input_ids=inputs(256),past_key_values=a,use_cache=True,logits_to_keep=1)
            model(input_ids=inputs(256),past_key_values=b,use_cache=True,logits_to_keep=1)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                captured=step(a)
            measurements=[];checks=[]
            for i in range(16):
                token.copy_(inputs(1)+i)
                pos.fill_(256+i);mask.copy_(torch.arange(512,device='cuda').view(1,1,1,-1)<=pos.view(1,1,1,1))
                pair={}
                order=('graph','eager') if i%2==0 else ('eager','graph')
                for label in order:
                    stream.synchronize();start=time.perf_counter()
                    x,y=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    x.record(stream)
                    if label=='graph':graph.replay()
                    else:expected=step(b)
                    y.record(stream);y.synchronize()
                    pair[label]={'cuda_stream_ms':x.elapsed_time(y),'synchronized_wall_ms':1000*(time.perf_counter()-start)}
                measurements.append({'order':list(order),**pair})
                checks.append({'step':i,'max_abs_logit_error':(captured.float()-expected.float()).abs().max().item(),
                               'argmax_equal':bool((captured.argmax(-1)==expected.argmax(-1)).all().item()),
                               'finite':bool(torch.isfinite(captured).all().item())})
            del graph
        torch.cuda.current_stream().wait_stream(stream)
    return {'status':'teacher_forced_static_cache_graph_probe','measurements':measurements,'checks':checks,
            'passed':all(x['finite'] and x['argmax_equal'] and x['max_abs_logit_error']==0 for x in checks),
            'limitations':'Alternating eager/graph measurement order, same static-cache path, independent static caches, explicit position/mask buffers, 16 changing teacher-forced token inputs, short context only. Not serving or dynamic-cache equivalence.'}
