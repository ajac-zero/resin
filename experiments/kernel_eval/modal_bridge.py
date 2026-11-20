"""Same-process, shared-buffer/stream experiment, not a hostile-code sandbox."""
import json
import modal
if modal.is_local():
    from modal_rust import image
else:
    image=modal.Image.debian_slim()
app=modal.App('resin-shared-buffer-study')

@app.function(image=image,gpu='B200',timeout=150,retries=0,max_containers=1,scaledown_window=2)
def study(seed: int=817, heldout: bool=False):
    import ctypes, importlib.util, random, statistics
    import torch
    import torch.nn.functional as F
    spec=importlib.util.spec_from_file_location('comparator','/opt/comparator.py')
    comparator=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparator)
    lib=ctypes.CDLL('/opt/rust/target/release/libresin_gelu_probe.so')
    lib.resin_create.argtypes=[ctypes.c_uint64]*3+[ctypes.c_int32,ctypes.c_void_p]
    lib.resin_create.restype=ctypes.c_void_p
    lib.resin_launch.argtypes=[ctypes.c_void_p,ctypes.c_int32];lib.resin_launch.restype=ctypes.c_int
    lib.resin_destroy.argtypes=[ctypes.c_void_p];lib.resin_destroy.restype=ctypes.c_int
    rng=random.Random(seed);torch.manual_seed(seed)
    stream=torch.cuda.Stream()
    results=[]
    def exact(a,b):
        return int((~((a.view(torch.int16)==b.view(torch.int16))|(torch.isnan(a)&torch.isnan(b)))).sum().item())
    with torch.cuda.stream(stream):
        for n in ((24576,86016,98304,190464,380928,393216,780288) if heldout else (257,6144,12288,43008,202752,405504,792576)):
            g=torch.randn(n,dtype=torch.bfloat16,device='cuda')
            u=torch.randn_like(g)
            original_g=g.clone();original_u=u.clone()
            storage=torch.full((n+16,),13,dtype=g.dtype,device=g.device)
            out=storage[8:-8] # 16-byte aligned, guard regions on both sides
            stream.synchronize()
            h=lib.resin_create(g.data_ptr(),u.data_ptr(),out.data_ptr(),n,stream.cuda_stream)
            if not h: raise RuntimeError('bridge create failed')
            graphs={};checks={}
            try:
                def rust(block):
                    rc=lib.resin_launch(h,block)
                    if rc: raise RuntimeError(f'bridge launch {rc}')
                def pt():
                    out.copy_(F.gelu(g,approximate='tanh')*u)
                funcs={'pytorch':pt,'triton':lambda:comparator.launch_gelu_tanh_mul(g,u,out),
                       **{f'rust_{b}':lambda b=b:rust(b) for b in (256,512,1024)}}
                for label,fn in funcs.items():
                    for _ in range(3):fn()
                    stream.synchronize()
                    checks[label]=exact(out,F.gelu(g,approximate='tanh')*u)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(100):fn()
                    graphs[label]=graph
                measured={k:[] for k in graphs};orders=[]
                for repeat in range(15):
                    order=list(graphs);rng.shuffle(order);orders.append(order)
                    for label in order:
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record(stream);graphs[label].replay();b.record(stream);b.synchronize()
                        measured[label].append(a.elapsed_time(b)*1000/100)
                inputs_unchanged=exact(g,original_g)==0 and exact(u,original_u)==0
                g.add_(0.5);u.mul_(-0.75)
                expected=F.gelu(g,approximate='tanh')*u
                replays={}
                for label,graph in graphs.items():
                    out.fill_(float('nan'));graph.replay();stream.synchronize()
                    replays[label]=exact(out,expected)
                guards=bool((storage[:8]==13).all().item() and (storage[-8:]==13).all().item())
                results.append({'elements':n,'initial_mismatches':checks,'changed_replay_mismatches':replays,
                                'inputs_unchanged':inputs_unchanged,'guards_unchanged':guards,
                                'us_per_call':measured,'orders':orders,
                                'medians':{k:statistics.median(v) for k,v in measured.items()}})
            finally:
                stream.synchronize()
                graphs.clear()
                if lib.resin_destroy(h):raise RuntimeError('bridge destroy failed')
    return {'status':'same_process_shared_buffer_study','gpu_qualified':False,'seed':seed,
            'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
            'notes':'Same process/context/stream/buffers, 16-byte-aligned output with guards, randomized 15-round orders; PyTorch includes explicit copy into shared output. Graph hot-cache kernel timing, not serving latency. All configurations prewarmed; no host timing in GPU measurements.',
            'heldout':heldout,
            'frozen_dispatch':'rust_256 n<=12288; rust_512 n<=405504; otherwise triton',
            'passed':all(all(v==0 for v in r['initial_mismatches'].values()) and all(v==0 for v in r['changed_replay_mismatches'].values()) and r['guards_unchanged'] and r['inputs_unchanged'] for r in results),
            'results':results}

@app.local_entrypoint()
def main(approve_spend:bool=False,seed:int=817,heldout:bool=False):
    if not approve_spend:raise RuntimeError('Explicit spending approval required')
    print(json.dumps(study.remote(seed,heldout),indent=2))
