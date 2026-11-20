"""Sweep repo M=1 Triton projection/MLP kernels against the cuBLAS path on B200."""
import json
import modal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt')
app=modal.App('resin-m1-kernel-sweep')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.15.0','numpy')
       .add_local_file(ROOT/'models/cuda/resin_models_cuda/gemma4/edge_kernels_triton.py','/opt/edge_kernels_triton.py') if modal.is_local()
       else modal.Image.debian_slim())

@app.function(image=image,gpu='B200',timeout=900,retries=0,max_containers=1,scaledown_window=2)
def sweep(seed:int=57, only: str='', rows: int=1):
    import sys,itertools,statistics,importlib.util,triton,torch
    import torch.nn.functional as F
    spec=importlib.util.spec_from_file_location('ek','/opt/edge_kernels_triton.py');ek=importlib.util.module_from_spec(spec);spec.loader.exec_module(ek)
    torch.manual_seed(seed)
    L2=torch.cuda.get_device_properties(0).L2_cache_size
    def timed(fn,calls=64):
        for _ in range(3):fn(0)
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(calls):fn(i)
        g.replay();torch.cuda.synchronize();v=[]
        for _ in range(9):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();v.append(a.elapsed_time(b)*1000/calls)
        return statistics.median(v)
    configs=[(bn,bk,nw) for bn,bk,nw in itertools.product((1,2,4,8),(64,128,256,512,1024),(2,4,8)) if bn*bk*rows<=16384]
    linear_shapes={'o_local':(1536,2048),'o_global':(1536,4096),'ple_gate':(256,1536),'ple_proj':(1536,256),
                   'qkv_local':(2560,1536),'qkv_global':(5120,1536),'q_local':(2048,1536),'q_global':(4096,1536),
                   'down_6144':(1536,6144),'down_12288':(1536,12288),
                   'asst_pre_proj':(256,3072),'asst_post_proj':(1536,256),'asst_q_local':(1024,256),'asst_q_global':(2048,256),
                   'asst_o_local':(256,1024),'asst_o_global':(256,2048),'asst_down':(256,2048)}
    out=[]
    def make(N,K,scale=1.0):
        c=max(8,int(4*L2//(N*K*2))+1);return c
    for name,(N,K) in linear_shapes.items():
        if only and not name.startswith(only):continue
        c=make(N,K);W=[torch.randn(N,K,dtype=torch.bfloat16,device='cuda')*0.05 for _ in range(c)]
        x=torch.randn(rows,K,dtype=torch.bfloat16,device='cuda');y=torch.empty(rows,N,dtype=torch.bfloat16,device='cuda')
        cub=timed(lambda i:F.linear(x,W[i%c]))
        best=None
        for bn,bk,nw in configs:
            f=(lambda i,bn=bn,bk=bk,nw=nw:ek.launch_linear_m1(x,W[i%c],y,block_n=bn,block_k=bk,num_warps=nw)) if rows==1 else (lambda i,bn=bn,bk=bk,nw=nw:ek.launch_linear_rows(x,W[i%c],y,rows=rows,block_n=bn,block_k=bk,num_warps=nw))
            t=timed(f)
            if best is None or t<best[0]:best=(t,bn,bk,nw)
        errs=[]
        for j in range(0,c,max(1,c//6)):
            (ek.launch_linear_m1(x,W[j],y,block_n=best[1],block_k=best[2],num_warps=best[3]) if rows==1 else ek.launch_linear_rows(x,W[j],y,rows=rows,block_n=best[1],block_k=best[2],num_warps=best[3]))
            ref=(x.double()@W[j].double().T);cp=F.linear(x,W[j])
            errs.append(((y.double()-ref).abs().max().item(),(cp.double()-ref).abs().max().item(),(y.double()-ref).abs().mean().item(),(cp.double()-ref).abs().mean().item()))
        out.append({'name':name,'N':N,'K':K,'cublas_us':cub,'triton_us':best[0],'best':best[1:],'speedup':cub/best[0],
                    'max_err_ratio':max(a/max(b,1e-12) for a,b,_,_ in errs),'mean_err_ratio':max(a/max(b,1e-12) for _,_,a,b in errs)})
        del W
    for name,(N,K) in {'mlp_6144':(6144,1536),'mlp_12288':(12288,1536),'asst_mlp':(2048,256)}.items():
        if only and not name.startswith(only):continue
        c=make(2*N,K);G=[torch.randn(N,K,dtype=torch.bfloat16,device='cuda')*0.05 for _ in range(c)];U=[torch.randn(N,K,dtype=torch.bfloat16,device='cuda')*0.05 for _ in range(c)]
        x=torch.randn(rows,K,dtype=torch.bfloat16,device='cuda');y=torch.empty(rows,N,dtype=torch.bfloat16,device='cuda')
        def cur(i):
            g=F.linear(x,G[i%c]);u=F.linear(x,U[i%c]);return F.gelu(g,approximate='tanh')*u
        # the existing fused activation kernel is the fair current path
        import torch as T
        def current(i):
            g=F.linear(x,G[i%c]);u=F.linear(x,U[i%c]);ek.launch_gelu_tanh_mul(g,u,y)
        cub=timed(current);best=None
        for bn,bk,nw in configs:
            f=(lambda i,bn=bn,bk=bk,nw=nw:ek.launch_gate_up_gelu_m1(x,G[i%c],U[i%c],y,block_n=bn,block_k=bk,num_warps=nw)) if rows==1 else (lambda i,bn=bn,bk=bk,nw=nw:ek.launch_gate_up_gelu_rows(x,G[i%c],U[i%c],y,rows=rows,block_n=bn,block_k=bk,num_warps=nw))
            t=timed(f)
            if best is None or t<best[0]:best=(t,bn,bk,nw)
        errs=[]
        for j in range(0,c,max(1,c//6)):
            (ek.launch_gate_up_gelu_m1(x,G[j],U[j],y,block_n=best[1],block_k=best[2],num_warps=best[3]) if rows==1 else ek.launch_gate_up_gelu_rows(x,G[j],U[j],y,rows=rows,block_n=best[1],block_k=best[2],num_warps=best[3]));mine=y.clone()
            g64=(x.double()@G[j].double().T);u64=(x.double()@U[j].double().T)
            ref=0.5*g64*(1+torch.tanh(0.7978845608028654*(g64+0.044715*g64**3)))*u64
            ek.launch_gelu_tanh_mul(F.linear(x,G[j]),F.linear(x,U[j]),y);cp=y.clone()
            errs.append(((mine.double()-ref).abs().max().item(),(cp.double()-ref).abs().max().item(),(mine.double()-ref).abs().mean().item(),(cp.double()-ref).abs().mean().item()))
        out.append({'name':name,'N':N,'K':K,'current_path_us':cub,'fused_us':best[0],'best':best[1:],'speedup':cub/best[0],
                    'max_err_ratio':max(a/max(b,1e-12) for a,b,_,_ in errs),'mean_err_ratio':max(a/max(b,1e-12) for _,_,a,b in errs)})
        del G,U
    return {'status':'m1_kernel_sweep_only','gpu':torch.cuda.get_device_name(),'triton':triton.__version__,'results':out,
            'method':'L2-cold rotating weights, graph of 64 calls, configs swept per fixed shape, fp64 oracle ratios (candidate/baseline) over sampled weight copies, synthetic weights/inputs, one seed'}

@app.local_entrypoint()
def main(approve_spend:bool=False,only:str='',rows:int=1):
    if not approve_spend:raise RuntimeError('Explicit spending approval required')
    print(json.dumps(sweep.remote(only=only,rows=rows),indent=2))
