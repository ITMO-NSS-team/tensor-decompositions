"""H10 true top-two traces and modeled expert placement; no physical communication."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
import torch
from torch import nn
from torch.nn import functional as F
import tensorly as tl

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git,guard,sha,sync,write_json
from run_h02_synthetic import tensor_hash
from tdecomp.tensor.tucker import HOOIDecomposition


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__();self.router=nn.Linear(32,4,bias=False)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32)) for _ in range(4)])
        self.head=nn.Linear(32,4)
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.weight[0,:2]=3;self.router.weight[1,:2]=2.9
            self.router.weight[2,2:4]=3;self.router.weight[3,2:4]=2.9
        self.router.weight.requires_grad_(False)

    def routes(self,x):
        logits=self.router(x)
        chosen=torch.argsort(logits,dim=-1,descending=True,stable=True)[:,:2]
        weights=logits.gather(1,chosen).softmax(-1)
        return chosen,weights

    def forward(self,x):
        chosen,weights=self.routes(x);result=torch.zeros_like(x)
        # Slot order is preserved across placement: no floating-point reorder.
        for slot in range(2):
            contribution=torch.zeros_like(x)
            for expert in range(4):
                mask=chosen[:,slot]==expert
                contribution[mask]=self.experts[expert](x[mask])
            result=result+contribution*weights[:,slot,None]
        return self.head(result)


def data(seed,device):
    output={}
    for index,(name,n) in enumerate([('train',2048),('calibration',512),('tuning',512),('replay',512),('test',512)]):
        gen=torch.Generator(device=device).manual_seed(seed+(index+1)*1000)
        labels=torch.randint(4,(n,),generator=gen,device=device)
        x=.2*torch.randn(n,32,generator=gen,device=device);x[torch.arange(n,device=device),labels]+=3
        output[name]=(x,labels,labels//2)
    return output


@torch.no_grad()
def trace(model,pair):
    x,_,origins=pair;chosen,_=model.routes(x)
    counts=torch.zeros(len(x)//32,2,4,1,device=x.device,dtype=torch.float32)
    windows=torch.arange(len(x),device=x.device)//32
    for slot in range(2):counts.index_put_((windows,origins,chosen[:,slot],torch.zeros_like(origins)),
                                          torch.ones_like(origins,dtype=torch.float32),accumulate=True)
    assert int(counts.sum())==2*len(x)
    return counts


def place(counts,mode):
    """Descending expert load, capacity two, cost or load-aware deterministic greedy."""
    rates=counts.sum((0,3));loads=rates.sum(0);device_load=[0.,0.];slots=[0,0];placement=[-1]*4
    order=sorted(range(4),key=lambda e:(-float(loads[e]),e))
    maximum=1.25*float(loads.sum())/2
    for expert in order:
        available=[d for d in range(2) if slots[d]<2 and device_load[d]+float(loads[expert])<=maximum+1e-6]
        if not available:return placement,False,device_load
        if mode=='balanced':
            destination=min(available,key=lambda d:(device_load[d],d))
        else:
            destination=min(available,key=lambda d:(float(rates[1-d,expert]),d))
        placement[expert]=destination;slots[destination]+=1;device_load[destination]+=float(loads[expert])
    feasible=max(device_load)<=1.25*sum(device_load)/2
    return placement,feasible,device_load


def price(counts,placement,initial):
    remote=counts.new_zeros(counts.shape[0])
    for e,d in enumerate(placement):remote+=counts[:,1-d,e,0]
    bytes_per_dispatch=2*32*4
    movement=sum(d!=initial[e] for e,d in enumerate(placement))*4192*4
    exchange=float(remote.sum())*bytes_per_dispatch
    # One request/return per remote dispatch; full computation/shared overhead absent.
    modeled=(remote*bytes_per_dispatch/(16*1024**3)+remote*2*10e-6)
    return {'remote_dispatches':int(remote.sum()),'exchange_bytes':exchange,'migration_bytes':movement,
            'total_exchange_and_migration_bytes':exchange+movement,
            'modeled_window_p95_seconds':float(torch.quantile(modeled,.95)),
            'latency_scope':'conditional communication only; GPU/compute/shared-layer latency absent'}


@torch.no_grad()
def replay(model,pair,placement):
    # Physical expert storage permutation; logical routes mapped to storage without reranking ties.
    physical_order=sorted(range(4),key=lambda e:(placement[e],e))
    physical=copy.deepcopy(model)
    physical.experts=nn.ModuleList([copy.deepcopy(model.experts[e]) for e in physical_order])
    logical_routes,weights=model.routes(pair[0]);inverse=torch.empty(4,dtype=torch.long,device=pair[0].device)
    for physical_index,logical in enumerate(physical_order):inverse[logical]=physical_index
    result=torch.zeros_like(pair[0])
    for slot in range(2):
        contribution=torch.zeros_like(result)
        routes=inverse[logical_routes[:,slot]]
        for e in range(4):
            mask=routes==e;contribution[mask]=physical.experts[e](pair[0][mask])
        result=result+contribution*weights[:,slot,None]
    output=physical.head(result);reference=model(pair[0])
    relative=float(torch.linalg.vector_norm(output-reference)/torch.linalg.vector_norm(reference))
    if relative>1e-6:raise ArithmeticError('placement changed model output')
    return relative


def admission():
    torch.manual_seed(123);model=ToyMoE().eval();pairs=data(11,'cpu')
    assert sum(p.numel() for p in model.parameters())==17028
    counts=trace(model,pairs['calibration']);assert counts.shape==(16,2,4,1)
    with torch.no_grad():assert model.routes(torch.zeros(1,32))[0].tolist()==[[0,1]]
    errors=[replay(model,pairs['replay'],placement) for placement in ([0,0,1,1],[1,0,1,0])]
    assert price(counts,[0,0,1,1],[0,0,1,1])['migration_bytes']==0
    return {'parameter_count':17028,'accepted_sends':int(counts.sum()),'replay_errors':errors}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--admission-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=['run_h10_synthetic.py','run_local.py','run_h02_synthetic.py'];source_hashes={}
    for name in sources:
        p=Path(__file__).with_name(name);source_hashes[name]=sha(p);(args.output/(name+'.source')).write_bytes(p.read_bytes())
    manifest={'state':'admission','hypothesis':'H10','seeds':[11,22,33,44,55],'setting':'synthetic',
              'source_hashes':source_hashes,'git_sha':git('rev-parse','HEAD'),'device':args.device,'torch':torch.__version__,
              'protocol_sha256':sha(next(Path(__file__).parent.glob('H10_*.md'))),
              'limits':{'gpu_bytes':12*1024**3,'rss_bytes':24*1024**3,'seconds_per_seed':900},
              'limitations':['two virtual devices only; no measured interdevice communication',
                             'physical stage C blocked by one GPU; whole H10 indeterminate',
                             'stable main regime only; independent-origin/future-drift controls pending',
                             'loads enforced on calibration, future violation reported separately',
                             'migration from same round-robin initial placement counted once',
                             'single rank fixed by protocol default; no test-driven tuning'],
              'command':sys.argv}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',admission())
        if args.admission_only:manifest['state']='admission_passed';return
        rows=[];manifest['state']='running'
        for seed in manifest['seeds']:
            start=time.perf_counter();torch.manual_seed(seed);model=ToyMoE().to(args.device)
            splits=data(seed,args.device);sd=args.output/f'seed-{seed}';sd.mkdir()
            write_json(sd/'inputs.json',{k:{'x':tensor_hash(x),'labels':tensor_hash(y),'origin':tensor_hash(o)} for k,(x,y,o) in splits.items()})
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
            gen=torch.Generator(device=args.device).manual_seed(seed+50000);x,y,_=splits['train']
            for step in range(500):
                ix=torch.randint(len(x),(32,),generator=gen,device=args.device);opt.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x[ix]),y[ix]);loss.backward()
                if not torch.isfinite(loss):raise ArithmeticError('nonfinite MoE loss')
                opt.step();guard(start,900)
            model.eval();torch.save(model.state_dict(),sd/'teacher.pt')
            counts=trace(model,splits['calibration']);sync();factor_start=time.perf_counter()
            with tl.backend_context('pytorch'):
                core,factors=HOOIDecomposition(rank=(4,2,4,1),random_state=seed).decompose(counts,n_iter_max=20,tol=1e-6)
                raw=tl.tucker_to_tensor((core,factors))
            corrected=raw.clamp_min(0)
            corrected=corrected*(counts.sum()/corrected.sum())
            sync();factor_seconds=time.perf_counter()-factor_start
            placements={}
            for method in ('balanced','full_statistics','tucker_statistics'):
                c=corrected if method=='tucker_statistics' else counts
                placements[method]=place(c,'balanced' if method=='balanced' else 'cost')
            write_json(sd/'fixed-placements.json',placements)
            future=trace(model,splits['test']);initial=[0,1,0,1]
            for method in ('balanced','full_statistics','tucker_statistics'):
                c=corrected if method=='tucker_statistics' else counts
                placement,valid,load=placements[method]
                if not valid:
                    rows.append({'seed':seed,'method':method,'placement':placement,'status':'inadmissible',
                                 'reason':'no greedy destination satisfying capacity and load bound; no rerouting',
                                 'calibration_device_load':load})
                    write_json(args.output/'runs.json',rows);continue
                error=replay(model,splits['replay'],placement)
                true_load=[sum(float(future[:,:,e,:].sum()) for e,d in enumerate(placement) if d==device) for device in range(2)]
                forecast=c.sum(0)/c.sum();actual=future.sum(0)/future.sum()
                row={'seed':seed,'method':method,'placement':placement,'calibration_feasible':valid,
                     'future_feasible':max(true_load)<=1.25*sum(true_load)/2,'replay_output_error':error,
                     'forecast_l1_error':float((forecast-actual).abs().sum()),
                     'factor_seconds':factor_seconds if method=='tucker_statistics' else 0,
                     'correction_relative_norm':float(torch.linalg.vector_norm(corrected-raw)/torch.linalg.vector_norm(raw)),
                     'checkpoint_sha256':sha(sd/'teacher.pt'),**price(future,placement,initial),
                     'status':'completed' if valid else 'inadmissible'}
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
            torch.save({'true':counts,'raw':raw,'corrected':corrected},sd/'statistics.pt')
            del model;guard(start,900)
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',error=str(error),traceback=traceback.format_exc());raise
    finally:write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
