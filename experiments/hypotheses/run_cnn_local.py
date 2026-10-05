"""H04 synthetic neural comparison and H06 residual-grid pilot."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import tensorly as tl
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import factorize, git, gpu_snapshot, guard, sha, sync, tensor_hash, write_json
from tdecomp.tensor.tucker import HOOIDecomposition, RSTHOSVDDecomposition


def synthetic(seed,device):
    gen=torch.Generator(device=device).manual_seed(seed)
    factors=[torch.linalg.qr(torch.randn(n,r,generator=gen,device=device)).Q
             for n,r in zip((16,8,3,3),(4,3,2,2))]
    core=torch.randn(4,3,2,2,generator=gen,device=device)
    signal=tl.tenalg.multi_mode_dot(core,factors)
    signal=signal/torch.linalg.vector_norm(signal)
    noise=torch.randn(signal.shape,generator=gen,device=device)
    weight=signal+.05*noise/torch.linalg.vector_norm(noise)
    teacher=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),
                          nn.Conv2d(8,16,3,padding=1),nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(16,4)).to(device)
    with torch.no_grad():
        for layer in (teacher[0],teacher[6]):
            fan_in=layer.weight[0].numel()
            layer.weight.copy_(torch.randn(layer.weight.shape,generator=gen,device=device)/fan_in**.5)
            layer.bias.zero_()
        teacher[2].weight.copy_(weight);teacher[2].bias.zero_()
    u=torch.linalg.qr(torch.randn(256,8,generator=gen,device=device)).Q
    scales=torch.tensor([3,2,1,.7,.5,.3,.2,.1],device=device)
    splits={}
    for k,(name,n) in enumerate([('recovery',4096),('calibration',512),('tuning',512),('test',1024)],1):
        rng=torch.Generator(device=device).manual_seed(seed+1000*k)
        x=((torch.randn(n,8,generator=rng,device=device)*scales)@u.T
           +.05*torch.randn(n,256,generator=rng,device=device)).reshape(n,1,16,16)
        with torch.no_grad():y=teacher(x)
        splits[name]=(x,y)
    return teacher.eval(),splits


class TuckerConv(nn.Module):
    def __init__(self,core,u,v,bias,direct):
        super().__init__()
        self.core=nn.Parameter(core.clone());self.u=nn.Parameter(u.clone());self.v=nn.Parameter(v.clone())
        self.bias=nn.Parameter(bias.clone());self.direct=direct

    def forward(self,x):
        if self.direct:
            return F.conv2d(F.conv2d(F.conv2d(x,self.v.T[:,:,None,None]),self.core,padding=1),
                            self.u[:,:,None,None],self.bias)
        weight=torch.einsum('oa,abhw,ib->oihw',self.u,self.core,self.v)
        return F.conv2d(x,weight,self.bias,padding=1)


class MatrixConv(nn.Module):
    def __init__(self,a,b,bias):
        super().__init__();self.a=nn.Parameter(a[:,:,None,None].clone())
        self.b=nn.Parameter(b.reshape(2,8,3,3).clone());self.bias=nn.Parameter(bias.clone())

    def forward(self,x):return F.conv2d(F.conv2d(x,self.b,padding=1),self.a,self.bias)


@torch.no_grad()
def mse(model,pair):
    x,y=pair;total=0.;energy=0.;count=0
    for start in range(0,len(x),128):
        target=y[start:start+128];prediction=model(x[start:start+128])
        total+=float((prediction-target).square().sum());energy+=float(target.square().sum());count+=target.numel()
    return {'mse':total/count,'normalized_mse':total/energy}


@torch.no_grad()
def latency(model,x):
    import numpy as np
    for _ in range(30):model(x)
    sync();values=[]
    for _ in range(200):
        sync();start=time.perf_counter();model(x);sync();values.append((time.perf_counter()-start)*1000)
    return {'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),
            'warmups':30,'measured':200,'batch':len(x)}


def rank_grid(weight,seed):
    rows=[]
    for ranks in itertools.product((2,4,8),(2,3,6),(1,2),(1,2)):
        start=time.perf_counter()
        dec=RSTHOSVDDecomposition(rank=ranks,oversampling=8,power_iteration=1,random_state=seed)
        core,factors=dec.decompose(weight)
        reconstructed=tl.tenalg.multi_mode_dot(core,factors)
        direct=float(torch.linalg.vector_norm(weight-reconstructed)/torch.linalg.vector_norm(weight))
        residual_energy=float(weight.square().sum()-core.square().sum())
        direct_energy=float((weight-reconstructed).square().sum())
        parameter_count=core.numel()+sum(f.numel() for f in factors)
        rows.append({'ranks':ranks,'relative_residual':direct,'parameters':parameter_count,
                     'admissible':direct<=.10,'energy_residual':residual_energy,
                     'direct_residual_energy':direct_energy,'seconds':time.perf_counter()-start})
        assert abs(residual_energy-direct_energy)<1e-5
    candidates=[r for r in rows if r['admissible']]
    return {'scope':'H06 grid residual pilot, not full adaptive allocation neural experiment',
            'seed':seed,'rows':rows,'minimal_storage_admissible':min(candidates,key=lambda x:x['parameters']) if candidates else None}


def run(out,seeds,device):
    rows=[]
    for seed in seeds:
        teacher,splits=synthetic(seed,device)
        sd=out/f'seed-{seed}';sd.mkdir();torch.save(teacher.state_dict(),sd/'teacher.pt')
        write_json(sd/'inputs.json',{name:{'x':tensor_hash(x),'y':tensor_hash(y),'count':len(x)} for name,(x,y) in splits.items()})
        w=teacher[2].weight.detach()
        sync();start=time.perf_counter()
        dec=HOOIDecomposition(rank=(4,3,3,3),random_state=seed)
        core,factors=dec.decompose(w,n_iter_max=20,tol=1e-6)
        spatial=tl.tenalg.multi_mode_dot(core,[factors[2],factors[3]],modes=[2,3])
        u,v=factors[:2]
        sync();tucker_setup=time.perf_counter()-start
        write_json(sd/'H06-rank-grid.json',rank_grid(w,seed))
        base_timing=latency(teacher,splits['tuning'][0][:64])
        for method in ('dense_reconstructed_tucker','direct_tucker','matrix_svd'):
            variant_start=time.perf_counter()
            student=copy.deepcopy(teacher)
            if method=='matrix_svd':
                sync();setup_start=time.perf_counter();a,b=factorize(w.reshape(16,72),2,'svd');sync()
                setup_seconds=time.perf_counter()-setup_start
                student[2]=MatrixConv(a,b,teacher[2].bias)
            else:
                student[2]=TuckerConv(spatial,u,v,teacher[2].bias,method=='direct_tucker')
                setup_seconds=tucker_setup
            for parameter in student.parameters():parameter.requires_grad_(False)
            for parameter in student[2].parameters():parameter.requires_grad_(True)
            tuning_before=mse(student,splits['tuning'])
            opt=torch.optim.AdamW(student[2].parameters(),lr=.001,weight_decay=0)
            sync();recovery_start=time.perf_counter();history=[]
            for phase,steps,offset in [('calibration',128,50000),('recovery',512,60000)]:
                x,y=splits[phase];gen=torch.Generator(device=device).manual_seed(seed+offset)
                for step in range(steps):
                    ix=torch.randint(len(x),(32,),generator=gen,device=device)
                    opt.zero_grad(set_to_none=True);loss=F.mse_loss(student(x[ix]),y[ix])
                    if not torch.isfinite(loss):raise ArithmeticError('nonfinite CNN loss')
                    loss.backward();opt.step()
                    if step%64==0:
                        guard(variant_start);history.append({'phase':phase,'step':step,'loss':float(loss.detach())})
            sync();recovery_seconds=time.perf_counter()-recovery_start
            student.eval();result=mse(student,splits['test']);timing=latency(student,splits['tuning'][0][:64])
            torch.save(student.state_dict(),sd/f'{method}.pt');write_json(sd/f'{method}-history.json',history)
            row={'hypothesis_id':'H04','setting':'synthetic','seed':seed,'method':method,
                 'rank_tuple':[4,3,3,3] if method!='matrix_svd' else None,'matrix_rank':2 if method=='matrix_svd' else None,
                 'compressed_weights':196 if method!='matrix_svd' else 176,
                 'trainable_parameters':sum(p.numel() for p in student.parameters() if p.requires_grad),
                 'tuning_before':tuning_before,'final_test':result,'latency':timing,'dense_original_latency':base_timing,
                 'setup_seconds':setup_seconds,'recovery_seconds':recovery_seconds,
                 'total_seconds':time.perf_counter()-variant_start+setup_seconds,
                 'checkpoint_sha256':sha(sd/f'{method}.pt'),'status':'completed'}
            rows.append(row);write_json(out/'runs.json',rows);print(json.dumps(row),flush=True)
    write_json(out/'summary.json',{'n_neural_variants':len(rows),'seeds':seeds,
               'scope':'first H04 synthetic series; H06 residual pilot; no real-data conclusions'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55]);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tl.set_backend('pytorch');torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest={'run_id':args.output.name,'state':'running','git_sha':git('rev-parse','HEAD'),
              'git_status':git('status','--porcelain'),'runner_sha256':sha(__file__),
              'protocol_hashes':{k:sha(Path(__file__).with_name(v)) for k,v in
                    [('H04','H04_direct_tensor_operator.md'),('H06','H06_residual_rank_budget.md')]},
              'torch':torch.__version__,'tensorly':tl.__version__,'cuda':torch.version.cuda,'device':device,
              'seeds':args.seeds,'command':sys.argv,'gpu_snapshot':gpu_snapshot(),
              'microbatch':32,'calibration_steps':128,'recovery_steps':512,'precision':'FP32',
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'limitations':['only compressed layer trainable','no real-data conclusions',
                  'H06 is only a grid/residual pilot','latency includes whole toy CNN, excludes CPU data loading',
                  'matrix factor and Tucker storage budgets differ (176 vs 196 weights)',
                  'no fused GPU kernel','WDDM process GPU peak not measured']}
    write_json(args.output/'manifest.json',manifest)
    try:run(args.output,args.seeds,device);manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
