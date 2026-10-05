"""First H11 toy-CNN comparison. Dense sketch references cannot prove speedup."""
from __future__ import annotations

import argparse
import copy
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
from run_local import git, guard, sha, sync, tensor_hash, write_json
from run_cnn_local import synthetic, mse, latency
from run_sketch_pilot import omega, rsvd_with_omega
from run_h11_confirmation import admission as implicit_admission, factors as implicit_factors


class FactorConv(nn.Module):
    def __init__(self,weight,bias):
        super().__init__()
        u,s,vh=torch.linalg.svd(weight,full_matrices=False)
        self.a=nn.Parameter((u[:,:4]*s[:4])[:,:,None,None])
        self.b=nn.Parameter(vh[:4].reshape(4,8,3,3))
        self.bias=nn.Parameter(bias.clone())

    def forward(self,x):return F.conv2d(F.conv2d(x,self.b,padding=1),self.a,self.bias)

    @classmethod
    def from_factors(cls,a,b,bias):
        result=cls.__new__(cls);nn.Module.__init__(result)
        result.a=nn.Parameter(a[:,:,None,None].clone())
        result.b=nn.Parameter(b.reshape(4,8,3,3).clone())
        result.bias=nn.Parameter(bias.clone())
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    p.add_argument('--methods',nargs='+',choices=['exact_svd','gaussian','countsketch','sparse_jl','srht_reference'],
                   default=['gaussian','countsketch','sparse_jl','srht_reference'])
    p.add_argument('--implicit',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['run_h11_neural.py','run_h11_confirmation.py','run_sketch_pilot.py','run_cnn_local.py','run_local.py']:
        (args.output/(name+'.source')).write_bytes(Path(__file__).with_name(name).read_bytes())
    tl.set_backend('pytorch');torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest={'state':'running','run_id':args.output.name,'git_sha':git('rev-parse','HEAD'),
              'git_status':git('status','--porcelain'),'runner_sha256':sha(__file__),
              'helper_sha256':{name:sha(Path(__file__).with_name(name)) for name in
                              ['run_local.py','run_cnn_local.py','run_sketch_pilot.py','run_h11_confirmation.py']},
              'protocol_sha256':sha(Path(__file__).with_name('H11_structured_sketches.md')),
              'seeds':args.seeds,'device':device,'torch':torch.__version__,'tensorly':tl.__version__,
              'rank':4,'ell':8,'power':1,'batch':32,'calibration_steps':128,'recovery_steps':512,
              'methods':args.methods,'implicit':args.implicit,
              'scope':'H11 first synthetic neural comparison after fixed ell=8 pilot selection',
              'limitations':['CountSketch and SRHT are dense reference matrices, not implicit optimized operators',
                  'factor initialization re-SVD uses an exact oracle and is counted; no speed benefit claimed',
                  'full 100-trial held-out reliability study not performed',
                  'no real-data conclusions; no fused kernels','only compressed layer trainable'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    if args.implicit:
        manifest['limitations']=['implicit Python operators; no fused kernels or speed claim',
                                'held-out 100-trial reliability is stored as a separate experiment',
                                'no real-data conclusions; only compressed layer trainable']
    write_json(args.output/'manifest.json',manifest)
    rows=[]
    try:
        if args.implicit:write_json(args.output/'implicit_admission.json',implicit_admission())
        for seed in args.seeds:
            teacher,splits=synthetic(seed,device);sd=args.output/f'seed-{seed}';sd.mkdir()
            torch.save(teacher.state_dict(),sd/'teacher.pt')
            write_json(sd/'inputs.json',{k:{'x':tensor_hash(x),'y':tensor_hash(y)} for k,(x,y) in splits.items()})
            w=teacher[2].weight.detach().reshape(16,72)
            for method in args.methods:
                start=time.perf_counter();sync();setup_start=time.perf_counter()
                if method=='exact_svd':
                    u,s,vh=torch.linalg.svd(w,full_matrices=False);a,b=u[:,:4]*s[:4],vh[:4]
                    approximation=a@b
                    replacement=FactorConv.from_factors(a,b,teacher[2].bias)
                elif args.implicit:
                    a,b=implicit_factors(w,8,method,seed+100000,4);approximation=a@b
                    replacement=FactorConv.from_factors(a,b,teacher[2].bias)
                else:
                    sketch=omega(72,8,method,seed+100000,dtype=torch.float32).to(device)
                    approximation=rsvd_with_omega(w,sketch,4)
                    replacement=FactorConv(approximation,teacher[2].bias)
                student=copy.deepcopy(teacher);student[2]=replacement
                sync();setup_seconds=time.perf_counter()-setup_start
                for parameter in student.parameters():parameter.requires_grad_(False)
                for parameter in student[2].parameters():parameter.requires_grad_(True)
                relative_error=float(torch.linalg.vector_norm(w-approximation)/torch.linalg.vector_norm(w))
                before=mse(student,splits['tuning'])
                opt=torch.optim.AdamW(student[2].parameters(),lr=.001,weight_decay=0)
                sync();recovery_start=time.perf_counter()
                for phase,steps,offset in [('calibration',128,50000),('recovery',512,60000)]:
                    x,y=splits[phase];gen=torch.Generator(device=device).manual_seed(seed+offset)
                    for step in range(steps):
                        ix=torch.randint(len(x),(32,),generator=gen,device=device)
                        opt.zero_grad(set_to_none=True);loss=F.mse_loss(student(x[ix]),y[ix])
                        if not torch.isfinite(loss):raise ArithmeticError('nonfinite H11 neural loss')
                        loss.backward();opt.step()
                        if step%64==0:guard(start)
                sync();recovery_seconds=time.perf_counter()-recovery_start
                student.eval();test=mse(student,splits['test']);timing=latency(student,splits['tuning'][0][:64])
                torch.save(student.state_dict(),sd/f'{method}.pt')
                row={'hypothesis_id':'H11','setting':'synthetic','seed':seed,'method':method,
                     'rank':4,'ell':8,'power':1,'setup_seconds':setup_seconds,'recovery_seconds':recovery_seconds,
                     'relative_weight_error':relative_error,'before_tuning':before,'final_test':test,
                     'latency':timing,'factor_weights':352,'trainable_parameters':368,
                     'checkpoint_sha256':sha(sd/f'{method}.pt'),'status':'completed'}
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='implementation_failure',error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
