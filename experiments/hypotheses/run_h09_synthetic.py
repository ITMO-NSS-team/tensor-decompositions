"""H09 explicit error-feedback compressed AdamW, fixed versus drift refresh."""
from __future__ import annotations

import argparse
import copy
import copy
import json
from pathlib import Path
import sys
import time
import traceback

import torch
import tensorly as tl
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git, guard, sha, tensor_hash, write_json
from tdecomp.matrix.decomposer import RandomizedSVD


class CompressedAdam:
    def __init__(self,shape,device,rank=8,refresh='adaptive',seed=0,solver='rsvd'):
        self.rank=rank;self.refresh=refresh;self.seed=seed
        self.e=torch.zeros(shape,device=device);self.q=None;self.m=None;self.v=None
        self.tau=0;self.updates=0;self.events=[]
        self.solver=solver

    @torch.no_grad()
    def step(self,p,step,lr=.001):
        original=p.grad.clone();h=original+self.e
        refresh=self.q is None;rho=None;diagnostic_seconds=0.;factor_seconds=0.
        if self.q is not None and step%10==0 and self.refresh in ['adaptive','fixed_paid']:
            begin=time.perf_counter()
            rng=torch.Generator(device=p.device).manual_seed(self.seed+900000+step//10)
            omega=torch.randn(h.shape[1],16,generator=rng,device=p.device)
            projected=h@omega;denominator=float(projected.square().sum())
            rho=float((self.q.T@projected).square().sum())/denominator if denominator else None
            diagnostic_seconds=time.perf_counter()-begin
        if self.q is not None:
            if self.refresh in ['fixed','fixed_paid']:refresh=step%40==0
            elif self.refresh=='adaptive':refresh=rho is not None and rho<.9
        refresh=refresh and self.updates<10
        if refresh:
            begin=time.perf_counter()
            if self.solver=='exact':self.q=torch.linalg.svd(h,full_matrices=False).U[:,:self.rank]
            else:
                with tl.backend_context('pytorch'):
                    self.q,_,_=RandomizedSVD(rank=self.rank,oversampling=8,power=1,
                        random_state=self.seed+950000+self.updates).decompose(h)
            factor_seconds=time.perf_counter()-begin
            self.m=torch.zeros(self.rank,h.shape[1],device=p.device)
            self.v=torch.zeros_like(self.m);self.tau=0;self.updates+=1
        g=self.q.T@h;self.e=h-self.q@g
        self.tau+=1;self.m.mul_(.9).add_(g,alpha=.1);self.v.mul_(.999).addcmul_(g,g,value=.001)
        direction=self.q@((self.m/(1-.9**self.tau))/(torch.sqrt(self.v/(1-.999**self.tau))+1e-8))
        p.add_(direction,alpha=-lr)
        if not torch.isfinite(p).all():raise ArithmeticError('nonfinite compressed Adam parameter')
        torch.testing.assert_close(p.grad,original,rtol=0,atol=0)
        self.events.append({'step':step,'rho':rho,'refreshed':refresh,'refresh_count':self.updates,
                            'tau':self.tau,'diagnostic_seconds':diagnostic_seconds,'factor_seconds':factor_seconds,
                            'ef_norm':float(self.e.norm())})

    def state_dict(self):
        return {'q':self.q,'m':self.m,'v':self.v,'e':self.e,'tau':self.tau,'updates':self.updates,
                'rank':self.rank,'refresh':self.refresh,'seed':self.seed,'events':self.events,'solver':self.solver}

    def load_state_dict(self,state):
        """Restore the full-coordinate residual and the local Adam clock."""
        for name in ('rank','refresh','seed','solver'):
            if state[name]!=getattr(self,name):
                raise ValueError(f'compressed optimizer {name} differs from saved state')
        if state['e'].shape!=self.e.shape:
            raise ValueError('compressed residual shape differs from saved state')
        q,m,v=state['q'],state['m'],state['v']
        if q is None:
            if m is not None or v is not None or state['tau']!=0 or state['updates']!=0:
                raise ValueError('uninitialized compressed optimizer has moments or clock')
        elif (q.shape!=(self.e.shape[0],self.rank) or m is None or v is None or
              m.shape!=(self.rank,self.e.shape[1]) or v.shape!=m.shape):
            raise ValueError('compressed basis or moment shape mismatch')
        if not 0<=state['updates']<=10 or state['tau']<0:
            raise ValueError('invalid compressed optimizer counters')
        tensors=[state['e'],q,m,v]
        if any(value is not None and not torch.isfinite(value).all() for value in tensors):
            raise ValueError('nonfinite compressed optimizer state')
        def restore(value):
            return None if value is None else value.detach().to(device=self.e.device,dtype=self.e.dtype).clone()
        self.e,self.q,self.m,self.v=[restore(value) for value in tensors]
        self.tau=state['tau'];self.updates=state['updates'];self.events=copy.deepcopy(state['events'])


def rotate(z,theta):
    x=z*torch.tensor([2.]*8+[.2]*24,device=z.device)
    result=x.clone();c,s=torch.cos(x.new_tensor(theta)),torch.sin(x.new_tensor(theta))
    result[:,:8]=c*x[:,:8]-s*x[:,8:16]
    result[:,8:16]=s*x[:,:8]+c*x[:,8:16]
    return result


def make_model(seed,device):
    rng=torch.Generator(device=device).manual_seed(seed)
    model=nn.Sequential(nn.Linear(32,64,bias=False),nn.GELU(),nn.Linear(64,4,bias=False)).to(device)
    with torch.no_grad():
        for layer in [model[0],model[2]]:
            layer.weight.copy_(torch.randn(layer.weight.shape,generator=rng,device=device)/layer.in_features**.5)
    return model


@torch.no_grad()
def metric(model,teacher,x):
    target=teacher(x);error=(model(x)-target).square().sum();energy=target.square().sum()
    return {'mse':float(error)/target.numel(),'normalized_mse':float(error/energy)}


def admission():
    p=nn.Parameter(torch.randn(12,5,generator=torch.Generator().manual_seed(7)))
    p.grad=torch.randn(12,5,generator=torch.Generator().manual_seed(9))
    before=p.grad.clone();method=CompressedAdam(p.shape,'cpu',rank=3,refresh='fixed')
    h=before.clone();method.step(p,0)
    torch.testing.assert_close(method.e+method.q@(method.q.T@h),h)
    torch.testing.assert_close(p.grad,before,rtol=0,atol=0)
    method.step(p,40);assert method.tau==1 and method.updates==2
    state=method.state_dict();assert set(['q','m','v','e','tau','updates'])<=state.keys()
    zero=nn.Parameter(torch.zeros(12,5));zero.grad=torch.zeros_like(zero)
    zmethod=CompressedAdam(zero.shape,'cpu',rank=3);zmethod.step(zero,0);zmethod.step(zero,10)
    assert zmethod.events[-1]['rho'] is None and zmethod.updates==1
    return {'gradient_immutable':'passed','error_feedback_identity':'passed','both_moments_reset':'passed',
            'zero_drift_no_refresh':'passed','explicit_serializable_state':'passed'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--subspace-solver',choices=['rsvd','exact'],default='rsvd')
    p.add_argument('--admission-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    (args.output/'run_h09_synthetic.py.source').write_bytes(Path(__file__).read_bytes())
    manifest={'state':'admission','hypothesis_id':'H09','setting':'synthetic','git_sha':git('rev-parse','HEAD'),
              'runner_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).with_name('H09_gradient_drift.md')),
              'torch':torch.__version__,'device':'cpu','precision':'FP32','seeds':[11,22,33,44,55],
              'methods':['dense','fixed','adaptive','fixed_paid'],'rank':8,'warmup_steps':20,'steps':400,
              'refresh_policy':'reset m/v and local bias counter; preserve full-coordinate E',
              'subspace_solver':args.subspace_solver,'oversampling':8,'power':1,
              'subspace_seed_rule':'model_seed + 950000 + refresh_index; independent diagnostic stream',
              'limitations':['only smooth rotation regime; abrupt/flat/no-EF controls pending',
                             'synthetic CPU only; no real ResNet optimizer conclusions',
                             'calibration512 input split is saved but unused; drift diagnostic uses current H and independent Omega',
                             'research optimizer; not existing TensorGRaD implementation'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',admission())
        if args.admission_only:manifest['state']='admission_passed';return
        rows=[];manifest['state']='running';write_json(args.output/'manifest.json',manifest)
        for seed in manifest['seeds']:
            teacher=make_model(seed,'cpu').eval();warm=make_model(seed+10000,'cpu')
            rng=torch.Generator().manual_seed(seed+1000);train=torch.randn(4096,32,generator=rng)
            tune=rotate(torch.randn(512,32,generator=torch.Generator().manual_seed(seed+3000)),.8)
            calibration=rotate(torch.randn(512,32,generator=torch.Generator().manual_seed(seed+2000)),.8)
            test=rotate(torch.randn(1024,32,generator=torch.Generator().manual_seed(seed+4000)),.8)
            sd=args.output/f'seed-{seed}';sd.mkdir()
            write_json(sd/'inputs.json',{'train_z':tensor_hash(train),'calibration_unused':tensor_hash(calibration),
                       'tuning_x':tensor_hash(tune),'test_x':tensor_hash(test)})
            warmopt=torch.optim.AdamW(warm.parameters(),lr=.001,weight_decay=0)
            batchrng=torch.Generator().manual_seed(seed+50000)
            for step in range(20):
                ix=torch.randint(len(train),(32,),generator=batchrng);x=rotate(train[ix],0)
                with torch.no_grad():y=teacher(x)
                warmopt.zero_grad(set_to_none=True);loss=F.mse_loss(warm(x),y);loss.backward();warmopt.step()
            for method in manifest['methods']:
                model=copy.deepcopy(warm);before=metric(model,teacher,tune)
                if method=='dense':
                    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
                    opt.load_state_dict(copy.deepcopy(warmopt.state_dict()));compressed=None
                else:
                    opt=torch.optim.AdamW([model[2].weight],lr=.001,weight_decay=0)
                    opt.state[model[2].weight]=copy.deepcopy(warmopt.state[warm[2].weight])
                    compressed=CompressedAdam(model[0].weight.shape,'cpu',refresh=method,seed=seed,solver=args.subspace_solver)
                batchrng=torch.Generator().manual_seed(seed+60000);history=[];start=time.perf_counter();hit=None
                for step in range(400):
                    ix=torch.randint(len(train),(32,),generator=batchrng);x=rotate(train[ix],.002*step)
                    with torch.no_grad():y=teacher(x)
                    model.zero_grad(set_to_none=True);loss=F.mse_loss(model(x),y)
                    if not torch.isfinite(loss):raise ArithmeticError('nonfinite H09 loss')
                    loss.backward()
                    if compressed is not None:compressed.step(model[0].weight,step)
                    opt.step()
                    if (step+1)%10==0:
                        quality=metric(model,teacher,tune)
                        elapsed=time.perf_counter()-start
                        history.append({'step':step+1,**quality,'seconds':elapsed})
                        if hit is None and quality['normalized_mse']<=.01:hit={'step':step+1,'seconds':elapsed}
                        guard(start,600)
                cp=sd/f'{method}.pt'
                torch.save({'model':model.state_dict(),'dense_optimizer':opt.state_dict(),
                            'compressed_optimizer':compressed.state_dict() if compressed else None},cp)
                write_json(sd/f'{method}-history.json',history)
                row={'seed':seed,'method':method,'initial_tuning':before,'final_test':metric(model,teacher,test),
                     'time_to_quality':hit if before['normalized_mse']>.01 else None,
                     'initial_quality_already_met':before['normalized_mse']<=.01,
                     'total_seconds':time.perf_counter()-start,'refresh_count':compressed.updates if compressed else None,
                     'diagnostic_seconds':sum(e['diagnostic_seconds'] for e in compressed.events) if compressed else 0,
                     'factor_seconds':sum(e['factor_seconds'] for e in compressed.events) if compressed else 0,
                     'checkpoint_sha256':sha(cp),'status':'completed'}
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
