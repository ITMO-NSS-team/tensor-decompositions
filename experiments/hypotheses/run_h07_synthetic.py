"""H07 two-probe contraction preparation versus column-sampling baselines."""
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


def normalize(x, fallback):
    return x/x.norm() if float(x.norm())>0 else fallback


def contraction_probabilities(w,gen):
    energy=w.new_zeros(w.shape[1])
    for _ in range(2):
        s=normalize(torch.randn(w.shape[2],generator=gen,device=w.device,dtype=w.dtype),w.new_ones(w.shape[2]))
        t=normalize(torch.randn(w.shape[3],generator=gen,device=w.device,dtype=w.dtype),w.new_ones(w.shape[3]))
        c=torch.einsum('oihw,h,w->oi',w,s,t)
        u,_,vh=torch.linalg.svd(c,full_matrices=False)
        # One common contraction pass gives both spatial gradients from OLD s/t.
        spatial=torch.einsum('oihw,o,i->hw',w,u[:,0],vh[0])
        gs,gt=spatial@t,spatial.T@s
        s,t=normalize(gs,s),normalize(gt,t)
        c=torch.einsum('oihw,h,w->oi',w,s,t)
        energy+=c.square().sum(0)
    q=energy/energy.sum() if float(energy.sum())>0 else torch.full_like(energy,1/len(energy))
    n=w.shape[1]*w.shape[2]*w.shape[3]
    return (.8*q[:,None,None].expand(-1,w.shape[2],w.shape[3]).reshape(-1)/(w.shape[2]*w.shape[3])+.2/n)


def probabilities(w,method,gen,rank=4):
    x=w.reshape(w.shape[0],-1);n=x.shape[1]
    if method=='uniform':return w.new_full((n,),1/n),0.
    if method=='column_norm':
        energy=x.square().sum(0)
        return (energy/energy.sum() if float(energy.sum()) else w.new_full((n,),1/n)),1.
    if method=='leverage':
        width=min(rank+4,x.shape[0],n)
        omega=torch.randn(n,width,generator=gen,device=w.device,dtype=w.dtype)
        q=torch.linalg.qr(x@omega,mode='reduced').Q
        _,_,vh=torch.linalg.svd(q.T@x,full_matrices=False)
        energy=vh[:rank].square().sum(0)/rank
        return energy/energy.sum(),2.
    if method=='contraction':return contraction_probabilities(w,gen),6.
    raise ValueError(method)


def sampled_tucker(w,method,seed,b=16,ranks=(4,3,2,2)):
    gen=torch.Generator(device=w.device).manual_seed(seed)
    p,prep=probabilities(w,method,gen,rank=ranks[0])
    if not torch.isfinite(p).all() or bool((p<0).any()):raise ArithmeticError('invalid probabilities')
    torch.testing.assert_close(p.sum(),p.new_tensor(1.))
    ix=torch.multinomial(p,b,replacement=True,generator=gen)
    x=w.reshape(w.shape[0],-1)
    columns=x[:,ix]/(b*p[ix]).sqrt()
    raw_q,r=torch.linalg.qr(columns,mode='reduced')
    rotation,singular,_=torch.linalg.svd(r,full_matrices=False)
    threshold=torch.finfo(w.dtype).eps*max(columns.shape)*singular.max()
    keep=singular>threshold
    if int(keep.sum())<ranks[0]:
        raise ArithmeticError('sampled column span rank below requested output rank; no null-space completion')
    q=raw_q@rotation[:,keep]
    u,_,_=torch.linalg.svd(q.T@x,full_matrices=False)
    first=(q@u)[:,:ranks[0]]
    core=tl.tenalg.mode_dot(w,first.T,mode=0);factors=[first]
    logical_elements=prep*w.numel()+columns.numel()+w.numel()*2
    for mode in [1,2,3]:
        unfolding=tl.unfold(core,mode)
        u,_,_=torch.linalg.svd(unfolding,full_matrices=False)
        factor=u[:,:ranks[mode]]
        logical_elements+=2*core.numel()
        core=tl.tenalg.mode_dot(core,factor.T,mode=mode);factors.append(factor)
    reconstructed=tl.tenalg.multi_mode_dot(core,factors)
    error=float((w-reconstructed).norm()/w.norm().clamp_min(1e-30))
    logical_elements+=w.numel()
    return core,factors,{'relative_tensor_error':error,'sampled_columns':ix.tolist(),
                         'unique_columns':len(ix.unique()),'preparation_full_passes':prep,
                         'sampled_span_rank':int(keep.sum()),'sampled_rank_threshold':float(threshold),
                         'logical_elements_read':logical_elements,'logical_full_pass_equivalents':logical_elements/w.numel(),
                         'passes_scope':'explicit tensor contractions/projections/sampling/reference residual; excludes opaque QR/SVD internal rereads'}


class FullTuckerConv(nn.Module):
    def __init__(self,core,factors,bias):
        super().__init__();self.core=nn.Parameter(core.clone())
        self.factors=nn.ParameterList([nn.Parameter(f.clone()) for f in factors])
        self.bias=nn.Parameter(bias.clone())
    def forward(self,x):
        weight=tl.tenalg.multi_mode_dot(self.core,list(self.factors))
        return F.conv2d(x,weight,self.bias,padding=1)


def admission():
    w=torch.zeros(3,2,2,2,dtype=torch.float64)
    p=contraction_probabilities(w,torch.Generator().manual_seed(7))
    torch.testing.assert_close(p,torch.full_like(p,1/8))
    # A spatial probe orthogonal to a rank-one signal removes it exactly.
    signal=torch.einsum('o,i,h,w->oihw',torch.ones(3),torch.ones(2),torch.tensor([1.,0.]),torch.tensor([1.,0.]))
    assert not torch.einsum('oihw,h,w->oi',signal,torch.tensor([0.,1.]),torch.tensor([1.,0.])).count_nonzero()
    gen=torch.Generator().manual_seed(77);weight=torch.randn(4,3,2,2,generator=gen,dtype=torch.float64)
    core,factors,info=sampled_tucker(weight,'uniform',8,b=64,ranks=(4,3,2,2))
    torch.testing.assert_close(tl.tenalg.multi_mode_dot(core,factors),weight,atol=1e-10,rtol=1e-10)
    assert info['unique_columns']<64
    adversarial=torch.zeros(16,8,3,3,dtype=torch.float64)
    adversarial.reshape(16,-1)[0,0]=1000.
    for i in range(1,16):adversarial.reshape(16,-1)[i,i]=1.
    try:sampled_tucker(adversarial,'column_norm',11,b=16)
    except ArithmeticError as error:assert 'sampled column span rank' in str(error)
    else:raise AssertionError('rank-deficient sample was completed with unsupported directions')
    return {'zero_probability_fallback':'passed','orthogonal_signal_probe':'passed',
            'duplicates_and_full_rank_roundtrip':'passed','rank_deficient_sample_rejected':'passed'}


def teacher_and_splits(seed,device):
    teacher,splits=synthetic(seed,device)
    # Reproduce the declared signal/noise draw, then anisotropically scale P only.
    gen=torch.Generator(device=device).manual_seed(seed)
    factors=[torch.linalg.qr(torch.randn(n,r,generator=gen,device=device)).Q for n,r in zip((16,8,3,3),(4,3,2,2))]
    core=torch.randn(4,3,2,2,generator=gen,device=device)
    p=tl.tenalg.multi_mode_dot(core,factors)
    p=p/p.norm()
    z=torch.randn(p.shape,generator=gen,device=device)
    p=p*torch.tensor([3,2,1,.5,.2,.1,.05,.02],device=device)[None,:,None,None];p=p/p.norm()
    with torch.no_grad():teacher[2].weight.copy_(p+.05*z/z.norm())
    with torch.no_grad():splits={k:(x,teacher(x)) for k,(x,_) in splits.items()}
    return teacher,splits


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--admission-only',action='store_true');parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--columns',type=int,choices=[8,16,32],default=16)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);tl.set_backend('pytorch');torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    (args.output/'run_h07_synthetic.py.source').write_bytes(Path(__file__).read_bytes())
    manifest={'state':'admission','hypothesis_id':'H07','setting':'synthetic','git_sha':git('rev-parse','HEAD'),
              'runner_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).with_name('H07_contraction_sampling.md')),
              'torch':torch.__version__,'tensorly':tl.__version__,'device':args.device,'rank':[4,3,2,2],'columns':args.columns,
              'stage':'primary b16' if args.columns==16 else 'predeclared secondary sample-size control',
              'seeds':[11,22,33,44,55],'calibration_steps':128,'recovery_steps':512,'precision':'FP32',
              'primary_scope':'tensor error and neural quality; full read-budget/performance criterion remains unverified',
              'limitations':['logical reads exclude opaque QR/SVD internal memory accesses; <=16|T| not certified',
                             'no real-model conclusions','dense differentiable reconstruction counted in execution',
                             'fixed b16 protocol setting; secondary b8/b32 and adversarial neural modes pending'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',admission())
        if args.admission_only:manifest['state']='admission_passed';return
        manifest['state']='running';write_json(args.output/'manifest.json',manifest);rows=[]
        for seed in manifest['seeds']:
            teacher,splits=teacher_and_splits(seed,args.device);sd=args.output/f'seed-{seed}';sd.mkdir()
            torch.save(teacher.state_dict(),sd/'teacher.pt')
            write_json(sd/'inputs.json',{k:{'x':tensor_hash(x),'y':tensor_hash(y)} for k,(x,y) in splits.items()})
            for method in ['uniform','column_norm','leverage','contraction']:
                start=time.perf_counter();sync();setup=time.perf_counter()
                core,factors,info=sampled_tucker(teacher[2].weight.detach(),method,seed+100000,b=args.columns)
                sync();factor_seconds=time.perf_counter()-setup
                student=copy.deepcopy(teacher);student[2]=FullTuckerConv(core,factors,teacher[2].bias)
                for p in student.parameters():p.requires_grad_(False)
                for p in student[2].parameters():p.requires_grad_(True)
                before=mse(student,splits['tuning']);opt=torch.optim.AdamW(student[2].parameters(),lr=.001,weight_decay=0)
                for phase,steps,offset in [('calibration',128,50000),('recovery',512,60000)]:
                    x,y=splits[phase];gen=torch.Generator(device=args.device).manual_seed(seed+offset)
                    for step in range(steps):
                        ix=torch.randint(len(x),(32,),generator=gen,device=args.device)
                        opt.zero_grad(set_to_none=True);loss=F.mse_loss(student(x[ix]),y[ix])
                        if not torch.isfinite(loss):raise ArithmeticError('nonfinite H07 loss')
                        loss.backward();opt.step()
                        if step%64==0:guard(start,600)
                row={'seed':seed,'method':method,**info,'tuning_before':before,'final_test':mse(student,splits['test']),
                     'factor_seconds':factor_seconds,'total_seconds':time.perf_counter()-start,
                     'latency':latency(student,splits['tuning'][0][:64]),'status':'completed'}
                torch.save(student.state_dict(),sd/f'{method}.pt');row['checkpoint_sha256']=sha(sd/f'{method}.pt')
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(TimeoutError,MemoryError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
