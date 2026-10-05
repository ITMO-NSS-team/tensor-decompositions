"""H11 held-out sketch reliability and implicit-operator admission."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.stats import beta
import torch
import tensorly as tl

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git, sha, write_json
from run_cnn_local import synthetic
from run_sketch_pilot import omega


def fwht(x):
    """Normalized Sylvester transform along the last axis, no dense H."""
    n=x.shape[-1]
    if n<1 or n&(n-1): raise ValueError('FWHT width must be a power of two')
    result=x.clone(); width=1
    while width<n:
        shaped=result.reshape(*result.shape[:-1],-1,2,width)
        a,b=shaped[...,0,:],shaped[...,1,:]
        result=torch.stack((a+b,a-b),dim=-2).reshape_as(result)
        width*=2
    return result/math.sqrt(n)


def apply_sketch(x,ell,method,seed):
    n=x.shape[1]; rng=np.random.default_rng(seed)
    if method=='gaussian':
        with tl.backend_context('pytorch'):
            sketch=omega(n,ell,'gaussian',seed,dtype=x.dtype).to(x.device)
        return x@sketch
    if method in ('countsketch','sparse_jl'):
        result=x.new_zeros(x.shape[0],ell)
        if method=='countsketch':
            buckets=rng.integers(0,ell,n)
            signs=rng.choice([-1.,1.],n)
            ids=torch.as_tensor(buckets,device=x.device)
            vals=torch.as_tensor(signs,device=x.device,dtype=x.dtype)
            result.scatter_add_(1,ids[None].expand(x.shape[0],-1),x*vals)
        else:
            # The normative generator draws distinct bins and signs row by row.
            if ell<4: raise ValueError('sparse JL requires ell >= 4')
            bins=[]; signs=[]
            for _ in range(n):
                bins.append(rng.choice(ell,4,replace=False))
                signs.append(rng.choice([-1.,1.],4)/2)
            bins=torch.as_tensor(np.asarray(bins),device=x.device)
            signs=torch.as_tensor(np.asarray(signs),device=x.device,dtype=x.dtype)
            for k in range(4):
                result.scatter_add_(1,bins[:,k][None].expand(x.shape[0],-1),x*signs[:,k])
        return result
    if method=='srht_reference':
        padded=1<<(n-1).bit_length()
        columns=torch.as_tensor(rng.choice(padded,ell,replace=False),device=x.device)
        signs=torch.as_tensor(rng.choice([-1.,1.],padded),device=x.device,dtype=x.dtype)
        xx=torch.nn.functional.pad(x,(0,padded-n))*signs
        return fwht(xx)[:,columns]*math.sqrt(padded/ell)
    raise ValueError(method)


def factors(x,ell,method,seed,rank):
    q=torch.linalg.qr(apply_sketch(x,ell,method,seed),mode='reduced').Q
    z=torch.linalg.qr(x.T@q,mode='reduced').Q
    q=torch.linalg.qr(x@z,mode='reduced').Q
    u,s,vh=torch.linalg.svd(q.T@x,full_matrices=False)
    return (q@u[:,:rank])*s[:rank],vh[:rank]


def admission():
    gen=torch.Generator().manual_seed(99); rows=[]
    for n in [7,8,72]:
        x=torch.randn(n,5,generator=gen,dtype=torch.float64).T  # noncontiguous
        ell=min(8,1<<(n-1).bit_length())
        for method in ['gaussian','countsketch','sparse_jl','srht_reference']:
            y=apply_sketch(x,ell,method,17)
            with tl.backend_context('pytorch'):
                reference=x@omega(n,ell,method,17)
            error=float((y-reference).norm()/reference.norm().clamp_min(1e-30))
            if error>1e-12: raise AssertionError((n,method,error))
            rows.append({'n':n,'method':method,'relative_error':error,'noncontiguous':True})
    z=torch.zeros(3,7,dtype=torch.float64)
    for method in ['gaussian','countsketch','sparse_jl','srht_reference']:
        assert not apply_sketch(z,8,method,17).count_nonzero()
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--admission-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);tl.set_backend('pytorch')
    (args.output/'run_h11_confirmation.py.source').write_bytes(Path(__file__).read_bytes())
    manifest={'state':'running','hypothesis_id':'H11','stage':'held-out matrix reliability',
              'git_sha':git('rev-parse','HEAD'),'git_status':git('status','--porcelain'),
              'runner_sha256':sha(__file__),'torch':torch.__version__,'numpy':np.__version__,
              'source_hashes':{n:sha(Path(__file__).with_name(n)) for n in ['run_local.py','run_cnn_local.py','run_sketch_pilot.py']},
              'protocol_sha256':sha(Path(__file__).with_name('H11_structured_sketches.md')),
              'device':'cpu','precision':'FP64','ell':8,'rank':4,'power':1,
              'trial_seeds':list(range(200000,200100)),
              'selection_provenance':'ell=8 fixed by prior pilot and first neural series; not changed by these trials',
              'limitations':['100 draws on one fixed toy weight; not 100 independent neural models',
                             'CPU implicit Python operators; no real-model or GPU speed conclusion'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',admission())
        if args.admission_only:manifest['state']='admission_passed';return
        teacher,_=synthetic(11,'cpu');x=teacher[2].weight.detach().reshape(16,72).double()
        u,s,vh=torch.linalg.svd(x,full_matrices=False)
        exact=(u[:,:4]*s[:4])@vh[:4]
        error_exact=float((x-exact).norm()/x.norm());rows=[];summary=[]
        for method in ['gaussian','countsketch','sparse_jl','srht_reference']:
            for seed in manifest['trial_seeds']:
                start=time.perf_counter();a,b=factors(x,8,method,seed,4)
                error=float((x-a@b).norm()/x.norm())
                rows.append({'method':method,'seed':seed,'relative_error':error,
                             'exact_error':error_exact,'failure':error>max(1.05*error_exact,1e-6),
                             'seconds':time.perf_counter()-start})
            group=[r for r in rows if r['method']==method];k=sum(r['failure'] for r in group);n=len(group)
            low=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
            high=1. if k==n else float(beta.ppf(.975,k+1,n-k))
            summary.append({'method':method,'failures':k,'n':n,'failure_rate':k/n,
                            'clopper_pearson_95':[low,high],'upper_bound_le_005':high<=.05,
                            'timing_scope':'diagnostic only; not warmed repeated performance benchmark'})
        write_json(args.output/'rows.json',rows);write_json(args.output/'summary.json',summary)
        manifest['state']='completed';print(json.dumps(summary),flush=True)
    except Exception as error:
        manifest.update(state='implementation_failure',error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
