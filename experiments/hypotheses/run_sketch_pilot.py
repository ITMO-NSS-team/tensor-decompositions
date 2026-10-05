"""H11 sketch-construction admission and bounded synthetic matrix pilot.

No neural H11 recovery/quality experiment or implicit SRHT speed claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import tensorly as tl

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git, sha, write_json
from run_cnn_local import synthetic
from tdecomp.matrix.decomposer import RandomizedSVD
from tdecomp.matrix.random_projections import normal, sparse_jl_matrix


def omega(n,ell,method,seed,dtype=torch.float64):
    rng=np.random.default_rng(seed)
    if method=='gaussian':return normal(n,ell,context={'dtype':dtype},random_state=rng)
    if method=='sparse_jl':return sparse_jl_matrix(n,ell,s=4,context={'dtype':dtype},random_state=rng)
    if method=='countsketch':
        matrix=torch.zeros(n,ell,dtype=dtype)
        buckets=torch.from_numpy(rng.integers(0,ell,n))
        signs=torch.from_numpy(rng.choice([-1.,1.],n)).to(dtype)
        matrix[torch.arange(n),buckets]=signs
        return matrix
    if method=='srht_reference':
        padded=1<<(n-1).bit_length()
        h=torch.ones(1,1,dtype=dtype)
        while len(h)<padded:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
        h/=math.sqrt(padded)
        columns=torch.from_numpy(rng.choice(padded,ell,replace=False))
        signs=torch.from_numpy(rng.choice([-1.,1.],padded)).to(dtype)
        return (math.sqrt(padded/ell)*signs[:,None]*h[:,columns])[:n]
    raise ValueError(method)


def rsvd_with_omega(x,sketch,rank):
    q=torch.linalg.qr(x@sketch,mode='reduced').Q
    z=torch.linalg.qr(x.T@q,mode='reduced').Q
    q=torch.linalg.qr(x@z,mode='reduced').Q
    u,s,vh=torch.linalg.svd(q.T@x,full_matrices=False)
    return (q@u[:,:rank]*s[:rank])@vh[:rank]


def admission():
    for method in ('countsketch','sparse_jl'):
        o=omega(72,8,method,17)
        assert torch.equal((o!=0).sum(1),torch.full((72,),1 if method=='countsketch' else 4))
        torch.testing.assert_close(o.square().sum(1),torch.ones(72,dtype=o.dtype))
    srht=omega(8,8,'srht_reference',17)
    torch.testing.assert_close(srht@srht.T,torch.eye(8,dtype=srht.dtype),atol=1e-12,rtol=1e-12)
    # Constructed adversarial cancellation is not a random failure frequency.
    x=torch.tensor([[1.,1.]],dtype=torch.float64)
    collision=torch.tensor([[1.],[-1.]],dtype=torch.float64)
    assert torch.equal(x@collision,torch.zeros(1,1,dtype=torch.float64))
    return {'sparse_row_energy':'passed','sparse_nonzeros':'passed','full_srht_isometry':'passed',
            'constructed_countsketch_cancellation':'passed'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);tl.set_backend('pytorch');torch.set_num_threads(8)
    manifest={'run_id':args.output.name,'state':'running','git_sha':git('rev-parse','HEAD'),
              'git_status':git('status','--porcelain'),'runner_sha256':sha(__file__),
              'protocol_sha256':sha(Path(__file__).with_name('H11_structured_sketches.md')),
              'torch':torch.__version__,'tensorly':tl.__version__,'device':'cpu','dtype':'float64',
              'scope':'admission plus width-selection matrix pilot only',
              'limitations':['SRHT materializes a dense reference: timing is not an implicit-transform benchmark',
                             'no neural recovery or real-data H11 experiment',
                             'pilot trials select width; cannot be reused as reliability confirmation'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',admission())
        teacher,_=synthetic(11,'cpu');x=teacher[2].weight.detach().reshape(16,72).double()
        u,s,vh=torch.linalg.svd(x,full_matrices=False);exact=(u[:,:4]*s[:4])@vh[:4]
        exact_error=float(torch.linalg.vector_norm(x-exact)/torch.linalg.vector_norm(x));rows=[]
        for method in ('gaussian','countsketch','sparse_jl','srht_reference'):
            for ell in (8,12,16):
                for trial in range(20):
                    start=time.perf_counter();sketch=omega(72,ell,method,1000+trial)
                    result=rsvd_with_omega(x,sketch,4)
                    error=float(torch.linalg.vector_norm(x-result)/torch.linalg.vector_norm(x))
                    rows.append({'method':method,'ell':ell,'rank':4,'power':1,'seed':1000+trial,
                                 'relative_error':error,'exact_error':exact_error,
                                 'failure':error>max(1.05*exact_error,1e-6),'total_seconds':time.perf_counter()-start})
        write_json(args.output/'pilot_rows.json',rows)
        write_json(args.output/'width_selection.json',[{'method':method,'ell':ell,
                    'failures':sum(r['failure'] for r in rows if r['method']==method and r['ell']==ell),
                    'n_trials':20} for method in ('gaussian','countsketch','sparse_jl','srht_reference') for ell in (8,12,16)])
        manifest['state']='completed';print(json.dumps({'matrix_pilot_runs':len(rows),'admission':'passed'}),flush=True)
    except Exception as error:manifest.update(state='implementation_failure',error=str(error));raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
