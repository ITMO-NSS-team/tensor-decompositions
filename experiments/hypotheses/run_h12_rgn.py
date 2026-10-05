"""H12 fixed-rank Tucker tangent GN with HOSVD retraction, independently of rank choice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import tensorly as tl
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git, sha, write_json
from run_cnn_local import synthetic
from tdecomp.tensor.tucker import HOOIDecomposition


def hosvd(x,ranks):
    factors=[]
    for mode,rank in enumerate(ranks):
        u,_,_=torch.linalg.svd(tl.unfold(x,mode),full_matrices=False)
        factors.append(u[:,:rank])
    core=tl.tenalg.multi_mode_dot(x,[u.T for u in factors])
    return core,factors


def reconstruct(core,factors):return tl.tenalg.multi_mode_dot(core,factors)


def tangent_matrix(core,factors):
    """Columns represent delta G and U_perp K; U.T delta U = 0 by construction."""
    columns=[];coordinates=[]
    for j in range(core.numel()):
        delta=torch.zeros(core.shape,dtype=core.dtype,device=core.device);delta.reshape(-1)[j]=1.
        columns.append(reconstruct(delta,factors).reshape(-1));coordinates.append(('core',j,None))
    complements=[]
    for mode,u in enumerate(factors):
        full=torch.linalg.qr(u,mode='complete').Q
        # Full QR may flip column signs; its complement is still perpendicular to ORIGINAL u.
        perpendicular=full[:,u.shape[1]:];complements.append(perpendicular)
        for a in range(perpendicular.shape[1]):
            for b in range(u.shape[1]):
                delta=torch.zeros_like(u);delta[:,b]=perpendicular[:,a]
                altered=list(factors);altered[mode]=delta
                columns.append(reconstruct(core,altered).reshape(-1));coordinates.append((mode,a,b))
    return torch.stack(columns,dim=1),coordinates,complements


def coordinate_perturbation(core,factors,coordinates,complements,vector,epsilon):
    g=core.contiguous().clone();u=[f.clone() for f in factors]
    for value,(mode,a,b) in zip(vector,coordinates):
        if mode=='core':g.reshape(-1)[a]+=epsilon*value
        else:u[mode][:,b]+=epsilon*value*complements[mode][:,a]
    return reconstruct(g,u)


def rgn(target,ranks,initial,iterations=20):
    core,factors=initial;history=[]
    for step in range(iterations):
        x=reconstruct(core,factors);residual=x-target;objective=.5*float(residual.square().sum())
        matrix,_,_=tangent_matrix(core,factors)
        # SVD-based minimum-norm solve, explicit threshold; no hidden damping.
        delta=torch.linalg.lstsq(matrix,-residual.reshape(-1),rcond=1e-6,driver='gelsd').solution
        direction=(matrix@delta).reshape_as(x)
        slope=float((residual*direction).sum())
        accepted=False
        for alpha in [1.,.5,.25,.125,.0625,.03125]:
            trial_core,trial_factors=hosvd(x+alpha*direction,ranks)
            trial=reconstruct(trial_core,trial_factors)
            value=.5*float((trial-target).square().sum())
            if value<=objective+1e-4*alpha*slope:
                core,factors=trial_core,trial_factors;accepted=True;break
        history.append({'step':step,'objective':objective,'tangent_dimension':matrix.shape[1],
                        'accepted':accepted,'alpha':alpha if accepted else None,
                        'direction_norm':float(direction.norm()),'slope':slope})
        if not accepted or float(direction.norm())<1e-10:break
    return core,factors,history


def admission():
    gen=torch.Generator().manual_seed(17)
    target=torch.randn(5,4,3,generator=gen,dtype=torch.float64);ranks=(2,2,2)
    core,factors=hosvd(target,ranks);matrix,coordinates,complements=tangent_matrix(core,factors)
    torch.testing.assert_close(matrix[:,:core.numel()].T@matrix[:,:core.numel()],
                              torch.eye(core.numel(),dtype=core.dtype),atol=1e-10,rtol=1e-10)
    # Independent core variation uses a tuple index, never the implementation's flattened writes.
    indexed=core.clone();index=(0,)*core.ndim;indexed[index]+=1e-6
    independent=(reconstruct(indexed,factors)-reconstruct(core,factors)).reshape(-1)/1e-6
    torch.testing.assert_close(independent,matrix[:,0],atol=1e-8,rtol=1e-8)
    assert int(torch.linalg.matrix_rank(matrix))==matrix.shape[1]
    for u,perp in zip(factors,complements):
        torch.testing.assert_close(u.T@perp,torch.zeros(u.shape[1],perp.shape[1],dtype=u.dtype),atol=1e-10,rtol=1e-10)
    v=torch.randn(matrix.shape[1],generator=gen,dtype=target.dtype)
    y=torch.randn(target.numel(),generator=gen,dtype=target.dtype)
    torch.testing.assert_close(torch.dot(matrix@v,y),torch.dot(v,matrix.T@y),atol=1e-10,rtol=1e-10)
    epsilon=1e-6
    plus=coordinate_perturbation(core,factors,coordinates,complements,v,epsilon)
    minus=coordinate_perturbation(core,factors,coordinates,complements,v,-epsilon)
    torch.testing.assert_close(((plus-minus)/(2*epsilon)).reshape(-1),matrix@v,atol=1e-8,rtol=1e-8)
    g,u=hosvd(reconstruct(core,factors),ranks)
    torch.testing.assert_close(reconstruct(g,u),reconstruct(core,factors),atol=1e-10,rtol=1e-10)
    fitted,fs,history=rgn(target,ranks,(core,factors),iterations=3)
    assert .5*float((reconstruct(fitted,fs)-target).square().sum())<=history[0]['objective']+1e-10
    return {'tangent_gauge':'passed','core_basis_gram_and_independent_indexed_derivative':'passed',
            'full_tangent_rank':'passed','adjoint':'passed','finite_difference':'passed',
            'fixed_rank_retraction':'passed','Armijo_descent':'passed'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--admission-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);tl.set_backend('pytorch');torch.set_num_threads(2)
    (args.output/'run_h12_rgn.py.source').write_bytes(Path(__file__).read_bytes())
    manifest={'state':'admission','hypothesis_id':'H12','stage':'fixed-rank RGN versus HOOI',
              'git_sha':git('rev-parse','HEAD'),'runner_sha256':sha(__file__),
              'protocol_sha256':sha(Path(__file__).with_name('H12_functional_rank_rgn.md')),
              'torch':torch.__version__,'tensorly':tl.__version__,'device':'cpu','precision':'FP64',
              'rank':[4,3,2,2],'seeds':[11,22,33,44,55],'iterations':20,
              'limitations':['GN+HOSVD+Armijo research variant; no automatic theorem transfer',
                             'A=identity weight approximation; not a functional rank selector',
                             'explicit tangent matrix and CPU SVD; no scalable real-model claim'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',admission())
        if args.admission_only:manifest['state']='admission_passed';return
        rows=[];manifest['state']='running';write_json(args.output/'manifest.json',manifest)
        for seed in manifest['seeds']:
            teacher,_=synthetic(seed,'cpu');target=teacher[2].weight.detach().double();ranks=(4,3,2,2)
            setup=time.perf_counter();initial=hosvd(target,ranks);setup_seconds=time.perf_counter()-setup
            for method in ['hooi','rgn']:
                start=time.perf_counter()
                if method=='rgn':core,factors,history=rgn(target,ranks,initial)
                else:
                    dec=HOOIDecomposition(rank=ranks,random_state=seed)
                    core,factors=dec.decompose(target,init=(initial[0].clone(),[u.clone() for u in initial[1]]),n_iter_max=20,tol=0.)
                    history=[]
                error=float((reconstruct(core,factors)-target).norm()/target.norm())
                row={'seed':seed,'method':method,'relative_error':error,'solver_seconds':time.perf_counter()-start,
                     'common_hosvd_seconds':setup_seconds,'status':'completed'}
                rows.append(row);write_json(args.output/'runs.json',rows)
                write_json(args.output/f'{seed}-{method}-history.json',history);print(json.dumps(row),flush=True)
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='implementation_failure',error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
