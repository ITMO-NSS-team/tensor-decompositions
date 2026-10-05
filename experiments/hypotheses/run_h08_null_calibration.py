"""H08 calibrate the entire fixed adaptive HOOI path on Gaussian null tensors."""
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
import tensorly as tl
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git, sha, write_json

RANKS=((2,2,1,1),(4,3,2,2),(8,6,3,3))


def initial_factors(x,ranks,seed):
    gen=torch.Generator(device=x.device).manual_seed(seed)
    result=[]
    for mode,rank in enumerate(ranks):
        matrix=tl.unfold(x,mode)
        width=min(rank+8,min(matrix.shape))
        omega=torch.randn(matrix.shape[1],width,generator=gen,device=x.device,dtype=x.dtype)
        q=torch.linalg.qr(matrix@omega,mode='reduced').Q
        z=torch.linalg.qr(matrix.T@q,mode='reduced').Q
        q=torch.linalg.qr(matrix@z,mode='reduced').Q
        u,_,_=torch.linalg.svd(q.T@matrix,full_matrices=False)
        result.append((q@u)[:,:rank])
    return result


def sweep(x,factors,sigma_entry):
    maximum=0.
    for mode in range(x.ndim):
        others=[k for k in range(x.ndim) if k!=mode]
        contracted=tl.tenalg.multi_mode_dot(x,[factors[k].T for k in others],modes=others)
        unfolded=tl.unfold(contracted,mode)
        u,s,_=torch.linalg.svd(unfolded,full_matrices=False)
        statistic=float(s[0])/(sigma_entry*(math.sqrt(unfolded.shape[0])+math.sqrt(unfolded.shape[1])))
        maximum=max(maximum,statistic)
        factors[mode]=u[:,:factors[mode].shape[1]]
    return factors,maximum


def core_and_residual(x,factors):
    core=tl.tenalg.multi_mode_dot(x,[f.T for f in factors])
    reconstructed=tl.tenalg.multi_mode_dot(core,factors)
    residual=float((x-reconstructed).norm()/x.norm().clamp_min(1e-30))
    return core,residual


def path(x,ranks,restart,passes=5):
    factors=initial_factors(x,ranks,780000+restart)
    maximum=0.;history=[]
    for step in range(passes):
        factors,stat=sweep(x,factors,1.)
        maximum=max(maximum,stat);history.append(stat)
    return maximum,history


def full_search(x,rank_candidates=RANKS):
    results=[]
    for ranks in rank_candidates:
        for restart in [0,1]:
            maximum,history=path(x,ranks,restart)
            results.append({'ranks':ranks,'restart':restart,'max_statistic':maximum,'history':history})
    return max(r['max_statistic'] for r in results),results


def working_rule(x,threshold,sigma_entry=1.,rank_candidates=RANKS):
    candidates=[]
    for ranks in rank_candidates:
        for restart in [0,1]:
            factors=initial_factors(x,ranks,780000+restart)
            factors,stat=sweep(x,factors,sigma_entry)
            core,residual=core_and_residual(x,factors)
            parameters=core.numel()+sum(f.numel() for f in factors)
            if stat>threshold:
                candidates.append({'ranks':ranks,'restart':restart,'factors':factors,'statistic':stat,
                                   'parameters':parameters,'residual':residual})
    if not candidates:
        return None,[],{'detected':False,'rank':None,'passes':1,'max_search_paths':2*len(rank_candidates),'reconstruction_energy':0.}
    ranks=min(candidates,key=lambda r:(r['parameters'],r['ranks']))['ranks']
    best=max((r for r in candidates if r['ranks']==ranks),key=lambda r:r['statistic'])
    factors=best['factors'];old=best['residual'];passes=1
    for _ in range(4):
        factors,_=sweep(x,factors,sigma_entry);core,new=core_and_residual(x,factors);passes+=1
        if abs(new-old)/max(abs(old),1e-30)<1e-4:break
        old=new
    core,_=core_and_residual(x,factors)
    return core,factors,{'detected':True,'rank':ranks,'restart':best['restart'],'passes':passes,
                         'parameters':best['parameters'],'reconstruction_energy':float(core.square().sum())}


def admission():
    gen=torch.Generator().manual_seed(15)
    x=torch.randn(16,8,3,3,generator=gen,dtype=torch.float64)
    maximum,history=full_search(x)
    assert len(history)==6 and all(len(r['history'])==5 for r in history)
    _,_,rejected=working_rule(x,maximum+1.)
    assert not rejected['detected']
    _,factors,detected=working_rule(x,0.)
    assert detected['detected'] and 1<=detected['passes']<=5
    for factor in factors:
        torch.testing.assert_close(factor.T@factor,torch.eye(factor.shape[1],dtype=factor.dtype),atol=1e-10,rtol=1e-10)
    twice,_=full_search(2*x)
    assert abs(twice/maximum-2)<1e-10
    return {'full_envelope_six_paths_five_sweeps':'passed','high_threshold_returns_zero':'passed',
            'selection_and_orthogonality':'passed','spectral_statistic_scales_with_noise':'passed'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--admission-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);tl.set_backend('pytorch')
    (args.output/'run_h08_null_calibration.py.source').write_bytes(Path(__file__).read_bytes())
    manifest={'state':'admission','hypothesis_id':'H08','stage':'whole-procedure Gaussian null calibration',
              'git_sha':git('rev-parse','HEAD'),'runner_sha256':sha(__file__),
              'protocol_sha256':sha(Path(__file__).with_name('H08_signal_denoising.md')),
              'torch':torch.__version__,'tensorly':tl.__version__,'device':'cpu','precision':'FP64',
              'shape':[16,8,3,3],'ranks':RANKS,'restarts':2,'max_sweeps':5,
              'initial_range_finder':{'oversampling':8,'power':1,'restart_seeds':[780000,780001]},
              'calibration_seeds':list(range(910000,910200)),'validation_seeds':list(range(920000,920200)),
              'limitations':['orthonormal modal contraction statistic is a declared concrete HOOI interpretation',
                             'not an MP asymptotic threshold or finite-sample FPR guarantee',
                             'shape-specific threshold; real activations require separate shapes and calibration',
                             'neural denoising and independent-probe baseline are separate stages'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',admission())
        if args.admission_only:manifest['state']='admission_passed';return
        manifest['state']='running';write_json(args.output/'manifest.json',manifest);values=[];start=time.perf_counter()
        for seed in manifest['calibration_seeds']:
            x=torch.randn(*manifest['shape'],generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
            maximum,_=full_search(x);values.append({'seed':seed,'max_statistic':maximum})
            if len(values)%25==0:
                write_json(args.output/'calibration.json',values);print(json.dumps({'calibrated':len(values)}),flush=True)
        threshold=float(np.quantile([r['max_statistic'] for r in values],.95,method='higher'))
        calibration_seconds=time.perf_counter()-start
        write_json(args.output/'threshold.json',{'threshold':threshold,'quantile':.95,'method':'higher',
                   'frozen_before_validation':True,'calibration_seconds':calibration_seconds,'noise_scale':1.})
        rows=[]
        for seed in manifest['validation_seeds']:
            x=torch.randn(*manifest['shape'],generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
            _,_,info=working_rule(x,threshold)
            rows.append({'seed':seed,**info,'energy_over_expected_noise':info['reconstruction_energy']/1152})
        k=sum(r['detected'] for r in rows);n=len(rows)
        interval=[0. if not k else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
        write_json(args.output/'validation.json',rows)
        summary={'false_detections':k,'n':n,'rate':k/n,'clopper_pearson_95':interval,
                 'upper_bound_le_005':interval[1]<=.05,'threshold':threshold,'seconds':time.perf_counter()-start}
        write_json(args.output/'summary.json',summary);print(json.dumps(summary),flush=True);manifest['state']='completed'
    except Exception as error:
        manifest.update(state='implementation_failure',error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
