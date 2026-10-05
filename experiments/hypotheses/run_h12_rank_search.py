"""H12 bounded functional/fractional/energy rank search with explicit search cost."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import traceback

import tensorly as tl
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git,guard,sha,sync,write_json
from run_cnn_local import synthetic,mse,latency
from run_h07_synthetic import FullTuckerConv
from tdecomp.tensor.tucker import HOOIDecomposition
from run_h06_synthetic import equal_budget
from run_h02_synthetic import tensor_hash

FUNCTIONAL=[(2,2,1,1),(4,2,2,2),(4,3,2,2),(8,3,2,2),(8,6,3,3),(16,8,3,3)]
FRACTIONS=[.125,.25,.375,.5,.625,.75]
EPSILONS=[.03,.05,.10,.15,.20,.30]


def update(student,pair,seed,steps,lr=.001,opt=None):
    x,y=pair;gen=torch.Generator(device=x.device).manual_seed(seed)
    if opt is None:opt=torch.optim.AdamW(student[2].parameters(),lr=lr,weight_decay=0)
    sync();start=time.perf_counter()
    for _ in range(steps):
        ix=torch.randint(len(x),(32,),generator=gen,device=x.device)
        opt.zero_grad(set_to_none=True);loss=F.mse_loss(student(x[ix]),y[ix])
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite H12 update')
        loss.backward();opt.step();guard(start,600)
    sync();return time.perf_counter()-start


def make_candidate(teacher,ranks,seed):
    weight=teacher[2].weight.detach()
    core,factors=HOOIDecomposition(rank=ranks,random_state=seed).decompose(weight,n_iter_max=20,tol=1e-6)
    student=copy.deepcopy(teacher);student[2]=FullTuckerConv(core,factors,teacher[2].bias)
    for p in student.parameters():p.requires_grad_(False)
    for p in student[2].parameters():p.requires_grad_(True)
    return student


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);tl.set_backend('pytorch');torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source_names=['run_h12_rank_search.py','run_cnn_local.py','run_h07_synthetic.py',
                  'run_h06_synthetic.py','synthetic_tucker_common.py','run_h02_synthetic.py','run_local.py']
    source_hashes={}
    for name in source_names:
        path=Path(__file__).with_name(name)
        (args.output/(name+'.source')).write_bytes(path.read_bytes())
        source_hashes[name]=sha(path)
    manifest={'state':'running','hypothesis_id':'H12','setting':'synthetic','git_sha':git('rev-parse','HEAD'),
              'runner_sha256':sha(__file__),'source_hashes':source_hashes,'protocol_sha256':sha(Path(__file__).with_name('H12_functional_rank_rgn.md')),
              'torch':torch.__version__,'tensorly':tl.__version__,'device':args.device,'seeds':[11,22,33,44,55],
              'functional_ranks':FUNCTIONAL,'fractions':FRACTIONS,'energy_epsilons':EPSILONS,
              'candidate_calibration_steps':32,'selected_calibration_steps':128,'recovery_steps':512,
              'quality_normalized_mse':.01,'amortization_images':10000,
              'limitations':['search objective uses remaining-training cost extrapolated from32 steps; exact C optimum not certified',
                             'each rule selects independently on tuning; no test selection',
                             'nuclear ADMM is separate pending secondary branch',
                             'RGN separately evaluated at fixed rank; not used as rank estimator',
                             'candidate has32 steps before selected128/512; that extra cost is in search',
                             'temporary dense Tucker reconstruction counted in measured execution'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        rows=[];search=[]
        for seed in manifest['seeds']:
            teacher,splits=synthetic(seed,args.device);sd=args.output/f'seed-{seed}';sd.mkdir()
            torch.save(teacher.state_dict(),sd/'teacher.pt')
            write_json(sd/'inputs.json',{name:{'x':tensor_hash(x),'y':tensor_hash(y)}
                                       for name,(x,y) in splits.items()})
            for rule in ['functional','fractional','energy']:
                search_start=time.perf_counter();candidates=[];models=[]
                for index in range(6):
                    start=time.perf_counter()
                    if rule=='functional':ranks=FUNCTIONAL[index]
                    elif rule=='fractional':ranks=tuple(min(n,max(1,math.ceil(n*FRACTIONS[index]))) for n in (16,8,3,3))
                    else:
                        _,_,info=equal_budget(teacher[2].weight.detach(),sequential=False,epsilon=EPSILONS[index])
                        ranks=tuple(info['ranks'])
                    sync();factor_start=time.perf_counter();student=make_candidate(teacher,ranks,seed);sync()
                    factor_seconds=time.perf_counter()-factor_start
                    calibration_seconds=update(student,splits['calibration'],seed+50000,32)
                    quality=mse(student,splits['tuning']);timing=latency(student,splits['tuning'][0][:1])
                    predicted_recovery_seconds=calibration_seconds*640/32
                    predicted_cost=predicted_recovery_seconds/10000+timing['p50_ms']/1000
                    row={'seed':seed,'rule':rule,'index':index,'rank':ranks,'quality':quality,
                         'admissible':quality['normalized_mse']<=.01,'factor_seconds':factor_seconds,
                         'candidate_seconds':time.perf_counter()-start,'calibration_seconds':calibration_seconds,
                         'predicted_remaining_recovery_seconds':predicted_recovery_seconds,
                         'predicted_cost_without_common_search':predicted_cost,'latency_batch1':timing,
                         'factor_parameters':sum(p.numel() for p in student[2].parameters())}
                    candidates.append(row);models.append(student);search.append(row)
                    write_json(args.output/'search.json',search)
                eligible=[index for index,c in enumerate(candidates) if c['admissible']]
                fallback=not eligible
                if eligible:chosen=min(eligible,key=lambda i:(candidates[i]['predicted_cost_without_common_search'],candidates[i]['rank']))
                else:
                    # Full-rank reserve is explicit, never described as successful compression.
                    sync();reserve_start=time.perf_counter()
                    student=make_candidate(teacher,(16,8,3,3),seed)
                    sync();reserve_factor_seconds=time.perf_counter()-reserve_start
                    reserve={'seed':seed,'rule':rule,'rank':(16,8,3,3),'index':'reserve',
                             'calibration_seconds':0,'calibration_steps':0,
                             'factor_seconds':reserve_factor_seconds,'quality':mse(student,splits['tuning']),
                             'scope':'untrained full-rank safety reserve; no seventh candidate training'}
                    candidates.append(reserve);search.append(reserve)
                    write_json(args.output/'search.json',search)
                    models.append(student);chosen=len(models)-1
                sync();search_seconds=time.perf_counter()-search_start
                student=models[chosen];selection=candidates[chosen]
                write_json(sd/f'{rule}-selection.json',{'selected':selection,'reserve_full_rank':fallback,
                           'search_seconds':search_seconds,'frozen_before_test':True,'objective':'forecast of C with measured candidate32-step cost'})
                for i,model in enumerate(models):
                    if i!=chosen:models[i]=None
                before=mse(student,splits['tuning']);start=time.perf_counter()
                selected_opt=torch.optim.AdamW(student[2].parameters(),lr=.001,weight_decay=0)
                calibration_seconds=update(student,splits['calibration'],seed+51000,128,opt=selected_opt)
                recovery_seconds=update(student,splits['recovery'],seed+60000,512,opt=selected_opt)
                student.eval();tuning_after=mse(student,splits['tuning'])
                timing=latency(student,splits['tuning'][0][:1])
                cp=sd/f'{rule}.pt';torch.save(student.state_dict(),cp)
                checkpoint_sha256=sha(cp)
                final=mse(student,splits['test'])
                measured_c=(search_seconds+calibration_seconds+recovery_seconds)/10000+timing['p50_ms']/1000
                row={'seed':seed,'rule':rule,'selected_rank':selection['rank'],'full_rank_reserve':fallback,
                     'tuning_before':before,'tuning_after':tuning_after,'final_test':final,
                     'search_seconds':search_seconds,'selected_calibration_seconds':calibration_seconds,
                     'recovery_seconds':recovery_seconds,'measured_cost_seconds_per_image':measured_c,
                     'latency_batch1':timing,'quality_passed':tuning_after['normalized_mse']<=.01,
                     'checkpoint_sha256':checkpoint_sha256,'status':'completed'}
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
                guard(start,600);del student,models,candidates
                if args.device=='cuda':torch.cuda.empty_cache()
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
