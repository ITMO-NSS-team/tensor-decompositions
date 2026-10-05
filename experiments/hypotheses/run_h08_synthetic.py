"""H08 neural denoising after a frozen whole-procedure Gaussian threshold."""
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
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git, guard, sha, sync, tensor_hash, write_json
from run_cnn_local import synthetic, mse, latency
from run_h07_synthetic import FullTuckerConv
from run_h08_null_calibration import RANKS, initial_factors, sweep, core_and_residual, working_rule


def fit(x,ranks):
    candidates=[]
    for restart in [0,1]:
        factors=initial_factors(x,ranks,780000+restart)
        for _ in range(5):factors,_=sweep(x,factors,1.)
        core,residual=core_and_residual(x,factors)
        candidates.append((residual,core,factors))
    return min(candidates,key=lambda c:c[0])


def signal_and_noise(seed,device):
    gen=torch.Generator(device=device).manual_seed(seed+11000)
    factors=[torch.linalg.qr(torch.randn(n,r,generator=gen,device=device)).Q for n,r in zip((16,8,3,3),(4,3,2,2))]
    core=torch.randn(4,3,2,2,generator=gen,device=device)
    signal=tl.tenalg.multi_mode_dot(core,factors);signal=signal/signal.norm()
    rng=torch.Generator(device=device).manual_seed(seed+12000)
    noise=.1/(30**.5)*torch.randn(signal.shape,generator=rng,device=device)
    return signal,signal+noise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--null-run',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tl.set_backend('pytorch');torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name in [Path(__file__).name,'run_h08_null_calibration.py','run_h07_synthetic.py','run_cnn_local.py','run_local.py']:
        (args.output/(name+'.source')).write_bytes(Path(__file__).with_name(name).read_bytes())
    manifest={'state':'admission','hypothesis_id':'H08','setting':'synthetic','git_sha':git('rev-parse','HEAD'),
              'runner_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).with_name('H08_signal_denoising.md')),
              'torch':torch.__version__,'tensorly':tl.__version__,'device':args.device,'seeds':[11,22,33,44,55],
              'sigma':.1,'beta':1.,'noise_rule':'IID Gaussian sigma/sqrt(30), no random norm division',
              'decomposition_precision':'CPU FP64, matches null calibration','training_precision':'FP32',
              'calibration_steps':128,'recovery_steps':512,
              'limitations':['independent spatial-probe baseline pending; three of four protocol branches',
                             'only beta1/sigma0.1; adversarial and transfer noise modes pending',
                             'whole-procedure null calibration applies to this shape, not real activations',
                             'threshold cost reported separately; no free amortization or general speed claim'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        nm=json.loads((args.null_run/'manifest.json').read_text(encoding='utf-8'))
        assert nm['state']=='completed' and nm['runner_sha256']==sha(Path(__file__).with_name('run_h08_null_calibration.py'))
        threshold_info=json.loads((args.null_run/'threshold.json').read_text(encoding='utf-8'))
        threshold=threshold_info['threshold'];manifest['null_manifest_sha256']=sha(args.null_run/'manifest.json')
        manifest['threshold']=threshold;manifest['threshold_calibration_seconds']=threshold_info['calibration_seconds']
        manifest['state']='running';write_json(args.output/'manifest.json',manifest);rows=[]
        for seed in manifest['seeds']:
            teacher,splits=synthetic(seed,args.device);signal,noisy=signal_and_noise(seed,args.device)
            with torch.no_grad():teacher[2].weight.copy_(signal)
            with torch.no_grad():splits={k:(x,teacher(x)) for k,(x,_) in splits.items()}
            sd=args.output/f'seed-{seed}';sd.mkdir();torch.save(teacher.state_dict(),sd/'teacher.pt')
            torch.save({'signal':signal.cpu(),'noisy':noisy.cpu()},sd/'signal_and_noisy.pt')
            write_json(sd/'inputs.json',{k:{'x':tensor_hash(x),'y':tensor_hash(y)} for k,(x,y) in splits.items()})
            x=noisy.cpu().double()
            for method in ['known_rank','tail_energy','adaptive_null']:
                start=time.perf_counter();setup=time.perf_counter()
                if method=='known_rank':
                    residual,core,factors=fit(x,(4,3,2,2));detail={'rank':(4,3,2,2),'relative_noisy_residual':residual}
                elif method=='tail_energy':
                    possibilities=[]
                    for ranks in [*RANKS,(16,8,3,3)]:
                        residual,core,factors=fit(x,ranks)
                        possibilities.append((core.numel()+sum(f.numel() for f in factors),residual,core,factors,ranks))
                    eligible=[r for r in possibilities if r[1]<=.1]
                    _,residual,core,factors,ranks=min(eligible,key=lambda r:(r[0],r[4]))
                    detail={'rank':ranks,'relative_noisy_residual':residual,'candidate_count':4}
                else:
                    core,factors,detail=working_rule(x,threshold,sigma_entry=.1/(30**.5))
                    if core is None:
                        factors=initial_factors(x,RANKS[0],780000)
                        core=torch.zeros(*RANKS[0],dtype=x.dtype)
                        detail['training_parameter_shape']=RANKS[0]
                factor_seconds=time.perf_counter()-setup
                approximation=tl.tenalg.multi_mode_dot(core,factors)
                signal_error=float((approximation-signal.cpu().double()).norm()/signal.norm())
                student=copy.deepcopy(teacher)
                student[2]=FullTuckerConv(core.float().to(args.device),[f.float().to(args.device) for f in factors],teacher[2].bias)
                for parameter in student.parameters():parameter.requires_grad_(False)
                for parameter in student[2].parameters():parameter.requires_grad_(True)
                before=mse(student,splits['tuning']);opt=torch.optim.AdamW(student[2].parameters(),lr=.001,weight_decay=0)
                sync();train_start=time.perf_counter()
                for phase,steps,offset in [('calibration',128,50000),('recovery',512,60000)]:
                    xx,yy=splits[phase];gen=torch.Generator(device=args.device).manual_seed(seed+offset)
                    for step in range(steps):
                        ix=torch.randint(len(xx),(32,),generator=gen,device=args.device)
                        opt.zero_grad(set_to_none=True);loss=F.mse_loss(student(xx[ix]),yy[ix])
                        if not torch.isfinite(loss):raise ArithmeticError('nonfinite H08 recovery')
                        loss.backward();opt.step()
                        if step%64==0:guard(start,600)
                sync();recovery_seconds=time.perf_counter()-train_start
                recovered=tl.tenalg.multi_mode_dot(student[2].core,list(student[2].factors))
                row={'seed':seed,'method':method,**detail,'signal_relative_error_before':signal_error,
                     'signal_relative_error_after':float((recovered-signal).norm()/signal.norm()),
                     'tuning_before':before,'final_test':mse(student,splits['test']),
                     'factor_seconds':factor_seconds,'recovery_seconds':recovery_seconds,
                     'threshold_calibration_seconds':threshold_info['calibration_seconds'] if method=='adaptive_null' else 0,
                     'total_seconds':time.perf_counter()-start,'latency':latency(student,splits['tuning'][0][:64]),
                     'status':'completed'}
                torch.save(student.state_dict(),sd/f'{method}.pt');row['checkpoint_sha256']=sha(sd/f'{method}.pt')
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
