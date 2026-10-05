"""H11 implicit sketch reliability and neural recovery on real ResNet-18 weights."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.stats import beta
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets,models,transforms

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git,gpu_snapshot,guard,sha,sync,write_json
from run_h04_real import MatrixConv,evaluate,latency,recover,seed_all
from run_h11_confirmation import admission,factors

METHODS=['gaussian','countsketch','sparse_jl','srht_reference']


def reference(x,rank=64):
    u,s,vh=torch.linalg.svd(x,full_matrices=False)
    return u[:,:rank]*s[:rank],vh[:rank]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for name in [Path(__file__).name,'run_h11_confirmation.py','run_h04_real.py','run_local.py']:
        (args.output/(name+'.source')).write_bytes(Path(__file__).with_name(name).read_bytes())
    manifest={'state':'admission','hypothesis_id':'H11','setting':'real','git_sha':git('rev-parse','HEAD'),
              'git_status':git('status','--porcelain'),'runner_sha256':sha(__file__),
              'protocol_sha256':sha(Path(__file__).with_name('H11_structured_sketches.md')),
              'torch':torch.__version__,'torchvision':torchvision.__version__,'python':sys.version,
              'cuda':torch.version.cuda,'device':device,'gpu_snapshot':gpu_snapshot(),'seeds':[101,202,303],
              'rank':64,'ell_candidates':[72,80,96,128],'power':1,'pilot_trials_per_point':10,
              'confirmation_trials_per_method':100,'microbatch':128,'calibration_steps':128,'recovery_steps':400,
              'final_test_opened':False,'command':sys.argv,'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'limitations':['width selection and reliability on fixed seed101 checkpoint only',
                             'implicit Python operators; scatter and parameter generation included',
                             '10-call cached-operator benchmark pending; these are fresh-operator calls',
                             'only compressed layer trained; BN and remaining weights frozen',
                             'three paired neural seeds; no global hypothesis confirmation',
                             'WDDM process peak unavailable; conservative device-usage guard']}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'operator_admission.json',admission())
        bm=json.loads((args.baseline/'manifest.json').read_text(encoding='utf-8'))
        assert bm['state']=='completed' and sha(args.data/'cifar-10-python.tar.gz')==bm['data_archive_sha256']
        assert sha(args.baseline/'splits.json')==bm['split_sha256']
        splits=json.loads((args.baseline/'splits.json').read_text(encoding='utf-8'));norm=bm['normalization']
        plain=transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm['mean'],norm['std'])])
        aug=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),transforms.Normalize(norm['mean'],norm['std'])])
        dp=datasets.CIFAR10(args.data,train=True,download=False,transform=plain)
        da=datasets.CIFAR10(args.data,train=True,download=False,transform=aug)
        tune=DataLoader(Subset(dp,splits['tuning']),batch_size=128,shuffle=False,num_workers=0)
        manifest.update(data_archive_sha256=bm['data_archive_sha256'],split_sha256=bm['split_sha256'],normalization=norm)
        checkpoint_hashes={}
        for seed in manifest['seeds']:
            result=json.loads((args.baseline/f'seed-{seed}'/'result.json').read_text(encoding='utf-8'))
            assert result['state']=='baseline_ready'
            cp=args.baseline/f'seed-{seed}'/'model.pt';assert sha(cp)==result['checkpoint_sha256']
            checkpoint_hashes[str(seed)]=sha(cp)
        manifest['baseline_checkpoint_hashes']=checkpoint_hashes;manifest['state']='width_selection'
        write_json(args.output/'manifest.json',manifest)
        pretrained=torch.load(args.baseline/'seed-101'/'model.pt',map_location=device,weights_only=True)
        x=pretrained['layer3.0.conv2.weight'].reshape(256,2304)
        a,b=reference(x);exact_error=float((x-a@b).norm()/x.norm());del a,b
        pilot=[];selected={};start=time.perf_counter()
        for method in METHODS:
            for ell in manifest['ell_candidates']:
                for trial in range(10):
                    sync();before=time.perf_counter();a,b=factors(x,ell,method,500000+trial,64);sync()
                    error=float((x-a@b).norm()/x.norm())
                    pilot.append({'method':method,'ell':ell,'seed':500000+trial,'relative_error':error,
                                  'exact_error':exact_error,'failure':error>max(1.05*exact_error,1e-6),
                                  'total_seconds':time.perf_counter()-before})
                write_json(args.output/'pilot.json',pilot);guard(start,3600)
            admissible=[ell for ell in manifest['ell_candidates'] if not any(r['failure'] for r in pilot if r['method']==method and r['ell']==ell)]
            if not admissible:raise ArithmeticError(f'no admissible pilot width for {method}; retain failed pilot')
            selected[method]=min(admissible)
        write_json(args.output/'selection.json',{'ell_by_method':selected,'frozen_before_confirmation':True,
                   'criterion':'smallest width with zero pilot failures, not a reliability guarantee','checkpoint_seed':101})
        manifest['state']='reliability';write_json(args.output/'manifest.json',manifest);reliability=[];summaries=[]
        for method in METHODS:
            method_start=time.perf_counter()
            for trial in range(100):
                sync();before=time.perf_counter();a,b=factors(x,selected[method],method,600000+trial,64);sync()
                error=float((x-a@b).norm()/x.norm())
                reliability.append({'method':method,'ell':selected[method],'seed':600000+trial,'relative_error':error,
                                    'failure':error>max(1.05*exact_error,1e-6),'seconds':time.perf_counter()-before})
                guard(method_start,900)
            group=[r for r in reliability if r['method']==method];k=sum(r['failure'] for r in group)
            ci=[0. if not k else float(beta.ppf(.025,k,101-k)),1. if k==100 else float(beta.ppf(.975,k+1,100-k))]
            summaries.append({'method':method,'ell':selected[method],'failures':k,'n':100,'clopper_pearson_95':ci,
                              'upper_bound_le_005':ci[1]<=.05})
            write_json(args.output/'reliability.json',reliability);write_json(args.output/'reliability_summary.json',summaries)
            print(json.dumps(summaries[-1]),flush=True)
        del pretrained,x,a,b
        if device=='cuda':torch.cuda.empty_cache()
        rows=[];manifest['state']='neural_recovery';write_json(args.output/'manifest.json',manifest)
        for seed in manifest['seeds']:
            teacher=models.resnet18(weights=None,num_classes=10).to(device)
            teacher.load_state_dict(torch.load(args.baseline/f'seed-{seed}'/'model.pt',map_location=device,weights_only=True))
            teacher.eval();w=teacher.layer3[0].conv2.weight.detach().reshape(256,2304);sd=args.output/f'seed-{seed}';sd.mkdir()
            for method in ['exact_svd',*METHODS]:
                start=time.perf_counter();seed_all(seed);sync();begin=time.perf_counter()
                if method=='exact_svd':a,b=reference(w)
                else:a,b=factors(w,selected[method],method,700000+seed,64)
                sync();setup_seconds=time.perf_counter()-begin
                student=copy.deepcopy(teacher);student.layer3[0].conv2=MatrixConv(a,b)
                before=evaluate(student,tune,device);sync();begin=time.perf_counter()
                history=recover(student,{'calibration':dp,'recovery':da},splits,seed,device,start)
                sync();recovery_seconds=time.perf_counter()-begin
                cp=sd/f'{method}.pt';torch.save(student.state_dict(),cp)
                write_json(sd/f'{method}-history.json',history)
                row={'seed':seed,'method':method,'ell':selected.get(method),'rank':64,'setup_seconds':setup_seconds,
                     'recovery_seconds':recovery_seconds,'tuning_before':before,'tuning_after':evaluate(student,tune,device),
                     'checkpoint_sha256':sha(cp),'status':'completed'}
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
                del student,a,b
                if device=='cuda':torch.cuda.empty_cache()
            del teacher,w
        # All widths, methods and checkpoints now fixed, no selection on official test.
        manifest['final_test_opened']=True;manifest['state']='final_evaluation';write_json(args.output/'manifest.json',manifest)
        testdata=datasets.CIFAR10(args.data,train=False,download=False,transform=plain)
        test=DataLoader(testdata,batch_size=128,shuffle=False,num_workers=0);final=[]
        inputs=next(iter(tune))[0][:64].to(device)
        for seed in manifest['seeds']:
            for method in ['original','exact_svd',*METHODS]:
                start=time.perf_counter();student=models.resnet18(weights=None,num_classes=10).to(device)
                if method=='original':cp=args.baseline/f'seed-{seed}'/'model.pt'
                else:
                    cp=args.output/f'seed-{seed}'/f'{method}.pt'
                    state=torch.load(cp,map_location=device,weights_only=True);prefix='layer3.0.conv2.'
                    student.layer3[0].conv2=MatrixConv(state[prefix+'a'][:,:,0,0],state[prefix+'b'].reshape(64,2304))
                student.load_state_dict(torch.load(cp,map_location=device,weights_only=True))
                row={'seed':seed,'method':method,'final_test':evaluate(student,test,device),
                     'latency':[latency(student,inputs[:batch],device,True) for batch in [64,1]]}
                final.append(row);write_json(args.output/'final.json',final);print(json.dumps(row),flush=True)
                guard(start,2700);del student
                if device=='cuda':torch.cuda.empty_cache()
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
