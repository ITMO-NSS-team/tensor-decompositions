"""H01 real input-weighted convolution compression, paired with common checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import psutil
import tensorly as tl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, models, transforms

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import factorize, git, gpu_snapshot, guard, invariants, sha, sync, write_json
from run_h04_real import MatrixConv, evaluate, latency, recover, seed_all


@torch.no_grad()
def moment(model,loader,device):
    matrix=torch.zeros(2304,2304,device=device);count=0
    def collect(_,inputs):
        nonlocal count
        patches=F.unfold(inputs[0].float(),kernel_size=3,padding=1).transpose(1,2).reshape(-1,2304)
        matrix.add_(patches.T@patches);count+=len(patches)
    hook=model.layer3[0].conv2.register_forward_pre_hook(collect)
    model.eval();start=time.perf_counter()
    try:
        for x,_ in loader:
            model(x.to(device));guard(start,2700)
    finally:hook.remove()
    if count!=8000:raise AssertionError(f'expected 8000 windows; got {count}')
    return matrix/count,count


@torch.no_grad()
def calibrate_bn(model,loader,device):
    """Calibrate the affected teacher BN once; copy common buffers to all students."""
    model.eval();batchnorms=[]
    module=model.layer3[0].bn2
    batchnorms.append((module,module.momentum));module.reset_running_stats()
    module.momentum=None;module.train()
    try:
        for x,_ in loader:
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                model(x.to(device))
    finally:
        for module,old in batchnorms:module.momentum=old;module.eval()
    model.eval()


@torch.no_grad()
def errors(teacher,student,loader,device):
    teacher.eval();student.eval();saved={};totals={k:[0.,0.,0] for k in ['layer','block','network']}
    hooks=[]
    for label,model in [('teacher',teacher),('student',student)]:
        for kind,module in [('layer',model.layer3[0].conv2),('block',model.layer3[0])]:
            def save(_,inputs,output,key=(label,kind)):saved[key]=output.detach().float()
            hooks.append(module.register_forward_hook(save))
    try:
        for x,_ in loader:
            x=x.to(device)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                ref=teacher(x).float();prediction=student(x).float()
            for kind in totals:
                a,b=(ref,prediction) if kind=='network' else (saved['teacher',kind],saved['student',kind])
                totals[kind][0]+=float((a-b).square().sum())
                totals[kind][1]+=float(a.square().sum());totals[kind][2]+=a.numel()
    finally:
        for hook in hooks:hook.remove()
    return {k:{'mse':e/n,'normalized_mse':e/max(energy,1e-30),'n_scalars':n}
            for k,(e,energy,n) in totals.items()}


def support_solution(w,m,rank):
    values,vectors=torch.linalg.eigh((m+m.T)/2)
    cutoff=float(values.max())*1e-6;keep=values>cutoff
    q=vectors[:,keep];root=values[keep].sqrt()
    u,s,vh=torch.linalg.svd((w@q)*root,full_matrices=False)
    a=u[:,:rank]*s[:rank];b=(vh[:rank]/root)@q.T
    return a,b,{'cutoff':cutoff,'support_dimension':int(keep.sum()),'ridge':0.,
                'scope':'numerical support diagnostic; threshold changes estimated support'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--pilot-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in [Path(__file__).name,'run_h04_real.py','run_local.py']:
        (args.output/(name+'.source')).write_bytes(Path(__file__).with_name(name).read_bytes())
    tl.set_backend('pytorch');torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest={'run_id':args.output.name,'hypothesis_id':'H01','setting':'real','state':'admission',
              'git_sha':git('rev-parse','HEAD'),'git_status':git('status','--porcelain'),
              'runner_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).with_name('H01_input_distribution.md')),
              'torch':torch.__version__,'torchvision':torchvision.__version__,'tensorly':tl.__version__,
              'python':sys.version,'cuda':torch.version.cuda,'device':device,'pid':os.getpid(),
              'gpu_snapshot':gpu_snapshot(),'seeds':[101,202,303],'rank_grid':[32,64,96],
              'ridge_rule':'1e-4 * trace(M) / 2304','microbatch':128,'calibration_steps':128,
              'recovery_steps':400,'final_test_opened':False,'command':sys.argv,
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'limitations':['compressed layer trainable; remaining parameters frozen',
                             'affected teacher BN calibrated once; common buffers copied and frozen in all branches',
                             'support-only solution diagnosed before recovery, not full secondary neural branch',
                             'three paired model seeds; exploratory uncertainty',
                             'torchvision differs from protocol interface reference; actual version recorded',
                             'WDDM process memory unavailable; total-device conservative guard']}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',invariants())
        bm=json.loads((args.baseline/'manifest.json').read_text(encoding='utf-8'))
        assert bm['state']=='completed'
        assert sha(args.data/'cifar-10-python.tar.gz')==bm['data_archive_sha256']
        assert sha(args.baseline/'splits.json')==bm['split_sha256']
        splits=json.loads((args.baseline/'splits.json').read_text(encoding='utf-8'))
        norm=bm['normalization'];plain=transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm['mean'],norm['std'])])
        aug=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),transforms.Normalize(norm['mean'],norm['std'])])
        dp=datasets.CIFAR10(args.data,train=True,download=False,transform=plain)
        da=datasets.CIFAR10(args.data,train=True,download=False,transform=aug)
        cal=DataLoader(Subset(dp,splits['calibration']),batch_size=128,shuffle=False,num_workers=0)
        tune=DataLoader(Subset(dp,splits['tuning']),batch_size=128,shuffle=False,num_workers=0)
        manifest.update(data_archive_sha256=bm['data_archive_sha256'],split_sha256=bm['split_sha256'],normalization=norm)
        rows=[];selected=None;checkpoint_hashes={}
        for seed in manifest['seeds']:
            result=json.loads((args.baseline/f'seed-{seed}'/'result.json').read_text(encoding='utf-8'))
            checkpoint=args.baseline/f'seed-{seed}'/'model.pt'
            assert result['state']=='baseline_ready' and sha(checkpoint)==result['checkpoint_sha256']
            checkpoint_hashes[str(seed)]=sha(checkpoint)
        manifest['baseline_checkpoint_hashes']=checkpoint_hashes;manifest['state']='running'
        write_json(args.output/'manifest.json',manifest)
        for seed in manifest['seeds']:
            seed_all(seed);teacher=models.resnet18(weights=None,num_classes=10).to(device)
            teacher.load_state_dict(torch.load(args.baseline/f'seed-{seed}'/'model.pt',map_location=device,weights_only=True))
            teacher.eval();sd=args.output/f'seed-{seed}';sd.mkdir()
            common_bn_start=time.perf_counter();calibrate_bn(teacher,cal,device);sync()
            common_bn_seconds=time.perf_counter()-common_bn_start
            calibrated_tuning=evaluate(teacher,tune,device)
            write_json(sd/'teacher_calibrated_tuning.json',calibrated_tuning)
            if calibrated_tuning['accuracy']<.70:
                raise ArithmeticError('common-BN teacher does not pass 70% tuning admission')
            torch.save(teacher.state_dict(),sd/'teacher_calibrated.pt')
            write_json(sd/'common_bn.json',{'policy':'teacher layer3.0.bn2 cumulative calibration; copied unchanged to all students',
                        'seconds':common_bn_seconds,'checkpoint_sha256':sha(sd/'teacher_calibrated.pt')})
            sync();stat_start=time.perf_counter();m,count=moment(teacher,cal,device);sync()
            stat_seconds=time.perf_counter()-stat_start;ridge=1e-4*float(torch.trace(m))/2304
            conditioned=m+ridge*torch.eye(2304,device=device)
            torch.save(m,sd/'moment.pt');write_json(sd/'moment.json',{'n_windows':count,'ridge':ridge,
                          'seconds':stat_seconds,'sha256':sha(sd/'moment.pt'),'uncentered':True})
            w=teacher.layer3[0].conv2.weight.detach().reshape(256,-1)
            a,b,support=support_solution(w,m,64)
            support['weighted_unregularized_risk']=float(((w-a@b)@m*(w-a@b)).sum())
            write_json(sd/'support_diagnostic.json',support)
            del a,b
            ranks=[32,64,96] if selected is None else [selected]
            methods=['svd','weighted_svd','weighted_rsvd'] if selected is None else ['svd','weighted_svd','weighted_rsvd','haar']
            if args.pilot_only:ranks=[64];methods=['svd','weighted_svd']
            for rank in ranks:
                for method in methods:
                    start=time.perf_counter();seed_all(seed);sync();setup=time.perf_counter()
                    if device=='cuda':torch.cuda.reset_peak_memory_stats()
                    a,b=factorize(w,rank,method,conditioned,seed=seed)
                    sync();setup_seconds=time.perf_counter()-setup
                    student=copy.deepcopy(teacher);student.layer3[0].conv2=MatrixConv(a,b)
                    before=errors(teacher,student,tune,device)
                    bn_seconds=common_bn_seconds
                    tuning_before=evaluate(student,tune,device)
                    recovery_start=time.perf_counter()
                    history=recover(student,{'calibration':dp,'recovery':da},splits,seed,device,start,
                                    steps=(20,0) if args.pilot_only else (128,400))
                    sync();recovery_seconds=time.perf_counter()-recovery_start
                    tuning=evaluate(student,tune,device);after=errors(teacher,student,tune,device)
                    cp=sd/f'{method}-r{rank}.pt';torch.save(student.state_dict(),cp)
                    write_json(sd/f'{method}-r{rank}-history.json',history)
                    row={'seed':seed,'method':method,'rank':rank,'ridge':ridge if method.startswith('weighted') else 0,
                         'moment_seconds':stat_seconds if method.startswith('weighted') else 0,
                         'setup_seconds':setup_seconds,'bn_seconds':bn_seconds,'recovery_seconds':recovery_seconds,
                         'elapsed_seconds':time.perf_counter()-start,'tuning_before':tuning_before,'tuning_after':tuning,
                         'errors_before':before,'errors_after':after,'parameters':rank*2560,
                         'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                         'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0,
                         'rss_bytes':psutil.Process().memory_info().rss,
                         'checkpoint_sha256':sha(cp),'status':'pilot_completed' if args.pilot_only else 'completed'}
                    rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
                    del student,a,b
                    if device=='cuda':torch.cuda.empty_cache()
                    guard(start,2700)
            if selected is None and not args.pilot_only:
                candidates=[r for r in rows if r['method']=='weighted_svd']
                selected=max(candidates,key=lambda r:(r['tuning_after']['accuracy'],-r['rank']))['rank']
                write_json(args.output/'selection.json',{'selected_rank':selected,'seed':101,'ridge_rule':manifest['ridge_rule'],
                           'criterion':'weighted SVD tuning accuracy; smaller rank breaks ties','frozen_before_test':True})
                # Haar is not part of the nine-point tuning grid; run the frozen rank now.
                start=time.perf_counter();sync();setup_start=time.perf_counter()
                a,b=factorize(w,selected,'haar',seed=seed);sync()
                setup_seconds=time.perf_counter()-setup_start
                student=copy.deepcopy(teacher);student.layer3[0].conv2=MatrixConv(a,b)
                before=errors(teacher,student,tune,device);tuning_before=evaluate(student,tune,device)
                recovery_start=time.perf_counter()
                history=recover(student,{'calibration':dp,'recovery':da},splits,seed,device,start)
                sync();recovery_seconds=time.perf_counter()-recovery_start
                write_json(sd/f'haar-r{selected}-history.json',history)
                cp=sd/f'haar-r{selected}.pt';torch.save(student.state_dict(),cp)
                row={'seed':seed,'method':'haar','rank':selected,'tuning_after':evaluate(student,tune,device),
                     'tuning_before':tuning_before,'errors_before':before,'errors_after':errors(teacher,student,tune,device),
                     'setup_seconds':setup_seconds,'recovery_seconds':recovery_seconds,'bn_seconds':common_bn_seconds,
                     'moment_seconds':0,
                     'elapsed_seconds':time.perf_counter()-start,'checkpoint_sha256':sha(cp),'status':'completed',
                     'parameters':selected*2560,'ridge':0,'scope':'frozen-rank control; no tuning selection'}
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
                del student,a,b
            del teacher,w,m,conditioned
            if device=='cuda':torch.cuda.empty_cache()
            if args.pilot_only:manifest['state']='pilot_completed';return
        manifest['final_test_opened']=True;manifest['selected_rank']=selected
        write_json(args.output/'manifest.json',manifest)
        testdata=datasets.CIFAR10(args.data,train=False,download=False,transform=plain)
        testloader=DataLoader(testdata,batch_size=128,shuffle=False,num_workers=0);final=[]
        x=next(iter(tune))[0][:64].to(device)
        for seed in manifest['seeds']:
            teacher=models.resnet18(weights=None,num_classes=10).to(device)
            teacher.load_state_dict(torch.load(args.output/f'seed-{seed}'/'teacher_calibrated.pt',map_location=device,weights_only=True))
            for method in ['original','svd','weighted_svd','weighted_rsvd','haar']:
                start=time.perf_counter()
                if method=='original':student=teacher
                else:
                    state=torch.load(args.output/f'seed-{seed}'/f'{method}-r{selected}.pt',map_location=device,weights_only=True)
                    student=copy.deepcopy(teacher);prefix='layer3.0.conv2.'
                    student.layer3[0].conv2=MatrixConv(state[prefix+'a'][:,:,0,0],state[prefix+'b'].reshape(selected,2304))
                    student.load_state_dict(state)
                row={'seed':seed,'method':method,'rank':selected,'final_test':evaluate(student,testloader,device),
                     'errors':errors(teacher,student,testloader,device),
                     'latency':[latency(student,x[:batch],device,True) for batch in [64,1]]}
                final.append(row);write_json(args.output/'final.json',final);print(json.dumps(row),flush=True)
                guard(start,2700)
                if method!='original':del student
            del teacher
            if device=='cuda':torch.cuda.empty_cache()
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
