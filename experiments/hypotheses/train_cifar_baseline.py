"""Shared ResNet-18/CIFAR-10 baseline. Official test split stays unopened."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, models, transforms

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git, gpu_snapshot, guard, sha, sync, write_json


def split_indices(labels):
    rng=np.random.default_rng(20261002)
    groups={'baseline':[],'calibration':[],'tuning':[],'recovery':[]}
    for label in range(10):
        ids=np.flatnonzero(np.asarray(labels)==label)
        rng.shuffle(ids)
        for name, lo, hi in [('baseline',0,4000),('calibration',4000,4200),
                              ('tuning',4200,4500),('recovery',4500,5000)]:
            groups[name].extend(ids[lo:hi].tolist())
    flattened=[i for group in groups.values() for i in group]
    assert len(flattened)==50000 and len(set(flattened))==50000
    return groups


def normalization(data, indices):
    count=0
    sums=np.zeros(3,dtype=np.float64)
    squares=np.zeros(3,dtype=np.float64)
    for start in range(0,len(indices),512):
        x=data[indices[start:start+512]].astype(np.float64)/255
        sums+=x.sum(axis=(0,1,2));squares+=(x*x).sum(axis=(0,1,2))
        count+=x.shape[0]*x.shape[1]*x.shape[2]
    mean=sums/count
    std=np.sqrt(squares/count-mean*mean)
    return mean.tolist(),std.tolist()


def initialize(seed, device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    return models.resnet18(weights=None,num_classes=10).to(device)


def training_schedule(optimizer, h09_warmup=False):
    """H09 needs a fresh five-epoch constant-lr optimizer state."""
    if h09_warmup:
        return 5, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.)
    return 30, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval();correct=0;count=0;total_loss=0.
    for x,y in loader:
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
            logits=model(x);loss=nn.functional.cross_entropy(logits,y,reduction='sum')
        correct+=int((logits.argmax(1)==y).sum());count+=len(y);total_loss+=float(loss)
    return {'accuracy':correct/count,'cross_entropy':total_loss/count,'n':count}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    parser.add_argument('--pilot-only',action='store_true')
    parser.add_argument('--h09-warmup',action='store_true',
                        help='fresh five-epoch constant-lr warmup; save full optimizer state, no test')
    args=parser.parse_args()
    if args.h09_warmup:
        git('merge-base','--is-ancestor','a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b','HEAD')
    args.output.mkdir(parents=True,exist_ok=False)
    args.data.mkdir(parents=True,exist_ok=True)
    (args.output/'train_cifar_baseline.py.source').write_bytes(Path(__file__).read_bytes())
    (args.output/'run_local.py.source').write_bytes(Path(__file__).with_name('run_local.py').read_bytes())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest={'run_id':args.output.name,'state':'preparing','git_sha':git('rev-parse','HEAD'),
              'git_status':git('status','--porcelain'),'runner_sha256':sha(__file__),
              'protocol_sha256':sha(Path(__file__).with_name('H09_gradient_drift.md' if args.h09_warmup else 'H01_input_distribution.md')),
              'torch':torch.__version__,'torchvision':torchvision.__version__,
              'python':sys.version,'cuda':torch.version.cuda,'device':device,
              'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
              'gpu_driver_snapshot':gpu_snapshot(),'command':sys.argv,'pid':os.getpid(),
              'seeds':args.seeds,'epochs':5 if args.h09_warmup else 30,'microbatch':128,'effective_batch':128,
              'mode':'h09_fresh_warmup' if args.h09_warmup else 'shared_baseline',
              'lr_schedule':'constant' if args.h09_warmup else 'cosine30',
              'optimizer':'AdamW','lr':.001,'weight_decay':.0001,
              'precision':'FP32 parameters/optimizer; BF16 autocast on CUDA',
              'data':'CIFAR-10 train only','architecture':'torchvision ResNet-18; original 7x7/stride2 stem',
              'baseline_admission_tuning_accuracy':.70,'final_test_opened':False,
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'limitations':['torchvision 0.17.0 replaces protocol interface reference 0.20; same declared architecture',
                             'WDDM process GPU memory unavailable; conservative total-device bound used']}
    write_json(args.output/'manifest.json',manifest)
    try:
        raw=datasets.CIFAR10(root=args.data,train=True,download=True)
        archive=args.data/'cifar-10-python.tar.gz'
        assert hashlib.md5(archive.read_bytes()).hexdigest()=='c58f30108f718f92721af3b95e74349a'
        manifest['data_archive_sha256']=sha(archive)
        splits=split_indices(raw.targets)
        write_json(args.output/'splits.json',splits)
        manifest['split_sha256']=sha(args.output/'splits.json')
        mean,std=normalization(raw.data,splits['baseline'])
        manifest['normalization']={'mean':mean,'std':std,'source':'40000 baseline train images only'}
        train_transform=transforms.Compose([transforms.RandomCrop(32,padding=4),
                                           transforms.RandomHorizontalFlip(.5),transforms.ToTensor(),
                                           transforms.Normalize(mean,std)])
        eval_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
        train_data=datasets.CIFAR10(root=args.data,train=True,transform=train_transform)
        tune_data=datasets.CIFAR10(root=args.data,train=True,transform=eval_transform)
        manifest['state']='running';write_json(args.output/'manifest.json',manifest)
        tune_loader=DataLoader(Subset(tune_data,splits['tuning']),batch_size=128,shuffle=False,
                               num_workers=0,pin_memory=device=='cuda')
        pilot_model=initialize(args.seeds[0],device)
        assert sum(p.numel() for p in pilot_model.parameters())==11181642
        pilot_loader=DataLoader(Subset(train_data,splits['baseline']),batch_size=128,shuffle=True,
                                generator=torch.Generator().manual_seed(args.seeds[0]+7000),
                                num_workers=0,pin_memory=device=='cuda')
        pilot_opt=torch.optim.AdamW(pilot_model.parameters(),lr=.001,weight_decay=.0001)
        sync();pilot_start=time.perf_counter()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for step,(x,y) in enumerate(pilot_loader):
            x=x.to(device);y=y.to(device);pilot_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                loss=nn.functional.cross_entropy(pilot_model(x),y)
            if not torch.isfinite(loss):raise ArithmeticError('nonfinite pilot loss')
            loss.backward();pilot_opt.step();guard(pilot_start,limit=600)
            if step==19:break
        sync()
        pilot={'steps':20,'seconds':time.perf_counter()-pilot_start,'last_loss':float(loss),
               'gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
               'gpu_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0,
               'rss_bytes':psutil.Process().memory_info().rss,'final_test_opened':False}
        write_json(args.output/'pilot.json',pilot);print(json.dumps({'pilot':pilot}),flush=True)
        del pilot_opt,pilot_model
        if device=='cuda':torch.cuda.empty_cache()
        if not args.pilot_only:
            for seed in args.seeds:
                seed_dir=args.output/f'seed-{seed}';seed_dir.mkdir()
                model=initialize(seed,device)
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                epoch_count,scheduler=training_schedule(opt,args.h09_warmup)
                loader=DataLoader(Subset(train_data,splits['baseline']),batch_size=128,shuffle=True,
                                  generator=torch.Generator().manual_seed(seed+7000),num_workers=0,
                                  pin_memory=device=='cuda')
                start=time.perf_counter();epochs=[]
                for epoch in range(epoch_count):
                    model.train();loss_sum=0.;seen=0
                    for step,(x,y) in enumerate(loader):
                        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                            loss=nn.functional.cross_entropy(model(x),y)
                        if not torch.isfinite(loss):raise ArithmeticError('nonfinite baseline loss')
                        loss.backward();opt.step();loss_sum+=float(loss)*len(y);seen+=len(y)
                        if step%25==0:guard(start,limit=7200 if args.h09_warmup else 21600)
                    scheduler.step()
                    metrics=evaluate(model,tune_loader,device)
                    row={'seed':seed,'epoch':epoch+1,'learning_rate':opt.param_groups[0]['lr'],'train_cross_entropy':loss_sum/seen,
                         'tuning':metrics,'elapsed_seconds':time.perf_counter()-start,
                         'gpu_reserved_bytes':torch.cuda.memory_reserved() if device=='cuda' else 0,
                         'rss_bytes':psutil.Process().memory_info().rss}
                    epochs.append(row);write_json(seed_dir/'epochs.json',epochs)
                    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                                'scheduler':scheduler.state_dict(),'epoch':epoch+1,
                                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                                'python_rng':random.getstate(),
                                'loader_rng':loader.generator.get_state(),
                                'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None},
                               seed_dir/'resume.pt')
                    print(json.dumps(row),flush=True)
                torch.save(model.state_dict(),seed_dir/'model.pt')
                write_json(seed_dir/'result.json',{'seed':seed,'checkpoint_sha256':sha(seed_dir/'model.pt'),
                           'resume_sha256':sha(seed_dir/'resume.pt'),
                           'state':'warmup_ready' if args.h09_warmup else ('baseline_ready' if metrics['accuracy']>=.70 else 'baseline_not_ready'),
                           'final_tuning':metrics,'epochs':epoch_count,'final_test_opened':False})
                del opt,model
                if device=='cuda':torch.cuda.empty_cache()
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
