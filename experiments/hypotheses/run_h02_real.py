"""H02 paired ResNet block calibration, preserving full-width BN/ReLU/residuals."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
import platform
import psutil
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_composition_core import ComposedBlock, admission
from run_h04_real import evaluate, latency, seed_all
from run_local import git, guard, sha, sync, write_json, gpu_snapshot
from run_h02_synthetic import tensor_hash

METHODS = ('independent', 'shared_initial', 'joint')


class IndexedCalibration(torch.utils.data.Dataset):
    def __init__(self,dataset,indices):self.dataset,self.indices=dataset,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        original=self.indices[index];x,y=self.dataset[original]
        return x,y,original


@torch.no_grad()
def common_bn(teacher, loader, device):
    teacher.eval()
    for bn in (teacher.layer3[1].bn1, teacher.layer3[1].bn2):
        bn.reset_running_stats(); bn.momentum = None; bn.train()
    for x, _ in loader: teacher(x.to(device))
    teacher.eval()


def block_inputs(teacher, x):
    values=[]
    hook=teacher.layer3[1].register_forward_pre_hook(lambda _, inp: values.append(inp[0].detach()))
    try:
        with torch.no_grad(): teacher(x)
    finally: hook.remove()
    return values[0]


@torch.no_grad()
def block_error(student, teacher, loader, device):
    error=norm=0.
    for x,_ in loader:
        z=block_inputs(teacher,x.to(device))
        target=teacher.layer3[1](z).float()
        output=student.layer3[1](z).float()
        error+=float((output-target).square().sum());norm+=float(target.square().sum())
    return error/max(norm,1e-30)


def fit(student, teacher, plain, aug, splits, seed, device, method, start, pilot):
    student.eval(); teacher.eval(); block=student.layer3[1]
    opt=torch.optim.AdamW(block.factors(),lr=.001,weight_decay=0)
    steps=20 if pilot else 128
    seed_all(seed+50000)
    loader=DataLoader(IndexedCalibration(plain,splits['calibration']),batch_size=128,shuffle=True,
                      generator=torch.Generator().manual_seed(seed+50000),num_workers=0)
    iterator=iter(loader);sync();cal_start=time.perf_counter();calibration_history=[]
    for step in range(steps):
        try:x,_,indices=next(iterator)
        except StopIteration:iterator=iter(loader);x,_,indices=next(iterator)
        z=block_inputs(teacher,x.to(device));opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            original=teacher.layer3[1]
            first=original.conv1(z)
            hidden=F.relu(original.bn1(first),inplace=False)
            second=original.conv2(hidden)
            target=original(z)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
            if method=='independent':
                loss=F.mse_loss(block.first(z).float(),first) if step<64 else F.mse_loss(block.second(hidden).float(),second)
            else:loss=F.mse_loss(block(z).float(),target)
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite block loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in block.factors()):
            raise ArithmeticError('nonfinite factor gradient')
        opt.step();guard(start,2700)
        calibration_history.append({'step':step,'loss':float(loss.detach()),'original_indices_sha256':tensor_hash(indices)})
    sync();cal_seconds=time.perf_counter()-cal_start
    tune=DataLoader(Subset(plain,splits['tuning']),batch_size=128,shuffle=False)
    primary_error=block_error(student,teacher,tune,device)
    prequality=evaluate(student,tune,device)
    recovery_start=time.perf_counter();history=[]
    opt=torch.optim.AdamW(block.factors(),lr=1e-4,weight_decay=1e-4)
    seed_all(seed+60000)
    loader=DataLoader(Subset(aug,splits['recovery']),batch_size=128,shuffle=True,
                      generator=torch.Generator().manual_seed(seed+60000),num_workers=0)
    iterator=iter(loader)
    for step in range(0 if pilot else 400):
        try:x,y=next(iterator)
        except StopIteration:iterator=iter(loader);x,y=next(iterator)
        x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
            loss=F.cross_entropy(student(x),y)
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite recovery loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in block.factors()):
            raise ArithmeticError('nonfinite recovery gradient')
        opt.step();guard(start,2700)
        if step%25==0:history.append({'step':step,'loss':float(loss.detach())})
    sync()
    return {'calibration_seconds':cal_seconds,'primary_block_normalized_mse':primary_error,
            'quality_before_recovery':prequality,'recovery_seconds':time.perf_counter()-recovery_start,
            'recovery_history':history,'calibration_history':calibration_history}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--pilot',action='store_true');p.add_argument('--admission-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    base_sha='a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b'
    import subprocess
    subprocess.run(['git','merge-base','--is-ancestor',base_sha,'HEAD'],check=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=next(Path(__file__).parent.glob('H02_*.md'))
    sources=['run_h02_real.py','real_composition_core.py','run_h04_real.py','run_local.py','run_h02_synthetic.py',protocol.name]
    hashes={}
    for name in sources:
        path=Path(__file__).with_name(name);hashes[name]=sha(path)
        (args.output/(name+'.source')).write_bytes(path.read_bytes())
    manifest={'state':'admission','hypothesis':'H02','setting':'real','git_sha':git('rev-parse','HEAD'),
              'source_hashes':hashes,'protocol_sha256':sha(protocol),'base_sha':base_sha,'torch':torch.__version__,'python':platform.python_version(),
              'cuda':torch.version.cuda,'device':args.device,'command':sys.argv,
              'gpu_snapshot':gpu_snapshot(),'ram_total_bytes':psutil.virtual_memory().total,
              'limits':{'gpu_bytes':12*1024**3,'rss_bytes':24*1024**3,'variant_seconds':2700},
              'seeds':[101] if args.pilot else [101,202,303],'primary_stage':'after128blockcalibration, before400CErecovery',
              'limitations':['only two compressed convolutions trained; all other parameters frozen',
                             'shared_initial has two independently trainable Q arrays; not parameter sharing',
                             'common target-block BN recalibration in teacher then identical frozen student buffers',
                             'illegal ReLU relocation and missing compensation negative neural controls pending',
                             'no full time-to-quality trajectory; final quality alone cannot confirm time criterion'],
              'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',admission())
        if args.admission_only:manifest['state']='admission_passed';return
        base=json.loads((args.baseline/'manifest.json').read_text(encoding='utf-8'))
        assert base['state']=='completed'
        assert sha(args.data/'cifar-10-python.tar.gz')==base['data_archive_sha256']
        split_path=args.baseline/'splits.json';assert sha(split_path)==base['split_sha256']
        splits=json.loads(split_path.read_text(encoding='utf-8'));norm=base['normalization']
        manifest.update(split_sha256=sha(split_path),archive_sha256=base['data_archive_sha256'])
        normalize=transforms.Normalize(norm['mean'],norm['std'])
        plain=datasets.CIFAR10(args.data,train=True,download=False,transform=transforms.Compose([transforms.ToTensor(),normalize]))
        aug=datasets.CIFAR10(args.data,train=True,download=False,transform=transforms.Compose([
            transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]))
        cal=DataLoader(Subset(plain,splits['calibration']),batch_size=128,shuffle=False)
        tune=DataLoader(Subset(plain,splits['tuning']),batch_size=128,shuffle=False)
        rows=[];selected=None;manifest['state']='running'
        for seed in manifest['seeds']:
            seed_all(seed);sd=args.output/f'seed-{seed}';sd.mkdir()
            cp=args.baseline/f'seed-{seed}'/'model.pt'
            expected=json.loads((cp.parent/'result.json').read_text(encoding='utf-8'))['checkpoint_sha256']
            assert sha(cp)==expected
            teacher=models.resnet18(weights=None,num_classes=10).to(args.device)
            teacher.load_state_dict(torch.load(cp,map_location=args.device,weights_only=True));teacher.eval()
            common_bn(teacher,cal,args.device);torch.save(teacher.state_dict(),sd/'common-teacher.pt')
            teacher_quality=evaluate(teacher,tune,args.device)
            if teacher_quality['accuracy']<.70:raise ArithmeticError('common teacher quality gate failed')
            write_json(sd/'teacher.json',{'original_checkpoint_sha256':expected,
                       'common_checkpoint_sha256':sha(sd/'common-teacher.pt'),'tuning':teacher_quality})
            ranks=[64] if args.pilot else ([32,64,96] if selected is None else [selected])
            for rank in ranks:
                for method in (['joint'] if args.pilot else METHODS):
                    start=time.perf_counter();student=copy.deepcopy(teacher)
                    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
                    for par in student.parameters():par.requires_grad_(False)
                    sync();factor_start=time.perf_counter()
                    student.layer3[1]=ComposedBlock(teacher.layer3[1],rank,shared_initialization=method=='shared_initial').eval()
                    sync();factor_seconds=time.perf_counter()-factor_start
                    result=fit(student,teacher,plain,aug,splits,seed,args.device,method,start,args.pilot)
                    row={'seed':seed,'rank':rank,'method':method,'factor_parameters':sum(p.numel() for p in student.layer3[1].factors()),
                         'factor_seconds':factor_seconds,**result,'tuning_after':evaluate(student,tune,args.device)}
                    target=sd/f'{method}-r{rank}.pt';torch.save(student.state_dict(),target)
                    row.update(checkpoint_sha256=sha(target),total_seconds=time.perf_counter()-start,status='completed')
                    row.update(rss_bytes=psutil.Process().memory_info().rss,
                               gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
                               gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else 0)
                    rows.append(row);write_json(args.output/'runs.json',rows)
                    print(json.dumps({key:row[key] for key in ('seed','rank','method','primary_block_normalized_mse','tuning_after','status')}),flush=True)
                    del student
            if selected is None:
                choices=[row for row in rows if row['seed']==seed and row['method']=='joint']
                selected=max(choices,key=lambda row:(row['tuning_after']['accuracy'],-row['rank']))['rank']
                manifest['selected_rank']=selected;write_json(args.output/'manifest.json',manifest)
            del teacher
        if not args.pilot:
            test=datasets.CIFAR10(args.data,train=False,download=False,transform=transforms.Compose([transforms.ToTensor(),normalize]))
            loader=DataLoader(test,batch_size=128,shuffle=False);final=[]
            x,_=next(iter(tune));x=x.to(args.device)
            for seed in manifest['seeds']:
                sd=args.output/f'seed-{seed}'
                for method in ('original',*METHODS):
                    model=models.resnet18(weights=None,num_classes=10).to(args.device).eval()
                    if method=='original':cp=sd/'common-teacher.pt'
                    else:
                        model.layer3[1]=ComposedBlock(model.layer3[1],selected).eval()
                        cp=sd/f'{method}-r{selected}.pt'
                    model.load_state_dict(torch.load(cp,map_location=args.device,weights_only=True))
                    result={'seed':seed,'method':method,'checkpoint_sha256':sha(cp),**evaluate(model,loader,args.device),
                            'latency_batch1':latency(model,x[:1],args.device,True),
                            'latency_batch64':latency(model,x[:64],args.device,True)}
                    final.append(result);write_json(args.output/'final.json',final);del model
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
