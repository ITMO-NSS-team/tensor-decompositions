"""H04 ResNet-18/CIFAR-10: paired execution, tuning before final evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_local import git, gpu_snapshot, guard, sha, sync, write_json
from tdecomp.tensor.tucker import HOOIDecomposition


class TuckerConv(nn.Module):
    def __init__(self, core, u, v, dense=False):
        super().__init__()
        self.core = nn.Parameter(core.clone())
        self.u = nn.Parameter(u.clone())
        self.v = nn.Parameter(v.clone())
        self.dense = dense

    def weight(self):
        return torch.einsum('oa,abhw,ib->oihw', self.u, self.core, self.v)

    def forward(self, x):
        if self.dense:
            return F.conv2d(x, self.weight(), padding=1)
        return F.conv2d(F.conv2d(F.conv2d(x, self.v.T[:, :, None, None]),
                                self.core, padding=1), self.u[:, :, None, None])


class MatrixConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = nn.Parameter(a[:, :, None, None].clone())
        self.b = nn.Parameter(b.reshape(b.shape[0], 256, 3, 3).clone())

    def forward(self, x):
        return F.conv2d(F.conv2d(x, self.b, padding=1), self.a)


def admission():
    """Independent dense convolution reference, including input/factor gradients."""
    gen = torch.Generator().manual_seed(9191)
    rows = []
    for dtype, tolerance in [(torch.float64, 1e-10), (torch.float32, 1e-5)]:
        core = torch.randn(4, 3, 3, 3, generator=gen, dtype=dtype)
        u = torch.randn(7, 4, generator=gen, dtype=dtype)
        v = torch.randn(5, 3, generator=gen, dtype=dtype)
        direct, dense = TuckerConv(core, u, v), TuckerConv(core, u, v, dense=True)
        x = torch.randn(2, 5, 2, 2, generator=gen, dtype=dtype, requires_grad=True)
        xx = x.detach().clone().requires_grad_(True)
        y, yy = direct(x), dense(xx)
        relative = lambda a, b: float((a-b).norm()/b.norm().clamp_min(1e-30))
        output_error = relative(y, yy)
        probe = torch.randn(y.shape, generator=gen, dtype=dtype)
        (y*probe).sum().backward(); (yy*probe).sum().backward()
        errors = [relative(x.grad, xx.grad)]
        errors += [relative(a.grad, b.grad) for a, b in zip(direct.parameters(), dense.parameters())]
        assert max([output_error, *errors]) <= tolerance, (dtype, output_error, errors)
        rows.append({'dtype': str(dtype), 'forward_relative': output_error,
                     'gradient_relative_max': max(errors), 'tolerance': tolerance})
    return rows


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, bf16=True):
    model.eval(); correct = count = 0; loss = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16 and device=='cuda'):
            logits = model(x)
        if not torch.isfinite(logits).all(): raise ArithmeticError('nonfinite evaluation')
        correct += int((logits.argmax(1)==y).sum()); count += len(y)
        loss += float(F.cross_entropy(logits.float(), y, reduction='sum'))
    return {'accuracy': correct/count, 'cross_entropy': loss/count, 'n': count}


@torch.no_grad()
def latency(model, x, device, bf16):
    model.eval(); values = []
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16 and device=='cuda'):
        for _ in range(30): model(x)
        sync()
        for _ in range(200):
            sync(); start = time.perf_counter(); model(x); sync()
            values.append((time.perf_counter()-start)*1000)
    return {'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95)),
            'warmups': 30, 'measured': 200, 'batch': len(x), 'precision': 'BF16' if bf16 else 'FP32'}


def recover(student, dataset, splits, seed, device, start, steps=(128, 400)):
    for p in student.parameters(): p.requires_grad_(False)
    for p in student.layer3[0].conv2.parameters(): p.requires_grad_(True)
    student.eval()  # Preserve all original BN buffers across the paired variants.
    opt = torch.optim.AdamW(student.layer3[0].conv2.parameters(), lr=1e-4, weight_decay=1e-4)
    history = []
    for phase, total, offset in [('calibration', steps[0], 50000), ('recovery', steps[1], 60000)]:
        seed_all(seed+offset)
        loader = DataLoader(Subset(dataset[phase], splits[phase]), batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(seed+offset), num_workers=0)
        iterator = iter(loader)
        for step in range(total):
            try: x, y = next(iterator)
            except StopIteration: iterator=iter(loader); x, y = next(iterator)
            x, y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device=='cuda'):
                loss = F.cross_entropy(student(x), y)
            if not torch.isfinite(loss): raise ArithmeticError('nonfinite recovery loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.layer3[0].conv2.parameters()):
                raise ArithmeticError('nonfinite recovery gradient')
            opt.step()
            if step%25==0:
                guard(start, limit=2700)
                history.append({'phase': phase, 'step': step, 'loss': float(loss.detach())})
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101,202,303])
    parser.add_argument('--admission-only', action='store_true')
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'run_h04_real.py.source').write_bytes(Path(__file__).read_bytes())
    (args.output/'run_local.py.source').write_bytes(Path(__file__).with_name('run_local.py').read_bytes())
    tl.set_backend('pytorch'); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    manifest = {'run_id': args.output.name, 'hypothesis_id':'H04', 'setting':'real', 'state':'admission',
                'git_sha':git('rev-parse','HEAD'), 'git_status':git('status','--porcelain'),
                'runner_sha256':sha(__file__), 'protocol_sha256':sha(Path(__file__).with_name('H04_direct_tensor_operator.md')),
                'source_hashes':{name:sha(Path(__file__).with_name(name)) for name in ['run_local.py']},
                'torch':torch.__version__, 'torchvision':torchvision.__version__, 'tensorly':tl.__version__,
                'python':sys.version, 'cuda':torch.version.cuda, 'device':device, 'pid':os.getpid(),
                'gpu':torch.cuda.get_device_name() if device=='cuda' else None, 'gpu_snapshot':gpu_snapshot(),
                'command':sys.argv, 'seeds':args.seeds, 'ranks':[32,64,96], 'matrix_ranks':[10,27,51],
                'microbatch':128, 'calibration_steps':128, 'recovery_steps':400,
                'final_test_opened':False, 'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'limitations':['only compressed layer trained; original BN and remaining parameters frozen',
                               'dense comparison is one-time materialization of the same recovered Tucker weight',
                               'torchvision actual version differs from protocol interface reference 0.20',
                               'WDDM process GPU memory unavailable; total-device conservative bound',
                               'single-device empirical result; no fused kernel',
                               'rank selection uses seed 101 tuning only; three paired seeds are exploratory']}
    write_json(args.output/'manifest.json', manifest)
    try:
        write_json(args.output/'operator_checks.json', admission())
        if args.admission_only:
            manifest['state']='admission_passed'; return
        base_manifest=json.loads((args.baseline/'manifest.json').read_text(encoding='utf-8'))
        assert base_manifest['state']=='completed'
        assert sha(args.data/'cifar-10-python.tar.gz')==base_manifest['data_archive_sha256']
        split_path=args.baseline/'splits.json'
        assert sha(split_path)==base_manifest['split_sha256']
        manifest['data_archive_sha256']=base_manifest['data_archive_sha256']
        manifest['split_sha256']=sha(split_path); splits=json.loads(split_path.read_text(encoding='utf-8'))
        norm=base_manifest['normalization']; manifest['normalization']=norm
        plain=transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm['mean'],norm['std'])])
        aug=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),transforms.Normalize(norm['mean'],norm['std'])])
        data_plain=datasets.CIFAR10(args.data,train=True,download=False,transform=plain)
        data_aug=datasets.CIFAR10(args.data,train=True,download=False,transform=aug)
        tune=DataLoader(Subset(data_plain,splits['tuning']),batch_size=128,shuffle=False,num_workers=0)
        checkpoint_hashes={}
        for seed in args.seeds:
            result=json.loads((args.baseline/f'seed-{seed}'/'result.json').read_text(encoding='utf-8'))
            cp=args.baseline/f'seed-{seed}'/'model.pt'
            assert result['state']=='baseline_ready' and sha(cp)==result['checkpoint_sha256']
            checkpoint_hashes[str(seed)]=sha(cp)
        manifest['baseline_checkpoint_hashes']=checkpoint_hashes
        manifest['state']='running'; write_json(args.output/'manifest.json',manifest)
        rows=[]; selected=None
        for seed in args.seeds:
            seed_all(seed); teacher=models.resnet18(weights=None,num_classes=10).to(device)
            teacher.load_state_dict(torch.load(args.baseline/f'seed-{seed}'/'model.pt',map_location=device,weights_only=True))
            teacher.eval(); sd=args.output/f'seed-{seed}'; sd.mkdir()
            write_json(sd/'original_tuning.json',evaluate(teacher,tune,device))
            w=teacher.layer3[0].conv2.weight.detach()
            assert tuple(w.shape)==(256,256,3,3)
            ranks=[32,64,96] if selected is None else [selected]
            if args.pilot_only: ranks=[64]
            for rank in ranks:
                for method in ['direct_tucker','matrix_svd']:
                    start=time.perf_counter(); seed_all(seed)
                    if device=='cuda': torch.cuda.reset_peak_memory_stats()
                    sync(); setup=time.perf_counter()
                    if method=='direct_tucker':
                        dec=HOOIDecomposition(rank=(rank,rank,3,3),random_state=seed)
                        core,factors=dec.decompose(w,n_iter_max=20,tol=1e-6)
                        spatial=tl.tenalg.multi_mode_dot(core,factors[2:],modes=[2,3])
                        replacement=TuckerConv(spatial,factors[0],factors[1])
                    else:
                        k=(512*rank+9*rank*rank)//2560
                        u,s,vh=torch.linalg.svd(w.reshape(256,-1),full_matrices=False)
                        replacement=MatrixConv(u[:,:k]*s[:k],vh[:k])
                    sync(); setup_seconds=time.perf_counter()-setup
                    student=copy.deepcopy(teacher); student.layer3[0].conv2=replacement
                    before=evaluate(student,tune,device)
                    sync(); recovery_start=time.perf_counter()
                    history=recover(student,{'calibration':data_plain,'recovery':data_aug},splits,
                                    seed,device,start,steps=(20,0) if args.pilot_only else (128,400))
                    sync(); recovery_seconds=time.perf_counter()-recovery_start
                    tuning=evaluate(student,tune,device)
                    cp=sd/f'{method}-r{rank}.pt'; torch.save(student.state_dict(),cp)
                    write_json(sd/f'{method}-r{rank}-history.json',history)
                    row={'seed':seed,'method':method,'rank':rank,'matrix_rank':k if method=='matrix_svd' else None,
                         'tuning_before':before,'tuning_after':tuning,'setup_seconds':setup_seconds,
                         'recovery_seconds':recovery_seconds,'elapsed_seconds':time.perf_counter()-start,
                         'parameters':sum(p.numel() for p in replacement.parameters()),
                         'checkpoint_sha256':sha(cp),'status':'pilot_completed' if args.pilot_only else 'completed',
                         'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                         'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0,
                         'rss_bytes':psutil.Process().memory_info().rss}
                    rows.append(row); write_json(args.output/'runs.json',rows); print(json.dumps(row),flush=True)
                    del student,replacement
                    if device=='cuda': torch.cuda.empty_cache()
                    guard(start,limit=2700)
                if args.pilot_only: break
            if selected is None and not args.pilot_only:
                candidates=[r for r in rows if r['seed']==seed and r['method']=='direct_tucker']
                selected=max(candidates,key=lambda r:(r['tuning_after']['accuracy'],-r['rank']))['rank']
                write_json(args.output/'selection.json',{'selected_rank':selected,'selection_seed':seed,
                           'criterion':'maximum tuning accuracy; smallest rank breaks ties',
                           'frozen_before_final_test':True,'tuning_candidates':candidates})
            del teacher
            if device=='cuda': torch.cuda.empty_cache()
            if args.pilot_only: break
        if args.pilot_only:
            manifest['state']='pilot_completed'; return
        # Open the official test only after the selection has been persisted.
        manifest['final_test_opened']=True; manifest['selected_rank']=selected
        write_json(args.output/'manifest.json',manifest)
        testdata=datasets.CIFAR10(args.data,train=False,download=False,transform=plain)
        testloader=DataLoader(testdata,batch_size=128,shuffle=False,num_workers=0)
        final=[]
        x=next(iter(tune))[0][:64].to(device)
        for seed in args.seeds:
            final_start=time.perf_counter()
            original=models.resnet18(weights=None,num_classes=10).to(device)
            original.load_state_dict(torch.load(args.baseline/f'seed-{seed}'/'model.pt',map_location=device,weights_only=True))
            direct=copy.deepcopy(original)
            state=torch.load(args.output/f'seed-{seed}'/f'direct_tucker-r{selected}.pt',map_location=device,weights_only=True)
            prefix='layer3.0.conv2.'
            direct.layer3[0].conv2=TuckerConv(state[prefix+'core'],state[prefix+'u'],state[prefix+'v'])
            direct.load_state_dict(state)
            dense=copy.deepcopy(direct)
            sync(); material_start=time.perf_counter()
            layer=nn.Conv2d(256,256,3,padding=1,bias=False).to(device)
            with torch.no_grad(): layer.weight.copy_(direct.layer3[0].conv2.weight())
            dense.layer3[0].conv2=layer; sync(); material_seconds=time.perf_counter()-material_start
            matrix=copy.deepcopy(original)
            state=torch.load(args.output/f'seed-{seed}'/f'matrix_svd-r{selected}.pt',map_location=device,weights_only=True)
            matrix.layer3[0].conv2=MatrixConv(state[prefix+'a'][:,:,0,0],state[prefix+'b'].reshape(-1,2304))
            matrix.load_state_dict(state)
            original.eval(); direct.eval(); dense.eval(); matrix.eval()
            with torch.no_grad():
                dy=direct(x); yy=dense(x)
                error=float((dy-yy).norm()/yy.norm().clamp_min(1e-30))
            if error>1e-5: raise ArithmeticError(f'recovered whole-network FP32 mismatch: {error}')
            for method,model in [('original',original),('direct_tucker',direct),('dense_materialized_tucker',dense),('matrix_svd',matrix)]:
                metric=evaluate(model,testloader,device)
                timings=[latency(model,x[:batch],device,bf16) for bf16 in [True,False] for batch in [64,1]]
                item={'seed':seed,'method':method,'selected_rank':selected,'final_test':metric,
                      'latency':timings,'direct_dense_fp32_relative':error,
                      'dense_materialization_seconds':material_seconds if method=='dense_materialized_tucker' else 0}
                final.append(item); write_json(args.output/'final.json',final); print(json.dumps(item),flush=True)
                guard(final_start,limit=2700)
            del original,direct,dense,matrix
            if device=='cuda':torch.cuda.empty_cache()
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__': main()
