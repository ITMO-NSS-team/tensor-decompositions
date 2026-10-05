"""Bounded local H01 experiment and H04/H06 admission checks.

Results are exploratory unless the complete protocol admission is satisfied.
No final CIFAR test data are read by this entry point.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import psutil
import tensorly as tl
import torch
from torch import nn
from torch.nn import functional as F
from tdecomp.matrix.decomposer import RandomizedSVD, SVDDecomposition


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def gpu_snapshot():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,driver_version',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'measurement_unavailable'


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def guard(start, limit=600):
    if time.perf_counter() - start > limit:
        raise TimeoutError('per-variant time limit reached')
    rss = psutil.Process().memory_info().rss
    if rss > 24 * 1024**3:
        raise MemoryError('24 GiB RSS limit reached')
    if torch.cuda.is_available():
        # WDDM does not expose process driver memory. Check reserved separately
        # and conservatively bound total device usage, including other processes.
        total = torch.cuda.get_device_properties(0).total_memory
        free, _ = torch.cuda.mem_get_info()
        if torch.cuda.memory_reserved() > 12 * 1024**3 or total - free > 12 * 1024**3:
            raise MemoryError('12 GiB device-usage/reserved limit reached')


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def factorize(w, rank, method, moment=None, seed=0):
    if method == 'haar':
        rng = torch.Generator(device=w.device).manual_seed(seed)
        q = torch.linalg.qr(torch.randn(w.shape[1], rank, generator=rng,
                                      device=w.device, dtype=w.dtype)).Q
        return w @ q, q.T
    dec = (RandomizedSVD(rank=rank, oversampling=8, power=1, random_state=seed)
           if method == 'weighted_rsvd' else SVDDecomposition(rank=rank))
    kwargs = {}
    if method.startswith('weighted'):
        values, vectors = torch.linalg.eigh(moment)
        if values.min() <= 0:
            raise ValueError('H01 synthetic second moment must be positive definite; no hidden ridge')
        c = (vectors * values.sqrt()) @ vectors.T
        kwargs['conditioner'] = c
    u, s, vh = dec.decompose(w, **kwargs)
    return u * s, vh


def invariants():
    """Independent FP64 checks before any neural experiment."""
    w = torch.diag(torch.tensor([10., 1.], dtype=torch.float64))
    moment = torch.diag(torch.tensor([1e-4, 1.], dtype=torch.float64))
    a, b = factorize(w, 1, 'weighted_svd', moment)
    c = torch.diag(torch.tensor([.01, 1.], dtype=torch.float64))
    risk = ((w - a @ b) @ c).square().sum().item()
    assert abs(risk - .01) < 1e-12
    plain = factorize(w, 1, 'svd')
    assert (((w - plain[0] @ plain[1]) @ c).square().sum() > risk)
    isotropic = factorize(w, 1, 'weighted_svd', torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(isotropic[0] @ isotropic[1], plain[0] @ plain[1])
    full = factorize(w, 2, 'weighted_svd', moment)
    torch.testing.assert_close(full[0] @ full[1], w, atol=1e-12, rtol=1e-12)
    return {'weighted_diagonal_risk': risk, 'identity_moment_control': 'passed',
            'conditioned_full_rank_roundtrip': 'passed'}


class FactorLinear(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = nn.Parameter(a.clone())
        self.b = nn.Parameter(b.clone())

    def forward(self, x):
        return F.linear(F.linear(x, self.b), self.a)


def h01_data(seed, device, control='aligned'):
    gen = torch.Generator(device=device).manual_seed(seed)
    u = torch.linalg.qr(torch.randn(64, 64, generator=gen, device=device)).Q
    v = torch.linalg.qr(torch.randn(64, 64, generator=gen, device=device)).Q
    teacher = nn.Sequential(nn.Linear(64, 64, bias=False), nn.GELU(),
                            nn.Linear(64, 16, bias=False), nn.GELU(),
                            nn.Linear(16, 4, bias=False)).to(device)
    with torch.no_grad():
        spectrum = torch.tensor([10.] * 8 + [1.] * 8 + [.2] * 48, device=device)
        teacher[0].weight.copy_((u * spectrum) @ v.T)
        for layer in (teacher[2], teacher[4]):
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen, device=device)
                               / layer.in_features**.5)
    scale = torch.tensor([.01] * 8 + [1.] * 8 + [.1**.5] * 48, device=device)
    orientation=v
    if control in ('rotated','test_drift'):
        orient_rng=torch.Generator(device=device).manual_seed(seed+200000)
        orientation=torch.linalg.qr(torch.randn(64,64,generator=orient_rng,device=device)).Q
    if control=='isotropic':scale=torch.ones_like(scale)
    splits = {}
    for k, (name, count) in enumerate([('recovery',4096), ('calibration',512),
                                       ('tuning',512), ('test',1024)], 1):
        rng = torch.Generator(device=device).manual_seed(seed + 1000*k)
        basis=orientation if control=='rotated' or (control=='test_drift' and name=='test') else v
        x = (torch.randn(count, 64, generator=rng, device=device) * scale) @ basis.T
        with torch.no_grad():
            splits[name] = (x, teacher(x))
    return teacher.eval(), splits


@torch.no_grad()
def errors(student, teacher, pair):
    x, y = pair
    predicted = student(x)
    exact_layer = teacher[0](x)
    predicted_layer = student[0](x)
    return {'network_normalized_mse': float((predicted-y).square().sum() / y.square().sum()),
            'layer_normalized_mse': float((predicted_layer-exact_layer).square().sum()
                                          / exact_layer.square().sum())}


def h01(out, seeds, device, rank=8, control='aligned'):
    rows = []
    for seed in seeds:
        teacher, splits = h01_data(seed, device, control)
        seed_dir = out / f'seed-{seed}'
        seed_dir.mkdir()
        torch.save(teacher.state_dict(), seed_dir / 'teacher.pt')
        hashes = {name: {'x':tensor_hash(x), 'y':tensor_hash(y), 'count':len(x)}
                  for name,(x,y) in splits.items()}
        write_json(seed_dir/'inputs.json', hashes)
        xcal, _ = splits['calibration']
        sync()
        moment_start = time.perf_counter()
        moment = xcal.T @ xcal / len(xcal)
        if control=='isotropic':moment=torch.eye(64,device=device)  # Declared exact M=I negative control.
        sync()
        moment_seconds = time.perf_counter() - moment_start
        for method in ('svd','weighted_svd','weighted_rsvd','haar'):
            start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            sync()
            factor_start = time.perf_counter()
            a,b = factorize(teacher[0].weight.detach(),rank,method,moment,seed+100000)
            sync()
            factor_seconds = time.perf_counter()-factor_start
            student = copy.deepcopy(teacher)
            student[0] = FactorLinear(a,b)
            # Shared policy: only the compressed layer is trainable.
            for parameter in student[2:].parameters():
                parameter.requires_grad_(False)
            tuning_before = errors(student, teacher, splits['tuning'])
            opt = torch.optim.AdamW(student[0].parameters(), lr=.001, weight_decay=0)
            histories = []
            sync()
            recovery_start = time.perf_counter()
            for phase, steps, offset in [('calibration',128,50000),('recovery',512,60000)]:
                x,y = splits[phase]
                batch_gen = torch.Generator(device=device).manual_seed(seed+offset)
                for step in range(steps):
                    ix = torch.randint(len(x),(32,),generator=batch_gen,device=device)
                    opt.zero_grad(set_to_none=True)
                    loss = F.mse_loss(student(x[ix]),y[ix])
                    if not torch.isfinite(loss):
                        raise ArithmeticError('nonfinite training loss')
                    loss.backward()
                    opt.step()
                    if step%64 == 0:
                        guard(start)
                        histories.append({'phase':phase,'step':step,'loss':float(loss.detach())})
            sync()
            recovery_seconds = time.perf_counter()-recovery_start
            student.eval()
            # Rank and hyperparameters are pre-fixed. Final split is read once
            # per method after all recovery, never used for selection.
            final = errors(student,teacher,splits['test'])
            torch.save(student.state_dict(),seed_dir/f'{method}.pt')
            write_json(seed_dir/f'{method}-history.json',histories)
            row = {'hypothesis_id':'H01','setting':'synthetic','seed':seed,'method':method,
                   'rank':rank,'control':control,'moment_mode':'exact_identity' if control=='isotropic' else 'calibration_empirical',
                   'sketch_width':rank+8 if method=='weighted_rsvd' else None,
                   'power':1 if method=='weighted_rsvd' else None,
                   'moment_seconds':moment_seconds if method.startswith('weighted') else 0,
                   'factor_seconds':factor_seconds,'recovery_seconds':recovery_seconds,
                   'total_seconds':time.perf_counter()-start + (moment_seconds if method.startswith('weighted') else 0),
                   'tuning_before_network_mse':tuning_before['network_normalized_mse'],
                   'tuning_before_layer_mse':tuning_before['layer_normalized_mse'],
                   **final,'trainable_parameters':sum(p.numel() for p in student.parameters() if p.requires_grad),
                   'checkpoint_sha256':sha(seed_dir/f'{method}.pt'),
                   'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                   'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0,
                   'rss_bytes':psutil.Process().memory_info().rss,'status':'completed'}
            rows.append(row)
            write_json(out/'runs.json',rows)
            print(json.dumps(row,allow_nan=False),flush=True)
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from scipy.stats import t
    summary=[]
    for method in ('weighted_svd','weighted_rsvd','haar'):
        differences=[]
        for seed in seeds:
            baseline=next(r for r in rows if r['seed']==seed and r['method']=='svd')
            candidate=next(r for r in rows if r['seed']==seed and r['method']==method)
            differences.append(candidate['layer_normalized_mse']-baseline['layer_normalized_mse'])
        mean=float(np.mean(differences))
        radius=float(t.ppf(.975,len(seeds)-1)*np.std(differences,ddof=1)/len(seeds)**.5) if len(seeds)>1 else None
        summary.append({'method':method,'reference':'svd','metric':'final layer normalized MSE',
                        'n_seeds':len(seeds),'paired_differences':differences,'mean_delta':mean,
                        'ci95_low':mean-radius if radius is not None else None,
                        'ci95_high':mean+radius if radius is not None else None,
                        'interpretation':'exploratory; conditional t interval; full hypothesis not confirmed'})
    write_json(out/'paired_summary.json',summary)


def h04_check():
    """Operator/output/input/core-factor gradient reference, not a speed claim."""
    gen=torch.Generator().manual_seed(73)
    dtype=torch.float64
    u=torch.linalg.qr(torch.randn(16,4,generator=gen,dtype=dtype)).Q.requires_grad_()
    v=torch.linalg.qr(torch.randn(8,3,generator=gen,dtype=dtype)).Q.requires_grad_()
    core=torch.randn(4,3,3,3,generator=gen,dtype=dtype,requires_grad=True)
    bias=torch.randn(16,generator=gen,dtype=dtype,requires_grad=True)
    x=torch.randn(2,8,9,9,generator=gen,dtype=dtype,requires_grad=True)
    w=torch.einsum('oa,abhw,ib->oihw',u,core,v)
    dense=F.conv2d(x,w,bias,padding=1)
    direct=F.conv2d(F.conv2d(F.conv2d(x,v.T[:,:,None,None]),core,padding=1),u[:,:,None,None],bias)
    torch.testing.assert_close(dense,direct,atol=1e-10,rtol=1e-10)
    probe=torch.randn(dense.shape,generator=gen,dtype=dtype)
    gd=torch.autograd.grad((dense*probe).sum(),(x,u,v,core,bias),retain_graph=True)
    gf=torch.autograd.grad((direct*probe).sum(),(x,u,v,core,bias))
    for left,right in zip(gd,gf):
        torch.testing.assert_close(left,right,atol=1e-10,rtol=1e-10)
    return {'status':'passed','dtype':'float64','output_max_abs_error':float((dense-direct).abs().max()),
            'gradient_max_abs_error':max(float((a-b).abs().max()) for a,b in zip(gd,gf)),
            'dense_weights':1152,'tucker_weights':196,
            'scope':'H04 admission only; no neural quality/latency experiment'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    parser.add_argument('--admission-only',action='store_true')
    parser.add_argument('--rank',type=int,choices=[4,8,16],default=8)
    parser.add_argument('--control',choices=['aligned','isotropic','rotated','test_drift'],default='aligned')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'run_local.py.source').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    tl.set_backend('pytorch')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest={'run_id':args.output.name,'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'git_sha':git('rev-parse','HEAD'),'git_status':git('status','--porcelain'),
              'runner_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).with_name('H01_input_distribution.md')),
              'python':platform.python_version(),'torch':torch.__version__,'tensorly':tl.__version__,
              'numpy':np.__version__,'cuda':torch.version.cuda,'device':device,
              'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
              'gpu_driver_snapshot':gpu_snapshot(),'ram_total_bytes':psutil.virtual_memory().total,
              'command':sys.argv,'seeds':args.seeds,'rank':args.rank,'control':args.control,'batch':32,'precision':'FP32',
              'calibration_steps':128,'recovery_steps':512,'state':'running',
              'scope':'H01 four-method synthetic first pass; H04 admission',
              'limits':{'gpu_bytes':12*1024**3,'rss_bytes':24*1024**3,'variant_seconds':600},
              'measurement_limitations':['WDDM process GPU memory unavailable; total device bound used',
                  'RSS sampled; not a continuous peak','no full recovery-time-to-threshold test',
                  'secondary methods and full hypothesis not confirmed']}
    write_json(args.output/'manifest.json',manifest)
    try:
        write_json(args.output/'admission.json',{'H01':invariants(),'H04':h04_check()})
        if not args.admission_only:
            h01(args.output,args.seeds,device,args.rank,args.control)
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        manifest['ended']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        manifest['gpu_driver_final']=gpu_snapshot()
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':
    main()
