"""H09: paired continuation from a fresh five-epoch ResNet warmup.

Save the same observation grid for every method. Freeze the earliest qualifying
checkpoint before test evaluation; a censored branch uses its final checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import psutil
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_h09_synthetic import CompressedAdam, admission
from run_h04_real import evaluate, seed_all
from run_local import git, gpu_snapshot, guard, sha, sync, write_json

BASE_SHA = 'a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b'
METHODS = ('dense', 'fixed', 'adaptive', 'fixed_paid')
TIMING_SCOPE = {
    'branch_origin': 'after prior-work synchronization, before model deepcopy and optimizer construction',
    'branch_setup_included': True,
    'quality_endpoint': 'tuning evaluation, model checkpoint save and current checkpoint SHA256 complete',
    'current_checkpoint_sha_included_in_observation': True,
    'observation_json': 'outside its own observation timestamp; included in later observations and branch total',
    'branch_total_endpoint': 'resume save, frozen-buffer checks, batch-index SHA256 and resume SHA256 complete',
    'batch_indices_and_resume_sha_included_in_branch_total': True,
    'final_reporting_excluded': True,
    'final_reporting': 'resource telemetry sampling and final projection-events.json/branch.json serialization',
    'common_warmup_excluded': True,
    'common_setup': 'shared five-epoch warmup, checkpoint loading, dataset construction and initial tuning evaluation; warmup cost reported separately',
}


def fixed_batches(indices,steps,seed):
    """Carry the epoch tail forward instead of shortening a neural update."""
    source=torch.as_tensor(indices,dtype=torch.int64)
    if len(source)==0 or steps<0:raise ValueError('nonempty split and nonnegative steps required')
    generator=torch.Generator().manual_seed(seed)
    permutations=[];count=0
    while count<steps*128:
        permutations.append(torch.randperm(len(source),generator=generator));count+=len(source)
    positions=torch.cat(permutations)[:steps*128].reshape(steps,128) if steps else torch.empty(0,128,dtype=torch.int64)
    return source[positions]


def continuation_optimizer(model, warm_state, method, rank, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(warm_state))
    for group in optimizer.param_groups:
        group['lr'] = 1e-4
        group['weight_decay'] = 1e-4
    target = model.layer3[0].conv2.weight
    compressed = None
    if method != 'dense':
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p is not target]
        optimizer.state.pop(target, None)
        compressed = CompressedAdam((256, 2304), target.device, rank=rank,
                                    refresh=method, seed=seed, solver='rsvd')
    return optimizer, compressed


@torch.no_grad()
def compressed_conv_step(target, compressed, step, lr=1e-4, weight_decay=1e-4):
    """Expose the same weight storage as a matrix, without editing its gradient."""
    if target.grad is None:
        raise ValueError('missing target convolution gradient')
    original = target.grad.detach().clone()
    matrix = target.detach().reshape(target.shape[0], -1)
    if matrix.data_ptr() != target.data_ptr():
        raise ValueError('target matrix must share convolution storage')
    matrix.grad = target.grad.detach().reshape_as(matrix)
    matrix.mul_(1-lr*weight_decay)
    compressed.step(matrix, step, lr=lr)
    torch.testing.assert_close(target.grad, original, rtol=0, atol=0)


def run_branch(warm, warm_state, data, splits, seed, method, rank, device, directory, steps):
    sync(); start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model = copy.deepcopy(warm)
    optimizer, compressed = continuation_optimizer(model, warm_state, method, rank, seed)
    seed_all(seed+60000)
    batches=fixed_batches(splits['recovery'],steps,seed+60000)
    torch.save(batches,directory/'batch-indices.pt')
    initial_bn={name:value.detach().clone() for name,value in model.named_buffers()}
    observations = []
    digest = hashlib.sha256()
    sync(); setup_seconds = time.perf_counter()-start
    for step in range(steps+1):
        if step % 10 == 0 or step == steps:
            quality = evaluate(model, data['tuning'], device)
            cp = directory/f'step-{step:03d}.pt'
            torch.save(model.state_dict(), cp)
            checkpoint_sha256 = sha(cp)
            sync()
            observations.append({'step': step, 'tuning': quality, 'elapsed_seconds': time.perf_counter()-start,
                                 'checkpoint': str(cp.resolve()), 'checkpoint_sha256': checkpoint_sha256})
            write_json(directory/'observations.json', observations)
        if step == steps:
            break
        # Eval freezes shared warmup BN buffers; all parameters remain trainable.
        model.eval()
        x,y=default_collate([data['augmented'][int(index)] for index in batches[step]])
        digest.update(x.contiguous().numpy().tobytes()); digest.update(y.numpy().tobytes())
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            loss = F.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise ArithmeticError('nonfinite H09 neural loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ArithmeticError('nonfinite H09 neural gradient')
        optimizer.step()
        if compressed is not None:
            compressed_conv_step(model.layer3[0].conv2.weight, compressed, step)
        guard(start, 3600)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'compressed': compressed.state_dict() if compressed is not None else None,
                'torch_rng': torch.get_rng_state(), 'batch_indices':batches,'next_step':steps}, directory/'resume.pt')
    for name,value in model.named_buffers():
        torch.testing.assert_close(value,initial_bn[name],rtol=0,atol=0)
    batch_indices_sha256 = sha(directory/'batch-indices.pt')
    resume_sha256 = sha(directory/'resume.pt')
    sync()
    branch_seconds = time.perf_counter()-start
    metadata = {'seed': seed, 'method': method, 'steps': steps, 'rank': rank if compressed else None,
                'batch_sha256': digest.hexdigest(), 'seconds': branch_seconds, 'setup_seconds': setup_seconds,
                'timing_scope': dict(TIMING_SCOPE),
                'batch_indices_sha256':batch_indices_sha256,'effective_batch':128,'shared_bn_frozen':True,
                'refresh_count': compressed.updates if compressed else 0,
                'rss_final_bytes': psutil.Process().memory_info().rss,
                'ram_process_peak_bytes': getattr(psutil.Process().memory_info(),'peak_wset',None),
                'ram_peak_scope': 'OS peak working set since process start, not reset per branch',
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else 0,
                'resume_sha256': resume_sha256}
    if compressed is not None:
        write_json(directory/'projection-events.json', compressed.events)
    write_json(directory/'branch.json', metadata)
    del model, optimizer, compressed
    if device == 'cuda':
        torch.cuda.empty_cache()
    return metadata, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--admission-only', action='store_true')
    args = parser.parse_args()
    if git('merge-base', '--is-ancestor', BASE_SHA, 'HEAD').strip():
        raise RuntimeError('base ancestry check unexpectedly produced output')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sources = ['run_h09_real.py', 'run_h09_synthetic.py', 'run_h04_real.py', 'run_local.py', 'train_cifar_baseline.py']
    for name in sources:
        (args.output/(name+'.source')).write_bytes(Path(__file__).with_name(name).read_bytes())
    manifest = {'hypothesis_id': 'H09', 'setting': 'real', 'state': 'admission', 'git_sha': git('rev-parse', 'HEAD'),
                'base_sha': BASE_SHA, 'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in sources},
                'protocol_sha256': sha(Path(__file__).with_name('H09_gradient_drift.md')),
                'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
                'device': args.device, 'gpu_snapshot': gpu_snapshot(), 'seeds': [101] if args.pilot else [101,202,303],
                'methods': METHODS, 'rank': 64, 'steps': 20 if args.pilot else 400, 'final_test_opened': False,
                'precision': 'FP32 states/gradients/QR/SVD; BF16 neural autocast',
                'batch_policy':'128constant; carry shuffled epoch tails into next epoch; no dropped examples',
                'batchnorm_policy':'shared fresh-warmup running statistics frozen; affine parameters trainable',
                'timing_scope': dict(TIMING_SCOPE),
                'quality_observation_interval': 10, 'initial_warmup': 'five epochs constant lr .001',
                'limits': {'gpu_bytes': 12*2**30, 'ram_bytes': 24*2**30, 'branch_seconds': 3600},
                'limitations': ['rank64 preregistered main; rank32 tuning is a separate pending branch',
                    'all methods save the same observation-grid model states; that cost is included',
                    'quality-time criterion uninformative if dense tuning gain is below2percentage points',
                    'one projected matrix; full-coordinate EF memory charged; no total model memory guarantee',
                    'n=3 paired t intervals conditional on approximate normality',
                    'observations every10steps; earliest checkpoint only resolved on that grid']}
    write_json(args.output/'manifest.json', manifest)
    try:
        write_json(args.output/'admission.json', admission())
        if args.admission_only:
            manifest['state'] = 'admission_passed'; return
        warm_manifest = json.loads((args.warmup/'manifest.json').read_text(encoding='utf-8'))
        if (warm_manifest['state'], warm_manifest.get('mode'), warm_manifest['epochs'], warm_manifest.get('lr_schedule')) != (
                'completed', 'h09_fresh_warmup', 5, 'constant'):
            raise ValueError('H09 requires fresh constant-lr five-epoch checkpoints')
        if sha(args.data/'cifar-10-python.tar.gz') != warm_manifest['data_archive_sha256']:
            raise ValueError('CIFAR archive hash changed')
        if sha(args.warmup/'splits.json') != warm_manifest['split_sha256']:
            raise ValueError('warmup split hash changed')
        splits = json.loads((args.warmup/'splits.json').read_text(encoding='utf-8'))
        mean, std = warm_manifest['normalization']['mean'], warm_manifest['normalization']['std']
        plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        augmented = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(.5),
                                        transforms.ToTensor(), transforms.Normalize(mean, std)])
        data = {'augmented': datasets.CIFAR10(args.data, train=True, download=False, transform=augmented)}
        data['tuning'] = DataLoader(Subset(datasets.CIFAR10(args.data, train=True, download=False, transform=plain),
                                          splits['tuning']), batch_size=128, shuffle=False)
        manifest.update(state='running', warmup_manifest_sha256=sha(args.warmup/'manifest.json'),
                        data_archive_sha256=warm_manifest['data_archive_sha256'], split_sha256=warm_manifest['split_sha256'])
        write_json(args.output/'manifest.json', manifest)
        rows = []
        for seed in manifest['seeds']:
            source = args.warmup/f'seed-{seed}'
            result = json.loads((source/'result.json').read_text(encoding='utf-8'))
            if result['state'] != 'warmup_ready' or sha(source/'resume.pt') != result['resume_sha256']:
                raise ValueError('warmup checkpoint failed provenance check')
            state = torch.load(source/'resume.pt', map_location=args.device)
            if state['epoch'] != 5:
                raise ValueError('warmup epoch mismatch')
            warm = models.resnet18(weights=None, num_classes=10).to(args.device)
            warm.load_state_dict(state['model'])
            sd = args.output/f'seed-{seed}'; sd.mkdir()
            before = evaluate(warm, data['tuning'], args.device)
            reference_batches = None
            threshold = None
            for method in METHODS:
                md = sd/method; md.mkdir()
                row, observations = run_branch(warm, state['optimizer'], data, splits, seed, method, 64,
                                               args.device, md, manifest['steps'])
                if reference_batches is None:
                    reference_batches = row['batch_sha256']
                elif row['batch_sha256'] != reference_batches:
                    raise ArithmeticError('paired training/augmentation inputs differ')
                if method == 'dense':
                    dense_end = observations[-1]['tuning']['accuracy']
                    threshold = dense_end-.01
                    dense_gain = dense_end-before['accuracy']
                    write_json(sd/'frozen-quality-boundary.json', {'threshold': threshold, 'dense_gain': dense_gain,
                        'criterion_informative': dense_gain >= .02, 'frozen_before_compressed_runs': True,
                        'dense_end_checkpoint_sha256': observations[-1]['checkpoint_sha256']})
                eligible = [item for item in observations if item['tuning']['accuracy'] >= threshold]
                chosen = eligible[0] if eligible else observations[-1]
                row.update(selected=chosen, quality_reached=bool(eligible), censored_at_steps=None if eligible else manifest['steps'],
                           time_to_quality_seconds=chosen['elapsed_seconds'] if eligible else None,
                           criterion_informative=dense_gain >= .02, threshold=threshold,
                           warmup_sha256=result['resume_sha256'], status='completed', frozen_before_test=True)
                rows.append(row); write_json(args.output/'runs.json', rows)
                print(json.dumps({'seed': seed, 'method': method, 'quality_reached': bool(eligible)}), flush=True)
            del warm, state
        if not args.pilot:
            for row in rows:
                if sha(Path(row['selected']['checkpoint']))!=row['selected']['checkpoint_sha256']:
                    raise ValueError('selected checkpoint changed before opening final test')
            manifest['all_selected_checkpoints_verified_before_test']=True
            test = DataLoader(datasets.CIFAR10(args.data, train=False, download=False, transform=plain), batch_size=128)
            manifest['final_test_opened'] = True
            write_json(args.output/'manifest.json', manifest)
            for row in rows:
                cp = Path(row['selected']['checkpoint'])
                if sha(cp) != row['selected']['checkpoint_sha256']:
                    raise ValueError('selected checkpoint changed before final test')
                model = models.resnet18(weights=None, num_classes=10).to(args.device)
                model.load_state_dict(torch.load(cp, map_location=args.device, weights_only=True))
                row['final_test'] = evaluate(model, test, args.device)
                write_json(args.output/'runs.json', rows); del model
        manifest['state'] = 'completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error, (MemoryError, TimeoutError)) else 'implementation_failure',
                        error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output/'manifest.json', manifest)


if __name__ == '__main__':
    main()
