"""H08 real activation intervention, inference only; adaptive versus energy.

Independent-probe baseline and correlated-noise controls remain separate stages.
No final-test labels participate in group-wise unsupervised reconstruction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import psutil
import tensorly as tl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets,models,transforms

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_h08_real_null import real_rank_candidates
from run_h08_null_calibration import working_rule
from synthetic_tucker_common import left_svd
from run_local import git,guard,gpu_snapshot,sha,sync,write_json


def energy_reconstruction(x,epsilon=.10):
    factors=[];ranks=[]
    energy=x.square().sum()
    if float(energy)==0.:return torch.zeros_like(x),{'ranks':None,'relative_error':None}
    for mode in range(4):
        u,s,_=left_svd(tl.unfold(x,mode),rank=min(tl.unfold(x,mode).shape))
        tails=torch.cat([s.square().flip(0).cumsum(0).flip(0),s.new_zeros(1)])
        eligible=torch.nonzero(tails[1:]<=epsilon**2*energy/4).flatten()
        rank=int(eligible[0])+1 if len(eligible) else len(s)
        ranks.append(rank);factors.append(u[:,:rank])
    core=tl.tenalg.multi_mode_dot(x,[factor.T for factor in factors])
    result=tl.tenalg.multi_mode_dot(core,factors)
    relative=float((result-x).norm()/x.norm())
    if relative>epsilon+1e-5:raise ArithmeticError('H08 energy reconstruction violated epsilon')
    return result,{'ranks':ranks,'relative_error':relative}


def prefix(model,x):
    x=model.maxpool(model.relu(model.bn1(model.conv1(x))))
    return model.layer3[0](model.layer2(model.layer1(x)))


def suffix(model,x):
    x=model.layer4(model.layer3[1](x))
    return model.fc(torch.flatten(model.avgpool(x),1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--null',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    git('merge-base','--is-ancestor','a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b','HEAD')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    names=['run_h08_real.py','run_h08_real_null.py','run_h08_null_calibration.py','synthetic_tucker_common.py','run_local.py']
    for name in names:(args.output/(name+'.source')).write_bytes(Path(__file__).with_name(name).read_bytes())
    manifest={'hypothesis_id':'H08','setting':'real','state':'preparing','git_sha':git('rev-parse','HEAD'),
              'source_hashes':{name:sha(Path(__file__).with_name(name)) for name in names},
              'protocol_sha256':sha(Path(__file__).with_name('H08_signal_denoising.md')),
              'torch':torch.__version__,'torchvision':torchvision.__version__,'tensorly':tl.__version__,'device':args.device,
              'gpu_snapshot':gpu_snapshot(),'seeds':[101] if args.pilot else [101,202,303],
              'methods':['full_noisy','energy','adaptive_null'],'controls':['clean'],
              'noise':'known IID Gaussian, sigma=.1*||clean||F/sqrt(numel)',
              'inference_only':True,'final_test_opened':False,'batch':128,'max_seconds_per_seed':1800,
              'limitations':['independent16probe baseline and correlated-noise/zero-noise controls pending',
                             'test-group reconstruction is unsupervised transductive intervention, registered in advance',
                             'Gaussian null envelope is shape-specific, not a model of natural activation noise',
                             'no network recovery training or native compressed convolution claims',
                             'null setup charged once in each full-cost comparison; amortization reported separately']}
    write_json(args.output/'manifest.json',manifest);tl.set_backend('pytorch')
    try:
        nm=json.loads((args.null/'manifest.json').read_text(encoding='utf-8'))
        if nm['state']!='completed' or nm['precision']!='FP32':raise ValueError('missing completed FP32 shape envelopes')
        if nm['device']!=args.device:raise ValueError('null and denoising procedure must use the same device')
        thresholds={}
        for batch in (128,80,56,16):
            path=args.null/f'batch-{batch}'/'frozen-threshold.json'
            value=json.loads(path.read_text(encoding='utf-8'))
            if tuple(tuple(rank) for rank in value['rank_candidates'])!=real_rank_candidates(batch):
                raise ValueError('null path candidates differ from inference candidates')
            if sha(args.null/f'batch-{batch}'/'calibration.json')!=value['calibration_sha256']:
                raise ValueError('noise calibration hash mismatch')
            thresholds[batch]=value['threshold']
        null_summary=json.loads((args.null/'summary.json').read_text(encoding='utf-8'))
        setup_seconds=sum(item['total_seconds'] for item in null_summary)
        base=json.loads((args.baseline/'manifest.json').read_text(encoding='utf-8'))
        if base['state']!='completed' or base['epochs']!=30:raise ValueError('shared baseline is not complete')
        if sha(args.data/'cifar-10-python.tar.gz')!=base['data_archive_sha256']:raise ValueError('CIFAR archive changed')
        mean,std=base['normalization']['mean'],base['normalization']['std']
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
        if args.pilot:
            from torch.utils.data import Subset
            splits=json.loads((args.baseline/'splits.json').read_text(encoding='utf-8'))
            source=Subset(datasets.CIFAR10(args.data,train=True,download=False,transform=transform),splits['tuning'][:128])
        else:
            source=datasets.CIFAR10(args.data,train=False,download=False,transform=transform)
            manifest['final_test_opened']=True
        loader=DataLoader(source,batch_size=128,shuffle=False)
        manifest.update(state='running',null_manifest_sha256=sha(args.null/'manifest.json'),null_setup_seconds=setup_seconds,
                        thresholds=thresholds,data_archive_sha256=base['data_archive_sha256'])
        write_json(args.output/'manifest.json',manifest);rows=[]
        with torch.inference_mode():
            for seed in manifest['seeds']:
                cp=args.baseline/f'seed-{seed}'/'model.pt'
                expected=json.loads((cp.parent/'result.json').read_text(encoding='utf-8'))['checkpoint_sha256']
                if sha(cp)!=expected:raise ValueError('baseline checkpoint changed')
                model=models.resnet18(weights=None,num_classes=10).to(args.device).eval()
                model.load_state_dict(torch.load(cp,map_location=args.device,weights_only=True))
                metrics={method:{'correct':0,'loss_sum':0.,'error_squared':0.,'clean_energy':0.,'n':0,'denoise_seconds':0.}
                         for method in ('clean','full_noisy','energy','adaptive_null')}
                details=[];start=time.perf_counter()
                for batch_index,(images,labels) in enumerate(loader):
                    images=images.to(args.device);labels=labels.to(args.device)
                    with torch.autocast(device_type=args.device,dtype=torch.bfloat16,enabled=args.device=='cuda'):
                        clean=prefix(model,images).float()
                    if tuple(clean.shape[1:])!=(256,2,2):raise ValueError('actual intervention shape differs')
                    sigma=.1*clean.norm()/clean.numel()**.5
                    generator=torch.Generator(device=args.device).manual_seed(seed+400000+batch_index)
                    noisy=clean+sigma*torch.randn(clean.shape,generator=generator,device=args.device)
                    for method in metrics:
                        sync();denoise_start=time.perf_counter();info={}
                        if method=='clean':value=clean
                        elif method=='full_noisy':value=noisy
                        elif method=='energy':value,info=energy_reconstruction(noisy)
                        else:
                            if float(sigma)==0.:value=torch.zeros_like(noisy);info={'zero_noise_branch':True}
                            else:
                                core,factors,info=working_rule(noisy,thresholds[len(images)],float(sigma),real_rank_candidates(len(images)))
                                value=tl.tenalg.multi_mode_dot(core,factors) if core is not None else torch.zeros_like(noisy)
                        sync();elapsed=time.perf_counter()-denoise_start
                        with torch.autocast(device_type=args.device,dtype=torch.bfloat16,enabled=args.device=='cuda'):
                            logits=suffix(model,value)
                        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():raise ArithmeticError('nonfinite intervention')
                        item=metrics[method];item['correct']+=int((logits.argmax(1)==labels).sum());item['n']+=len(images)
                        item['loss_sum']+=float(F.cross_entropy(logits.float(),labels,reduction='sum'))
                        item['error_squared']+=float((value-clean).square().sum());item['clean_energy']+=float(clean.square().sum())
                        item['denoise_seconds']+=elapsed
                        details.append({'batch':batch_index,'shape':list(clean.shape),'method':method,'sigma':float(sigma),
                                        'seconds':elapsed,**info})
                    guard(start,1800)
                for method,item in metrics.items():
                    row={'seed':seed,'method':method,'status':'completed','checkpoint_sha256':expected,
                         'n':item['n'],'accuracy':item['correct']/item['n'],'cross_entropy':item['loss_sum']/item['n'],
                         'relative_clean_error':(item['error_squared']/item['clean_energy'])**.5 if item['clean_energy'] else None,
                         'denoise_seconds':item['denoise_seconds'],
                         'null_setup_seconds':setup_seconds if method=='adaptive_null' else 0.}
                    rows.append(row)
                write_json(args.output/f'seed-{seed}-batches.json',details);write_json(args.output/'runs.json',rows)
                memory=psutil.Process().memory_info()
                manifest['ram_process_peak_bytes']=getattr(memory,'peak_wset',memory.rss)
                print(json.dumps({'seed':seed,'state':'completed'}),flush=True);del model
        manifest['state']='completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
