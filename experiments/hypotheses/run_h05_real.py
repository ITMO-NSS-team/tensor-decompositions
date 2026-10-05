"""H05 paired exact/sketched Tucker LS on real CIFAR checkpoint weights."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import subprocess
import platform
import torchvision
import psutil
import torch
import tensorly as tl
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,models,transforms

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local import git,guard,sha,sync,write_json,gpu_snapshot
from run_h04_real import TuckerConv,admission as conv_admission,evaluate,latency,recover,seed_all
from run_h05_synthetic import METHODS,factorize,admission as ls_admission
from run_h07_synthetic import sampled_tucker,admission as sample_admission


def main(hypothesis='H05'):
    if hypothesis not in ('H05','H07'):raise ValueError(hypothesis)
    methods=METHODS if hypothesis=='H05' else ('uniform','column_norm','leverage','contraction')
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--admission-only',action='store_true');parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    base_sha='a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b'
    subprocess.run(['git','merge-base','--is-ancestor',base_sha,'HEAD'],check=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=['run_h05_real.py','run_h05_synthetic.py','run_h07_synthetic.py','synthetic_tucker_common.py','run_h02_synthetic.py','run_h04_real.py','run_local.py']
    if hypothesis=='H07':sources.append('run_h07_real.py')
    sources.append(next(Path(__file__).parent.glob(hypothesis+'_*.md')).name)
    hashes={}
    for name in sources:
        path=Path(__file__).with_name(name);hashes[name]=sha(path);(args.output/(name+'.source')).write_bytes(path.read_bytes())
    manifest={'state':'admission','hypothesis':hypothesis,'setting':'real','seeds':[101] if args.pilot else [101,202,303],
              'rank':[64,64,3,3],'sketch_rows':256,'max_sweeps':10,'torch':torch.__version__,'device':args.device,
              'source_hashes':hashes,'git_sha':git('rev-parse','HEAD'),'command':sys.argv,
              'base_sha':base_sha,'python':platform.python_version(),'torchvision':torchvision.__version__,
              'tensorly':tl.__version__,'cuda':torch.version.cuda,'gpu_driver_snapshot':gpu_snapshot(),
              'protocol_sha256':sha(next(Path(__file__).parent.glob(hypothesis+'_*.md'))),
              'limitations':['protocol-default sketch/sampling size256 used;128/512 secondary tuning not run',
                             'true LS validation and exact core projection costs included',
                             'BN frozen to common original teacher buffers; only target factors/core trained',
                             'full3x3 spatial factors absorbed into trainable core identically in all neural branches',
                             'full time-to-quality recovery trajectory not available'],
              'limits':{'gpu_bytes':12*1024**3,'rss_bytes':24*1024**3,'variant_seconds':2700}}
    if hypothesis=='H07':
        manifest['columns']=256
        manifest['limitations'][1]='logical read counter excludes opaque QR/SVD internal rereads; complete16T pass bound not certified'
    write_json(args.output/'manifest.json',manifest)
    try:
        with tl.backend_context('pytorch'):
            admission_result={'algorithm':ls_admission() if hypothesis=='H05' else sample_admission(),'convolution':conv_admission()}
        write_json(args.output/'admission.json',admission_result)
        if args.admission_only:manifest['state']='admission_passed';return
        base=json.loads((args.baseline/'manifest.json').read_text(encoding='utf-8'))
        assert base['state']=='completed'
        assert sha(args.data/'cifar-10-python.tar.gz')==base['data_archive_sha256']
        assert sha(args.baseline/'splits.json')==base['split_sha256']
        splits=json.loads((args.baseline/'splits.json').read_text(encoding='utf-8'));norm=base['normalization']
        manifest.update(archive_sha256=base['data_archive_sha256'],split_sha256=base['split_sha256'])
        normalize=transforms.Normalize(norm['mean'],norm['std'])
        plain_transform=transforms.Compose([transforms.ToTensor(),normalize])
        plain=datasets.CIFAR10(args.data,train=True,download=False,transform=plain_transform)
        aug=datasets.CIFAR10(args.data,train=True,download=False,transform=transforms.Compose([
            transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]))
        tune=DataLoader(Subset(plain,splits['tuning']),batch_size=128,shuffle=False)
        rows=[];manifest['state']='running'
        for seed in manifest['seeds']:
            sd=args.output/f'seed-{seed}';sd.mkdir();cp=args.baseline/f'seed-{seed}'/'model.pt'
            expected=json.loads((cp.parent/'result.json').read_text(encoding='utf-8'))['checkpoint_sha256']
            assert sha(cp)==expected
            teacher=models.resnet18(weights=None,num_classes=10).to(args.device)
            teacher.load_state_dict(torch.load(cp,map_location=args.device,weights_only=True));teacher.eval()
            write_json(sd/'teacher.json',{'checkpoint_sha256':expected,'tuning':evaluate(teacher,tune,args.device)})
            for method in ([methods[1]] if args.pilot else methods):
                start=time.perf_counter();seed_all(seed)
                if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
                try:
                    with tl.backend_context('pytorch'):
                        if hypothesis=='H05':
                            core,factors,ls_rows,info=factorize(teacher.layer3[0].conv2.weight.detach(),(64,64,3,3),method,seed,
                                                              sketch_rows=256,max_sweeps=1 if args.pilot else 10,
                                                              guard=SimpleNamespace(check=lambda:guard(start,2700)))
                        else:
                            sync();factor_start=time.perf_counter()
                            core,factors,info=sampled_tucker(teacher.layer3[0].conv2.weight.detach(),method,seed,b=256,ranks=(64,64,3,3))
                            sync();info['factor_seconds']=time.perf_counter()-factor_start;ls_rows=[]
                        spatial=tl.tenalg.multi_mode_dot(core,factors[2:],modes=[2,3])
                    guard(start,2700);write_json(sd/f'{method}-LS.json',ls_rows)
                    student=copy.deepcopy(teacher);student.layer3[0].conv2=TuckerConv(spatial,factors[0],factors[1])
                    before=evaluate(student,tune,args.device)
                    history=recover(student,{'calibration':plain,'recovery':aug},splits,seed,args.device,start,
                                    steps=(20,0) if args.pilot else (128,400))
                    cp=sd/f'{method}.pt';torch.save(student.state_dict(),cp)
                    row={'seed':seed,'method':method,'status':'completed',**info,'tuning_before':before,
                         'tuning_after':evaluate(student,tune,args.device),'checkpoint_sha256':sha(cp),
                         'parameters':sum(p.numel() for p in student.layer3[0].conv2.parameters()),
                         'total_seconds':time.perf_counter()-start}
                    row.update(rss_bytes=psutil.Process().memory_info().rss,
                               gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
                               gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else 0)
                    write_json(sd/f'{method}-history.json',history);del student
                except ArithmeticError as error:
                    row={'seed':seed,'method':method,'status':'stopped','reason':str(error),
                         'total_seconds':time.perf_counter()-start,'scope':'failed numerical/true LS guard; no hidden redraw'}
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps(row),flush=True)
            del teacher
        if not args.pilot:
            test=datasets.CIFAR10(args.data,train=False,download=False,transform=plain_transform)
            loader=DataLoader(test,batch_size=128,shuffle=False);final=[];x=next(iter(tune))[0][:64].to(args.device)
            for seed in manifest['seeds']:
                for method in ('original',*methods):
                    if method!='original' and not any(row['seed']==seed and row['method']==method and row['status']=='completed' for row in rows):continue
                    model=models.resnet18(weights=None,num_classes=10).to(args.device).eval()
                    cp=args.baseline/f'seed-{seed}'/'model.pt' if method=='original' else args.output/f'seed-{seed}'/f'{method}.pt'
                    state=torch.load(cp,map_location=args.device,weights_only=True)
                    if method!='original':
                        prefix='layer3.0.conv2.'
                        model.layer3[0].conv2=TuckerConv(state[prefix+'core'],state[prefix+'u'],state[prefix+'v'])
                    model.load_state_dict(state);model.eval()
                    item={'seed':seed,'method':method,'checkpoint_sha256':sha(cp),'test':evaluate(model,loader,args.device),
                          'latency_batch1':latency(model,x[:1],args.device,True),'latency_batch64':latency(model,x,args.device,True)}
                    final.append(item);write_json(args.output/'final.json',final);del model
        manifest['state']='completed' if all(row['status']=='completed' for row in rows) else 'completed_with_stops'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(TimeoutError,MemoryError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
