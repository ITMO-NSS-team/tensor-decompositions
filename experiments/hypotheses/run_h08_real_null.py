"""H08: separately frozen Gaussian null envelopes for all real batch shapes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.stats import beta
import tensorly as tl
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_h08_null_calibration import full_search,working_rule
from run_local import git,guard,gpu_snapshot,sha,sync,tensor_hash,write_json


def real_rank_candidates(batch):
    if batch not in (128,80,56,16):
        raise ValueError('batch shape outside the registered real protocol')
    # Clamping can produce identical tuples for the final sixteen examples.
    return tuple(dict.fromkeys((min(a,batch),min(c,256,4*batch),2,2)
                               for a,c in ((8,32),(16,64),(32,128))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--admission-only',action='store_true')
    args=parser.parse_args()
    git('merge-base','--is-ancestor','a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b','HEAD')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    sources=['run_h08_real_null.py','run_h08_null_calibration.py','run_local.py']
    for name in sources:
        (args.output/(name+'.source')).write_bytes(Path(__file__).with_name(name).read_bytes())
    manifest={'hypothesis_id':'H08','stage':'real-shape Gaussian null calibration','state':'running',
              'git_sha':git('rev-parse','HEAD'),'source_hashes':{name:sha(Path(__file__).with_name(name)) for name in sources},
              'protocol_sha256':sha(Path(__file__).with_name('H08_signal_denoising.md')),
              'torch':torch.__version__,'tensorly':tl.__version__,'device':args.device,'precision':'FP32',
              'gpu_snapshot':gpu_snapshot(),'batch_shapes':[128,80,56,16],'calibration_samples':200,'validation_samples':200,
              'restarts':2,'sweeps':5,'max_seconds_per_shape':1800,'final_test_opened':False,
              'limitations':['pure Gaussian shape-specific envelope, not a neural experiment or RMT guarantee',
                             'deduplicate identical clamped rank tuples on tail16, in both calibration and working rule',
                             'single envelope reused across checkpoints; setup cost must be charged explicitly']}
    write_json(args.output/'manifest.json',manifest);tl.set_backend('pytorch')
    try:
        summaries=[]
        for index,batch in enumerate(manifest['batch_shapes']):
            ranks=real_rank_candidates(batch)
            if args.admission_only:
                # Small independent envelope check uses real tail dimensions and fixed paths.
                if batch!=16:continue
                x=torch.randn(batch,256,2,2,generator=torch.Generator().manual_seed(9917),device='cpu')
                statistic,paths=full_search(x,ranks)
                assert len(paths)==2*len(ranks)
                assert all(len(path['history'])==5 for path in paths)
                _,_,rejected=working_rule(x,statistic+1.,rank_candidates=ranks)
                assert not rejected['detected']
                write_json(args.output/'admission.json',{'shape':list(x.shape),'ranks':ranks,'paths':len(paths),
                    'high_threshold_zero':'passed','input_sha256':tensor_hash(x)})
                continue
            sd=args.output/f'batch-{batch}';sd.mkdir();values=[];validation=[]
            sync();start=time.perf_counter()
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            for seed in range(910000+index*10000,910200+index*10000):
                x=torch.randn(batch,256,2,2,generator=torch.Generator(device=args.device).manual_seed(seed),device=args.device)
                maximum,_=full_search(x,ranks)
                values.append({'seed':seed,'max_statistic':maximum})
                guard(start,1800)
                if len(values)%25==0:
                    write_json(sd/'calibration.json',values)
                    print(json.dumps({'batch':batch,'calibrated':len(values)}),flush=True)
            threshold=float(np.quantile([item['max_statistic'] for item in values],.95,method='higher'))
            sync();calibration_seconds=time.perf_counter()-start
            frozen={'shape':[batch,256,2,2],'rank_candidates':ranks,'threshold':threshold,
                    'calibration_seconds':calibration_seconds,'frozen_before_validation':True,
                    'calibration_sha256':sha(sd/'calibration.json'),'dtype':'FP32'}
            write_json(sd/'frozen-threshold.json',frozen)
            for seed in range(1010000+index*10000,1010200+index*10000):
                x=torch.randn(batch,256,2,2,generator=torch.Generator(device=args.device).manual_seed(seed),device=args.device)
                _,_,result=working_rule(x,threshold,rank_candidates=ranks)
                validation.append({'seed':seed,**result})
                guard(start,1800)
                if len(validation)%25==0:write_json(sd/'validation.json',validation)
            false=sum(item['detected'] for item in validation);n=len(validation)
            interval=[float(beta.ppf(.025,false,n-false+1)) if false else 0.,
                      float(beta.ppf(.975,false+1,n-false)) if false<n else 1.]
            sync();summary={**frozen,'false_positives':false,'validation_samples':n,'false_positive_fraction':false/n,
                    'clopper_pearson_95':interval,'total_seconds':time.perf_counter()-start,
                    'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if args.device=='cuda' else 0}
            summaries.append(summary);write_json(args.output/'summary.json',summaries)
        manifest['state']='admission_passed' if args.admission_only else 'completed'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',
                        error=str(error),traceback=traceback.format_exc());raise
    finally:write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
