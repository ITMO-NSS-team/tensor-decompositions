"""H10 pinned Switch/WikiText denoising traces and virtual four-device placement."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
import tensorly as tl
import transformers
from transformers import AutoTokenizer,SwitchTransformersForConditionalGeneration

sys.path.insert(0,str(Path(__file__).resolve().parent))
from switch_trace_core import corrupt,AcceptedTrace,sparse_modules,physical_permutation,greedy_placement,validate_actual_load
from run_h03_real import documents_from_rows
from run_local import git,guard,sha,sync,write_json,gpu_snapshot
from run_h02_synthetic import tensor_hash
from tdecomp.tensor.tucker import HOOIDecomposition

REVISION='92fe2d22b024d9937146fe097ba3d3a7ba146e1b'
RANKS=[(4,4,4,4),(2,4,4,4),(4,4,6,4),(4,4,8,4),(8,4,4,4),(4,4,4,8)]


def load_chunks(root,out,pilot):
    provenance=json.loads((root/'switch-wikitext-provenance.json').read_text(encoding='utf-8'))
    if provenance['switch_revision']!=REVISION or provenance['wikitext_revision']!='b08601e04326c79dfdd32d625aee71d232d685c3':
        raise ArithmeticError('pinned Switch/WikiText provenance revision mismatch')
    for record in provenance['files']:
        if sha(record['path'])!=record['sha256']:raise ArithmeticError('pinned input hash mismatch')
    tokenizer=AutoTokenizer.from_pretrained(root/'switch-base-8',local_files_only=True,use_fast=True)
    tokenizer.model_max_length=10**9;partitions={};metadata={}
    for name in ('train','validation','test'):
        if pilot and name=='test':continue
        path=root/'wikitext2/wikitext-2-raw-v1'/f'{name}-00000-of-00001.parquet'
        raw=pq.read_table(path,columns=['text']).column('text').to_pylist()
        nonempty=[row for row in raw if row.strip()];documents=documents_from_rows(nonempty)
        chunks=[];tokens_all=[];tails=[]
        for doc_id,document in enumerate(documents):
            tokens=tokenizer.encode(document,add_special_tokens=False,truncation=False)
            tokens_all.extend(tokens);tails.append(len(tokens)%128)
            for offset in range(0,len(tokens),128):
                chunks.append({'document':doc_id,'offset':offset,'index':len(chunks),'tokens':tokens[offset:offset+128]})
        partitions[name]=chunks
        metadata[name]={'source_sha256':sha(path),'raw_rows':len(raw),'excluded_empty_rows':len(raw)-len(nonempty),
                        'documents':len(documents),'tokens':len(tokens_all),'tokens_sha256':tensor_hash(torch.tensor(tokens_all)),
                        'chunks':len(chunks),'full_chunks':sum(len(chunk['tokens'])==128 for chunk in chunks),
                        'document_tail_tokens':tails,'tail_policy':'retained as variable source length, encoder padding masked'}
        torch.save(chunks,out/f'{name}-chunks.pt')
    full=[chunk for chunk in partitions['train'] if len(chunk['tokens'])==128]
    if len(full)<1536:raise ArithmeticError('fewer than1536fulltrainchunks; sample not reduced')
    if len(partitions['validation'])<256:raise ArithmeticError('fewer than256validationchunks')
    data={'calibration':full[:1024],'tuning':full[1024:1536],'replay':partitions['validation'][:256]}
    if not pilot:data['test']=partitions['test']
    if pilot:data={name:chunks[:16] for name,chunks in data.items()}
    write_json(out/'inputs.json',metadata)
    return tokenizer,data,metadata


def corruption_seed(seed,partition,index):
    offsets={'calibration':100000,'tuning':200000,'replay':300000,'test':400000}
    if seed not in (101,202,303) or partition not in offsets or not 0<=index<100000:
        raise ValueError('corruption stream outside registered nonoverlapping seed/partition ranges')
    return seed*1000000+offsets[partition]+index


def payload_for(chunk,partition,seed,tokenizer):
    sentinels=[tokenizer.convert_tokens_to_ids(f'<extra_id_{index}>') for index in range(100)]
    return corrupt(chunk['tokens'],corruption_seed(seed,partition,chunk['index']),sentinels,tokenizer.eos_token_id,tokenizer.pad_token_id)


@torch.no_grad()
def collect(model,chunks,partition,seed,tokenizer,device,start):
    modules=sparse_modules(model);trace=AcceptedTrace(model)
    counts=torch.zeros(len(chunks),4,8,len(modules));padding=[];dropped=[];nll_sum=0.;tokens=0;hashes=[];state_bytes=set()
    try:
        for index,chunk in enumerate(chunks):
            payload,info=payload_for(chunk,partition,seed,tokenizer);origin=chunk['index']%4
            hashes.append({k:tensor_hash(v) for k,v in payload.items()})
            payload={k:v.to(device) for k,v in payload.items()};trace.begin(origin,payload)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                output=model(**payload,use_cache=False,return_dict=True)
            counts[index,origin]=trace.finish().float()
            state_bytes.add(tuple(trace.state_bytes[i] for i in range(len(modules))))
            if not torch.isfinite(output.logits).all():raise ArithmeticError('nonfinite Switch output')
            loss=F.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),payload['labels'].reshape(-1),reduction='sum')
            nll_sum+=float(loss);tokens+=payload['labels'].numel()
            padding.append(sum(trace.padding.values()));dropped.append(sum(trace.dropped.values()))
            guard(start,3600)
            if (index+1)%256==0:print(json.dumps({'seed':seed,'partition':partition,'chunks':index+1}),flush=True)
    finally:trace.close()
    if len(state_bytes)!=1:raise ArithmeticError('sparse state transport dtype changes across chunks')
    return counts,{'chunks':len(chunks),'accepted_dispatches':int(counts.sum()),'accepted_padding_dispatches':sum(padding),
                   'state_element_bytes':list(next(iter(state_bytes))),
                   'capacity_dropped_dispatches':sum(dropped),'denoising_nll':nll_sum/max(tokens,1),
                   'decoder_tokens':tokens,'payload_sha256':sha_json(hashes),'source_data_scope':'not causal LM perplexity'}


def sha_json(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def price(counts,placements,expert_bytes,state_bytes):
    if placements is None:return None
    byte_cost=counts.new_zeros(len(counts));messages=counts.new_zeros(len(counts));moved=0
    for layer,placement in enumerate(placements):
        for expert,destination in enumerate(placement):
            moved+=expert_bytes[layer][expert] if destination!=expert%4 else 0
            for origin in range(4):
                distance=min((origin-destination)%4,(destination-origin)%4)
                if distance:
                    byte_cost+=distance*counts[:,origin,expert,layer]*2*768*state_bytes[layer]
                    messages+=2*counts[:,origin,expert,layer]
    # Transport retains the observed native residual-state dtype; no extra quantization.
    modeled=byte_cost/(16*1024**3)+messages*10e-6
    return {'ring_weighted_exchange_bytes':float(byte_cost.sum()),'migration_bytes':moved,
            'total_exchange_and_migration_bytes':float(byte_cost.sum())+moved,
            'modeled_window_p95_seconds':float(torch.quantile(modeled,.95)),
            'latency_scope':'communication-only network model16GiB/s10us; computation/shared layers absent'}


@torch.no_grad()
def replay_admission(model,chunks,seed,tokenizer,device,placements,start):
    errors=[];route_hashes=[];trace=AcceptedTrace(model)
    try:
      for chunk in chunks:
        payload,_=payload_for(chunk,'replay',seed,tokenizer);payload={k:v.to(device) for k,v in payload.items()}
        trace.begin(chunk['index']%4,payload)
        reference=model(**payload,use_cache=False).logits
        expected_counts=trace.finish().clone();expected_drop=trace.dropped.copy();expected_pad=trace.padding.copy()
        expected_masks={i:mask.clone() for i,mask in trace.route_masks.items()}
        trace.begin(chunk['index']%4,payload)
        with physical_permutation(model,placements):actual=model(**payload,use_cache=False).logits
        if not torch.equal(expected_counts,trace.finish()) or trace.dropped!=expected_drop or trace.padding!=expected_pad:
            raise ArithmeticError('Switch storage permutation changed accepted/dropped routing counts')
        if any(not torch.equal(mask,trace.route_masks[index]) for index,mask in expected_masks.items()):
            raise ArithmeticError('Switch storage permutation changed token route mask')
        route_hashes.append({i:tensor_hash(mask) for i,mask in expected_masks.items()})
        error=float((reference-actual).norm()/reference.norm().clamp_min(1e-30))
        if error>1e-6:raise ArithmeticError('Switch storage permutation changed full logits')
        errors.append(error)
        guard(start,3600)
    finally:trace.close()
    return {'maximum_output_error':max(errors,default=0),'chunks':len(chunks),'route_masks_equal':True,
            'accepted_and_dropped_counts_equal':True,'reference_routes_sha256':sha_json(route_hashes)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[3]/'audit/datasets')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--pilot',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if transformers.__version__!='4.39.1':raise RuntimeError('reviewednativeSwitchversion4.39.1required')
    source_names=['run_h10_real.py','switch_trace_core.py','run_h03_real.py','gpt2_rotation_core.py',
                  'run_h03_synthetic.py','run_h02_synthetic.py','run_local.py',next(Path(__file__).parent.glob('H10_*.md')).name]
    hashes={}
    for name in source_names:
        path=Path(__file__).with_name(name);hashes[name]=sha(path);(args.output/(name+'.source')).write_bytes(path.read_bytes())
    base_sha='a5eaec03c82d8d1c6ce11bdbb47622e5fd71219b'
    import subprocess
    subprocess.run(['git','merge-base','--is-ancestor',base_sha,'HEAD'],check=True)
    manifest={'state':'preparation','hypothesis':'H10','setting':'real','git_sha':git('rev-parse','HEAD'),'base_sha':base_sha,
              'source_hashes':hashes,'model_revision':REVISION,'data_revision':'b08601e04326c79dfdd32d625aee71d232d685c3',
              'torch':torch.__version__,'transformers':transformers.__version__,'device':args.device,'command':sys.argv,
               'gpu_snapshot':gpu_snapshot(),'seeds':[101] if args.pilot else [101,202,303],
               'corruption_seed_rule':'seed*1000000 + partition_offset + chunk_index; index in[0,100000)',
              'limitations':['physical communication stage C blocked by one GPU; whole H10 indeterminate',
                             'forward parametersFP32, BF16autocast; physical permutation admissionFP32; transport retains observed sparse-state dtype',
                             'accepted padding dispatches included because native routers actually dispatch them',
                             'load/capacity gate enforced onestimatedselection then truecalibration; future violations disclosed',
                             'fixed logical router/capacity decisions preserved through expert storage mapping',
                             'network sensitivity8/32GiB5/20us secondary branches pending'],
              'limits':{'gpu_bytes':12*1024**3,'rss_bytes':24*1024**3,'seconds_per_seed':3600}}
    write_json(args.output/'manifest.json',manifest)
    try:
        tokenizer,data,metadata=load_chunks(args.data,args.output,args.pilot)
        manifest['checkpoint_sha256']=sha(args.data/'switch-base-8/pytorch_model.bin')
        model=SwitchTransformersForConditionalGeneration.from_pretrained(args.data/'switch-base-8',local_files_only=True).to(args.device).eval()
        manifest['unique_parameters']=sum(par.numel() for par in model.parameters())
        if manifest['unique_parameters']!=619339008 or len(sparse_modules(model))!=12:raise ArithmeticError('pinnedSwitcharchitecturemismatch')
        manifest['native_switch_source_sha256']=sha(inspect.getfile(type(model)))
        expert_bytes=[[sum(par.numel()*par.element_size() for par in expert.parameters()) for expert in module.experts.values()] for _,module in sparse_modules(model)]
        manifest['state']='running';rows=[];selected=None
        for seed in manifest['seeds']:
            start=time.perf_counter();sd=args.output/f'seed-{seed}';sd.mkdir()
            cal,calinfo=collect(model,data['calibration'],'calibration',seed,tokenizer,args.device,start)
            tune,tuneinfo=collect(model,data['tuning'],'tuning',seed,tokenizer,args.device,start)
            state_bytes=calinfo['state_element_bytes']
            if state_bytes!=tuneinfo['state_element_bytes']:raise ArithmeticError('transport dtype differs across data splits')
            torch.save({'calibration':cal,'tuning':tune},sd/'statistics.pt');write_json(sd/'trace-info.json',{'calibration':calinfo,'tuning':tuneinfo})
            uniform=[[expert%4 for expert in range(8)] for _ in range(12)]
            full,fulldetails=greedy_placement(cal,'cost');candidates=[]
            for rank in ([RANKS[0]] if selected is not None or args.pilot else RANKS):
                if selected is not None:rank=selected
                sync();factor_start=time.perf_counter()
                with tl.backend_context('pytorch'):
                    core,factors=HOOIDecomposition(rank=rank,random_state=seed).decompose(cal.to(args.device),n_iter_max=20,tol=1e-6)
                    raw=tl.tucker_to_tensor((core,factors)).cpu()
                corrected=raw.clamp_min(0)
                if not float(corrected.sum()):raise ArithmeticError('zero corrected statistics')
                corrected*=cal.sum()/corrected.sum();sync();factor_seconds=time.perf_counter()-factor_start
                pi,details=greedy_placement(corrected,'cost')
                actual_ok,actual_details=validate_actual_load(cal,pi) if pi is not None else (False,[])
                tuning_ok=validate_actual_load(tune,pi)[0] if pi is not None else False
                record={'rank':rank,'placement':pi,'estimated_admission':details,'actual_calibration_admissible':actual_ok,
                        'tuning_load_admissible':tuning_ok,
                        'actual_load_details':actual_details,'factor_seconds':factor_seconds,
                        'coefficients':core.numel()+sum(f.numel() for f in factors),
                        'correction_relative_norm':float((corrected-raw).norm()/raw.norm().clamp_min(1e-30)),
                        'tuning_price':price(tune,pi,expert_bytes,state_bytes) if actual_ok and tuning_ok else None}
                candidates.append(record);guard(start,3600)
            write_json(sd/'rank-search.json',candidates)
            if selected is None:
                reference=price(tune,full,expert_bytes,state_bytes) if full is not None and validate_actual_load(tune,full)[0] else None
                eligible=[row for row in candidates if row['tuning_price'] is not None and reference is not None and
                          row['tuning_price']['total_exchange_and_migration_bytes']<=1.05*reference['total_exchange_and_migration_bytes']]
                selected=min(eligible,key=lambda row:(row['coefficients'],row['rank']))['rank'] if eligible else RANKS[0]
                manifest.update(selected_rank=selected,rank_selection_admissible=bool(eligible));write_json(args.output/'manifest.json',manifest)
            chosen=next(row for row in candidates if tuple(row['rank'])==tuple(selected))
            placements={'uniform':uniform,'full_statistics':full,'tucker_statistics':chosen['placement']}
            gates={method:validate_actual_load(cal,pi)[0] if pi is not None else False for method,pi in placements.items()}
            write_json(sd/'fixed-placements.json',{'placements':placements,'true_calibration_gates':gates,'fixed_before_test':True})
            replay_checks={method:replay_admission(model,data['replay'],seed,tokenizer,args.device,pi,start)
                           for method,pi in placements.items() if gates[method]}
            write_json(sd/'replay-admission.json',replay_checks)
            if not args.pilot and any(gates.values()):
                future,futureinfo=collect(model,data['test'],'test',seed,tokenizer,args.device,start)
                if state_bytes!=futureinfo['state_element_bytes']:raise ArithmeticError('future transport dtype changed')
                torch.save(future,sd/'future-statistics.pt');write_json(sd/'future-info.json',futureinfo)
            else:future=None
            for method,pi in placements.items():
                if not gates[method]:
                    row={'seed':seed,'method':method,'status':'inadmissible','reason':'calibrationload/capacity or greedy gate','placement':pi}
                else:
                    row={'seed':seed,'method':method,'status':'pilot_completed' if args.pilot else 'completed',
                         'placement':pi,'replay_admission':replay_checks[method],'checkpoint_sha256':manifest['checkpoint_sha256'],
                         'future_price':price(future,pi,expert_bytes,state_bytes) if future is not None else None,
                         'future_load_admissible':validate_actual_load(future,pi)[0] if future is not None else None}
                rows.append(row);write_json(args.output/'runs.json',rows);print(json.dumps({'seed':seed,'method':method,'status':row['status']}),flush=True)
            guard(start,3600)
        manifest['state']='completed';manifest['physical_stage_C']='blocked_by_hardware'
    except Exception as error:
        manifest.update(state='resource_failure' if isinstance(error,(MemoryError,TimeoutError)) else 'implementation_failure',error=str(error),traceback=traceback.format_exc());raise
    finally:write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
