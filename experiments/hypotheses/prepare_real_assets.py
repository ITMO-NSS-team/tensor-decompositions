"""Fetch immutable model/data revisions, verify SHA256, write local provenance.

Use --local-only to verify existing assets without any network request.
Never replace an existing file whose checksum differs from the lock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[3]/'audit/datasets')
    parser.add_argument('--local-only',action='store_true')
    args=parser.parse_args();args.data=args.data.resolve()
    lock_path=Path(__file__).with_name('asset-lock.json')
    lock=json.loads(lock_path.read_text(encoding='utf-8'))
    groups={
        'gpt2':('gpt2','model',lock['gpt2_revision']),
        'switch-base-8':('google/switch-base-8','model',lock['switch_revision']),
        'wikitext2':('Salesforce/wikitext','dataset',lock['wikitext_revision']),
    }
    records=[]
    for relative,expected in lock['files'].items():
        relative_path=Path(relative);group=relative_path.parts[0]
        path=args.data/relative_path
        if not path.exists():
            if args.local_only:raise FileNotFoundError(f'missing locked asset: {relative}')
            from huggingface_hub import hf_hub_download
            repo,kind,revision=groups[group]
            filename='/'.join(relative_path.parts[1:])
            downloaded=Path(hf_hub_download(repo_id=repo,repo_type=kind,revision=revision,filename=filename,
                                          local_dir=args.data/group,local_dir_use_symlinks=False))
            if downloaded.resolve()!=path.resolve():
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(downloaded,path)
        if digest(path)!=expected:raise ValueError(f'locked checksum mismatch, existing asset preserved: {relative}')
        records.append({'dataset':group,'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':expected})
    for group,name,key in [('gpt2','gpt2-wikitext-provenance.json','gpt2_revision'),
                           ('switch-base-8','switch-wikitext-provenance.json','switch_revision')]:
        target=args.data/name
        value={key:lock[key],'wikitext_revision':lock['wikitext_revision'],
               'files':[record for record in records if record['dataset'] in (group,'wikitext2')]}
        if target.exists():
            previous=json.loads(target.read_text(encoding='utf-8'))
            signature=lambda entries:sorted((entry['path'],entry['bytes'],entry['sha256']) for entry in entries)
            if (previous[key]!=value[key] or previous['wikitext_revision']!=value['wikitext_revision'] or
                signature(previous['files'])!=signature(value['files'])):
                raise ValueError(f'existing provenance differs; preserved: {target}')
        else:target.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({'verified_assets':len(records),'data':str(args.data),'local_only':args.local_only}))


if __name__=='__main__':main()
