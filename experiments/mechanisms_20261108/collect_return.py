"""Collect only this campaign's new runs, with per-file hashes and task accounting."""
import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path


def collect(repo,output,checkpoints=False):
    repo=Path(repo).resolve();output=Path(output).resolve();root=repo/'results/mechanisms_20261108'
    sys.path.insert(0,str(repo))
    try:
        from cs622.mechanism_jobs import jobs
    finally:
        sys.path.pop(0)
    if output.exists():raise FileExistsError(output)
    if output.is_relative_to(root):raise ValueError('Return archive must be outside collected results')
    files=[];tasks=[]
    for j in jobs():
        d=root/'runs'/j['output']
        complete=all((d/k).exists() for k in ['predictions.parquet','config.json','metadata.json'])
        if complete and json.loads((d/'config.json').read_text())!=j['config']:raise ValueError(f'Config differs: {d}')
        tasks.append(dict(task=j['task'],campaign=j['campaign'],complete=complete))
    candidates=[p for folder in [root/'runs',root/'slurm'] if folder.exists() for p in folder.rglob('*')]
    for p in sorted(candidates):
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):continue
        if p.name=='predictions_with_truth.parquet' or p.name=='latest.pt':continue
        keep=p.suffix in ['.json','.csv','.out','.log'] or p.name=='predictions.parquet'
        if checkpoints and p.name in ['best.pt','features.json']:keep=True
        if not checkpoints and p.name=='features.json':keep=False
        if keep:files.append(p)
    manifest=dict(tasks=tasks,complete_tasks=sum(x['complete'] for x in tasks),files=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w:gz') as tar:
        for p in files:
            name=p.relative_to(repo).as_posix();sha=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
            manifest['files'].append(dict(path=name,bytes=p.stat().st_size,sha256=sha));tar.add(p,arcname=name,recursive=False)
        data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('return_manifest.json');info.size=len(data)
        tar.addfile(info,io.BytesIO(data))
    print(json.dumps(dict(archive=str(output),bytes=output.stat().st_size,sha256=hashlib.file_digest(output.open('rb'),'sha256').hexdigest(),
                         complete_tasks=manifest['complete_tasks'],incomplete_task_ids=[x['task'] for x in tasks if not x['complete']]),indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.');p.add_argument('--output',required=True)
    p.add_argument('--checkpoints',action='store_true');a=p.parse_args();collect(a.repo,a.output,a.checkpoints)
