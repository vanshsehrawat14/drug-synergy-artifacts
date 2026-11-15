"""Frozen matched-seed closure and two targeted follow-up experiments; no search."""
import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
from .io import read_json,write_json,digest
from .research_jobs import MASKS
from .runner import run

SEED_ARMS=['full','no_drug_identity','no_cell_identity','no_dose']
DATASET_SHA256='ea588405019971906ac99c3caa547d5f4fac5b1be340356f0a8fe17a6aa3ce8b'


def config_for(arm,seed,fold):
    c=deepcopy(read_json(Path(__file__).resolve().parents[1]/'configs/neural_monotherapy.json'))
    c.update(seed=seed,test_fold=fold)
    c['features'].update(mono_missing='interpolate')
    c['training'].update(device='cuda',eval_batch_size=16384)
    if arm in MASKS:c['features'].update(MASKS[arm])
    elif arm=='joint_identity':c['features'].update(drug_identity=False,cell_identity=False)
    elif arm=='residual_no_drug':c['features'].update(drug_identity=False,reference_offset='bliss')
    elif arm!='full':raise ValueError('Unknown frozen arm')
    return c


def jobs():
    rows=[]
    for campaign,arms,seeds in [('seed2029',SEED_ARMS,[2029]),('joint_identity',['joint_identity'],[2026,2027,2028]),
                                ('residual_drug_identity',['residual_no_drug'],[2026,2027,2028])]:
        for arm in arms:
            for seed in seeds:
                for fold in range(5):
                    c=config_for(arm,seed,fold)
                    rows.append(dict(task=len(rows),campaign=campaign,arm=arm,seed=seed,fold=fold,config=c,
                        config_sha256=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                        output=f'{campaign}/{arm}/seed_{seed}/fold_{fold}'))
    return rows


def required_controls(campaign):
    families=['f2_base'] if campaign=='seed2029' else ['f2_base'] if campaign=='joint_identity' else ['f2_base','f2_residual']
    return [dict(family=f,seed=s,fold=k,path=f'results/neural/runs/sprint_v1/runs/{f}/seed_{s}/fold_{k}')
            for f in families for s in [2026,2027,2028] for k in range(5)]


def trigger_audit(comparisons):
    """Reproduce the historical coded trigger, retaining its extrema-based definition."""
    rows=[]
    for r in comparisons.itertuples():
        if r.model not in MASKS:continue
        bounds={k:getattr(r,k) for k in ['pair_low','pair_high','drug_low','drug_high']}
        close=[k for k,v in bounds.items() if abs(abs(v)-.05)<.005]
        opposite_extreme=(sum((v>0)!=(r.change>0) for v in [r.fold_min,r.fold_max])==1 and abs(r.change)>=.05)
        rows.append(dict(mask=r.model,endpoint=r.level,trigger=bool(close or opposite_extreme),
            near_bounds=close,opposite_fold_extreme=bool(opposite_extreme),
            historical_trigger=bool(r.confirmatory_seed_trigger),
            note='Fold min/max detects sign disagreement; literal exactly-one-fold dependence requires all five fold results.'))
    if any(r['trigger']!=r['historical_trigger'] for r in rows):raise ValueError('Trigger reconstruction mismatch')
    return rows


def check_controls(campaign,baseline_root,ablations):
    root=Path(baseline_root)
    if digest('data/processed/oneil_v1/observations.parquet')!=DATASET_SHA256:
        raise ValueError('Active prepared dataset differs from the frozen O\'Neil artifact')
    for item in required_controls(campaign):
        d=root/item['path']
        c=read_json(d/'config.json')
        expected=config_for('full',item['seed'],item['fold'])
        if item['family']=='f2_residual':expected['features']['reference_offset']='bliss'
        if c!=expected:raise ValueError(f'Control config mismatch: {d}')
        if not (d/'metadata.json').exists() or not (d/'predictions.parquet').exists():raise ValueError(f'Incomplete control: {d}')
        if read_json(d/'metadata.json')['dataset_sha256']!=DATASET_SHA256:raise ValueError(f'Control dataset differs: {d}')
    if campaign in ['joint_identity','residual_drug_identity']:
        arms=['no_drug_identity','no_cell_identity'] if campaign=='joint_identity' else ['no_drug_identity']
        for arm in arms:
            for seed in [2026,2027,2028]:
                for fold in range(5):
                    d=Path(ablations)/arm/f'seed_{seed}'/f'fold_{fold}'
                    if read_json(d/'config.json')!=config_for(arm,seed,fold):raise ValueError(f'Ablation control differs: {d}')
                    if not (d/'metadata.json').exists() or not (d/'predictions.parquet').exists():raise ValueError(f'Incomplete ablation: {d}')
                    if read_json(d/'metadata.json')['dataset_sha256']!=DATASET_SHA256:raise ValueError(f'Ablation dataset differs: {d}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=int)
    p.add_argument('--manifest');p.add_argument('--output');p.add_argument('--check-only',action='store_true')
    p.add_argument('--baseline-root',default='.');p.add_argument('--ablations',default='results/research_20261107/runs')
    a=p.parse_args();alljobs=jobs()
    if a.manifest:
        write_json(a.manifest,dict(protocol_sha256=digest('configs/gpu_mechanisms_20261108.json'),jobs=alljobs,
            task_ranges={'seed2029':[0,19],'joint_identity':[20,34],'residual_drug_identity':[35,49]},
            controls={c:required_controls(c) for c in ['seed2029','joint_identity','residual_drug_identity']}));return
    if a.task is None or not 0<=a.task<len(alljobs):p.error('Task must be 0..49')
    j=alljobs[a.task];check_controls(j['campaign'],a.baseline_root,a.ablations)
    if a.check_only:print('Controls valid',j['campaign']);return
    if not a.output:p.error('--output is required for training')
    stop=Path(a.output)/j['campaign']/'STOP.json'
    if stop.exists():raise RuntimeError(f'Campaign stopped; inspect preserved failure record {stop}')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Requires exactly one allocated visible CUDA GPU')
    d=Path(a.output)/j['output']
    if (d/'metadata.json').exists():
        if read_json(d/'config.json')!=j['config'] or not (d/'predictions.parquet').exists():raise ValueError('Existing result is incomplete or differs')
        print('Completed',d);return
    last=d/'checkpoints/latest.pt'
    try:
        run(j['config'],d,resume=str(last) if last.exists() else None)
    except (ValueError,RuntimeError,AssertionError) as exc:
        write_json(stop,dict(task=j['task'],error_type=type(exc).__name__,error=str(exc),
            instruction='Stop this campaign and diagnose; preserve this record. No automatic configuration change or additional search.'))
        raise
    write_json(d/'mechanism_job.json',{k:v for k,v in j.items() if k!='config'})


if __name__=='__main__':main()
