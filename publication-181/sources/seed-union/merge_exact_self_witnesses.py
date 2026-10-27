#!/usr/bin/env python3
"""Extend candidate-matched 026 self seeds with exact rejected-fit witnesses."""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np

OWNER=Path('/Users/numi/Work/numi-human-connected-surface-069/publication-101/evidence/204/source/fit/self_separation_026.py')
OWNER_SHA='c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--baseline-sha256',required=True)
    p.add_argument('--prior-seeds',type=Path,required=True)
    p.add_argument('--prior-receipt',type=Path,required=True)
    p.add_argument('--witness',nargs=3,action='append',required=True,metavar=('CANDIDATE','SUMMARY','PAIRS'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('output must be new')
    if sha(a.baseline)!=a.baseline_sha256 or sha(OWNER)!=OWNER_SHA: raise ValueError('baseline or owner pin mismatch')
    prior=read(a.prior_receipt)
    if (prior.get('candidate_sha256')!=a.baseline_sha256 or prior.get('status')!='PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY'
        or prior.get('zero_step_feasible') is not True or prior.get('output_pairs_sha256')!=sha(a.prior_seeds)):
        raise ValueError('prior seed certificate mismatch')
    with np.load(a.baseline,allow_pickle=False) as z:
        x=z['points_world_m'].astype(np.float64)*1000
        faces=z['source_O_faces'].copy()
    old=np.load(a.prior_seeds,allow_pickle=False)
    if old.ndim!=2 or old.shape[1]!=2: raise ValueError('bad prior seed shape')
    prior_set={tuple(sorted(map(int,pair))) for pair in old}
    all_new=set(); sources=[]
    for cand_path,summary_path,pair_path in a.witness:
        cand_path,summary_path,pair_path=map(Path,(cand_path,summary_path,pair_path))
        d=read(summary_path); pairs=np.load(pair_path,allow_pickle=False)
        if (d.get('candidate_sha256')!=sha(cand_path) or d.get('complete_surface_audit') is not True
            or d.get('status')!='STOPPED_AT_SURFACE_GATE' or pairs.ndim!=2 or pairs.shape[1]!=2
            or len(pairs)!=d.get('exact_PLC_surface_pairs')): raise ValueError('self witness does not match exact trial audit')
        with np.load(cand_path,allow_pickle=False) as z:
            if not np.array_equal(faces,z['source_O_faces']): raise ValueError('trial topology changed; exact face remapping required')
        rows={tuple(sorted(map(int,v))) for v in pairs}
        if len(rows)!=len(pairs) or any(i<0 or j>=len(faces) or i==j for i,j in rows): raise ValueError('invalid/duplicate witness pairs')
        all_new.update(rows)
        sources.append({'candidate':str(cand_path),'candidate_sha256':sha(cand_path),'summary':str(summary_path),'summary_sha256':sha(summary_path),'pairs':str(pair_path),'pairs_sha256':sha(pair_path),'witness_count':len(pairs)})
    additions=sorted(all_new-prior_set)
    if not additions: raise ValueError('no new witness pairs')
    spec=importlib.util.spec_from_file_location('owner026',OWNER); owner=importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    new_rows,records=owner.build_pair_rows(x,faces,np.asarray(additions,dtype=np.int64),margin_fraction=.25)
    lower=max((float(row['lower']) for row in new_rows),default=0.0)
    if not np.isfinite(lower) or lower>2e-8: raise ValueError(f'new constraints fail zero-step preflight: {lower}')
    merged=np.asarray(sorted(prior_set|all_new),dtype=np.int64)
    a.output.mkdir(parents=True)
    outfile=a.output/'merged-self-pair-seeds.npy'; np.save(outfile,merged)
    receipt={'status':'PASS_NEW_EXACT_SELF_WITNESS_ZERO_STEP_PREFLIGHT','baseline':str(a.baseline),'baseline_sha256':a.baseline_sha256,
       'prior_seed':str(a.prior_seeds),'prior_seed_sha256':sha(a.prior_seeds),'prior_receipt':str(a.prior_receipt),'prior_receipt_sha256':sha(a.prior_receipt),
       'sources':sources,'new_witness_pairs':len(additions),'new_rows':len(new_rows),'new_records':len(records),'new_rows_max_lower_mm':lower,
       'prior_pair_count':len(prior_set),'merged_pair_count':len(merged),'merged_seed':str(outfile),'merged_seed_sha256':sha(outfile),
       'pair_owner':str(OWNER),'pair_owner_sha256':sha(OWNER),'topology_exactly_matched':True,'native_admission':False,'accepted_physical_steps':0,
       'limitation':'Construction constraints only; full emitted geometry and physical checks still required.'}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','new_witness_pairs','new_rows','new_rows_max_lower_mm','merged_pair_count','merged_seed_sha256')}))
if __name__=='__main__':main()
