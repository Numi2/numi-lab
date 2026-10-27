#!/usr/bin/env python3
"""Retry the pinned three-bank fit with exact newly observed fold barriers."""
from pathlib import Path
import hashlib, json, subprocess, sys, time

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
HERE=ROOT/'fit-from-663-001/seed-retry-663-001'
PREVIOUS=ROOT/'combined-fit-663-001'
SOURCE=PREVIOUS/'fit-001-combined-bank-663.launch.json'
SEED_RECEIPT=HERE/'seeds-001/receipt.json'
OUT=HERE/'weighted-fit-001'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    src=json.loads(SOURCE.read_text()); seed=json.loads(SEED_RECEIPT.read_text())
    if OUT.exists(): raise ValueError('retry output already exists')
    for key in ('audit159','bank_config','candidate','required_pairs','selector_npz','selector_report'):
        q=src[key]
        if sha(q['path'])!=q['sha256']: raise ValueError('input pin changed: '+key)
    runner=Path(src['launcher_path'])
    if sha(runner)!=src['launcher_sha256']: raise ValueError('launcher changed')
    if seed.get('status')!='PASS_NEW_EXACT_SELF_WITNESS_ZERO_STEP_PREFLIGHT' or seed['baseline_sha256']!=src['candidate']['sha256']:
        raise ValueError('seed preflight not matched to baseline')
    if sha(seed['merged_seed'])!=seed['merged_seed_sha256']: raise ValueError('seed payload changed')
    args=list(src['fit_argv_after_adapter_flags'])
    for flag in ('--self-pairs','--output','--base-candidate','--raw-trial'):
        if args.count(flag)!=1: raise ValueError('unexpected previous fit flags: '+flag)
    args[args.index('--self-pairs')+1]=str(Path(seed['merged_seed']).resolve())
    args[args.index('--output')+1]=str(OUT)
    cmd=[sys.executable,str(runner),*args,'--selector-npz',src['selector_npz']['path'],
         '--selector-report',src['selector_report']['path'],'--bank-balanced-config',src['bank_config']['path'],
         '--required-pairs',src['required_pairs']['path']]
    record={'status':'LAUNCHING','command':cmd,'previous_launch':str(SOURCE),'previous_launch_sha256':sha(SOURCE),
      'retry_source':str(Path(__file__).resolve()),'retry_source_sha256':sha(__file__),
      'seed_receipt':str(SEED_RECEIPT),'seed_receipt_sha256':sha(SEED_RECEIPT),'seed_sha256':seed['merged_seed_sha256'],
      'baseline_sha256':src['candidate']['sha256'],'construction_only':True,'native_admission':False,'accepted_physical_steps':0}
    (HERE/'weighted-retry-launch.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    start=time.monotonic()
    with (HERE/'weighted-retry.log').open('wb') as f:
        cp=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    result={'status':'FINISHED' if cp.returncode==0 else 'FAILED','returncode':cp.returncode,
      'seconds':time.monotonic()-start,'native_admission':False,'accepted_physical_steps':0}
    (HERE/'weighted-retry-execution.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(cp.returncode)
if __name__=='__main__':main()
