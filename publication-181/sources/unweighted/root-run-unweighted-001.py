#!/usr/bin/env python3
"""Run the existing 638 fit for the matched 663 state; construction comparison only."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
import numpy as np
R=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/fit-from-663-001')
W=R.parent/'run_axis_map_sat_fit638.py'; A=R/'matched-prepared-613/fit-argv.json'; SELECT=R/'selector-001/selection.npz'; REPORT=SELECT.with_name('report.json'); OUT=R/'root-unweighted-fit-001'; BASE=R.parent/'outward-boneface-filter-640/root-bank43-selfcut-001/retained-631v2-component-5883-only-001/candidate.npz'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(BASE)=='f740435b74315d258a0bc9ea58255f6f676a437cf3a39709f9eafb8029b5b70b'
args=json.loads(A.read_text()); assert len(args)>2 and args[0].endswith('/python') and args[1].endswith('run_joint_full_coordinate_fit603.py')
args=args[2:]
for flag in ['--base-candidate','--raw-trial','--output','--self-pairs','--matched-inputs-json-490','--audit159-490']:
 assert args.count(flag)==1,flag
assert Path(args[args.index('--base-candidate')+1]).resolve()==BASE.resolve()
assert Path(args[args.index('--raw-trial')+1]).resolve()==BASE.resolve()
assert not OUT.exists()
args[args.index('--output')+1]=str(OUT)
sel=json.loads(REPORT.read_text());assert sel['selection_npz_sha256']==sha(SELECT)
with np.load(SELECT,allow_pickle=False) as z:
 ids=z['pair_identity']; active=z['selected']; cover={}
 for body,bone,expected in [(43,16,153),(109,114,120),(65,99,86)]:
  m=(ids[:,1]==body)&(ids[:,2]==bone);assert int(m.sum())==expected and int((m&active).sum())==expected
  cover[f'{body}/{bone}']={'total':int(m.sum()),'selected':int((m&active).sum())}
cmd=[sys.executable,str(W),*args,'--selector-npz',str(SELECT),'--selector-report',str(REPORT)]
launch={'status':'LAUNCHING','argv':cmd,'base_sha256':sha(BASE),'fit_argv_sha256':sha(A),'selector_sha256':sha(SELECT),'selector_report_sha256':sha(REPORT),'target_selector_coverage':cover,'claim_limit':'Unweighted construction fit; exact 545 and all physical checks remain required.','native_admission':False,'accepted_physical_steps':0}
(R/'root-unweighted-001-launch.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
t=time.monotonic()
with (R/'root-unweighted-001.log').open('wb') as f:cp=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
res={'status':'FINISHED' if cp.returncode==0 else 'FAILED','returncode':cp.returncode,'seconds':time.monotonic()-t,'output_exists':OUT.exists(),'native_admission':False,'accepted_physical_steps':0}
(R/'root-unweighted-001-execution.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print(json.dumps(res),flush=True);raise SystemExit(cp.returncode)
