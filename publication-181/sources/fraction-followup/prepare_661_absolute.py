#!/usr/bin/env python3
"""Prepare new 661-state constraints and exact failed 15-fold witnesses."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546')
R=N/'component-sat-selector-638';A=R/'fit-from-663-001/seed-retry-663-001'
OUT=R/'fit-from-661-001';BASE=A/'fraction-small-001';W=R/'combined-fit-663-001/fit-001'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    state=BASE/'final-absolute-546/state.json';previous=R/'fit-from-663-001/matched-prepared-613/constraints/constraint-context.json'
    old=R/'outward-boneface-filter-640/root-bank43-selfcut-001/retained-631v2-component-5883-only-001'
    summary=W/'full-check-545-001/audit-130/summary.json'
    inputs=[state,previous,old/'candidate.npz',old/'full-check/audit-159/report.json',W/'candidate.npz',summary,summary.parent/'surface-self-pairs.npy',N/'prepare_surface546.py']
    if sha(BASE/'candidate.npz')!='beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35':raise ValueError('661 baseline hash changed')
    if (OUT/'prepared-002').exists():raise ValueError('prepared output already exists')
    cmd=[sys.executable,str(N/'prepare_surface546.py'),'--state',str(state),'--out',str(OUT/'prepared-002'),
         '--prior-context',str(previous),'--seed-source-candidate',str(W/'candidate.npz'),'--seed-source-audit130',str(summary),
         '--prior-surface-report',str(old/'full-check/audit-159/report.json'),'--ancestry-parent-candidate',str(old/'candidate.npz')]
    pins={str(p):sha(p) for p in inputs};(OUT/'prepare-002-launch.json').write_text(json.dumps({'argv':cmd,'input_sha256':pins,'native_admission':False,'accepted_physical_steps':0},indent=2,sort_keys=True)+'\n')
    start=time.monotonic()
    with (OUT/'prepare-002.log').open('wb') as f:cp=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    changed=[p for p,h in pins.items() if sha(p)!=h]
    result={'status':'FINISHED' if cp.returncode==0 and not changed else 'FAILED','returncode':cp.returncode,'changed_pins':changed,'seconds':time.monotonic()-start,'native_admission':False,'accepted_physical_steps':0}
    (OUT/'prepare-002-execution.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='FINISHED':raise SystemExit(1)
if __name__=='__main__':main()
