#!/usr/bin/env python3
"""Bounded Float32 interpolation from the valid 675 geometry toward one folded proposal.
This is construction exploration; it neither approves geometry nor advances physics.
"""
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546')
base=S/'connected-topology-580/run-010-from689/root-combined-001/candidate.npz'
trial=S/'component-sat-selector-638/outward-boneface-filter-640/root-bank43-component-001/candidate.npz'
out=S/'component-sat-selector-638/outward-boneface-filter-640/root-component-linesearch-002'
py='/Users/numi/numi-human-prep-venv-20261005/bin/python'
slot='/tmp/numi-human-geometry-slot.py'
audit='/Users/numi/Work/numi-human-connected-surface-069/audit_connected_outer_geometry_130.py'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base)=='b6bde2d74ea54cc3cbbd332cfd5e331b08771c6a2554d9aacc176d06db189257'
assert sha(trial)=='b890a17a66596e92bf7cd7fd7770414fed28f3c15f889daaae68537c41ecf29c'
assert not out.exists(); out.mkdir()
with np.load(base,allow_pickle=False) as z: b={k:z[k] for k in z.files}
with np.load(trial,allow_pickle=False) as z: t={k:z[k] for k in z.files}
assert list(b)==list(t)
for k in b:
    if k!='points_world_m':
        assert np.array_equal(b[k],t[k],equal_nan=True) if b[k].dtype.kind in 'fc' else np.array_equal(b[k],t[k]),k
v=b['points_world_m'].astype(np.float64)
d=t['points_world_m'].astype(np.float64)-v
assert np.count_nonzero(np.any(d!=0,axis=1))==249
result={'baseline_sha256':sha(base),'trial_sha256':sha(trial),'audit_script_sha256':sha(Path(audit)), 'meaning':'Float32 construction interpolation only; no physical steps or admission','samples':[]}
for a in [0.05,0.15,0.30,0.50,0.75]:
    name=f'alpha-{int(a*100):03d}'; q=out/name; q.mkdir()
    c=dict(b); c['points_world_m']=(v+a*d).astype(np.float32)
    path=q/'candidate.npz'; np.savez_compressed(path,**c)
    start=time.monotonic()
    with (q/'audit.log').open('w') as f: cp=subprocess.run([py,slot,py,audit,str(path),str(q/'audit-130')],stdout=f,stderr=subprocess.STDOUT)
    item={'alpha':a,'candidate_sha256':sha(path),'audit_returncode':cp.returncode,'wall_seconds':time.monotonic()-start}
    sp=q/'audit-130/summary.json'
    if sp.exists():
        s=json.loads(sp.read_text()); item['summary']=s
    result['samples'].append(item)
    (out/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'alpha':a,'returncode':cp.returncode,'summary':item.get('summary')}),flush=True)
