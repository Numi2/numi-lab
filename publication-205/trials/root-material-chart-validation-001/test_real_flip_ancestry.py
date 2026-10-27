#!/usr/bin/env python3
"""Construct an independently mapped copy of the six previously audited flips."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,subprocess,sys
import numpy as np
import validate_charts as v
W=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027")
D=Path(__file__).parent
base=W/"b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz"
candidate=W/"root-topology-a69-safe-subset-001/candidate.npz"
out=D/"d071-overlap-mapped-v2.npz"; report=D/"d071-chart-validation-v2.json"
assert v.digest(base)=="a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db"
assert v.digest(candidate)=="d071f04fa64ce602f08dbf7859c45b4518ef2032ac92af0c1100d69dbdd59826"
assert not out.exists() and not report.exists()
x=v.load(base);y=v.load(candidate)
xp,xf,xi,xb,xc,xa=v.scene(x);yp,yf,yi,yb,yc,ya=v.scene(y)
assert len(xp)==len(yp) and len(xf)==len(yf) and np.array_equal(xi,yi)
changed=np.flatnonzero(np.any(xf!=yf,axis=1)|np.any(xb!=yb,axis=(1,2)))
assert len(changed)==12
groups=defaultdict(list)
for i,p in enumerate(xi):groups[int(p)].append(i)
rows=[];ptr=[0];changedset=set(map(int,changed))
for n,p in enumerate(yi):
 if n not in changedset: q=[n]
 else:q=[i for i in groups[int(p)] if v.area(v.clip(xc[i],yc[n]))>1e-10]
 assert q
 rows.extend(q);ptr.append(len(rows))
y["remesh806_baseline_sha256"]=np.array([v.digest(base)])
y["remesh806_old_to_new_node_rows"]=np.arange(len(xp),dtype=np.int64)
y["remesh806_new_to_old_node_rows"]=np.arange(len(xp),dtype=np.int64)
y["remesh806_face_parent_indptr"]=np.array(ptr,dtype=np.int64)
y["remesh806_face_parent_face_rows"]=np.array(rows,dtype=np.int64)
np.savez_compressed(out,**y)
r=subprocess.run([sys.executable,str(D/"validate_charts.py"),str(base),str(out),str(report),"--expected-baseline",v.digest(base),"--expected-candidate",v.digest(out)],capture_output=True,text=True)
print(r.stdout,r.stderr)
if r.returncode:raise SystemExit(r.returncode)
j=json.loads(report.read_text());assert j["altered_charts"]==6 and j["deleted_nodes"]==0
print("PASS: six genuine same-parent flips have independently verified material ancestry")
