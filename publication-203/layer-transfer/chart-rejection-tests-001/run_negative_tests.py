#!/usr/bin/env python3
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
S=ROOT/"per-bank-workers-20261027"
BASE=S/"combined-d071-b109-b61-001/candidate.npz"
BASE_SHA="806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756"
CAND=S/"b43-16/attempt-028-edge-collapse-001/candidate.npz"
CAND_SHA="dd31bdffbb689c8caed7bcc46229b89e23f77096aca6e780005c6a69529e432c"
XFER=S/"b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py"
CHART=S/"root-material-chart-validation-001/validate_charts.py"
OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location("transfer541",XFER)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
with np.load(CAND,allow_pickle=False) as z: c={k:np.asarray(z[k]).copy() for k in z.files}
with np.load(BASE,allow_pickle=False) as z: b={k:np.asarray(z[k]).copy() for k in z.files}
assert sha(CAND)==CAND_SHA and sha(BASE)==BASE_SHA
F=np.asarray(c["source_O_faces"],dtype=np.int64); B=np.asarray(b["source_O_faces"],dtype=np.int64)
nmap=np.asarray(c["remesh806_new_to_old_node_rows"],dtype=np.int64)
base_by_key={tuple(sorted(map(int,t))):i for i,t in enumerate(B)}
changed=[]
for fi,t in enumerate(F):
 old=nmap[t]
 if np.any(old<0): changed.append(fi); continue
 bi=base_by_key.get(tuple(sorted(map(int,old))))
 if bi is None or not any(tuple(map(int,old))==tuple(map(int,np.roll(B[bi],k))) for k in range(3)):
  changed.append(fi)
assert changed
bb0=np.asarray(c["refinement_face_parent_barycentric"],dtype=np.float64)
choices=[]
for fi in changed:
 for vi in range(3):
  for ia in range(3):
   for ib in range(ia+1,3):
    lim=min(float(bb0[fi,vi,ia]),float(bb0[fi,vi,ib]))
    if lim>1e-12: choices.append((lim,fi,vi,ia,ib))
assert choices
lim,f,v,ia,ib=max(choices)
c1={k:v0.copy() for k,v0 in c.items()}
bb=np.asarray(c1["refinement_face_parent_barycentric"],dtype=np.float64).copy()
eps=min(1e-4,0.25*lim)
bb[f,v,ia]+=eps; bb[f,v,ib]-=eps
assert bb[f,v].min()>=0 and abs(float(bb[f,v].sum())-1)<1e-12
c1["refinement_face_parent_barycentric"]=bb
stale=OUT/"stale-legacy-barycentrics.npz"
np.savez_compressed(stale,**c1)
stale_out=OUT/"stale-chart-validator.json"
p=subprocess.run([sys.executable,str(CHART),str(BASE),str(stale),str(stale_out),
 "--expected-baseline",BASE_SHA,"--expected-candidate",sha(stale)],capture_output=True,text=True)
stale_result={"rejected":p.returncode!=0,"returncode":p.returncode,"changed_face_row":f,
 "candidate_sha256":sha(stale),"stderr":p.stderr.strip()[-1500:],"stdout":p.stdout.strip()[-500:]}
assert stale_result["rejected"],"stale legacy barycentrics unexpectedly accepted"
c2={k:v.copy() for k,v in c.items()}
ip=np.asarray(c2["remesh806_face_parent_indptr"],dtype=np.int64)
rows=np.asarray(c2["remesh806_face_parent_face_rows"],dtype=np.int64).copy()
srcip=np.asarray(b["face_source_ancestry_indptr"],dtype=np.int64)
srcrows=np.asarray(b["face_source_ancestry_face_rows"],dtype=np.int64)
a,z=map(int,ip[f:f+2]); actual=set(map(int,rows[a:z]))
def ancestry_union(parent_rows):
 out=set()
 for r in parent_rows: out.update(map(int,srcrows[srcip[int(r)]:srcip[int(r)+1]]))
 return out
actual_union=ancestry_union(actual)
alt=next(i for i in range(len(B)) if i not in actual and ancestry_union([i])!=actual_union)
rows[a]=alt
c2["remesh806_face_parent_face_rows"]=rows
wrong=OUT/"inconsistent-overlap-ancestry.npz"
np.savez_compressed(wrong,**c2)
try:
 mod.validate_candidate(c2,wrong,b,BASE_SHA)
 rejected={"rejected":False,"error":None}
except mod.TransferError as e:
 rejected={"rejected":True,"error":str(e)}
assert rejected["rejected"],"inconsistent overlap ancestry unexpectedly accepted"
report={"schema":"numi.human.topology-independent-transfer-negative-tests.v1",
 "status":"PASS_NEGATIVE_TESTS","pins":{"baseline_sha256":BASE_SHA,"candidate_sha256":CAND_SHA,
 "transfer_script_sha256":sha(XFER),"chart_validator_sha256":sha(CHART)},
 "stale_legacy_barycentric_test":stale_result,
 "inconsistent_overlap_ancestry_test":{"rejected":True,"changed_face_row":f,
  "wrong_direct_parent_face_row":int(alt),"transfer_error":rejected["error"],
  "candidate_sha256":sha(wrong)},
 "scope":["Private corrupted copies only.","Tests rejection behavior; no candidate was modified or promoted.","No physical/native admission."]}
rp=OUT/"negative-tests-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(rp),"sha256":sha(rp),"status":report["status"],
 "stale_chart_rejected":stale_result["rejected"],"overlap_rejected":rejected["rejected"],
 "changed_face_row":f,"transfer_script_sha256":sha(XFER)},indent=2))
