#!/usr/bin/env python3
"""Rebase 026 self-separation seed pairs by exact triangle identity onto joint checkpoint c1e."""
import hashlib,json,importlib.util
from pathlib import Path
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
C=D/"D/focused-clearance-541/joint/area-damping-14187/run-001/candidate.npz"
A130=D/"D/focused-clearance-541/joint/area-damping-14187/check-001-full/audit-130/summary.json"
A159=D/"D/focused-clearance-541/joint/area-damping-14187/check-001-full/audit-159/report.json"
HIST=D/"D/refresh-515/run-001/connected-continuation-prep491/run-001/self-seeds/merged-self-pair-seeds.npy"
HISTREPORT=HIST.with_name("report.json")
HISTCAND=D/"D/finite-triangle-backtrack-512/all-pairs-001/backtrack-373/pass-008/candidate.npz"
TRIAL=D/"D/focused-clearance-541/saved-reuse/full-amplitude/edge-clipped-001/candidate.npz"
TRIAL130=D/"D/focused-clearance-541/saved-reuse/full-amplitude/edge-clipped-exact-001/audit-130/summary.json"
TRIALPAIRS=TRIAL130.parent/"surface-self-pairs.npy"
OWNER=D/"publication-101/evidence/204/source/fit/self_separation_026.py"
OUT=D/"D/focused-clearance-541/constraints/topology-007-c1e02/seed-refresh-001"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mod(path):
 s=importlib.util.spec_from_file_location("self_sep_026_seed_rebase",path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def loadj(p):return json.loads(Path(p).read_text())
assert not OUT.exists()
OUT.mkdir(parents=True)
pins={str(p):sha(p) for p in [C,A130,A159,HIST,HISTREPORT,HISTCAND,TRIAL,TRIAL130,TRIALPAIRS,OWNER]}
want={str(C):"c1e02aaaacebe571370e5b6273501fbb95d1d938cf4292025163495415468939",
str(A130):"1be729eebf68d78b4d8bc5a707a762bca079a469f230f21aea59b7b5dca24f12",
str(A159):"1c97d856712f34ca4fce684bb6ca6914d4df0b21da7c9c1a17dfbe9fbffa683c"}
for p,h in want.items():assert pins[p]==h
current130=loadj(A130);current159=loadj(A159)
assert current130["candidate_sha256"]==current159["candidate_sha256"]==pins[str(C)]
assert current130["complete_surface_audit"] is True and current130["exact_PLC_surface_pairs"]==0
assert current159["complete"] is True and len(current159["banks"])==185
base=np.load(C,allow_pickle=False);hist=np.load(HISTCAND,allow_pickle=False);trial=np.load(TRIAL,allow_pickle=False)
F=np.asarray(base["source_O_faces"],dtype=np.int64)
def face_index(faces):
 out={}
 for i,t in enumerate(np.asarray(faces)):
  key=tuple(sorted(map(int,t)))
  out.setdefault(key,[]).append(i)
 return out
ix=face_index(F)
amb=[k for k,v in ix.items() if len(v)!=1]
assert not amb,"current candidate contains duplicate face identities"
def map_pairs(srcfaces,pairs,label):
 srcfaces=np.asarray(srcfaces,dtype=np.int64);pairs=np.asarray(pairs,dtype=np.int64)
 assert pairs.ndim==2 and pairs.shape[1]==2 and pairs.dtype.kind in "iu"
 fmap=np.full(len(srcfaces),-1,dtype=np.int64)
 missing_source=[]
 for i,t in enumerate(srcfaces):
  hit=ix.get(tuple(sorted(map(int,t))),[])
  if len(hit)==1:fmap[i]=hit[0]
  elif len(hit)>1:raise ValueError(f"ambiguous destination match for {label} face row {i}")
 out=[];dropped=[]
 for i,j in pairs:
  i=int(i);j=int(j)
  if i<0 or j<0 or i>=len(srcfaces) or j>=len(srcfaces) or i==j:raise ValueError(f"invalid {label} pair")
  mi,mj=int(fmap[i]),int(fmap[j])
  if mi<0 or mj<0:
   dropped.append({"source_pair":[i,j],"triangle_a":sorted(map(int,srcfaces[i])),"triangle_b":sorted(map(int,srcfaces[j])),
                  "reason":"changed_or_unmapped_triangle","a_mapped":mi>=0,"b_mapped":mj>=0})
   continue
  if mi==mj:raise ValueError(f"{label} pair collapsed to one current face")
  out.append(sorted((mi,mj)))
 q=np.unique(np.asarray(out,dtype=np.int64).reshape(-1,2),axis=0) if out else np.empty((0,2),np.int64)
 if len(q)!=len(out):raise ValueError(f"{label} remap collapsed duplicate source identities")
 return q,dropped,int(np.count_nonzero(fmap>=0))
histpairs=np.asarray(np.load(HIST,allow_pickle=False),dtype=np.int64)
trialpairs=np.asarray(np.load(TRIALPAIRS,allow_pickle=False),dtype=np.int64)
hp,hd,hf=map_pairs(hist["source_O_faces"],histpairs,"historical")
tp,td,tf=map_pairs(trial["source_O_faces"],trialpairs,"edge-clipped trial")
allpairs=np.unique(np.vstack((hp,tp)),axis=0).astype(np.int64)
assert len(allpairs)>0 and np.all(allpairs[:,0]<allpairs[:,1]) and np.all(allpairs>=0) and np.all(allpairs<len(F))
np.save(OUT/"merged-self-pair-seeds.npy",allpairs,allow_pickle=False)
seedpath=OUT/"merged-self-pair-seeds.npy"
owner=mod(OWNER)
P=np.asarray(base["points_world_m"],dtype=np.float32)
rows,records=owner.build_pair_rows(P.astype(np.float64)*1000,F,allpairs,margin_fraction=.25)
maxrhs=float(max((float(r["lower"]) for r in rows),default=-float("inf")))
assert maxrhs<=2e-8
indptr=np.zeros(len(rows)+1,dtype=np.int64);nodes=[];coeff=[];lower=[]
for i,row in enumerate(rows):
 n=np.asarray(row["nodes"],np.int64);c=np.asarray(row["coefficients"],np.float64)
 assert c.shape==(len(n),3)
 nodes.extend(n.tolist());coeff.extend(c.tolist());lower.append(float(row["lower"]));indptr[i+1]=len(nodes)
np.savez_compressed(OUT/"self-seed-026-rows.npz",indptr=indptr,nodes=np.asarray(nodes,np.int64),
 coefficients=np.asarray(coeff,np.float64).reshape(-1,3),lower=np.asarray(lower,np.float64))
# Correct fixed-037 magnitude gate and preserve the independent current-orientation gate.
ref=np.asarray(base["area_reference_037_points_world_m_f64"],dtype=np.float64)
rf=np.asarray(base["area_reference_037_source_O_faces"],dtype=np.int64)
assert np.array_equal(F,rf)
co=np.cross(P[F[:,1]].astype(float)-P[F[:,0]],P[F[:,2]].astype(float)-P[F[:,0]])
cr=np.cross(ref[F[:,1]]-ref[F[:,0]],ref[F[:,2]]-ref[F[:,0]])
mag=np.linalg.norm(co,axis=1)/np.linalg.norm(cr,axis=1)
assert float(mag.min())>=.1
dropped={"historical":hd,"edge_clipped_trial":td}
(OUT/"dropped-seed-pair-identities.json").write_text(json.dumps(dropped,indent=2,sort_keys=True)+"\n")
report={
 "schema":"numi.focused-clearance-541.current-026-seeds.v1","status":"PASS_ZERO_STEP_REBUILT_026_ROWS",
 "candidate_path":str(C),"candidate_sha256":pins[str(C)],
 "current_exact_self_pairs":0,"current_exact_bone_crossing_pairs":sum(int(b["result"]["count"]) for b in current159["banks"] if b.get("result") is not None),
 "source_triangle_mapping":"unique exact unordered vertex-triple identity; no row-index reuse",
 "historical":{"candidate_path":str(HISTCAND),"candidate_sha256":pins[str(HISTCAND)],
  "seed_pairs_path":str(HIST),"seed_pairs_sha256":pins[str(HIST)],"seed_report_path":str(HISTREPORT),
  "seed_report_sha256":pins[str(HISTREPORT)],"source_pair_count":int(len(histpairs)),
  "source_face_rows_mapped":hf,"mapped_pair_count":int(len(hp)),"dropped_pair_count":len(hd)},
 "edge_clipped_trial":{"candidate_path":str(TRIAL),"candidate_sha256":pins[str(TRIAL)],
  "audit130_path":str(TRIAL130),"audit130_sha256":pins[str(TRIAL130)],
  "source_pair_path":str(TRIALPAIRS),"source_pair_sha256":pins[str(TRIALPAIRS)],
  "source_pair_count":int(len(trialpairs)),"source_face_rows_mapped":tf,
  "mapped_pair_count":int(len(tp)),"dropped_pair_count":len(td)},
 "merged_unique_pair_count":int(len(allpairs)),"linear_row_count":int(len(rows)),
 "maximum_zero_step_rhs_mm":maxrhs,"zero_step_feasible":True,"margin_fraction":.25,
 "seed_pairs_path":str(seedpath),"seed_pairs_sha256":sha(seedpath),
 "row_cache_path":str(OUT/"self-seed-026-rows.npz"),"row_cache_sha256":sha(OUT/"self-seed-026-rows.npz"),
 "dropped_identity_path":str(OUT/"dropped-seed-pair-identities.json"),
 "dropped_identity_sha256":sha(OUT/"dropped-seed-pair-identities.json"),
 "fixed037_reference":{"points_key":"area_reference_037_points_world_m_f64","faces_key":"area_reference_037_source_O_faces",
  "minimum_magnitude_ratio_at_base":float(mag.min()),"floor":.1},
 "source_pins":pins,"native_admission":False,"accepted_physical_steps":0}
(OUT/"seed-refresh-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:report[k] for k in ["status","current_exact_self_pairs","current_exact_bone_crossing_pairs",
 "historical","edge_clipped_trial","merged_unique_pair_count","linear_row_count","maximum_zero_step_rhs_mm",
 "fixed037_reference","seed_pairs_path","row_cache_path","dropped_identity_path"]},indent=2))
