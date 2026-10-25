#!/usr/bin/env python3
"""Build baseline separating-plane constraints for pinned 077 detail pairs."""
from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT=Path("/Users/n/Work/numi-human-connected-surface-069")
EVID=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=ROOT/"E/embedded-self-separator-078/run-002"
BASE=EVID/"E/connected-volume-refinement-037/run-006/candidate.npz"
CAND=ROOT/"repair-run-embedded-077/iteration-004/candidate.npz"
LOCAL=ROOT/"self-intersection-077/run-003/new-pair-localization.json"
TD=EVID/"accepted-transfer-row-export-075/run-001/source-preflight"
DETAIL=EVID/"bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"
ADAPTER=EVID/"E/accepted-transfer-row-adapter-075/load_transfer_075.py"
PINS={
"baseline_037":"52c81dbe323eab2240bc315f7b7bde605a201fa8e928206f27000de7e5d7a492",
"candidate_077":"484bb4939493abe0daf28c9ded2f33625b4b2424bc9ee01d4939dae47692156a",
"localization":"e2232567c1864b0858143dcd789c463950544b173c61dd7038dcb909f32383fe",
"adapter":"6f1c8887ab33d75bf5dd6241ec7a4a4ae8bc21d2d7573a8ba36b417f303c1117",
"transfer_rows":"71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14",
"coarse_reference":"9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2",
"coarse_faces":"7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2",
"detail_payload":"7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb",
"mapped_surface_owner":"31d7f036c9795b90e761e5713250e6a5ddbf07a71b7d3bec9e344e043d0cb60e"}

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def unit(v,tol=1e-14):
 n=float(np.linalg.norm(v))
 return None if not np.isfinite(n) or n<=tol else np.asarray(v,dtype=np.float64)/n
def sat_axes(A,B):
 eA=np.roll(A,-1,axis=0)-A;eB=np.roll(B,-1,axis=0)-B
 nA=unit(np.cross(eA[0],eA[1]));nB=unit(np.cross(eB[0],eB[1]))
 raw=[]
 if nA is not None:raw.append(("triangle_normal_A",nA))
 if nB is not None:raw.append(("triangle_normal_B",nB))
 for i,a in enumerate(eA):
  for j,b in enumerate(eB):raw.append((f"edge_cross_{i}_{j}",np.cross(a,b)))
 for side,(edges,n) in enumerate(((eA,nA),(eB,nB))):
  if n is not None:
   for i,e in enumerate(edges):raw.append((f"coplanar_edge_normal_{side}_{i}",np.cross(n,e)))
 out=[];seen=[]
 for label,x in raw:
  u=unit(x)
  if u is None or any(abs(float(np.dot(u,v)))>1-1e-12 for v in seen):continue
  seen.append(u);out.append((label,u))
 return out
def select_separator(A,B,idsA,idsB,WA,WB,candA,candB):
 nonshared=[(i,j) for i in range(3) for j in range(3) if int(idsA[i])!=int(idsB[j])]
 if not nonshared:raise ValueError("pair has no nonshared indexed vertex pairs")
 choices=[]
 for label,axis in sat_axes(A,B):
  for sign in (1.,-1.):
   n=axis*sign
   gaps=np.asarray([np.dot(n,B[j]-A[i]) for i,j in nonshared],dtype=float)
   g=float(gaps.min())
   if g < -1e-12:continue
   margin=.5*g if g>1e-12 else 0.
   sens=max((float(np.linalg.norm((WB.getrow(j)-WA.getrow(i)).data)) for i,j in nonshared),default=0.)
   cg=float(min(np.dot(n,candB[j]-candA[i]) for i,j in nonshared))
   violation=max(0.,margin-cg)
   choices.append((violation,margin,sens,-cg,label,n,g,cg))
 return max(choices,key=lambda x:(x[0]>1e-12,x[0],x[1],x[2],x[3],x[4])) if choices else None
def pair_rows(A,B,idsA,idsB,WA,WB,X0,candA,candB,pi):
 selected=select_separator(A,B,idsA,idsB,WA,WB,candA,candB)
 if selected is None:return [],{"pair_index":pi,"status":"no_baseline_separator"}
 violation,margin,sensitivity,_,label,n,gmin,cgmin=selected
 rows=[]; provenance=[]; shared=sorted(set(map(int,idsA))&set(map(int,idsB)))
 xflat=X0.astype(np.float64).reshape(-1)
 for i in range(3):
  for j in range(3):
   va,vb=int(idsA[i]),int(idsB[j])
   d=(WB.getrow(j)-WA.getrow(i)).tocsr();d.sum_duplicates();d.eliminate_zeros()
   gap=float(np.dot(n,B[j]-A[i]))
   if va==vb:
    if d.nnz and np.any(d.data!=0):raise ValueError("shared ID rows are not transfer-identical")
    provenance.append({"pair_index":pi,"vertex_a":va,"vertex_b":vb,"status":"omitted_shared_identity_zero_row",
      "baseline_gap_m":0.,"margin_m":0.,"target":0.0,"candidate_gap_m":0.})
    continue
   coo=d.tocoo();cols=[];vals=[]
   for col,val in zip(coo.col,coo.data):
    for k in range(3):
     z=float(val*n[k])
     if z:cols.append(3*int(col)+k);vals.append(z)
   if not cols:
    provenance.append({"pair_index":pi,"vertex_a":va,"vertex_b":vb,"status":"omitted_fixed_transfer_zero_row",
      "baseline_gap_m":gap,"margin_m":margin,"target":None,
      "candidate_gap_m":float(np.dot(n,candB[j]-candA[i]))})
    continue
   row=sparse.csr_matrix((vals,([0]*len(cols),cols)),shape=(1,X0.size))
   c0=float((row@xflat).item())
   target=float(margin-gap+c0)
   slack=c0-target
   if slack < -1e-12:raise ValueError(f"baseline inequality fails: {pi}/{va}/{vb}: {slack}")
   rows.append(row)
   provenance.append({"pair_index":pi,"vertex_a":va,"vertex_b":vb,"status":"constraint",
     "baseline_gap_m":gap,"margin_m":margin,"target":target,"baseline_slack_m":slack,
     "candidate_gap_m":float(np.dot(n,candB[j]-candA[i]))})
 record={"pair_index":pi,"status":"ok","axis_source":label,"normal_world":n.tolist(),
  "baseline_min_nonshared_gap_m":gmin,"margin_m":margin,"projection_sensitivity":sensitivity,
  "candidate_min_nonshared_gap_m":cgmin,"candidate_margin_violation_m":violation,
  "shared_vertex_ids":shared,"constraint_rows":len(rows)}
 return rows,record,provenance

def build():
 paths={"baseline_037":BASE,"candidate_077":CAND,"localization":LOCAL,"adapter":ADAPTER,
  "transfer_rows":TD/"accepted-transfer-rows.csv","coarse_reference":TD/"accepted-coarse-reference.csv",
  "coarse_faces":TD/"accepted-coarse-faces.csv","detail_payload":DETAIL,
  "mapped_surface_owner":ROOT/"canonical/matter/tools/human_skin_fat_embedded_surface.py"}
 for k,p in paths.items():
  if sha(p)!=PINS[k]:raise RuntimeError(f"hash pin mismatch {k}: {p}")
 sp=importlib.util.spec_from_file_location("adapter075",ADAPTER);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 W,D0,C0,faces=mod.load_transfer075(TD/"accepted-transfer-rows.csv",TD/"accepted-coarse-reference.csv",
  TD/"accepted-coarse-faces.csv",DETAIL,expected_sha256={k:PINS[k] for k in ("transfer_rows","coarse_reference","coarse_faces","detail_payload")})
 base=np.load(BASE);cand=np.load(CAND);X0=base["current_points_f32"];Xc=cand["points_world_f32"]
 if X0.shape!=(9320,3) or Xc.shape!=X0.shape or not np.array_equal(X0[:len(C0)],C0):
  raise RuntimeError("075 source origin does not match 037 mechanical origin")
 Dbase=(D0.astype(np.float64)+W@(X0[:W.shape[1]].astype(np.float64)-C0.astype(np.float64))).astype(np.float32)
 Dcand=(D0.astype(np.float64)+W@(Xc[:W.shape[1]].astype(np.float64)-C0.astype(np.float64))).astype(np.float32)
 loc=json.loads(LOCAL.read_text());rows=[];records=[];provenance=[]
 for pi,item in enumerate(loc["pairs"]):
  fa,fb=map(int,item["face_rows"]);idsA,idsB=faces[fa],faces[fb]
  result=pair_rows(Dbase[idsA].astype(float),Dbase[idsB].astype(float),idsA,idsB,W[idsA],W[idsB],
   X0,Dcand[idsA].astype(float),Dcand[idsB].astype(float),pi)
  rr,rec,prov=result
  if rec["status"]!="ok":raise RuntimeError(f"no baseline separator for faces {fa},{fb}")
  rows.extend(rr);records.append({"face_rows":[fa,fb],**rec});provenance.extend(prov)
 if not rows:raise RuntimeError("no variable constraints emitted")
 A=sparse.vstack(rows,format="csr");A.sum_duplicates();A.eliminate_zeros();A.sort_indices()
 target=np.asarray([p["target"] for p in provenance if p["status"]=="constraint"],dtype=np.float64)
 slack=np.asarray(A@X0.astype(float).reshape(-1)-target)
 if A.shape!=(len(target),27960) or slack.min() < -1e-12:raise RuntimeError(f"baseline infeasible/malformed CSR: {A.shape} min={slack.min()}")
 OUT.mkdir(parents=True,exist_ok=False)
 sparse.save_npz(OUT/"separator-inequalities.npz",A)
 np.savez_compressed(OUT/"separator-targets.npz",target=target,baseline_slack_m=slack)
 report={"status":"PASS","scope":"3 exact 077 detail pairs; linear separator inequalities for joint mechanical fit only",
  "not_clearance_certificate":True,"pair_count":len(records),"matrix_shape":list(A.shape),"matrix_nnz":int(A.nnz),
  "constraint_count":len(target),"omitted_zero_rows":sum(p["status"]!="constraint" for p in provenance),
  "baseline_min_slack_m":float(slack.min()),"baseline_max_slack_m":float(slack.max()),
  "pairs":records,"row_provenance":provenance,"input_sha256":PINS}
 (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 pinout={"paths":{k:str(v) for k,v in paths.items()},"sha256":{k:sha(v) for k,v in paths.items()}}
 (OUT/"input-pins.json").write_text(json.dumps(pinout,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"out":str(OUT),"shape":A.shape,"nnz":int(A.nnz),"baseline_min_slack_m":float(slack.min()),
  "rows":len(target),"omitted":report["omitted_zero_rows"],
  "pair_violations":[x["candidate_margin_violation_m"] for x in records]},indent=2))

if __name__=="__main__":build()

