#!/usr/bin/env python3
"""Diffuse one measured 65/99 face motion over its neighborhood, keeping physical construction gates."""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
from scipy import sparse
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638");w=S/"partial-midpoint-continuation-root-001";a=S/"edge-refinement-661-threebank-001/generation-002-all-banks-001";out=w/"connected-witness65-001";out.mkdir(exist_ok=False)
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def mod(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
start=time.monotonic();base=load(a/"candidate.npz");x=base["points_world_m"];faces=base["source_O_faces"];ref=base["area_reference_037_points_world_m_f64"]*1000.;anchors=base["typed_coarse_anchor_output_rows_018"];assert sha(a/"candidate.npz")=="d367af1383d920645c22552569e95614797de8aea8a757b658def68bad4412d8"
search=w/"outward-face-65-99-001/report.json";j=json.loads(search.read_text());r=next(r for r in j["results"] if r["trial_index"]==1138);assert r["face"]==14438 and r["area_pass"] and r["edge_slack_mm"]<0 and r["target_parent"]==84
selected=np.array(r["support"],dtype=np.int64);motion=np.array(r["motion_mm"],dtype=float);assert not np.intersect1d(selected,anchors).size
qpath=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py");assert sha(qpath)=="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2";q=mod("quality408_65root",qpath);area_owner=mod("area65root",q.DEFAULT_AREA_OWNER);q.verify_area_linearization_layout(area_owner)
source=Path("/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz");assert sha(source)=="70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
with np.load(a/"candidate.npz",allow_pickle=False) as z:refsrc,_=q.compose_source_reference(z,source)
bm=x.astype(float)*1000.;srcmm=refsrc*1000.;dp=a/"depth-413/point-depth-rows.npz";dr=json.loads((a/"depth-413/365-compatible-report.json").read_text());nodes,normals,lower_m,ids,records,_=q.validate_point_depth_rows_payload(load(dp),dr,base_candidate_sha256=sha(a/"candidate.npz"),base_points=x,rows_npz_path=dp,owner_path=q.DEFAULT_POINT_DEPTH_OWNER,candidate_path=a/"candidate.npz")
edge=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);adj=sparse.csr_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(x),len(x)));support=set(map(int,selected));front=set(support)
for _ in range(3):front=set(map(int,adj[list(front)].indices))-support;support.update(front)
support=np.array(sorted(support),dtype=np.int64);local={int(v):i for i,v in enumerate(support)};affected=np.flatnonzero(np.any(np.isin(faces,support),axis=1));normalall,_,_,req=q.fixed_reference_area_requirements(bm,ref,faces);prows,_=q.build_scoped_output_node_direction_rows(len(x),support,anchors,nodes,normals,lower_m,ids);G,rhs,_,_=area_owner.area_linearization(bm,faces,affected,normalall[affected],req[affected],minimum_ratio=.1);A,b=q._owner_matrix(prows,G,rhs,support);EG,eb,_=q.build_edge_gradient_constraints(bm,faces,support,.5);A=sparse.vstack((A,EG),format="csr");b=np.r_[b,eb];lo0,hi0,_=q.source_safe_trust_bounds(bm,srcmm,support,anchors,4.,12.)
results=[]
for alpha in (1.,.75,.5,.25):
 lo=lo0.copy();hi=hi0.copy();target=(x[selected].astype(float)+alpha*motion/1000.).astype(np.float32);delta=(target.astype(float)-x[selected].astype(float))*1000.
 for i,row in enumerate(selected):
  k=local[int(row)]*3;lo[k:k+3]=hi[k:k+3]=delta[i]
 sol=q.solve_sparse_l1(A,b,np.zeros(3*len(support)),lo,hi,10.);item={"alpha":alpha,"solver":{k:v for k,v in sol.items() if k!="delta"}}
 if not sol["success"]:item["status"]="INFEASIBLE_OR_SOLVE_FAILURE";results.append(item);continue
 y=x.copy();y[support]=(x[support].astype(float)+sol["delta"].reshape(-1,3)/1000.).astype(np.float32);y[selected]=target;actual=(y.astype(float)-x.astype(float))*1000.;area=q.full_f32_area_gate(y,bm,ref,faces,.1);ls=float((A@actual[support].reshape(-1)-b).min(initial=1e30));es=float((EG@actual[support].reshape(-1)-eb).min(initial=1e30));mx=float(np.linalg.norm(actual,axis=1).max());sm=float((np.linalg.norm(y.astype(float)-refsrc,axis=1)*1000.).max());changed=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1));item.update(area_pass=area["pass"],linear_slack=ls,edge_slack_mm=es,max_step_mm=mx,max_source_mm=sm,changed_count=len(changed))
 if not area["pass"] or ls < -1e-4 or es < -1e-4 or mx>4.000001 or sm>12.000001 or not np.array_equal(y[anchors],x[anchors]):item["status"]="POST_EMISSION_GATE_FAIL";results.append(item);continue
 payload={k:v.copy() for k,v in base.items()};payload["points_world_m"]=y;p=out/("candidate-alpha-"+str(alpha)+".npz")
 with p.open("xb") as fp:np.savez_compressed(fp,**payload)
 item.update(status="CHEAP_GATE_PASS_ONLY",candidate=str(p),candidate_sha256=sha(p),changed_rows=changed.tolist());results.append(item)
(out/"report.json").write_text(json.dumps({"baseline_sha256":sha(a/"candidate.npz"),"search_report_sha256":sha(search),"support_nodes":len(support),"affected_faces":len(affected),"depth_rows":len(prows),"results":results,"elapsed_s":time.monotonic()-start,"native_admission":False,"accepted_physical_steps":0,"limitations":"No exact postfit self, 185-bone, solid containment, transfer or native audit."},indent=2,allow_nan=False)+"\n");print(json.dumps({"support_nodes":len(support),"results":results,"elapsed_s":time.monotonic()-start},indent=2))
