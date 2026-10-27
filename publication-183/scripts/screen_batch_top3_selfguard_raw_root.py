#!/usr/bin/env python3
"""Screen bounded independent fractions of a three-node SAT witness.

Evidence-only: incremental exact screening, never admission.
"""
from pathlib import Path
from itertools import product
import hashlib, importlib.util, json, sys, time
import numpy as np
N=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546"); S=N/"component-sat-selector-638"; w=S/"partial-midpoint-continuation-root-001"; a=w/"combined-43-57-65smooth-109-124-001"; original=S/"edge-refinement-661-threebank-001/generation-002-all-banks-001"; out=w/"batch-top3-selfguard-raw-screen-001"; out.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pairs(rep):return {(int(b["body"]),int(b["stable_bone_id"])):{tuple(map(int,v)) for v in (b.get("result") or {}).get("triangle_pairs",[])} for b in rep["banks"]}
base=load(a/"candidate.npz"); other=load(w/"combined-43-57-65smooth-109-001/candidate.npz"); trial=load(w/"sat-face57-witness-002/diagnostic/candidate.npz"); x=base["points_world_m"]; f=base["source_O_faces"]; parent=base["refinement_face_parent_face_row"]
assert sha(a/"candidate.npz")=="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
assert sha(w/"sat-face57-witness-002/diagnostic/candidate.npz")=="565374058766c8aa08c1977af49dd5770ba74bc53888c8399e7c287a840c00a8"
assert np.array_equal(f,trial["source_O_faces"])
bef=pairs(json.loads((a/"audit-159/report.json").read_text())); expected=pairs(json.loads((w/"combined-43-57-65smooth-109-001/audit-159/report.json").read_text()))
topo=mod("topogrid57",N/"topology/topology-bone-aware-025.py"); assert sha(topo.OWNER370)==topo.EXPECTED["owner370"]; owner=mod("countergrid57",topo.OWNER370);_,surf,bh,exact,runtime=owner.load_pinned_runtime();banks,keys=owner.typed_banks_from_159_inputs(bh.read_bank_inputs(owner.BANK_DIR_DEFAULT,owner.EVIDENCE_ROOT_DEFAULT));assert set(keys)==set(bef)
bp={k:{(int(parent[i]),j) for i,j in v} for k,v in bef.items()}
def calc(y,sel):
 nodes=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1)); faces=np.flatnonzero(np.any(np.isin(f,nodes),axis=1)); affected=set(map(int,faces)); new=owner.Exact159PatchCounter(y,sel,exact,surf.records).count_faces(faces,f[faces]); return {b.key:{v for v in bef[b.key] if v[0] not in affected}|set(map(tuple,new[b.key]["triangle_pairs"])) for b in sel}
check=calc(other["points_world_m"],banks);assert all(check[k]==expected[k] for k in keys)
qb=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py");assert sha(qb)=="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2";q=mod("qgrid57",qb);source=Path("/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz");assert sha(source)=="70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
with np.load(a/"candidate.npz",allow_pickle=False) as z:src,_=q.compose_source_reference(z,source)
predp=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py");assert sha(predp)=="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d";sys.path.insert(0,str(predp.parent.parent));pred=mod("numilab_human._grid57",predp);br,_=pred.float32_surface_records(x,f,source_seam_groups=());bs,_=pred._prepare_complete_self_pair_baseline(br);assert not bs.triangle_pairs
na=json.loads((original/"point-targets-589.current-source-normal-audit.json").read_text());nr=np.array([v["query_row"] for v in na["rows"]]);nn={int(v["query_row"]):np.array(v["registered_source_outward_normal_world_unit"]) for v in na["rows"]};ns={int(v["query_row"]):v["status"] for v in na["rows"]};ad=mod("grid57adapter",w/"run_partial_depthmax_targets589_root.py");current=ad.current_source_alignment(x,f,nr,nn,-1);ns={int(v["query_row"]):v["status"] for v in current["rows"]}


from scipy import sparse
from scipy.sparse.csgraph import connected_components
from itertools import product
selected=load(w/"targets-top-three-from653-001/nearest-union-exits.npz")["query_rows"];edges=np.unique(np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0);adj=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(x),len(x)));selfpairs=np.load(w/"batch-top3-self-witness-109-001/self-pairs.npy",allow_pickle=False);support=set(map(int,selected))|set(map(int,f[selfpairs.reshape(-1)].reshape(-1)));front=set(support)
for _ in range(3):front=set(map(int,adj[list(front)].indices))-support;support.update(front)
support=np.array(sorted(support));ncomp,labels=connected_components(adj[support][:,support],directed=False);assert ncomp==3
paths=[w/"batch-top3-selfguard-lp653-001/candidate-alpha-0.5.npz"]
trials=[load(p)["points_world_m"] for p in paths];results=[];best={};start=time.monotonic()
for ci in range(ncomp):
 section=support[labels==ci]
 for pidx,fraction in product(range(len(paths)),(1.,.75,.5,.25,.125)):
  y=x.copy();y[section]=(x[section].astype(float)+fraction*(trials[pidx][section].astype(float)-x[section].astype(float))).astype(np.float32);changed=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1));
  if not len(changed):continue
  assert not np.intersect1d(changed,base["typed_coarse_anchor_output_rows_018"]).size
  EG,eb,_=q.build_edge_gradient_constraints(x.astype(float)*1000.,f,changed,.5);delta=(y.astype(float)-x.astype(float))*1000.;area=q.full_f32_area_gate(y,x.astype(float)*1000.,base["area_reference_037_points_world_m_f64"]*1000.,f,.1);slack=float((EG@delta[changed].reshape(-1)-eb).min(initial=1e30));source_max=float((np.linalg.norm(y.astype(float)-src,axis=1)*1000).max());r={"component":ci,"source_candidate":str(paths[pidx]),"source_sha256":sha(paths[pidx]),"fraction":fraction,"changed_nodes":len(changed),"area_pass":area["pass"],"edge_slack_mm":slack,"source_max_mm":source_max}
  if not area["pass"] or slack < -1e-4 or source_max>12.000001:r["status"]="CHEAP_GATE_REJECT";results.append(r);continue
  fs=np.flatnonzero(np.any(np.isin(f,changed),axis=1));u=x.astype(float)[f[fs]];v=y.astype(float)[f[fs]];n0=np.cross(u[:,1]-u[:,0],u[:,2]-u[:,0]);n1=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);m0=np.linalg.norm(n0,axis=1);m1=np.linalg.norm(n1,axis=1);rat=m1/m0;cos=np.sum(n0*n1,axis=1)/(m0*m1);r.update(local_min_area=float(rat.min()),local_max_area=float(rat.max()),local_min_normal_cos=float(cos.min()))
  if r["local_min_area"]<.5 or r["local_max_area"]>2. or r["local_min_normal_cos"]<.85:r["status"]="LOCAL_SHAPE_REJECT";results.append(r);continue
  after=calc(y,banks);ap={key:{(int(parent[i]),j) for i,j in v} for key,v in after.items()};r.update(raw_total=sum(map(len,after.values())),parent_total=sum(map(len,ap.values())),raw_delta={f"{key[0]}/{key[1]}":len(after[key])-len(bef[key]) for key in keys if len(after[key])!=len(bef[key])},parent_delta={f"{key[0]}/{key[1]}":len(ap[key])-len(bp[key]) for key in keys if len(ap[key])!=len(bp[key])})
  if (r["parent_total"]>653 or (r["parent_total"]==653 and r["raw_total"]>=985)) or any(v>0 for v in r["raw_delta"].values()) or any(v>0 for v in r["parent_delta"].values()):r["status"]="NO_SAFE_COUNT_IMPROVEMENT";results.append(r);continue
  rec,_=pred.float32_surface_records(y,f,source_seam_groups=());_,sr=pred._update_complete_self_pair_baseline(bs,rec);r["self_pairs"]=int(sr["count"]);norm=ad.current_source_alignment(y,f,nr,nn,-1);r["new_normal_conflicts"]=[int(z["query_row"]) for z in norm["rows"] if z["status"]!="PASS" and ns[int(z["query_row"])]=="PASS"];r["status"]="SCREEN_PASS_ONLY" if not r["self_pairs"] and not r["new_normal_conflicts"] else "SELF_OR_NORMAL_REJECT";results.append(r)
  if r["status"]=="SCREEN_PASS_ONLY" and (ci not in best or (r["parent_total"],r["raw_total"])<(best[ci][0]["parent_total"],best[ci][0]["raw_total"])):best[ci]=(r,y.copy())
if best:
 y=x.copy()
 for ci,(item,proposal) in best.items():
  section=support[labels==ci];y[section]=proposal[section]
 after=calc(y,banks);ap={key:{(int(parent[i]),j) for i,j in v} for key,v in after.items()};rec,_=pred.float32_surface_records(y,f,source_seam_groups=());_,sr=pred._update_complete_self_pair_baseline(bs,rec);norm=ad.current_source_alignment(y,f,nr,nn,-1);conf=[int(z["query_row"]) for z in norm["rows"] if z["status"]!="PASS" and ns[int(z["query_row"])]=="PASS"];combined={"raw_total":sum(map(len,after.values())),"parent_total":sum(map(len,ap.values())),"raw_delta":{f"{key[0]}/{key[1]}":len(after[key])-len(bef[key]) for key in keys if len(after[key])!=len(bef[key])},"parent_delta":{f"{key[0]}/{key[1]}":len(ap[key])-len(bp[key]) for key in keys if len(ap[key])!=len(bp[key])},"self_pairs":int(sr["count"]),"new_normal_conflicts":conf}
 if (combined["parent_total"]<653 or (combined["parent_total"]==653 and combined["raw_total"]<985)) and not any(v>0 for v in combined["raw_delta"].values()) and not any(v>0 for v in combined["parent_delta"].values()) and not combined["self_pairs"] and not conf:
  base["points_world_m"]=y
  with (out/"candidate.npz").open("xb") as fp:np.savez_compressed(fp,**base)
 else:best={}
else:combined=None
summary={"status":"SCREEN_PASS_ONLY" if best else "NO_SAFE_IMPROVEMENT","results":results,"best_components":{str(k):v[0] for k,v in best.items()},"combined":combined,"baseline_candidate_sha256":sha(a/"candidate.npz"),"candidate_sha256":sha(out/"candidate.npz") if best else None,"elapsed_s":time.monotonic()-start,"geometry_admitted":False,"accepted_physical_steps":0,"limitations":"Incremental exact screen; full 130/159, closed-solid containment, transfer, and native qualification remain."};(out/"report.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n");print(json.dumps({k:v for k,v in summary.items() if k!="results"},indent=2),flush=True)
