#!/usr/bin/env python3
"""Screen bounded independent fractions of a three-node SAT witness.

Evidence-only: incremental exact screening, never admission.
"""
from pathlib import Path
from itertools import product
import hashlib, importlib.util, json, sys, time
import numpy as np
N=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546"); S=N/"component-sat-selector-638"; w=S/"partial-midpoint-continuation-root-001"; a=S/"edge-refinement-661-threebank-001/generation-002-all-banks-001"; out=w/"connected65-blend075-screen-001"; out.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pairs(rep):return {(int(b["body"]),int(b["stable_bone_id"])):{tuple(map(int,v)) for v in (b.get("result") or {}).get("triangle_pairs",[])} for b in rep["banks"]}
base=load(a/"candidate.npz"); other=load(w/"proposal-six-001/candidate.npz"); trial=load(w/"sat-face57-witness-002/diagnostic/candidate.npz"); x=base["points_world_m"]; f=base["source_O_faces"]; parent=base["refinement_face_parent_face_row"]
assert sha(a/"candidate.npz")=="d367af1383d920645c22552569e95614797de8aea8a757b658def68bad4412d8"
assert sha(w/"sat-face57-witness-002/diagnostic/candidate.npz")=="565374058766c8aa08c1977af49dd5770ba74bc53888c8399e7c287a840c00a8"
assert np.array_equal(f,trial["source_O_faces"])
bef=pairs(json.loads((a/"audit-159/report.json").read_text())); expected=pairs(json.loads((w/"proposal-six-001/audit-159/report.json").read_text()))
topo=mod("topogrid57",N/"topology/topology-bone-aware-025.py"); assert sha(topo.OWNER370)==topo.EXPECTED["owner370"]; owner=mod("countergrid57",topo.OWNER370);_,surf,bh,exact,runtime=owner.load_pinned_runtime();banks,keys=owner.typed_banks_from_159_inputs(bh.read_bank_inputs(owner.BANK_DIR_DEFAULT,owner.EVIDENCE_ROOT_DEFAULT));assert set(keys)==set(bef)
bp={k:{(int(parent[i]),j) for i,j in v} for k,v in bef.items()}
def calc(y,sel):
 nodes=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1)); faces=np.flatnonzero(np.any(np.isin(f,nodes),axis=1)); affected=set(map(int,faces)); new=owner.Exact159PatchCounter(y,sel,exact,surf.records).count_faces(faces,f[faces]); return {b.key:{v for v in bef[b.key] if v[0] not in affected}|set(map(tuple,new[b.key]["triangle_pairs"])) for b in sel}
check=calc(other["points_world_m"],banks);assert all(check[k]==expected[k] for k in keys)
qb=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py");assert sha(qb)=="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2";q=mod("qgrid57",qb);source=Path("/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz");assert sha(source)=="70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
with np.load(a/"candidate.npz",allow_pickle=False) as z:src,_=q.compose_source_reference(z,source)
predp=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py");assert sha(predp)=="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d";sys.path.insert(0,str(predp.parent.parent));pred=mod("numilab_human._grid57",predp);br,_=pred.float32_surface_records(x,f,source_seam_groups=());bs,_=pred._prepare_complete_self_pair_baseline(br);assert not bs.triangle_pairs
na=json.loads((a/"point-targets-589.current-source-normal-audit.json").read_text());nr=np.array([v["query_row"] for v in na["rows"]]);nn={int(v["query_row"]):np.array(v["registered_source_outward_normal_world_unit"]) for v in na["rows"]};ns={int(v["query_row"]):v["status"] for v in na["rows"]};ad=mod("grid57adapter",w/"run_partial_depthmax_targets589_root.py")

sparse_trial=load(w/"connected-witness65-001/candidate-alpha-0.75.npz");smooth_trial=load(w/"connected-witness65-smooth-001/candidate-alpha-0.75.npz");assert np.array_equal(sparse_trial["source_O_faces"],f) and np.array_equal(smooth_trial["source_O_faces"],f);xs=sparse_trial["points_world_m"];xt=smooth_trial["points_world_m"];support=np.flatnonzero(np.any(xs.view(np.uint32)!=x.view(np.uint32),axis=1)|np.any(xt.view(np.uint32)!=x.view(np.uint32),axis=1));assert not np.intersect1d(support,base["typed_coarse_anchor_output_rows_018"]).size;EG,eb,_=q.build_edge_gradient_constraints(x.astype(float)*1000.,f,support,.5)
results=[];best=None;start=time.monotonic()
for beta in np.linspace(0.,1.,33):
 y=x.copy();y[support]=((1.-beta)*xs[support].astype(float)+beta*xt[support].astype(float)).astype(np.float32);delta=(y.astype(float)-x.astype(float))*1000.;area=q.full_f32_area_gate(y,x.astype(float)*1000.,base["area_reference_037_points_world_m_f64"]*1000.,f,.1);slack=float((EG@delta[support].reshape(-1)-eb).min(initial=1e30));source_max=float((np.linalg.norm(y.astype(float)-src,axis=1)*1000).max());r={"beta":float(beta),"area_pass":area["pass"],"edge_slack_mm":slack,"source_max_mm":source_max}
 if not area["pass"] or slack < -1e-4 or source_max>12.000001:r["status"]="CHEAP_GATE_REJECT";results.append(r);continue
 after=calc(y,banks);ap={k:{(int(parent[i]),j) for i,j in v} for k,v in after.items()};r.update(raw_total=sum(map(len,after.values())),parent_total=sum(map(len,ap.values())),raw_delta={f"{k[0]}/{k[1]}":len(after[k])-len(bef[k]) for k in keys if len(after[k])!=len(bef[k])},parent_delta={f"{k[0]}/{k[1]}":len(ap[k])-len(bp[k]) for k in keys if len(ap[k])!=len(bp[k])})
 if r["parent_total"]>=661 or any(v>0 for v in r["raw_delta"].values()) or any(v>0 for v in r["parent_delta"].values()):r["status"]="NO_SAFE_COUNT_IMPROVEMENT";results.append(r);continue
 rec,_=pred.float32_surface_records(y,f,source_seam_groups=());_,sr=pred._update_complete_self_pair_baseline(bs,rec);r["self_pairs"]=int(sr["count"]);norm=ad.current_source_alignment(y,f,nr,nn,-1);r["new_normal_conflicts"]=[int(z["query_row"]) for z in norm["rows"] if z["status"]!="PASS" and ns[int(z["query_row"])]=="PASS"];r["status"]="SCREEN_PASS_ONLY" if not r["self_pairs"] and not r["new_normal_conflicts"] else "SELF_OR_NORMAL_REJECT";results.append(r)
 if r["status"]=="SCREEN_PASS_ONLY" and (best is None or (r["parent_total"],r["raw_total"],-r["beta"])<(best[0]["parent_total"],best[0]["raw_total"],-best[0]["beta"])):best=(r,y.copy())
if best:
 base["points_world_m"]=best[1]
 with (out/"candidate.npz").open("xb") as fp:np.savez_compressed(fp,**base)
summary={"status":"SCREEN_PASS_ONLY" if best else "NO_SAFE_IMPROVEMENT","results":results,"selected":best[0] if best else None,"baseline_candidate_sha256":sha(a/"candidate.npz"),"sparse_sha256":sha(w/"connected-witness65-001/candidate-alpha-0.75.npz"),"smooth_sha256":sha(w/"connected-witness65-smooth-001/candidate-alpha-0.75.npz"),"full_six_159_parity":True,"candidate_sha256":sha(out/"candidate.npz") if best else None,"elapsed_s":time.monotonic()-start,"geometry_admitted":False,"accepted_physical_steps":0,"limitations":"Incremental screen only; full exact 130/159, containment, skin/fat transfer, native qualification still required."};(out/"report.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n");print(json.dumps({k:v for k,v in summary.items() if k!="results"},indent=2))
