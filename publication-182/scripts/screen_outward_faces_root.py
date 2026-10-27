#!/usr/bin/env python3
"""Screen bounded source-informed tangential continuations of a three-node SAT witness.

Evidence-only: incremental exact screening, never admission.
"""
from pathlib import Path
from itertools import product
import hashlib, importlib.util, json, sys, time
import numpy as np
N=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546"); S=N/"component-sat-selector-638"; w=S/"partial-midpoint-continuation-root-001"; a=S/"edge-refinement-661-threebank-001/generation-002-all-banks-001"; out=w/("outward-face-"+sys.argv[1]+"-"+sys.argv[2]+"-001"); out.mkdir(exist_ok=False)
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

body=int(sys.argv[1]);bone=int(sys.argv[2]); key=(body,bone); target=[b for b in banks if b.key==key];assert len(target)==1 and key in bef
byface={}
for face,tri in bef[key]:byface.setdefault(int(face),[]).append(int(tri))
# Highest-witness faces drive trials; anchored nodes do not move.
face_ids=sorted(byface,key=lambda i:(-len(byface[i]),i))[:8];ref=base["area_reference_037_points_world_m_f64"]*1000.;bm=x.astype(float)*1000.;anchor=set(map(int,base["typed_coarse_anchor_output_rows_018"]))
contexts=[]
for face in face_ids:
 allnodes=np.asarray(f[face],dtype=np.int64);support=np.array([int(n) for n in allnodes if int(n) not in anchor],dtype=np.int64)
 if not len(support):continue
 n=-np.cross(ref[allnodes[1]]-ref[allnodes[0]],ref[allnodes[2]]-ref[allnodes[0]]);length=np.linalg.norm(n)
 if length<1e-12:continue
 EG,eb,_=q.build_edge_gradient_constraints(bm,f,support,.5);contexts.append((face,support,n/length,EG,eb))
assert contexts
results=[];best=None;start=time.monotonic();rng=np.random.default_rng(20261027+body*1000+bone)
for trial_index in range(4000):
 if time.monotonic()-start>50:break
 face,support,normal,EG,eb=contexts[trial_index%len(contexts)]
 # Bias motion outward from the registered source face, with bounded tangential
 # adjustments. Half the trials reverse the normal to test local alternatives.
 sign=1. if trial_index%4 else -1.;mag=rng.uniform(0.,1.8,size=(len(support),1));scale=(.2,.4,.7)[trial_index%3];noise=rng.normal(0.,scale,size=(len(support),3));motion=sign*mag*normal+noise
 y=x.copy();y[support]=(x[support].astype(float)+motion/1000.).astype(np.float32);delta=(y.astype(float)-x.astype(float))*1000
 if not np.any(delta[support]):continue
 after=calc(y,target);raw=len(after[key]);ap={(int(parent[i]),j) for i,j in after[key]};r={"trial_index":trial_index,"face":face,"support":support.tolist(),"motion_mm":motion.tolist(),"target_raw":raw,"target_parent":len(ap),"removed_parent":len(bp[key]-ap),"added_parent":len(ap-bp[key])}
 if len(ap)>=len(bp[key]) or raw>len(bef[key]):continue
 area=q.full_f32_area_gate(y,bm,ref,f,.1);slack=float((EG@delta[support].reshape(-1)-eb).min(initial=1e30));source_max=float((np.linalg.norm(y.astype(float)-src,axis=1)*1000).max());max_step=float(np.linalg.norm(delta,axis=1).max());r.update(area_pass=area["pass"],edge_slack_mm=slack,source_max_mm=source_max,max_step_mm=max_step)
 if not area["pass"] or slack < -1e-7 or source_max>12.000001 or max_step>4.000001:r["status"]="CHEAP_GATE_REJECT";results.append(r);continue
 after=calc(y,banks);allp={k:{(int(parent[i]),j) for i,j in v} for k,v in after.items()};r.update(raw_total=sum(map(len,after.values())),parent_total=sum(map(len,allp.values())),raw_delta={f"{k[0]}/{k[1]}":len(after[k])-len(bef[k]) for k in keys if len(after[k])!=len(bef[k])},parent_delta={f"{k[0]}/{k[1]}":len(allp[k])-len(bp[k]) for k in keys if len(allp[k])!=len(bp[k])})
 if r["parent_total"]>=661 or any(v>0 for v in r["raw_delta"].values()) or any(v>0 for v in r["parent_delta"].values()):r["status"]="OTHER_BANK_REGRESSION";results.append(r);continue
 rec,_=pred.float32_surface_records(y,f,source_seam_groups=());_,sr=pred._update_complete_self_pair_baseline(bs,rec);r["self_pairs"]=int(sr["count"]);norm=ad.current_source_alignment(y,f,nr,nn,-1);r["new_normal_conflicts"]=[int(z["query_row"]) for z in norm["rows"] if z["status"]!="PASS" and ns[int(z["query_row"])]=="PASS"];r["status"]="SCREEN_PASS_ONLY" if not r["self_pairs"] and not r["new_normal_conflicts"] else "SELF_OR_NORMAL_REJECT";results.append(r)
 if r["status"]=="SCREEN_PASS_ONLY" and (best is None or (r["parent_total"],r["raw_total"])<(best[0]["parent_total"],best[0]["raw_total"])):best=(r,y.copy())
if best:
 base["points_world_m"]=best[1]
 with (out/"candidate.npz").open("xb") as fp:np.savez_compressed(fp,**base)
summary={"status":"SCREEN_PASS_ONLY" if best else "NO_SAFE_IMPROVEMENT","target_bank":list(key),"witness_faces":face_ids,"trial_count":trial_index+1,"results":results,"baseline_candidate_sha256":sha(a/"candidate.npz"),"full_six_159_parity":True,"candidate_sha256":sha(out/"candidate.npz") if best else None,"elapsed_s":time.monotonic()-start,"geometry_admitted":False,"accepted_physical_steps":0,"limitations":"Incremental screen only; full exact 130/159, containment, skin/fat transfer, native qualification still required."}
(out/"report.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n");print(json.dumps({"status":summary["status"],"target_bank":summary["target_bank"],"trial_count":summary["trial_count"],"passing":len([z for z in results if z["status"]=="SCREEN_PASS_ONLY"]),"best":best[0] if best else None,"candidate_sha256":summary["candidate_sha256"],"elapsed_s":summary["elapsed_s"]},indent=2),flush=True)
