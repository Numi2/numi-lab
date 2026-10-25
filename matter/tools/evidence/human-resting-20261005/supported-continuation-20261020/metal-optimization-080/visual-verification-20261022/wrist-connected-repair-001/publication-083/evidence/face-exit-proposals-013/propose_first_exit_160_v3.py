#!/usr/bin/env python3
"""Proposal-only first SAT overlap exits for exact skin/bone face witnesses."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, sys, time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
CAND=D/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
FACE= D/"full-connected-clearance-150/run-013-first-exit-proposals/face-audit-001/report.json"
COARSE=D/"publication-072/evidence/whole-reference-TetGen-084/whole-skin-fat-reference-084-candidate.npz"
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
TRI=BANKDIR/"exact-emitted.triangles.csv"
BODY=BANKDIR/"exact-emitted.bodies.csv"
HELPER=D/"full-connected-clearance-150/triangle_translation_intervals_160.py"
OUT=D/"full-connected-clearance-150/run-013-first-exit-proposals/proposal-002"
EXPECTED={
 str(CAND):"70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31",
 str(TRI):"0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac",
 str(BODY):"dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90",
 str(HELPER):"b9e2dc94bdd4a49d3c2635e312a5d5b5b9a35786ed68695fb4fd96f7317e865d",
 str(COARSE):"d3d0abb590f5858de4a27e0362e98bc9b3088831183ea045096cefc4ebba26e1",
 str(FACE):"26083de305f88c310ea12c159a6eb8e6ec2504bcd08b3d37eb5589f8ceea8171",
}
CAP_M=0.006

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def module(path):
 spec=importlib.util.spec_from_file_location("sat_translation_160",path)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def cyc_equal(a,b):
 return any(np.array_equal(a,np.roll(b,k)) for k in range(3))

def cyc_reverse(a,b):
 br=b[[0,2,1]]
 return any(np.array_equal(a,np.roll(br,k)) for k in range(3))

def orient_surface(F,kind,parent,node_source_row,source_faces):
 edge_rows=defaultdict(list)
 for fi,t in enumerate(F):
  for a,b in ((int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))):
   key=(min(a,b),max(a,b));sgn=1 if (a,b)==key else -1
   edge_rows[key].append((fi,sgn))
 adj=[[] for _ in range(len(F))]
 nonmanifold=0
 for vals in edge_rows.values():
  if len(vals)>2: nonmanifold+=1
  if len(vals)==2:
   (i,si),(j,sj)=vals
   rel=-si*sj
   adj[i].append((j,rel));adj[j].append((i,rel))
 flip=np.full(len(F),-1,dtype=np.int8);comps=[]
 for root in range(len(F)):
  if flip[root]>=0: continue
  flip[root]=0;q=deque([root]);comp=[];conflicts=0
  while q:
   i=q.popleft();comp.append(i)
   for j,rel in adj[i]:
    want=flip[i] if rel==1 else 1-flip[i]
    if flip[j]<0: flip[j]=want;q.append(j)
    elif flip[j]!=want: conflicts+=1
  comps.append((comp,conflicts))
 if nonmanifold or any(c for _,c in comps):
  raise RuntimeError(f"surface orientation topology invalid nonmanifold={nonmanifold} conflicts={sum(c for _,c in comps)}")
 component_by_face=np.empty(len(F),dtype=np.int32); report=[]
 for ci,(comp,_) in enumerate(comps):
  component_by_face[comp]=ci
  seeds=[];signs=[]
  for fi in comp:
   if int(kind[fi])!=0: continue
   pr=int(parent[fi])
   if pr<0 or pr>=len(source_faces): raise RuntimeError(f"bad coarse face parent at {fi}: {pr}")
   a=np.asarray(node_source_row[F[fi]]);b=np.asarray(source_faces[pr])
   if cyc_equal(a,b): parity=1
   elif cyc_reverse(a,b): parity=-1
   else: raise RuntimeError(f"kind0 face {fi} not same triangle as foundation parent {pr}")
   corrected=parity*(-1 if int(flip[fi]) else 1)
   seeds.append(int(fi));signs.append(corrected)
  if not seeds:
   raise RuntimeError(f"surface component {ci} has no foundation coarse-face orientation seed")
  if len(set(signs))!=1:
   raise RuntimeError(f"surface component {ci} coarse seed orientation disagreement")
  report.append({"component":ci,"face_count":len(comp),"coarse_orientation_seed_count":len(seeds),"coarse_orientation_parity_after_consistency":signs[0]})
 return flip,component_by_face,report

def main():
 t0=time.monotonic()
 OUT.mkdir(parents=True,exist_ok=False)
 for p,h in EXPECTED.items():
  got=sha(p)
  if got!=h: raise RuntimeError(f"input SHA mismatch {p}: {got} expected {h}")
 face_doc=json.load(open(FACE))
 if face_doc.get("complete") is not True or face_doc.get("expected_bank_count")!=185: raise RuntimeError("matched complete 185-bank exact face audit required")
 if face_doc.get("candidate_sha256")!=EXPECTED[str(CAND)]: raise RuntimeError("face audit does not match certified candidate")
 z=np.load(CAND,allow_pickle=False); P=np.asarray(z["points_world_m"],dtype=np.float32).astype(np.float64);F=np.asarray(z["source_O_faces"],dtype=np.int64)
 kind=np.asarray(z["face_kind"],dtype=np.int64);parent=np.asarray(z["parent_coarse_face_row"],dtype=np.int64)
 if P.ndim!=2 or P.shape[1]!=3 or F.ndim!=2 or F.shape[1]!=3 or len(F)!=len(kind) or len(F)!=len(parent):raise RuntimeError("candidate schema mismatch")
 if not np.isfinite(P).all() or F.min()<0 or F.max()>=len(P):raise RuntimeError("invalid candidate geometry")
 coarse=np.load(COARSE,allow_pickle=False)
 flip,component,orientation_report=orient_surface(F,kind,parent,np.asarray(z["node_source_row"],dtype=np.int64),np.asarray(coarse["source_O_faces"],dtype=np.int64))
 crosses=defaultdict(set)
 exact_pair_count=0
 for b in face_doc["banks"]:
  body0,bone0=int(b["body"]),int(b["stable_bone_id"])
  rr=b.get("result") or {}
  for face,triid in rr.get("triangle_pairs",[]):
   crosses[int(face)].add((body0,bone0,int(triid)));exact_pair_count+=1
 if not crosses: raise RuntimeError("exact face audit has no crossings; do not fabricate exit proposals")
 for fi in crosses:
  ci=int(component[fi]);parity=orientation_report[ci]["coarse_orientation_parity_after_consistency"]
  t=F[fi];n=np.cross(P[t[1]]-P[t[0]],P[t[2]]-P[t[0]])
  n*=(-1 if int(flip[fi]) else 1)
  # Foundation O face winding points into its material-1 skin tet.
  n*=(-parity)
  nn=float(np.linalg.norm(n))
  if nn==0 or not math.isfinite(nn):raise RuntimeError(f"degenerate crossed face {fi}")
  n/=nn
  # Outward is a world vector; do not transform and renormalize in bone-local space.
  crosses[fi]={"expected_pairs":crosses[fi],"outward_world_unit":n}
 # Parse native-emitted F32 bank corners; exact values are decimal encodings of F32.
 raw=np.loadtxt(TRI,delimiter=",",skiprows=1,usecols=(1,2,4,5,6,7,8,9,10,11,12,13),dtype=np.float64)
 if raw.ndim!=2 or raw.shape[1]!=12 or len(raw)!=469136:raise RuntimeError("unexpected emitted triangle table shape/count")
 body_doc={}
 with open(BODY,newline="") as f:
  for row in csv.DictReader(f):
   bi=int(row["body_index"])
   body_doc[bi]={
    "position":np.array([float(row[f"position_{a}_m"]) for a in "xyz"],dtype=np.float64),
    "M":np.array([[float(row[f"world_to_body_m{i}{j}"]) for j in range(3)] for i in range(3)],dtype=np.float64),
    "offset":np.array([float(row[f"com_offset_{a}_m"]) for a in "xyz"],dtype=np.float64),
   }
 banks={}
 for key in sorted(set((int(r[0]),int(r[1])) for r in raw)):
  body,bone=key; rr=raw[(raw[:,0]==body)&(raw[:,1]==bone)]
  ids=rr[:,2].astype(np.int64)
  if len(np.unique(ids))!=len(ids):raise RuntimeError(f"duplicate stable triangle ID in bank {key}")
  frame=body_doc[body];M=frame["M"]
  local=rr[:,3:12].reshape(-1,3,3)
  # Exact raw native frame inverse: local = M*(world-position)-COMoffset.
  world=(np.linalg.solve(M,(local+frame["offset"]).reshape(-1,3).T).T.reshape(-1,3,3)+frame["position"])
  banks[key]={"ids":ids,"id_to_row":{int(v):i for i,v in enumerate(ids)},"tri":world,"lo":world.min(axis=1),"hi":world.max(axis=1)}
 if len(banks)!=185:raise RuntimeError(f"expected 185 typed banks, got {len(banks)}")
 report_banks={(int(b["body"]),int(b["stable_bone_id"])):b for b in face_doc["banks"]}
 if set(report_banks)!=set(banks):raise RuntimeError("face audit and exact bank owner keysets differ")
 for key,b in banks.items():
  if int(report_banks[key]["bank_faces"])!=len(b["ids"]):raise RuntimeError(f"bank face count mismatch {key}")
 sat=module(HELPER)
 summaries=[];bank_candidates=defaultdict(int);bank_examined=defaultdict(int)
 sat_total=0;aabb_total=0;forced_total=0;miss_total=0
 ordered_banks=sorted(banks)
 for index,fi in enumerate(sorted(crosses)):
  w=crosses[fi];d=w["outward_world_unit"];verts=P[F[fi]]
  end=verts+CAP_M*d
  lo=np.nextafter(np.minimum(verts,end).min(axis=0),-np.inf)
  hi=np.nextafter(np.maximum(verts,end).max(axis=0),np.inf)
  expected=set(w["expected_pairs"])
  raw_intervals=[];exact_misses=[];forced=[];local_candidates=0
  for key in ordered_banks:
   b=banks[key];bank_examined[key]+=len(b["ids"])
   mask=np.all(b["hi"]>=lo,axis=1)&np.all(b["lo"]<=hi,axis=1)
   selected=np.flatnonzero(mask)
   expected_ids={tid for eb,eo,tid in expected if (eb,eo)==key}
   selected_ids=set(map(int,b["ids"][selected]))
   missing=expected_ids-selected_ids
   if missing:
    forced.extend([{"bank":[key[0],key[1]],"triangle_stable_id":tid} for tid in sorted(missing)])
    add=np.array([b["id_to_row"][tid] for tid in sorted(missing)],dtype=np.int64)
    selected=np.concatenate([selected,add])
   bank_candidates[key]+=len(selected)
   aabb_total+=int(np.count_nonzero(mask));forced_total+=len(missing);local_candidates+=len(selected)
   for row in selected:
    tid=int(b["ids"][row]);pair=(key[0],key[1],tid);is_exact=pair in expected
    interval=sat.triangle_translation_interval(verts,b["tri"][row],d,CAP_M)
    if interval is None:
     if is_exact: exact_misses.append({"body":key[0],"stable_bone_id":key[1],"triangle_stable_id":tid,"reason":"SAT returned no interval for exact origin witness"})
     continue
    raw_entry=float(interval["entry_distance_m"]);raw_exit=float(interval["exit_distance_m"])
    if is_exact and raw_entry>0.0:
     # The exact audit is authoritative that this pair overlaps at s=0.
     # Preserve raw SAT start and include a separate evidence-backed zero-start proposal.
     eff=dict(interval);eff["entry_distance_m"]=0.0
     exact_misses.append({"body":key[0],"stable_bone_id":key[1],"triangle_stable_id":tid,"reason":"positive floating SAT entry despite exact current intersection","raw_entry_distance_m":raw_entry})
     raw_intervals.append({"pair_id":f"{key[0]}/{key[1]}/{tid}","interval":eff,"raw_entry_distance_m":raw_entry,"raw_exit_distance_m":raw_exit,"exact_current_intersection":True,"origin_inclusion_override_from_exact_audit":True,"entry_axes":interval["entry_axes"],"exit_axes":interval["exit_axes"]})
    else:
     raw_intervals.append({"pair_id":f"{key[0]}/{key[1]}/{tid}","interval":interval,"raw_entry_distance_m":raw_entry,"raw_exit_distance_m":raw_exit,"exact_current_intersection":bool(is_exact),"origin_inclusion_override_from_exact_audit":False,"entry_axes":interval["entry_axes"],"exit_axes":interval["exit_axes"]})
  merged=sat.union_triangle_translation_intervals(raw_intervals)
  first=sat.first_current_overlap_exit(merged)
  first_ids=set(first["pair_ids"]) if first else set()
  metadata={r["pair_id"]:r for r in raw_intervals}
  first_limiters=[]
  for pid in sorted(first_ids):
   rec=metadata.get(pid)
   if rec: first_limiters.append({"pair_id":pid,"raw_entry_distance_m":rec["raw_entry_distance_m"],"raw_exit_distance_m":rec["raw_exit_distance_m"],"exact_current_intersection":rec["exact_current_intersection"],"origin_inclusion_override_from_exact_audit":rec["origin_inclusion_override_from_exact_audit"],"exit_axes":rec["exit_axes"]})
  summaries.append({
   "surface_face_row":int(fi),"face_kind":int(kind[fi]),"vertex_rows":F[fi].astype(int).tolist(),
   "outward_world_unit":d.tolist(),"exact_current_pair_count":len(expected),
   "exact_current_pairs":[{"body":x[0],"stable_bone_id":x[1],"triangle_stable_id":x[2]} for x in sorted(expected)],
   "banks_examined":185,"bank_triangle_rows_examined":int(sum(len(banks[k]["ids"]) for k in ordered_banks)),
   "swept_aabb_candidate_pair_count_before_forced_witnesses":int(sum(int(np.count_nonzero(np.all(banks[k]["hi"]>=lo,axis=1)&np.all(banks[k]["lo"]<=hi,axis=1))) for k in ordered_banks)),
   "swept_aabb_candidate_pair_count_including_forced_witnesses":int(local_candidates),
   "forced_exact_witnesses_missing_from_aabb":forced,
   "sat_interval_pair_count":len(raw_intervals),
   "exact_witness_sat_origin_anomalies":exact_misses,
   "merged_candidate_intervals":merged,
   "first_current_overlap_exit":first,
   "first_exit_limiting_pair_records":first_limiters,
   "candidate_only":True,"clearance_certificate":False
  })
  sat_total+=len(raw_intervals);miss_total+=len(exact_misses)
  if (index+1)%25==0:
   (OUT/"progress.json").write_text(json.dumps({"completed_faces":index+1,"total_faces":len(crosses),"sat_interval_pairs":sat_total,"aabb_candidates":aabb_total,"exact_witness_sat_anomalies":miss_total,"elapsed_seconds":time.monotonic()-t0})+"\n")
 # Cross-check every exact input pair was retained in candidate's per-face record.
 out_expected={(r["surface_face_row"],x["body"],x["stable_bone_id"],x["triangle_stable_id"]) for r in summaries for x in r["exact_current_pairs"]}
 in_expected={(fi,b,bo,tid) for fi in crosses for b,bo,tid in crosses[fi]["expected_pairs"]}
 if out_expected!=in_expected:raise RuntimeError("exact face witness inventory changed")
 report={
  "schema":"numi.human.skin-bone-first-exit-proposals.v1",
  "status":"COMPLETE_PROPOSAL_ONLY",
  "candidate_sha256":sha(CAND),"candidate_path":str(CAND),
  "face_audit_sha256":sha(FACE),"face_audit_path":str(FACE),
  "surface_point_count":len(P),"surface_face_count":len(F),"exact_current_unique_face_count":len(crosses),
  "exact_current_pair_count":exact_pair_count,"typed_bank_count":len(banks),"expected_typed_bank_count":185,
  "bank_inputs":{"triangles_sha256":sha(TRI),"bodies_sha256":sha(BODY),"triangle_rows":len(raw),"bank_keys":[list(k) for k in ordered_banks]},
  "foundation_sha256":sha(COARSE),"orientation":{"method":"edge-adjacency orientation propagation, seeded per component by exact cyclic comparison of retained kind0 face row to pinned 084 source_O_faces; pinned 084 source_O winding points into material-1 skin tet, so proposal direction is opposite","candidate_components":orientation_report,"nonmanifold_edges":0,"winding_conflicts":0},
  "sat_helper_path":str(HELPER),"sat_helper_sha256":sha(HELPER),
  "coordinate_method":"candidate vertices cast to F32 then promoted to F64, matching exact 159 audit; bank corners reconstructed from exact emitted F32 local corners through pinned raw world_to_body matrix and COM offset; directions normalized in world only",
  "sweep_cap_m":CAP_M,"per_face_summaries":summaries,
  "aggregate":{"faces_with_current_exact_crossings":len(crosses),"aabb_candidates_before_forced_witnesses":aabb_total,"forced_exact_witnesses_missing_from_aabb":forced_total,"sat_interval_pairs":sat_total,"exact_witness_sat_origin_anomalies":miss_total,"faces_with_first_current_overlap_exit":sum(x["first_current_overlap_exit"] is not None for x in summaries),"faces_without_first_exit":sum(x["first_current_overlap_exit"] is None for x in summaries),"faces_with_later_union_components":sum(len(x["merged_candidate_intervals"])>1 for x in summaries),"per_bank_sweep_aabb_candidates":[{"body":k[0],"stable_bone_id":k[1],"triangle_rows_examined":bank_examined[k],"candidate_pairs":bank_candidates[k]} for k in ordered_banks]},
  "limitations":["Floating SAT intervals are construction proposals only, not exact collision-clearance proofs.","All 185 typed banks were swept over [0,6 mm] for each exact current-crossing surface face; this does not certify unqueried faces or translated geometry.","Exact current face audit and exact full-surface/self audits remain required after any fit."],
  "elapsed_seconds":time.monotonic()-t0,"geometry_admitted":False,"physical_steps":0
 }
 (OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"status":report["status"],"candidate_sha256":report["candidate_sha256"],"faces":len(crosses),"exact_pairs":exact_pair_count,"banks":len(banks),"aabb_candidates":aabb_total,"sat_pairs":sat_total,"origin_anomalies":miss_total,"faces_with_exit":report["aggregate"]["faces_with_first_current_overlap_exit"],"elapsed_seconds":report["elapsed_seconds"]},sort_keys=True))
if __name__=="__main__":main()

