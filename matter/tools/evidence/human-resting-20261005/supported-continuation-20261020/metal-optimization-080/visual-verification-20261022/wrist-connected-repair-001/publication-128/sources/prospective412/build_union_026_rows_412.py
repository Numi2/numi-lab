#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
CAND=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/candidate.npz"
SEEDS=ROOT/"D/self-witness-refresh-410/run-001/merged-self-pair-seeds.npy"
NEW=ROOT/"D/prospective-self-guards-412/run-005-witness-one-ring/new-pair-ids.npy"
SCOPE=ROOT/"D/prospective-self-guards-412/run-005-witness-one-ring/report.json"
FAILED=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-001/audit-130/surface-self-pairs.npy"
FAILED_SUMMARY=FAILED.with_name("summary.json")
CUR_AUDIT=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/audit-130/summary.json"
CUR_SURFACE=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/audit-159/report.json"
OWNER_PATH=ROOT/"barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"
OUT=ROOT/"D/prospective-self-guards-412/run-007-selected410-union"
MAX_SECONDS=90.0; MARGIN_FRACTION=0.25; ZERO_TOL_MM=2e-8; CHECKPOINT_EVERY=1000
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def loadmod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 assert s.loader is not None;s.loader.exec_module(m);return m
def writejson(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n");tmp.replace(path)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);t0=time.monotonic()
 cand_sha=sha(CAND);owner_sha=sha(OWNER_PATH)
 with np.load(CAND,allow_pickle=False) as z:p=z["points_world_m"];f=z["source_O_faces"]
 if p.dtype!=np.float32 or p.shape!=(20889,3) or f.shape!=(41733,3):raise ValueError("selected 410 shape/dtype mismatch")
 seeds=np.load(SEEDS,allow_pickle=False);new=np.load(NEW,allow_pickle=False);failed=np.load(FAILED,allow_pickle=False)
 if seeds.shape!=(18069,2) or new.shape!=(38156,2) or failed.shape!=(23,2):raise ValueError("unexpected source pair-set shape")
 if any(a.dtype.kind not in "iu" for a in (seeds,new,failed)):raise ValueError("pair sources must be integer")
 seed_set={tuple(map(int,x)) for x in seeds};new_set={tuple(map(int,x)) for x in new};failed_set={tuple(map(int,x)) for x in failed}
 union_set=seed_set|new_set|failed_set
 union=np.asarray(sorted(union_set),dtype=np.int64).reshape((-1,2))
 membership=np.asarray([[int(tuple(x) in seed_set),int(tuple(x) in new_set),int(tuple(x) in failed_set)] for x in union],dtype=np.uint8)
 np.save(OUT/"guard-pair-ids.npy",union,allow_pickle=False);np.save(OUT/"guard-pair-membership.npy",membership,allow_pickle=False)
 with np.load(CAND,allow_pickle=False) as z: points_hash=arrsha(z["points_world_m"]);faces_hash=arrsha(z["source_O_faces"])
 owner=loadmod(OWNER_PATH,"self_separation_026_412_union");pmm=p.astype(np.float64)*1000.0
 ptr=[0];node_ids=[];coeffs=[];lowers=[];row_pairs=[];compatible=[];incompatible=[];processed=0
 max_rhs=-float("inf");min_margin=float("inf");bad_rhs_count=0
 def snapshot(status,reason=None):
  indptr=np.asarray(ptr,np.int64);ni=np.concatenate(node_ids).astype(np.int32,copy=False) if node_ids else np.empty(0,np.int32)
  cf=np.concatenate(coeffs,axis=0).astype(np.float64,copy=False) if coeffs else np.empty((0,3),np.float64)
  np.savez_compressed(OUT/"guard-026-rows.npz",row_indptr=indptr,row_node_ids=ni,row_coefficients=cf,
      row_lower_mm=np.asarray(lowers,np.float64),row_pair_ids=np.asarray(row_pairs,np.int32).reshape((-1,2)))
  remaining=union[processed:].astype(np.int64,copy=False)
  np.save(OUT/"unprocessed-pair-ids.npy",remaining,allow_pickle=False)
  np.save(OUT/"compatible-pair-ids.npy",np.asarray(compatible,np.int64).reshape((-1,2)),allow_pickle=False)
  rep={
   "schema":"numi.human.prospective-self-guards-412.union-current-rows.v1","status":status,"stop_reason":reason,
   "geometry_admitted":False,"physical_steps":0,"candidate_path":str(CAND),"candidate_sha256":cand_sha,
   "candidate_points_array_sha256":points_hash,"candidate_faces_array_sha256":faces_hash,
   "candidate_exact_self_audit_path":str(CUR_AUDIT),"candidate_exact_self_audit_sha256":sha(CUR_AUDIT),
   "candidate_exact_self_pair_count":int(json.loads(CUR_AUDIT.read_text()).get("exact_PLC_surface_pairs",-1)),
   "candidate_surface_audit_path":str(CUR_SURFACE),"candidate_surface_audit_sha256":sha(CUR_SURFACE),
   "candidate_surface_audit_complete":bool(json.loads(CUR_SURFACE.read_text()).get("complete",False)),
   "candidate_surface_bank_count":int(json.loads(CUR_SURFACE.read_text()).get("bank_count",-1)),
   "candidate_surface_pair_count":sum(int(b["result"]["count"]) for b in json.loads(CUR_SURFACE.read_text())["banks"] if b.get("result") is not None),
   "source_pair_sets":{
    "historical_seed_ids":{"path":str(SEEDS),"sha256":sha(SEEDS),"count":int(len(seeds)),"candidate_origin":json.loads(SEEDS.with_name("report.json").read_text()).get("candidate_sha256")},
    "witness_one_ring_new_ids":{"path":str(NEW),"sha256":sha(NEW),"count":int(len(new)),"scope_report_path":str(SCOPE),"scope_report_sha256":sha(SCOPE)},
    "failed_410_pass001_ids":{"path":str(FAILED),"sha256":sha(FAILED),"count":int(len(failed)),"audit_summary_path":str(FAILED_SUMMARY),"audit_summary_sha256":sha(FAILED_SUMMARY)}
   },
   "union_pair_ids_path":str(OUT/"guard-pair-ids.npy"),"union_pair_ids_sha256":sha(OUT/"guard-pair-ids.npy"),
   "union_pair_count":int(len(union)),"membership_path":str(OUT/"guard-pair-membership.npy"),
   "membership_sha256":sha(OUT/"guard-pair-membership.npy"),"membership_columns":["historical_seed","new_witness_one_ring","failed_410_pass001"],
   "union_source_overlap_counts":{
    "historical_and_new":int(len(seed_set&new_set)),"historical_and_failed":int(len(seed_set&failed_set)),
    "new_and_failed":int(len(new_set&failed_set)),"all_three":int(len(seed_set&new_set&failed_set))},
   "row_owner_path":str(OWNER_PATH),"row_owner_sha256":owner_sha,"margin_fraction":MARGIN_FRACTION,
   "zero_feasibility_tolerance_mm":ZERO_TOL_MM,"max_seconds":MAX_SECONDS,"elapsed_seconds":time.monotonic()-t0,
   "processed_pair_count":int(processed),"compatible_pair_count":int(len(compatible)),
   "incompatible_pair_count":int(len(incompatible)),"unprocessed_pair_count":int(len(remaining)),
   "emitted_row_count":int(len(lowers)),"max_baseline_rhs_mm":float(max_rhs) if np.isfinite(max_rhs) else None,
   "positive_rhs_within_tolerance_count":int(bad_rhs_count),"minimum_preserved_margin_mm":float(min_margin) if np.isfinite(min_margin) else None,
   "all_union_pair_ids_classified":bool(processed==len(union)),"seed_ids_dropped":False,
   "compatible_pair_ids_path":str(OUT/"compatible-pair-ids.npy"),"compatible_pair_ids_sha256":sha(OUT/"compatible-pair-ids.npy"),
   "unprocessed_pair_ids_path":str(OUT/"unprocessed-pair-ids.npy"),"unprocessed_pair_ids_sha256":sha(OUT/"unprocessed-pair-ids.npy"),
   "row_payload_path":str(OUT/"guard-026-rows.npz"),"row_payload_sha256":sha(OUT/"guard-026-rows.npz"),
   "incompatible_pairs":incompatible,
   "scope_limit":"Current-410-coordinate 026 guards for the deduplicated union of retained historical IDs, witness-focused prospective IDs, and failed pass-001 pair IDs. This is a local linear construction guard set, not exact-clearance evidence."
  }
  writejson(OUT/"report.json",rep);return rep
 failure=None
 try:
  for idx,(a,b) in enumerate(union.tolist()):
   if time.monotonic()-t0>=MAX_SECONDS:failure="90_second_wall_limit";break
   processed=idx+1
   try:
    rows,recs=owner.build_pair_rows(pmm,f,np.asarray([[a,b]],np.int64),margin_fraction=MARGIN_FRACTION)
    if len(recs)!=1 or list(map(int,recs[0].get("faces",())))!=[int(a),int(b)]:raise ValueError("owner pair record mismatch")
    if not rows:raise ValueError("owner returned no rows")
    pending=[];localmax=-float("inf")
    for r in rows:
     nodes=np.asarray(r.get("nodes"));c=np.asarray(r.get("coefficients"),np.float64);lower=float(r.get("lower"))
     if nodes.ndim!=1 or not np.issubdtype(nodes.dtype,np.integer) or len(nodes)==0 or c.shape!=(len(nodes),3):raise ValueError("malformed row")
     if nodes.min()<0 or nodes.max()>=len(p) or not np.isfinite(c).all() or not np.isfinite(lower):raise ValueError("nonfinite/out-of-range row")
     if lower>ZERO_TOL_MM:raise ValueError(f"zero-step infeasible RHS {lower:.17g} mm")
     if lower>0:bad_rhs_count+=1
     pending.append((nodes.astype(np.int32,copy=True),c.copy(),lower));localmax=max(localmax,lower)
    for nodes,c,lower in pending:
     node_ids.append(nodes);coeffs.append(c);lowers.append(lower);row_pairs.append((int(a),int(b)));ptr.append(ptr[-1]+len(nodes))
    compatible.append((int(a),int(b)));max_rhs=max(max_rhs,localmax)
    for rec in recs:
     m=rec.get("preserved_margin_mm")
     if m is not None and np.isfinite(float(m)):min_margin=min(min_margin,float(m))
   except Exception as exc:
    incompatible.append({"pair":[int(a),int(b)],"source_membership":membership[idx].astype(int).tolist(),
                         "reason":f"{type(exc).__name__}: {exc}"})
   if processed%CHECKPOINT_EVERY==0:snapshot("IN_PROGRESS")
  rep=snapshot("COMPLETE" if processed==len(union) else "BOUNDED_PARTIAL",failure)
  print(json.dumps({k:rep[k] for k in ["status","elapsed_seconds","union_pair_count","processed_pair_count","compatible_pair_count",
      "incompatible_pair_count","unprocessed_pair_count","emitted_row_count","max_baseline_rhs_mm","minimum_preserved_margin_mm",
      "union_source_overlap_counts"]},sort_keys=True),flush=True)
 except BaseException as exc:
  snapshot("FAILED_PARTIAL",f"{type(exc).__name__}: {exc}")
  raise
if __name__=="__main__":main()
