#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
CAND=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/candidate.npz"
SEEDS=ROOT/"D/self-witness-refresh-410/run-001/merged-self-pair-seeds.npy"
SCOPE=ROOT/"D/prospective-self-guards-412/run-005-witness-one-ring"
NEWPAIRS=SCOPE/"new-pair-ids.npy"
OUT=ROOT/"D/prospective-self-guards-412/run-006-new-026-rows"
OWNER_PATH=ROOT/"barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"
MAX_SECONDS=90.0
MARGIN_FRACTION=0.25
ZERO_TOL_MM=2e-8
CHECKPOINT_EVERY=1000
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
 tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
 tmp.replace(path)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 t0=time.monotonic()
 cand_sha=sha(CAND);seedsha=sha(SEEDS);newsha=sha(NEWPAIRS);owner_sha=sha(OWNER_PATH)
 with np.load(CAND,allow_pickle=False) as z:
  points=z["points_world_m"];faces=z["source_O_faces"]
 seeds=np.load(SEEDS,allow_pickle=False);pairs=np.load(NEWPAIRS,allow_pickle=False)
 if points.dtype!=np.float32 or points.shape!=(20889,3):raise ValueError("candidate Float32 point contract failed")
 if faces.shape!=(41733,3) or not np.issubdtype(faces.dtype,np.integer):raise ValueError("candidate face contract failed")
 if seeds.shape!=(18069,2) or not np.issubdtype(seeds.dtype,np.integer):raise ValueError("historical seed contract failed")
 if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.issubdtype(pairs.dtype,np.integer):raise ValueError("new pair contract failed")
 seedset={tuple(map(int,p)) for p in seeds}
 if any(tuple(map(int,p)) in seedset for p in pairs):raise ValueError("new row input overlaps preserved seed identities")
 if len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError("new pair list contains duplicates")
 owner=loadmod(OWNER_PATH,"self_separation_026_412")
 pmm=points.astype(np.float64)*1000.0
 # Flatten sparse rows incrementally; no geometry is changed by this audit.
 ptr=[0];node_ids=[];coeffs=[];lowers=[];row_pair_ids=[]
 compatible=[];incompatible=[];processed=0;max_rhs=-float("inf");min_margin=float("inf")
 def snapshot(status,reason=None):
  indptr=np.asarray(ptr,dtype=np.int64)
  ni=np.concatenate(node_ids).astype(np.int32,copy=False) if node_ids else np.empty(0,np.int32)
  cf=np.concatenate(coeffs,axis=0).astype(np.float64,copy=False) if coeffs else np.empty((0,3),np.float64)
  np.savez_compressed(OUT/"new-pair-026-rows.npz",
    row_indptr=indptr,row_node_ids=ni,row_coefficients=cf,row_lower_mm=np.asarray(lowers,np.float64),
    row_pair_ids=np.asarray(row_pair_ids,np.int32).reshape((-1,2)),
    compatible_pair_ids=np.asarray(compatible,np.int32).reshape((-1,2)))
  remaining=pairs[processed:].astype(np.int64,copy=False)
  np.save(OUT/"unprocessed-pair-ids.npy",remaining,allow_pickle=False)
  np.save(OUT/"compatible-pair-ids.npy",np.asarray(compatible,np.int64).reshape((-1,2)),allow_pickle=False)
  rep={
   "schema":"numi.human.prospective-self-guards-412.new-026-rows.v1","status":status,"stop_reason":reason,
   "geometry_admitted":False,"physical_steps":0,"candidate_path":str(CAND),"candidate_sha256":cand_sha,
   "candidate_points_array_sha256":arrsha(points),"candidate_faces_array_sha256":arrsha(faces),
   "historical_seed_pairs_path":str(SEEDS),"historical_seed_pairs_sha256":seedsha,"historical_seed_pair_count":int(len(seeds)),
   "new_pair_ids_path":str(NEWPAIRS),"new_pair_ids_sha256":newsha,"new_pair_count_input":int(len(pairs)),
   "row_owner_path":str(OWNER_PATH),"row_owner_sha256":owner_sha,"margin_fraction":MARGIN_FRACTION,
   "zero_feasibility_tolerance_mm":ZERO_TOL_MM,"max_seconds":MAX_SECONDS,"elapsed_seconds":time.monotonic()-t0,
   "processed_pair_count":int(processed),"compatible_pair_count":int(len(compatible)),
   "incompatible_pair_count":int(len(incompatible)),"unprocessed_pair_count":int(len(remaining)),
   "emitted_row_count":int(len(lowers)),"max_baseline_rhs_mm":(float(max_rhs) if np.isfinite(max_rhs) else None),
   "minimum_preserved_margin_mm":(float(min_margin) if np.isfinite(min_margin) else None),
   "all_input_pair_ids_classified":bool(processed==len(pairs)),
   "seed_ids_dropped":False,
   "compatible_pair_ids_path":str(OUT/"compatible-pair-ids.npy"),
   "compatible_pair_ids_file_sha256":sha(OUT/"compatible-pair-ids.npy"),
   "unprocessed_pair_ids_path":str(OUT/"unprocessed-pair-ids.npy"),
   "unprocessed_pair_ids_file_sha256":sha(OUT/"unprocessed-pair-ids.npy"),
   "row_payload_path":str(OUT/"new-pair-026-rows.npz"),"row_payload_sha256":sha(OUT/"new-pair-026-rows.npz"),
   "incompatible_pairs":incompatible,
   "scope_limit":"Only new witness-one-ring prospective pairs were linearized. Existing 18,069 historical seed identities remain preserved and outside this row payload; rows are local construction guards and do not certify exact self-clearance."
  }
  writejson(OUT/"report.json",rep)
  return rep
 failure=None
 try:
  for idx,(a,b) in enumerate(pairs.tolist()):
   if time.monotonic()-t0>=MAX_SECONDS:
    failure="90_second_wall_limit";break
   processed=idx+1
   try:
    rows,recs=owner.build_pair_rows(pmm,faces,np.asarray([[a,b]],dtype=np.int64),margin_fraction=MARGIN_FRACTION)
    if len(recs)!=1 or list(map(int,recs[0].get("faces",())))!=[int(a),int(b)]:
     raise ValueError("owner returned missing/mismatched pair record")
    if not rows:raise ValueError("owner returned no separation rows")
    pending=[]
    local_max=-float("inf");local_margin=float("inf")
    for r in rows:
     nodes=np.asarray(r.get("nodes"));c=np.asarray(r.get("coefficients"),dtype=np.float64);lower=float(r.get("lower"))
     if nodes.ndim!=1 or not np.issubdtype(nodes.dtype,np.integer) or len(nodes)==0 or c.shape!=(len(nodes),3):
      raise ValueError("malformed row node/coefficient shape")
     if nodes.min()<0 or nodes.max()>=len(points) or not np.isfinite(c).all() or not np.isfinite(lower):
      raise ValueError("nonfinite/out-of-range row data")
     if lower>ZERO_TOL_MM:raise ValueError(f"row not zero-feasible: {lower:.17g} mm")
     pending.append((nodes.astype(np.int32,copy=True),c.copy(),lower))
     local_max=max(local_max,lower)
    for nodes,c,lower in pending:
     node_ids.append(nodes);coeffs.append(c);lowers.append(lower);row_pair_ids.append((int(a),int(b)));ptr.append(ptr[-1]+len(nodes))
    compatible.append((int(a),int(b)));max_rhs=max(max_rhs,local_max)
    for rec in recs:
     m=rec.get("preserved_margin_mm")
     if m is not None and np.isfinite(float(m)):min_margin=min(min_margin,float(m))
   except Exception as exc:
    incompatible.append({"pair":[int(a),int(b)],"reason":f"{type(exc).__name__}: {exc}"})
   if processed%CHECKPOINT_EVERY==0:
    snapshot("IN_PROGRESS",None)
  status="COMPLETE" if processed==len(pairs) else "BOUNDED_PARTIAL"
  rep=snapshot(status,failure)
  print(json.dumps({k:rep[k] for k in ["status","elapsed_seconds","new_pair_count_input","processed_pair_count","compatible_pair_count","incompatible_pair_count","unprocessed_pair_count","emitted_row_count","max_baseline_rhs_mm","minimum_preserved_margin_mm"]},sort_keys=True),flush=True)
 except BaseException as exc:
  try:snapshot("FAILED_PARTIAL",f"{type(exc).__name__}: {exc}")
  finally:raise
if __name__=="__main__":main()
