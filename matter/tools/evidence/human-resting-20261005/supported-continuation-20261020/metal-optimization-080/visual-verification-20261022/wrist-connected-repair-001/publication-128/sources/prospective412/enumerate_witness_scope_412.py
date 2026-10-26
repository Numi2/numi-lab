#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=ROOT/"D/local-two-parent-retriangulation-409-next"
CAND=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/candidate.npz"
FAIL_CAND=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-001/candidate.npz"
FAIL_PAIRS=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-001/audit-130/surface-self-pairs.npy"
FAIL_SUMMARY=FAIL_PAIRS.with_name("summary.json")
SEEDS=ROOT/"D/self-witness-refresh-410/run-001/merged-self-pair-seeds.npy"
SEED_REPORT=SEEDS.with_name("report.json")
CUR_AUDIT=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/audit-130/summary.json"
CUR_SURFACE=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/audit-159/report.json"
OUT=ROOT/"D/prospective-self-guards-412/run-005-witness-one-ring"
RADIUS_M=0.0005
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def loadmod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 assert s.loader is not None;s.loader.exec_module(m);return m
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);t0=time.monotonic()
 with np.load(CAND,allow_pickle=False) as z:p=z["points_world_m"];f=z["source_O_faces"]
 with np.load(FAIL_CAND,allow_pickle=False) as z:ff=z["source_O_faces"]
 if p.dtype!=np.float32 or p.shape!=(20889,3) or f.shape!=(41733,3):raise ValueError("unexpected current selected candidate arrays")
 if f.dtype!=ff.dtype or f.shape!=ff.shape or f.tobytes()!=ff.tobytes():raise ValueError("failed witness face row identities differ from current topology")
 oldpairs=np.load(FAIL_PAIRS,allow_pickle=False)
 if oldpairs.shape!=(23,2) or oldpairs.dtype.kind not in "iu":raise ValueError(f"expected 23 failed exact witness pairs, got {oldpairs.shape}/{oldpairs.dtype}")
 fs=json.loads(FAIL_SUMMARY.read_text())
 if fs.get("candidate_sha256")!=sha(FAIL_CAND) or int(fs.get("exact_PLC_surface_pairs",-1))!=23 or not fs.get("complete_surface_audit",False):raise ValueError("failed 410 witness receipt does not pin complete 23-pair source")
 seeds=np.load(SEEDS,allow_pickle=False);sr=json.loads(SEED_REPORT.read_text())
 if seeds.shape!=(18069,2) or sr.get("candidate_sha256")!=sha(ROOT/"D/postfit-backtrack-408/run-001/backtrack-373/pass-003/candidate.npz"):raise ValueError("historical seed source mismatch")
 if np.any(oldpairs<0) or np.any(oldpairs>=len(f)):raise ValueError("witness pair face out of range")
 witness_face_rows=np.unique(oldpairs.reshape(-1)).astype(np.int64)
 witness_nodes=np.unique(f[witness_face_rows].reshape(-1)).astype(np.int64)
 scope=np.flatnonzero(np.any(np.isin(f,witness_nodes),axis=1)).astype(np.int64)
 if len(scope)==0:raise ValueError("witness one-ring is empty")
 exact_path=BASE/"exact_global_incremental_flip_search_409_next.py";exact=loadmod(exact_path,"exact_global_incremental_flip_search_409_next")
 owner=exact.exact_owner();pred_sha=exact.sha_file(exact.PRED)
 tb=time.monotonic();records,_=owner.float32_surface_records(p,f,source_seam_groups=());index=owner._prepare_surface_aabb(records);build_s=time.monotonic()-tb
 helper_path=ROOT/"D/local-two-parent-retriangulation-409-next/prospective_self_guards_409_next.py";helper=loadmod(helper_path,"prospective_self_guards_409_next")
 seeds_norm=helper._pair_array(seeds,len(f));seedset={tuple(map(int,x)) for x in seeds_norm}
 near=set();edge_skips=0;query_candidate_hits=0
 te=time.monotonic()
 for i in scope.tolist():
  rec=records[int(i)];tri,_,_,row,eff_ids=rec
  xyz=p[f[int(i)]].astype(np.float64)
  lo=helper._outward_f32(xyz.min(axis=0)-RADIUS_M,-1);hi=helper._outward_f32(xyz.max(axis=0)+RADIUS_M,1)
  q=(tri,owner.float32_point_lattice_key(lo),owner.float32_point_lattice_key(hi),int(i),tuple(eff_ids))
  for _,other in owner._query_surface_aabb((q,),index):
   j=int(other[3])
   if j==i:continue
   query_candidate_hits+=1
   common=set(eff_ids).intersection(other[4])
   if len(common)>=2:
    edge_skips+=1;continue
   near.add((min(int(i),j),max(int(i),j)))
 near_s=time.monotonic()-te
 near_array=np.asarray(sorted(near),dtype=np.int64).reshape((-1,2))
 # Explicitly derive only pairs absent from the immutable historical seed set.
 new=np.asarray(sorted(near-seedset),dtype=np.int64).reshape((-1,2))
 union=np.asarray(sorted(near|seedset),dtype=np.int64).reshape((-1,2))
 np.save(OUT/"witness-one-ring-aabb-pair-ids.npy",near_array,allow_pickle=False)
 np.save(OUT/"new-pair-ids.npy",new,allow_pickle=False)
 np.save(OUT/"union-pair-ids.npy",union,allow_pickle=False)
 cur=json.loads(CUR_AUDIT.read_text());surf=json.loads(CUR_SURFACE.read_text())
 empty=[{"body":int(b["body"]),"stable_bone_id":int(b["stable_bone_id"])} for b in surf["banks"] if b.get("result") is None]
 spairs=sum(int(b["result"]["count"]) for b in surf["banks"] if b.get("result") is not None)
 report={
  "schema":"numi.human.prospective-self-guards-412.witness-one-ring.v1",
  "status":"WITNESS_ONE_RING_EXPANDED_AABB_PROPOSAL",
  "candidate_path":str(CAND),"candidate_sha256":sha(CAND),"candidate_points_array_sha256":arrsha(p),
  "candidate_faces_array_sha256":arrsha(f),"candidate_point_count":len(p),"candidate_face_count":len(f),
  "candidate_exact_self_audit_path":str(CUR_AUDIT),"candidate_exact_self_audit_sha256":sha(CUR_AUDIT),
  "candidate_exact_self_audit_complete":bool(cur.get("complete_surface_audit",False)),
  "candidate_exact_self_pairs":int(cur.get("exact_PLC_surface_pairs",-1)),
  "candidate_surface_audit_path":str(CUR_SURFACE),"candidate_surface_audit_sha256":sha(CUR_SURFACE),
  "candidate_surface_audit_complete":bool(surf.get("complete",False)),
  "candidate_surface_bank_count":int(surf.get("bank_count",-1)),
  "candidate_surface_expected_bank_count":int(surf.get("expected_bank_count",-1)),
  "candidate_surface_pair_count":int(spairs),"candidate_surface_empty_bank_count":len(empty),
  "failed_witness_candidate_path":str(FAIL_CAND),"failed_witness_candidate_sha256":sha(FAIL_CAND),
  "failed_witness_pairs_path":str(FAIL_PAIRS),"failed_witness_pairs_sha256":sha(FAIL_PAIRS),
  "failed_witness_pair_count":len(oldpairs),"failed_witness_source_audit_sha256":sha(FAIL_SUMMARY),
  "failed_witness_face_rows":witness_face_rows.tolist(),"witness_node_count":int(len(witness_nodes)),
  "query_scope":"All current candidate faces incident to any node of the 23 failed-pass witness triangles (one vertex ring); each scoped face queried against the complete current candidate face AABB index.",
  "query_face_rows_path":str(OUT/"query-face-rows.npy"),"query_face_count":int(len(scope)),
  "seed_pairs_path":str(SEEDS),"seed_pairs_sha256":sha(SEEDS),"seed_historical_source_candidate_sha256":sr.get("candidate_sha256"),
  "seed_pair_count":len(seeds_norm),"seed_pair_ids_sha256":helper._sha_pairs(seeds_norm),
  "radius_m":RADIUS_M,"predicate_owner_path":str(exact.PRED),"predicate_owner_sha256":pred_sha,
  "enumerator_helper_path":str(helper_path),"enumerator_helper_sha256":sha(helper_path),
  "record_index_build_seconds":build_s,"scoped_near_pair_enumeration_seconds":near_s,
  "elapsed_seconds":time.monotonic()-t0,"scoped_query_count":int(len(scope)),
  "raw_scoped_aabb_candidate_hits":int(query_candidate_hits),"shared_edge_pair_skips":int(edge_skips),
  "prospective_scoped_pair_count":int(len(near_array)),"new_pair_count":int(len(new)),
  "union_pair_count":int(len(union)),"historical_seed_pairs_preserved":True,
  "prospective_pair_ids_sha256":helper._sha_pairs(near_array),"new_pair_ids_sha256":helper._sha_pairs(new),
  "union_pair_ids_sha256":helper._sha_pairs(union),
  "prospective_pair_ids_file_sha256":sha(OUT/"witness-one-ring-aabb-pair-ids.npy"),
  "new_pair_ids_file_sha256":sha(OUT/"new-pair-ids.npy"),"union_pair_ids_file_sha256":sha(OUT/"union-pair-ids.npy"),
  "scope_limit":"This is a witness-focused prospective guard set, not complete self-clearance coverage. Existing seed IDs are retained. No exact intersections, 026 rows, or geometry admission are claimed."
 }
 np.save(OUT/"query-face-rows.npy",scope,allow_pickle=False)
 report["query_face_rows_sha256"]=sha(OUT/"query-face-rows.npy")
 (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:report[k] for k in ["status","candidate_sha256","failed_witness_pair_count","witness_node_count","query_face_count","raw_scoped_aabb_candidate_hits","prospective_scoped_pair_count","seed_pair_count","new_pair_count","union_pair_count","record_index_build_seconds","scoped_near_pair_enumeration_seconds","elapsed_seconds","candidate_surface_pair_count"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
