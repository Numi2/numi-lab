#!/usr/bin/env python3
"""402 fixed-physical-O gate on the current 398 geometry.

This is an executable preflight, not an all-bank solve. It consumes only the
pinned 398 mesh, the 395 unsolved initializer, and one exactly replayed witness
from the partial 399 chunks. A fixed-O conflict exits before clipping or LP.
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
OUT=E/"global-inner-p1-field-402-absolute-variable-integration/current-398-fixed-O-preflight-002"
BG=E/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz"
INIT=E/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/p1-interior-face-support-395/refined-background-unsolved-initializer.npz"
CACHE399=E/"global-inner-p1-field-399-398-refresh/run-002/global-field-cache.npz"
PARTIAL399=E/"global-inner-p1-field-399-398-refresh/run-002/partial-fixed-boundary-conflict-classification.json"
CHECKPOINT399=E/"global-inner-p1-field-399-398-refresh/run-002/partial-constraint-checkpoint.json"
PARTS399=E/"global-inner-p1-field-399-398-refresh/run-002/constraint-parts"
SOURCE399=E/"global-inner-p1-field-399-398-refresh/build_global_p1_399.py"
BANK=E/"registered-surrounding-tissue-native013-191/registered-tissue-banks.npz"
OBS=E/"robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py"
ABS=E/"global-inner-p1-field-402-absolute-variable-integration/absolute_field_402.py"
TARGET=0.341

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def require(ok,message):
 if not bool(ok): raise RuntimeError(message)
def main():
 if OUT.exists(): raise SystemExit(f"refuse overwrite {OUT}")
 OUT.mkdir(parents=True)
 started=time.monotonic()
 for p in (BG,INIT,CACHE399,PARTIAL399,CHECKPOINT399,SOURCE399,BANK,OBS,ABS):
  require(p.is_file(),f"missing pinned input {p}")
 # Preserve only the 398 mesh and initializer. No 293/296b/310/311 geometry or
 # constraint part is read by this producer.
 b=np.load(BG,allow_pickle=False); i=np.load(INIT,allow_pickle=False); c=np.load(CACHE399,allow_pickle=False)
 X=np.asarray(b["points_world_m"],np.float32); T=np.asarray(b["tetrahedra"],np.int64)
 phi0=np.asarray(i["phi_initializer_unsolved"],np.float64)
 require(X.shape==(49951,3) and T.shape==(211439,4),"unexpected 398 current geometry row domain")
 require(np.array_equal(X,np.asarray(i["points_world_m"],np.float32)),"395 initializer point rows do not match 398")
 require(np.array_equal(X,np.asarray(c["points_world_m"],np.float32)),"399 cache point rows do not match 398")
 require(np.array_equal(T,np.asarray(c["tetrahedra"],np.int64)),"399 cache tet rows do not match 398")
 require(np.array_equal(phi0,np.asarray(c["phi0"],np.float64)),"395 initializer values do not match the 399 cache")
 require(str(c["initializer_sha256"])==sha(INIT),"399 cache initializer file pin does not match 395")
 outer=np.asarray(b["physical_outer_node_ids"],np.int64)
 require(np.array_equal(np.sort(outer),np.sort(np.asarray(c["physical_outer_node_ids"],np.int64))),"physical O node IDs differ from 399 cache")
 require(phi0.shape==(len(X),) and np.isfinite(phi0).all() and np.all(phi0>=0) and np.all(phi0<=1),"invalid unsolved 395 initializer")
 require(np.all(phi0[outer]==0),"physical O initializer is not exactly zero")
 # 399 used the current 398 mesh and 395 initializer. Check its source entry
 # pins, then independently replay the selected witness with the pinned clipper.
 source_text=SOURCE399.read_text()
 require("local-quality-cavities-398/candidate-with-transfer.npz" in source_text,"399 producer source does not pin 398")
 require("p1-interior-face-support-395/refined-background-unsolved-initializer.npz" in source_text,"399 producer source does not pin 395")
 require("robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py" in source_text,"399 producer source does not pin robust clip helper 302")
 meta=json.loads(PARTIAL399.read_text());checkpoint=json.loads(CHECKPOINT399.read_text())
 require(meta["background_sha256"]==sha(BG),"399 partial witness geometry hash mismatch")
 require(meta["initializer_sha256"]==sha(INIT),"399 partial witness initializer hash mismatch")
 require(meta["bank_cache_sha256"]==sha(BANK),"399 partial witness bank-cache hash mismatch")
 require(meta["field_cache_sha256"]==sha(CACHE399),"399 partial witness field-cache hash mismatch")
 require(checkpoint["status"]=="bounded_partial_checkpoint" and int(checkpoint["new_parts"])>0,"399 chunks are not a partial checkpoint")
 # Verify hashes and domain IDs for every retained partial chunk; do not treat
 # these rows as complete all-bank coverage.
 hashes={q["file"]:q["sha256"] for q in meta["part_hashes"]}
 part_paths=sorted(PARTS399.glob("part-*.npz"))
 require(len(part_paths)==len(hashes)==int(checkpoint["new_parts"]),"399 partial chunk inventory mismatch")
 offsets=np.asarray(c["bank_face_offsets"],np.int64)
 require(offsets.shape==(690,) and offsets[0]==0,"399 logical bank-face offsets invalid")
 witness=meta["first_witness"]
 for path in part_paths:
  require(path.name in hashes and sha(path)==hashes[path.name],f"399 chunk hash mismatch: {path.name}")
  q=np.load(path,allow_pickle=False)
  ids=np.asarray(q["tet_ids"],np.int64);face=np.asarray(q["face_ids"],np.int64);bank=np.asarray(q["bank_ids"],np.int64)
  off=np.asarray(q["poly_offsets"],np.int64);B=np.asarray(q["polygon_barycentric"],np.float64)
  require(ids.shape==face.shape==bank.shape and off.shape==(len(ids)+1,) and off[0]==0 and off[-1]==len(B),f"bad 399 chunk layout: {path.name}")
  require(np.all((ids>=0)&(ids<len(T))) and np.all((bank>=0)&(bank<689)) and np.all((face>=offsets[bank])&(face<offsets[bank+1])),f"399 chunk IDs outside 398/689 domain: {path.name}")
  require(np.isfinite(B).all() and np.all(B>=0) and (not len(B) or np.max(np.abs(B.sum(axis=1)-1))<=2e-12),f"invalid 399 barycentrics: {path.name}")
 # Find the reported first witness in the exact saved part and ensure the
 # source-level triangle clipper reproduces that polygon on current 398.
 expected=(int(witness["tet_row"]),int(witness["flat_bank_face_row"]),int(witness["bank_id"]))
 found=None
 for path in part_paths:
  q=np.load(path,allow_pickle=False)
  rows=np.flatnonzero((q["tet_ids"]==expected[0])&(q["face_ids"]==expected[1])&(q["bank_ids"]==expected[2]))
  if len(rows):
   require(found is None,"duplicate 399 fixed-O witness records")
   r=int(rows[0]);lo=int(q["poly_offsets"][r]);hi=int(q["poly_offsets"][r+1])
   found=(path,q,r,np.asarray(q["polygon_barycentric"][lo:hi],np.float64))
 require(found is not None,"399 report witness absent from its pinned partial chunks")
 path,q,row,poly=found
 bary=np.asarray(witness["barycentric"],np.float64)
 require(np.any(np.all(poly==bary,axis=1)),"reported fixed-O witness point absent from saved polygon")
 bankz=np.load(BANK,allow_pickle=False)
 trirow=int(witness["bank_local_face_row"])
 triids=np.asarray(bankz["bank_0004_faces"],np.int64)[trirow]
 P=np.asarray(bankz["bank_0004_vertices"],np.float32)[triids].astype(np.float64)
 clip_ns={"__name__":"pinned_clip_replay","__file__":str(OBS)}
 exec(compile(OBS.read_text(),str(OBS),"exec"),clip_ns)
 replay,_=clip_ns["triangle_tet_barycentric_polygon"](X[T[expected[0]]].astype(np.float64),P)
 canon=lambda a:np.asarray(sorted(map(tuple,np.asarray(a,np.float64))),np.float64)
 require(replay.shape==poly.shape and np.allclose(canon(replay),canon(poly),rtol=0,atol=2e-13),"399 witness polygon did not replay with pinned helper 302")
 require(int(witness["bank_stable_id"])==5 and expected[1]==16346 and expected[2]==4,"unexpected witness identity")
 support=np.flatnonzero(bary>1e-14)
 tet_nodes=T[expected[0]]
 support_nodes=tet_nodes[support]
 require(set(map(int,support_nodes)).issubset(set(map(int,outer))),"reported witness is not supported only on physical O")
 point=bary@X[tet_nodes].astype(np.float64)
 require(np.max(np.abs(point-np.asarray(witness["world_m"],np.float64)))<=2e-8,"witness coordinates do not match current 398 tetrahedron")
 # The solver performs its capacity precheck before invoking HiGHS/linprog.
 sys.path.insert(0,str(ABS.parent))
 from absolute_field_402 import solve_absolute_obstacle_field
 field,solve_report=solve_absolute_obstacle_field(phi0,T,[expected[0]],bary[None,:],outer,np.zeros(len(outer)),np.ones(len(X)),TARGET)
 require(field is None and solve_report.get("status")=="fixed_support_conflict" and solve_report.get("minimum_capacity")==0.0,"expected fixed-O conflict did not fail closed")
 report={
  "schema":"numi.global-p1-absolute-field-402-current-398-preflight.v1",
  "status":"blocked_before_clipping_or_LP_by_verified_fixed_physical_O_witness",
  "full_scene_solve_run":False,
  "geometry":{"398_file_sha256":sha(BG),"399_cache_sha256":sha(CACHE399),"point_rows_bitwise_equal":True,"tetrahedra_bitwise_equal":True,"points":len(X),"tets":len(T),"physical_outer_nodes":len(outer)},
  "initializer":{"395_file_sha256":sha(INIT),"399_cached_initializer_sha256":str(c["initializer_sha256"]),"values_bitwise_equal_to_399_cache":True,"status":"unsolved initializer only; never a lower bound or objective",
   "closed_union_diagnostic_inside_nodes":int(np.count_nonzero(i["inside_closed_union"])),
   "closed_union_flags_used_as_constraints":False},
  "partial_399":{"producer_source_sha256_current":sha(SOURCE399),"bank_cache_sha256":sha(BANK),"bank_count":689,"partial_chunk_count":len(part_paths),"partial_polygon_count":int(sum(len(np.load(p,allow_pickle=False)["tet_ids"]) for p in part_paths)),"all_chunk_sha256_verified":True,"helper_302_sha256_current":sha(OBS),"witness_polygon_replayed_with_current_helper":True,"coverage":"partial only; not all-bank constraint coverage"},
  "fixed_O_witness":{"bank_id":expected[2],"bank_stable_id":int(witness["bank_stable_id"]),"bank_local_face_row":int(witness["bank_local_face_row"]),"flat_bank_face_row":expected[1],"tet_row":expected[0],"tet_nodes":tet_nodes.tolist(),"barycentric":bary.tolist(),"positive_support_nodes":support_nodes.tolist(),"world_m":point.tolist(),"target":TARGET,"capacity":0.0,"solver_precheck":solve_report},
  "reused_legacy_parts":False,
  "limitations":["399 chunks are partial and used only for a witness, not as complete coverage.","399 run output did not store the executed builder/helper hashes; current source and helper are pinned here and the selected witness was independently replayed.","No obstacle LP, level extraction, material-volume generation, or physical field admission was run.","A successful obstacle LP with only sampled surface constraints would still not certify a physically valid closed-solid skin/fat field.",
   "395 inside_closed_union flags are diagnostic only, not certified node constraints; interior pockets remain possible."],
  "elapsed_s":time.monotonic()-started,
  "input_sha256":{"398":sha(BG),"395":sha(INIT),"399_cache":sha(CACHE399),"399_partial_report":sha(PARTIAL399),"399_partial_checkpoint":sha(CHECKPOINT399),"399_builder_current":sha(SOURCE399),"registered_banks":sha(BANK),"robust_clip_302_current":sha(OBS),"absolute_solver_402":sha(ABS)}}
 (OUT/"preflight-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":report["status"],"output":str(OUT/"preflight-report.json"),"witness_tet":expected[0],"partial_chunks":len(part_paths),"elapsed_s":report["elapsed_s"]},sort_keys=True),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(main())
