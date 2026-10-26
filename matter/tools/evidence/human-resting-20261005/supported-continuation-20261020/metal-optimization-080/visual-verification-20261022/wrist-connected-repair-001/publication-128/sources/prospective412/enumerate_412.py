#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=ROOT/"D/local-two-parent-retriangulation-409-next"
CAND=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/candidate.npz"
HIST_CAND=ROOT/"D/postfit-backtrack-408/run-001/backtrack-373/pass-003/candidate.npz"
SEEDS=ROOT/"D/self-witness-refresh-410/run-001/merged-self-pair-seeds.npy"
SEED_REPORT=SEEDS.with_name("report.json")
BASELINE_AUDIT=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/audit-130/summary.json"
SURFACE_AUDIT=ROOT/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/audit-159/report.json"
OUT=ROOT/"D/prospective-self-guards-412/run-002-selected410"
RADIUS_M=0.0005
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    start=time.monotonic();cand_sha=sha(CAND);hist_sha=sha(HIST_CAND)
    with np.load(CAND,allow_pickle=False) as z:
        points=z["points_world_m"];faces=z["source_O_faces"]
    with np.load(HIST_CAND,allow_pickle=False) as z:
        hist_faces=z["source_O_faces"]
    if points.dtype!=np.float32 or points.shape!=(20889,3): raise ValueError(f"unexpected points {points.shape} {points.dtype}")
    if faces.shape!=(41733,3) or not np.issubdtype(faces.dtype,np.integer): raise ValueError(f"unexpected faces {faces.shape} {faces.dtype}")
    if faces.dtype!=hist_faces.dtype or faces.shape!=hist_faces.shape or faces.tobytes()!=hist_faces.tobytes():
        raise ValueError("historical pair face identities do not match current 410 topology bytes")
    seed_report=json.loads(SEED_REPORT.read_text());seeds=np.load(SEEDS,allow_pickle=False)
    if seed_report.get("candidate_sha256")!=hist_sha: raise ValueError("seed refresh historical candidate pin mismatch")
    if seeds.shape!=(18069,2) or not np.issubdtype(seeds.dtype,np.integer): raise ValueError(f"unexpected seeds {seeds.shape} {seeds.dtype}")
    exact_path=BASE/"exact_global_incremental_flip_search_409_next.py"
    exact=load_module(exact_path,"exact_global_incremental_flip_search_409_next")
    owner=exact.exact_owner();pred_sha=exact.sha_file(exact.PRED)
    t_build=time.monotonic()
    records,_=owner.float32_surface_records(points,faces,source_seam_groups=())
    index=owner._prepare_surface_aabb(records)
    index_elapsed=time.monotonic()-t_build
    state=SimpleNamespace(owner=owner,points=points,faces=faces,records=records,index=index,candidate_sha256=cand_sha,predicate_sha256=pred_sha)
    helper_path=ROOT/"D/local-two-parent-retriangulation-409-next/prospective_self_guards_409_next.py"
    helper=load_module(helper_path,"prospective_self_guards_409_next")
    t_enum=time.monotonic();result=helper.enumerate_prospective_pairs(state,RADIUS_M,seeds);enum_elapsed=time.monotonic()-t_enum
    prospective=result["prospective_pair_ids"];seedset={tuple(map(int,x)) for x in seeds}
    new_pairs=np.asarray([p for p in prospective.tolist() if tuple(map(int,p)) not in seedset],dtype=np.int64).reshape((-1,2))
    union=result["pair_ids"]
    np.save(OUT/"prospective-aabb-pair-ids.npy",prospective,allow_pickle=False)
    np.save(OUT/"new-pair-ids.npy",new_pairs,allow_pickle=False)
    np.save(OUT/"union-pair-ids.npy",union,allow_pickle=False)
    point_audit=json.loads(BASELINE_AUDIT.read_text());surface=json.loads(SURFACE_AUDIT.read_text())
    report=result["report"]
    report.update({
      "schema":"numi.human.prospective-self-guards-412.v2",
      "candidate_path":str(CAND),"candidate_sha256":cand_sha,"candidate_points_array_sha256":array_sha(points),
      "candidate_faces_array_sha256":array_sha(faces),"candidate_point_count":int(len(points)),"candidate_face_count":int(len(faces)),
      "historical_seed_source_candidate_path":str(HIST_CAND),"historical_seed_source_candidate_sha256":hist_sha,
      "historical_seed_source_faces_array_sha256":array_sha(hist_faces),
      "historical_seed_source_topology_byte_identical":True,
      "candidate_exact_self_audit_path":str(BASELINE_AUDIT),"candidate_exact_self_audit_sha256":sha(BASELINE_AUDIT),
      "candidate_exact_self_audit_complete":bool(point_audit.get("complete_surface_audit",False)),
      "candidate_exact_self_pair_count":int(point_audit.get("pair_count",-1)),
      "candidate_exact_surface_audit_path":str(SURFACE_AUDIT),"candidate_exact_surface_audit_sha256":sha(SURFACE_AUDIT),
      "candidate_exact_surface_audit_complete":bool(surface.get("complete",False)),
      "candidate_exact_surface_audit_bank_count":int(surface.get("bank_count",-1)),
      "candidate_exact_surface_audit_expected_bank_count":int(surface.get("expected_bank_count",-1)),
      "candidate_exact_surface_pair_count":int(sum(int(b["result"]["count"]) for b in surface["banks"])),
      "seed_pairs_path":str(SEEDS),"seed_pairs_sha256":sha(SEEDS),"seed_report_path":str(SEED_REPORT),
      "seed_report_sha256":sha(SEED_REPORT),"seed_report_candidate_sha256":seed_report.get("candidate_sha256"),
      "seed_pair_array_sha256":array_sha(seeds),"seed_pair_count":int(len(seeds)),
      "seed_interpretation":"Historical pair identities were generated from 408 and reused only because current 410 source_O_faces are byte-identical; pair rows must be rebuilt at current 410 coordinates.",
      "predicate_owner_path":str(exact.PRED),"predicate_owner_sha256":pred_sha,
      "enumerator_path":str(helper_path),"enumerator_sha256":sha(helper_path),
      "baseline_record_build_seconds":index_elapsed,"prospective_enumeration_seconds":enum_elapsed,
      "total_elapsed_seconds":time.monotonic()-start,"new_pair_count":int(len(new_pairs)),
      "new_pair_ids_path":str(OUT/"new-pair-ids.npy"),"new_pair_ids_sha256":sha(OUT/"new-pair-ids.npy"),
      "prospective_pair_ids_path":str(OUT/"prospective-aabb-pair-ids.npy"),
      "prospective_pair_ids_file_sha256":sha(OUT/"prospective-aabb-pair-ids.npy"),
      "union_pair_ids_path":str(OUT/"union-pair-ids.npy"),"union_pair_ids_file_sha256":sha(OUT/"union-pair-ids.npy"),
      "scope_limit":"Expanded-AABB proposal enumeration only. No 026 rows built; candidates are not exact collision findings or admission evidence."
    })
    (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    keys=["status","candidate_sha256","seed_pair_count","prospective_aabb_pair_count","prospective_shared_edge_pair_skips","new_pair_count","union_pair_count","baseline_record_build_seconds","prospective_enumeration_seconds","total_elapsed_seconds","candidate_exact_self_audit_complete","candidate_exact_self_pair_count","candidate_exact_surface_audit_complete","candidate_exact_surface_pair_count"]
    print(json.dumps({k:report[k] for k in keys},sort_keys=True),flush=True)
if __name__=="__main__": main()
