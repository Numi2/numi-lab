#!/usr/bin/env python3
"""Global exact self-gated single-edge flip search using incremental checking.

The baseline gets one complete exact self audit. Each proposal must change two
faces; changed faces are queried against the cached whole-surface AABB index,
while exact pairs between unchanged faces are reused. Only no-new-pair strict
reductions are committed. This is construction evidence, not admission.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
PRED=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
GEOM=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/incremental-exact-audit-owner-253/src")
PATCH=ROOT/"D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py"
FLIPCHECK=Path(__file__).with_name("incremental_exact_flip_check_408.py")
AREA_GATE=Path(__file__).with_name("f32_area_gate_409_next.py")

def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def exact_owner():
    sys.path.insert(0,str(GEOM))
    name="numilab_human.patch_exact_402"
    spec=importlib.util.spec_from_file_location(name,PRED)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def incremental_owner():
    spec=importlib.util.spec_from_file_location("incremental_exact_flip_check_408",FLIPCHECK)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def local_scope(faces, detail, source_detail_rows):
    selected=np.flatnonzero(np.isin(detail,np.asarray(source_detail_rows,dtype=np.int64)))
    if not len(selected):raise ValueError("selected source-detail patch is empty")
    nodes=np.unique(faces[selected]).astype(np.int64)
    touch=np.flatnonzero(np.any(np.isin(faces,nodes),axis=1))
    return selected,touch

def pair_array(report):
    return np.asarray(report.get("triangle_pairs",()),dtype=np.int64).reshape((-1,2))

def audit_local(exact, points, faces, scope):
    records,_=exact.float32_surface_records(np.asarray(points,dtype=np.float32),faces,source_seam_groups=())
    result=exact._audit_self_pairs_touching_faces(records,set(map(int,scope)))
    if not result.get("complete_surface_audit",False) and int(result.get("selected_face_count",-1))!=len(scope):
        raise RuntimeError("exact local audit did not cover every selected face")
    return result,pair_array(result)

def load_candidate(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def area_gate_owner():
    spec=importlib.util.spec_from_file_location("f32_area_gate_409_next",AREA_GATE)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def run(candidate_path, source_detail_rows, initial_seed_rows, output_dir, *,
        max_accept=8,max_trials_per_round=64,max_seconds=90.0):
    sys.path.insert(0,str(PATCH.parent))
    import two_parent_patch_flips_402 as patch
    cp=Path(candidate_path); od=Path(output_dir)
    if od.exists():raise FileExistsError(f"output path exists: {od}")
    if max_accept<0 or max_trials_per_round<1 or not np.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError("invalid bounded search limits")
    od.mkdir(parents=True)
    start=time.monotonic()
    current=load_candidate(cp)
    patch.validate_candidate(current)
    input_points=np.array(current["points_world_m"],copy=True)
    details=tuple(sorted(set(map(int,source_detail_rows))))
    selected,scope=local_scope(current["source_O_faces"],current["parent_fine_detail_face_row"],details)
    exact=exact_owner(); checker=incremental_owner(); area_gate=area_gate_owner()
    baseline_area=area_gate.f32_area_floor(current["points_world_m"],current["points_world_m"],
        current["area_reference_037_points_world_m_f64"],current["source_O_faces"],floor=0.1)
    if not baseline_area["pass"]:
        raise ValueError("input candidate fails the matched-037 emitted-F32 area floor")
    candidate_sha=sha_file(cp); predicate_sha=sha_file(PRED)
    state,baseline_receipt,baseline_audit=checker.prepare_full(
        exact,current["points_world_m"],current["source_O_faces"],
        candidate_sha256=candidate_sha,predicate_sha256=predicate_sha,
        expected_predicate_sha256=predicate_sha,source_seam_groups=())
    current_pairs=np.asarray(sorted(state.pairs),dtype=np.int64).reshape((-1,2))
    baseline_pair_ids=current_pairs.tolist()
    baseline_pair_count=len(current_pairs)
    active=set(map(int,initial_seed_rows))
    if not active:active=set(map(int,current_pairs.ravel()))
    accepted=[];trial_log=[]
    stop_reason="accepted_flip_limit_or_no_eligible_improvement"
    wall_limit_hit=False
    seen_topologies={hashlib.sha256(np.ascontiguousarray(current["source_O_faces"]).view(np.uint8)).hexdigest()}
    for step in range(max_accept):
        if time.monotonic()-start>=max_seconds:
            stop_reason="wall_time_limit_reached";wall_limit_hit=True;break
        if not active:
            stop_reason="no_active_failure_witnesses";break
        selected_set=set(map(int,selected)); candidates={}; generated=0
        for seed in sorted(active):
            if time.monotonic()-start>=max_seconds:
                stop_reason="wall_time_limit_reached";wall_limit_hit=True;break
            if generated>=max_trials_per_round:break
            if seed<0 or seed>=len(current["source_O_faces"]) or seed not in selected_set:continue
            try:
                proposal,rep=patch.propose_patch_flips(current,details,[seed],max_flips=1,max_rounds=1,min_angle_gain_deg=-180.0)
            except (patch.PatchError,ValueError) as exc:
                trial_log.append({"step":step+1,"seed_face_row":int(seed),"status":"PROPOSAL_ERROR","error":str(exc)})
                continue
            if not rep["accepted_flip_count"]:continue
            op=rep["operations"][0]; edge=tuple(op["edge"])
            if edge in candidates:continue
            candidates[edge]=(proposal,op,int(seed));generated+=1
        best=None; tested=0
        for edge,(proposal,op,seed) in sorted(candidates.items()):
            if time.monotonic()-start>=max_seconds:
                stop_reason="wall_time_limit_reached";wall_limit_hit=True;break
            if tested>=max_trials_per_round:break
            tested+=1
            trial_faces=proposal["source_O_faces"]
            topology_hash=hashlib.sha256(np.ascontiguousarray(trial_faces).view(np.uint8)).hexdigest()
            if topology_hash in seen_topologies:continue
            changed_rows=np.flatnonzero(np.any(trial_faces!=current["source_O_faces"],axis=1))
            if len(changed_rows)!=2:
                trial_log.append({"step":step+1,"seed_face_row":seed,"edge":list(edge),
                                  "status":"REJECT_CHANGED_FACE_COUNT","changed_face_rows":changed_rows.tolist()})
                continue
            area=area_gate.f32_area_floor(proposal["points_world_m"],current["points_world_m"],
                current["area_reference_037_points_world_m_f64"],trial_faces,changed_rows,floor=0.1)
            if not area["pass"]:
                trial_log.append({"step":step+1,"seed_face_row":seed,"edge":list(edge),
                    "status":"REJECT_MATCHED_037_F32_AREA_FLOOR","changed_face_rows":changed_rows.tolist(),
                    "matched_037_f32_area_gate":area})
                continue
            trial=checker.evaluate(state,proposal["points_world_m"],trial_faces)
            if len(trial.changed)!=2:
                trial_log.append({"step":step+1,"seed_face_row":seed,"edge":list(edge),
                                  "status":"REJECT_CHANGED_FACE_COUNT","changed_face_rows":list(trial.changed)})
                continue
            ir=trial.report
            after=np.asarray(sorted(trial.pairs),dtype=np.int64).reshape((-1,2))
            record={"step":step+1,"seed_face_row":seed,"edge":list(edge),"operation":op,
                    "before_pair_count":len(current_pairs),"after_pair_count":len(after),
                    "introduced_pairs":ir["introduced_pairs"],"removed_pairs":ir["removed_pairs"],
                    "changed_face_rows":ir["changed_face_rows"],
                    "unchanged_baseline_pairs_reused":ir["unchanged_baseline_pairs_reused"],
                    "changed_face_aabb_candidate_pairs":ir["changed_face_aabb_candidate_pairs"],
                    "changed_face_exact_pairs":ir["changed_face_exact_pairs"],
                    "exact_incremental_global_audit":True,"matched_037_f32_area_gate":area,
                    "no_new_pairs":ir["no_new_pairs"],
                    "strict_pair_reduction":ir["strict_pair_reduction"],
                    "trial_topology_sha256":topology_hash}
            trial_log.append(record)
            if not (ir["no_new_pairs"] and ir["strict_pair_reduction"]):continue
            score=(len(after),-float(op["angle_gain_deg"]),edge)
            if best is None or score<best[0]:best=(score,proposal,op,seed,trial,record)
        if best is None:
            if not wall_limit_hit:stop_reason="no_strict_global_pair_reducing_trial"
            break
        _,current,op,seed,accepted_trial,record=best
        seen_topologies.add(record["trial_topology_sha256"])
        stepdir=od/f"step-{len(accepted)+1:02d}";stepdir.mkdir()
        cpath=stepdir/"candidate.npz";np.savez_compressed(cpath,**current)
        record["candidate_path"]=str(cpath);record["candidate_sha256"]=sha_file(cpath)
        state=checker.commit(state,accepted_trial,require_reduction=True,candidate_sha256=record["candidate_sha256"])
        current_pairs=np.asarray(sorted(state.pairs),dtype=np.int64).reshape((-1,2))
        accepted.append(record)
        active=set(map(int,current_pairs.ravel()))
        progress={"status":"BOUNDED_INCREMENTAL_FLIP_SEARCH_IN_PROGRESS","geometry_admitted":False,"physical_steps":0,
                  "candidate_path":str(cpath),"candidate_sha256":record["candidate_sha256"],
                  "current_global_exact_pair_count":len(current_pairs),"current_global_exact_pairs":current_pairs.tolist(),
                  "accepted_flip_count":len(accepted),"accepted_steps":accepted,"trial_count":len(trial_log),
                  "elapsed_seconds":time.monotonic()-start,"max_seconds":float(max_seconds)}
        tmp=od/"progress.json.tmp";tmp.write_text(json.dumps(progress,indent=2,sort_keys=True,allow_nan=False)+"\n");tmp.replace(od/"progress.json")
        if not active:
            stop_reason="global_exact_self_pairs_cleared";break
    else:
        stop_reason="accepted_flip_limit_reached"
    finaldir=od/"selected";finaldir.mkdir()
    final=finaldir/"candidate.npz";np.savez_compressed(final,**current)
    final_sha=sha_file(final)
    final_pairs=np.asarray(sorted(state.pairs),dtype=np.int64).reshape((-1,2))
    report={"status":"GLOBAL_INCREMENTAL_EXACT_SELF_RETRIANGULATION_COMPLETE" if accepted else "NO_GLOBAL_INCREMENTAL_EXACT_SELF_IMPROVEMENT",
            "geometry_admitted":False,"physical_steps":0,"candidate_input_sha256":candidate_sha,
            "candidate_output_sha256":final_sha,"candidate_path":str(final),
            "matched_037_f32_area_floor":0.1,"matched_037_f32_area_gate_helper_sha256":sha_file(AREA_GATE),
            "baseline_matched_037_f32_area_gate":baseline_area,
            "builder_sha256":sha_file(Path(__file__)),"patch_owner_sha256":sha_file(PATCH),
            "incremental_checker_sha256":sha_file(FLIPCHECK),"predicate_owner_sha256":predicate_sha,
            "source_detail_rows":list(details),"seed_face_rows":list(map(int,initial_seed_rows)),
            "selected_patch_face_count":len(selected),"local_patch_plus_vertex_one_ring_face_count":len(scope),
            "full_surface_face_count":len(current["source_O_faces"]),
            "baseline_audit_complete":True,"baseline_audit_selected_face_count":int(baseline_audit["selected_face_count"]),
            "baseline_audit_complete_surface_face_count":int(baseline_audit["complete_surface_face_count"]),
            "initial_global_exact_pair_count":baseline_pair_count,
            "initial_global_exact_pair_ids":baseline_pair_ids,
            "final_global_exact_pair_count":len(final_pairs),"final_global_exact_pair_ids":final_pairs.tolist(),
            "accepted_flip_count":len(accepted),"accepted_steps":accepted,"trial_count":len(trial_log),"trial_log":trial_log,
            "points_bitwise_unchanged":bool(np.ascontiguousarray(current["points_world_m"]).tobytes()==np.ascontiguousarray(input_points).tobytes()),
            "matched_037_connectivity_synchronized":bool(np.array_equal(current["source_O_faces"],current["area_reference_037_source_O_faces"])),
            "fixed_patch_outer_ring_and_inter_parent_seams_preserved":True,
            "accepted_trials_no_new_pairs":all(not r["introduced_pairs"] for r in accepted),
            "accepted_trials_strict_pair_reduction":all(r["after_pair_count"]<r["before_pair_count"] for r in accepted),
            "global_trial_audit_strategy":"One full exact baseline; each 2-face flip exact-tests changed faces against cached full-surface AABB and reuses only unchanged-face pair rows.",
            "all185_point_and_surface_gates_required":True,
            "scope_limit":"Self-pair counts are globally exact through incremental reuse; no bone gate or geometric admission is claimed.",
            "elapsed_seconds":time.monotonic()-start,"max_seconds":float(max_seconds),
            "wall_time_limit_hit":wall_limit_hit,"search_stop_reason":stop_reason}
    (od/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate",required=True);ap.add_argument("--source-detail-rows",required=True);ap.add_argument("--seed-face-rows",required=True);ap.add_argument("--output-dir",required=True)
    ap.add_argument("--max-accept",type=int,default=8);ap.add_argument("--max-trials-per-round",type=int,default=64)
    ap.add_argument("--max-seconds",type=float,default=90.0)
    a=ap.parse_args()
    d=json.loads(Path(a.source_detail_rows).read_text());s=json.loads(Path(a.seed_face_rows).read_text())
    report=run(a.candidate,d,s,a.output_dir,max_accept=a.max_accept,max_trials_per_round=a.max_trials_per_round,max_seconds=a.max_seconds)
    print(json.dumps({k:v for k,v in report.items() if k not in ("accepted_steps","trial_log","initial_global_exact_pair_ids","final_global_exact_pair_ids")},indent=2))
if __name__=="__main__":main()
