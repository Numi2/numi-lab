from pathlib import Path
src=Path("/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-408/exact_global_incremental_flip_search_409.py")
dst=Path("/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-409-next/exact_global_incremental_flip_search_409_next.py")
s=src.read_text()
s=s.replace('FLIPCHECK=Path(__file__).with_name("incremental_exact_flip_check_408.py")',
'''FLIPCHECK=Path(__file__).with_name("incremental_exact_flip_check_408.py")
AREA_GATE=Path(__file__).with_name("f32_area_gate_409_next.py")''')
s=s.replace('def run(candidate_path, source_detail_rows, initial_seed_rows, output_dir, *,',
'''def area_gate_owner():
    spec=importlib.util.spec_from_file_location("f32_area_gate_409_next",AREA_GATE)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def run(candidate_path, source_detail_rows, initial_seed_rows, output_dir, *,''')
s=s.replace('''    exact=exact_owner(); checker=incremental_owner()
    candidate_sha=sha_file(cp); predicate_sha=sha_file(PRED)''','''    exact=exact_owner(); checker=incremental_owner(); area_gate=area_gate_owner()
    baseline_area=area_gate.f32_area_floor(current["points_world_m"],current["points_world_m"],
        current["area_reference_037_points_world_m_f64"],current["source_O_faces"],floor=0.1)
    if not baseline_area["pass"]:
        raise ValueError("input candidate fails the matched-037 emitted-F32 area floor")
    candidate_sha=sha_file(cp); predicate_sha=sha_file(PRED)''')
s=s.replace('''            trial_faces=proposal["source_O_faces"]
            topology_hash=hashlib.sha256(np.ascontiguousarray(trial_faces).view(np.uint8)).hexdigest()
            if topology_hash in seen_topologies:continue
            trial=checker.evaluate(state,proposal["points_world_m"],trial_faces)''','''            trial_faces=proposal["source_O_faces"]
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
            trial=checker.evaluate(state,proposal["points_world_m"],trial_faces)''')
s=s.replace('''                    "exact_incremental_global_audit":True,"no_new_pairs":ir["no_new_pairs"],''','''                    "exact_incremental_global_audit":True,"matched_037_f32_area_gate":area,
                    "no_new_pairs":ir["no_new_pairs"],''')
s=s.replace('''            "candidate_output_sha256":final_sha,"candidate_path":str(final),
            "builder_sha256":sha_file(Path(__file__)),''','''            "candidate_output_sha256":final_sha,"candidate_path":str(final),
            "matched_037_f32_area_floor":0.1,"matched_037_f32_area_gate_helper_sha256":sha_file(AREA_GATE),
            "baseline_matched_037_f32_area_gate":baseline_area,
            "builder_sha256":sha_file(Path(__file__)),''')
dst.parent.mkdir(parents=True,exist_ok=True)
dst.write_text(s)
print(dst)
