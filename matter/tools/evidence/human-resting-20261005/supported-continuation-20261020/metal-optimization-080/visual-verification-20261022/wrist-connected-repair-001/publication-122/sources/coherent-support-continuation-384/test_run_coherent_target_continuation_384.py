from pathlib import Path
import importlib.util
import pytest
import numpy as np

RUNNER = Path(__file__).with_name("run_coherent_target_continuation_384.py")
spec = importlib.util.spec_from_file_location("runner384_tests", RUNNER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SELECTED_362 = mod.H / "coherent-support-continuation-362/uniform-scale-recovery/backtrack/selected-result.json"
SELECTED_362_SHA256 = "7397899413a48679b6d118938814b1419ec6ac261602525a0e0c01d85dca4582"
SELECTED_362_CANDIDATE_SHA256 = "814b879cb8f3af988aeaa0540d183222d942e95f5597434b6328f714d35e14d3"
SELECTED_362_ROWS_NPZ = mod.POINT_DEPTH_ROWS_NPZ.parent.parent / "candidate-362-814b879-near-001/point-depth-rows.npz"
SELECTED_362_ROWS_REPORT = mod.POINT_DEPTH_ROWS_REPORT.parent.parent / "candidate-362-814b879-near-001/report.json"
SELECTED_362_ROWS_NPZ_SHA256 = "d0c8712a8bb33adb46bd4c90c84010e78ec123279be23d4c9ac7192ed8c57577"
SELECTED_362_ROWS_REPORT_SHA256 = "75b648488567bc4e59cfea813532c4269ce00c14ad9aec06ebfde6b3fadc85aa"


def test_default_357_7042_preflight_is_exact_candidate_and_185_complete():
    state = mod.read(mod.DEFAULT_STATE)
    candidate = Path(state["candidate"])
    result = mod.validate_point_depth_preflight(
        candidate, mod.POINT_DEPTH_ROWS_NPZ, mod.POINT_DEPTH_ROWS_REPORT, mod.POINT_DEPTH_OWNER)
    assert result["candidate_sha256"] == mod.PINS["283_candidate_sha256"]
    assert result["row_count"] == result["schema_row_count"] == 7042
    assert result["unique_node_count"] == 6952
    assert result["inside_row_count"] == 489
    assert result["outside_row_count"] == 6553
    assert result["bank_count"] == 185
    assert result["complete_bank_accounting_verified"] is True
    assert result["baseline_feasibility"]["candidate_sha256"] == result["candidate_sha256"]
    assert result["selection_policy_correction"]["applies"] is True
    assert result["selection_policy_correction"]["sha256"] == mod.POINT_DEPTH_POLICY_CORRECTION_SHA256


def test_additional_exact_self_audit_roots_are_in_configured_inventory():
    roots = mod.configured_seed_roots()
    names = {p.name for p in roots}
    assert {"coherent-support-continuation-344", "coherent-support-continuation-348",
            "coherent-support-continuation-352", "coherent-support-continuation-362"} <= names
    assert len(roots) == len(set(map(lambda p: p.resolve(), roots)))


def test_sha_pinned_selected_362_state_and_matching_fresh_rows():
    assert mod.sha(SELECTED_362) == SELECTED_362_SHA256
    assert mod.sha(SELECTED_362_ROWS_NPZ) == SELECTED_362_ROWS_NPZ_SHA256
    assert mod.sha(SELECTED_362_ROWS_REPORT) == SELECTED_362_ROWS_REPORT_SHA256
    state = mod.read(SELECTED_362)
    point_sha = mod.sha(Path(state["point_summary"]))
    surface_sha = mod.sha(Path(state["surface_report"]))
    resumed, candidate, faces, anchors = mod.validate_state(
        SELECTED_362, resume_state_sha256=SELECTED_362_SHA256,
        resume_point_summary_sha256=point_sha, resume_surface_report_sha256=surface_sha)
    assert resumed["_resume_verified_hashes"]["state_sha256"] == SELECTED_362_SHA256
    assert resumed["_resume_verified_hashes"]["candidate_sha256"] == SELECTED_362_CANDIDATE_SHA256
    assert len(faces) == 41733 and len(anchors) == 237
    with np.load(candidate, allow_pickle=False) as z:
        assert np.asarray(z["points_world_m"]).shape == (20889, 3)
    depth = mod.validate_point_depth_preflight(
        candidate, SELECTED_362_ROWS_NPZ, SELECTED_362_ROWS_REPORT, mod.POINT_DEPTH_OWNER)
    assert depth["candidate_sha256"] == SELECTED_362_CANDIDATE_SHA256
    assert depth["row_count"] == depth["schema_row_count"] == 7033
    assert depth["inside_row_count"] == 486
    assert depth["outside_row_count"] == 6547
    assert depth["bank_count"] == 185 and depth["complete_bank_accounting_verified"] is True
    assert depth["baseline_feasibility"]["candidate_sha256"] == SELECTED_362_CANDIDATE_SHA256
    assert depth["selection_policy_correction"]["applies"] is False
    assert depth["selection_policy_correction"]["path"] is None


def test_default_322_rows_are_rejected_for_selected_362_candidate():
    state = mod.read(SELECTED_362)
    with pytest.raises(ValueError):
        mod.validate_point_depth_preflight(
            Path(state["candidate"]), mod.POINT_DEPTH_ROWS_NPZ,
            mod.POINT_DEPTH_ROWS_REPORT, mod.POINT_DEPTH_OWNER)


def test_global_first_backtracking_audits_eight_exact_steps():
    assert mod.GLOBAL_FIRST_MAXIMUM_EXACT_AUDITS == 8
    source = RUNNER.read_text()
    assert "'maximum_exact_audits': GLOBAL_FIRST_MAXIMUM_EXACT_AUDITS" in source
    assert "reduce_global_step_scales', GLOBAL_FIRST_MAXIMUM_EXACT_AUDITS, 16" in source


MATCHED_375 = mod.MATCHED_375_CACHE_DIR
MATCHED_375_STATE = mod.MATCHED_375_STATE
MATCHED_375_ROWS_NPZ = mod.POINT_DEPTH_ROWS_NPZ.parent.parent / "candidate-362-814b879-near-001/point-depth-rows.npz"
MATCHED_375_ROWS_REPORT = mod.POINT_DEPTH_ROWS_REPORT.parent.parent / "candidate-362-814b879-near-001/report.json"


def test_matched_375_cache_has_exact_candidate_and_complete_point_face_receipts():
    state = mod.read(MATCHED_375_STATE)
    receipt = mod.validate_375_target_cache(
        Path(state["candidate"]), Path(state["point_summary"]), Path(state["surface_report"]),
        MATCHED_375_ROWS_NPZ, MATCHED_375_ROWS_REPORT)
    assert receipt["point"]["targets"] == 486
    assert receipt["face"]["targets"] == 3588
    assert receipt["point"]["matched_complete"] is True
    assert receipt["face"]["matched_complete"] is True
    assert receipt["classification"].endswith("no clearance admission")


def test_matched_375_target_adapters_copy_validated_outputs_without_running_builders(tmp_path):
    state = mod.read(MATCHED_375_STATE)
    candidate = Path(state["candidate"])
    point_summary = Path(state["point_summary"])
    surface = Path(state["surface_report"])
    face_out = tmp_path / "face"; face_out.mkdir()
    point_out = tmp_path / "point"; point_out.mkdir()
    face_npz, face_report = mod.run_face_targets(
        candidate, surface, face_out, matched_375_cache_dir=MATCHED_375)
    point_npz, point_report = mod.run_point_targets(
        None, state, candidate, point_out, None,
        MATCHED_375_ROWS_NPZ, MATCHED_375_ROWS_REPORT, MATCHED_375)
    assert mod.sha(face_npz) == mod.MATCHED_375_PINS["face_npz_sha256"]
    assert mod.sha(face_report) == mod.MATCHED_375_PINS["face_report_sha256"]
    assert mod.sha(face_npz.parent / "target-records.jsonl") == mod.MATCHED_375_PINS["face_rows_sha256"]
    assert mod.sha(point_npz) == mod.MATCHED_375_PINS["point_npz_sha256"]
    assert mod.sha(point_report) == mod.MATCHED_375_PINS["point_report_sha256"]


def test_matched_375_cache_rejects_wrong_candidate_and_wrong_depth_rows():
    state = mod.read(MATCHED_375_STATE)
    with pytest.raises(ValueError):
        mod.validate_375_target_cache(
            Path(mod.read(mod.DEFAULT_STATE)["candidate"]), Path(state["point_summary"]),
            Path(state["surface_report"]), MATCHED_375_ROWS_NPZ, MATCHED_375_ROWS_REPORT)
    with pytest.raises(ValueError):
        mod.validate_375_point_cache(
            Path(state["candidate"]), Path(state["point_summary"]),
            mod.POINT_DEPTH_ROWS_NPZ, mod.POINT_DEPTH_ROWS_REPORT)


def test_382_global_and_local_fallback_replay_same_uniform_candidates():
    base = mod.H / "coherent-support-continuation-382/round-001/attempt-001"
    global_root = base / "global-275/backtrack"
    local_root = base / "local-339-fallback/backtrack"
    global_hashes = []
    local_hashes = []
    for index in range(1, 7):
        g = mod.read(global_root / f"pass-{index:03d}/result.json")
        l = mod.read(local_root / f"pass-{index:03d}/result.json")
        assert g["candidate_sha256"] == l["candidate_sha256"]
        assert g["emitted_points_sha256"] == l["emitted_points_sha256"]
        global_hashes.append(g["candidate_sha256"])
        local_hashes.append(l["candidate_sha256"])
    assert global_hashes == local_hashes
    assert len(set(global_hashes)) == 6


def test_skip_receipt_requires_all_eight_exact_global_audits_and_terminal_budget(tmp_path):
    trial = tmp_path / "trial.npz"
    trial.write_bytes(b"trial bytes")
    global_root = tmp_path / "global" / "backtrack"
    global_root.mkdir(parents=True)
    cfg = {
        "trial": str(trial), "maximum_exact_audits": 8,
        "uniform_failure_backtracking": True,
        "native_admission": False, "accepted_physical_steps": 0,
    }
    mod.write(global_root.parent / "backtrack-config.json", cfg)
    for index in range(1, 9):
        pass_dir = global_root / f"pass-{index:03d}"
        (pass_dir / "audit-130").mkdir(parents=True)
        candidate = pass_dir / "candidate.npz"
        candidate.write_bytes(f"candidate-{index}".encode())
        candidate_sha = mod.sha(candidate)
        mod.write(pass_dir / "result.json", {
            "candidate_sha256": candidate_sha,
            "emitted_points_sha256": f"emitted-{index}",
            "native_admission": False, "accepted_physical_steps": 0,
            "exact_self_pairs": 1,
        })
        mod.write(pass_dir / "audit-130/summary.json", {
            "candidate_sha256": candidate_sha, "complete_surface_audit": True,
        })
    exhausted = global_root / "pass-009"
    exhausted.mkdir()
    mod.write(exhausted / "result.json", {"status": "EXACT_AUDIT_BUDGET_REACHED"})
    receipt = mod.exhausted_uniform_global_budget_receipt(global_root, trial, 8)
    assert receipt["completed_exact_audits"] == 8
    assert receipt["local_fallback_launched"] is False
    assert receipt["learned_global_witnesses_retained"] is True


def test_skip_receipt_fails_closed_on_partial_mismatched_or_selected_global_run(tmp_path):
    trial = tmp_path / "trial.npz"
    trial.write_bytes(b"trial bytes")
    root = tmp_path / "global" / "backtrack"
    root.mkdir(parents=True)
    mod.write(root.parent / "backtrack-config.json", {
        "trial": str(trial), "maximum_exact_audits": 8,
        "uniform_failure_backtracking": True,
    })
    assert mod.exhausted_uniform_global_budget_receipt(root, trial, 8) is None
    assert mod.exhausted_uniform_global_budget_receipt(root, trial, 6) is None
    assert mod.exhausted_uniform_global_budget_receipt(root, trial, 8) is None
    (root / "selected-result.json").write_text("{}")
    assert mod.exhausted_uniform_global_budget_receipt(root, trial, 8) is None
