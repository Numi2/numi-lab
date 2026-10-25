from pathlib import Path
import importlib.util
import pytest
import numpy as np

RUNNER = Path(__file__).with_name("run_coherent_target_continuation_363.py")
spec = importlib.util.spec_from_file_location("runner363_tests", RUNNER)
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


def test_global_first_backtracking_audits_six_exact_steps():
    assert mod.GLOBAL_FIRST_MAXIMUM_EXACT_AUDITS == 6
    source = RUNNER.read_text()
    assert "'maximum_exact_audits': GLOBAL_FIRST_MAXIMUM_EXACT_AUDITS" in source
    assert "reduce_global_step_scales', GLOBAL_FIRST_MAXIMUM_EXACT_AUDITS, 16" in source
