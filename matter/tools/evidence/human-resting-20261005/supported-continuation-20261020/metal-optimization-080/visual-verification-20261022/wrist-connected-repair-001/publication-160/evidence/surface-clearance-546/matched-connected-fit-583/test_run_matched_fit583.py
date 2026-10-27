import json
from pathlib import Path
import numpy as np
import pytest

import run_matched_fit583 as m


def _fit_bundle(tmp_path, *, successful=True):
    base = tmp_path / "base.npz"
    emitted = tmp_path / "emitted.npz"
    raw = tmp_path / "raw.npz"
    points = np.zeros((3, 3), dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    np.savez(base, points_world_m=points, source_O_faces=faces,
             ancestry=np.asarray([2], dtype=np.int64),
             local_witness_cluster_222_delta_m=np.zeros((3, 3), dtype=np.float64))
    np.savez(emitted, points_world_m=points + 0.25, source_O_faces=faces,
             ancestry=np.asarray([2], dtype=np.int64),
             local_witness_cluster_222_delta_m=np.ones((3, 3), dtype=np.float64))
    np.savez(raw, points_world_m=points + 0.5, source_O_faces=faces,
             ancestry=np.asarray([2], dtype=np.int64),
             local_witness_cluster_222_delta_m=np.zeros((3, 3), dtype=np.float64))
    fit = tmp_path / "fit"
    fit.mkdir()
    capture_dir = fit / "raw-joint-capture-563"
    raw_dir = capture_dir / "raw-direction-001"
    raw_dir.mkdir(parents=True)
    raw_report = raw_dir / "report.json"
    raw_report.write_text(json.dumps({"candidate_sha256": m.sha(raw)}))
    capture_report = capture_dir / "report.json"
    capture_report.write_text(json.dumps({
        "base_candidate": str(base.resolve()),
        "base_candidate_sha256": m.sha(base),
        "solver_returns": [{
            "candidate_path": str(raw.resolve()), "candidate_sha256": m.sha(raw),
            "proposal_report_path": str(raw_report.resolve()),
            "proposal_report_sha256": m.sha(raw_report),
        }],
    }))
    fit_report = {
        "status": "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED" if successful else "LP_NO_PROPOSAL",
        "candidate_path": str(emitted.resolve()), "candidate_sha256": m.sha(emitted),
        "inputs": {"base_candidate_sha256": m.sha(base)},
        "solver": {"success": successful},
    }
    (fit / "report.json").write_text(json.dumps(fit_report))
    return base, fit, emitted, raw


def test_default_prefers_successful_emitted_endpoint(tmp_path):
    base, fit, emitted, raw = _fit_bundle(tmp_path)
    result = m.choose_target(base, fit, "emitted")
    assert result["selected_target"] == str(emitted)
    assert "successful emitted" in result["selection_reason"]


def test_raw_last_is_available_only_by_explicit_policy(tmp_path):
    base, fit, emitted, raw = _fit_bundle(tmp_path)
    result = m.choose_target(base, fit, "raw-last")
    assert result["selected_target"] == str(raw)
    assert "explicit raw-last" in result["selection_reason"]


def test_failed_fit_falls_back_to_captured_raw_direction(tmp_path):
    base, fit, emitted, raw = _fit_bundle(tmp_path, successful=False)
    result = m.choose_target(base, fit, "emitted")
    assert result["selected_target"] == str(raw)
    assert result["emitted_endpoint_available"] is False


def test_wrong_fit_baseline_is_rejected(tmp_path):
    base, fit, emitted, raw = _fit_bundle(tmp_path)
    report = json.loads((fit / "report.json").read_text())
    report["inputs"]["base_candidate_sha256"] = "0" * 64
    (fit / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="does not match requested state"):
        m.choose_target(base, fit, "emitted")


def test_emitted_fit_rejects_nonreceipt_payload_mutation(tmp_path):
    base, fit, emitted, raw = _fit_bundle(tmp_path)
    np.savez(emitted, points_world_m=np.ones((3, 3), np.float32),
             source_O_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
             ancestry=np.asarray([3], dtype=np.int64),
             local_witness_cluster_222_delta_m=np.ones((3, 3), dtype=np.float64))
    report = json.loads((fit / "report.json").read_text())
    report["candidate_sha256"] = m.sha(emitted)
    (fit / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="nonpoint payload ancestry"):
        m.choose_target(base, fit, "emitted")
