from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

RUNNER = Path("/Users/n/Work/numi-human-connected-surface-069/D/coherent-support-continuation-391/run_coherent_target_continuation_391.py")
SPEC = importlib.util.spec_from_file_location("continuation_391_tested", RUNNER)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_pinned_365_390_335_and_382_seed_root():
    pins = MOD.verify_file_pins()
    assert pins["solver_365_sha256"] == "b06af3a336fbf23fddfdd75f45353d4b27d1354978472cc5dbb3a4b7c7527203"
    assert pins["coherent_point_builder_390_sha256"] == "15be65c3f3863998a2fb709e70e22a20402ab1580e84960aeb1a3f2301c70b00"
    assert pins["face_builder_335_sha256"] == "1c3468b6e6285afed77e8ae7ca463abb797f6a22d562ccf878734c39161947c9"
    assert MOD.SIXTH_SEED_ROOT in MOD.configured_seed_roots()
    assert MOD.SIXTH_SEED_ROOT.is_dir()


def test_390_depth_maximizing_targets_validate_against_actual_matched_artifact(tmp_path, monkeypatch):
    state_path = MOD.H / "coherent-support-continuation-362/uniform-scale-recovery/backtrack/selected-result.json"
    state = MOD.read(state_path)
    candidate = Path(state["candidate"])
    point_npz = MOD.POINT_DEPTH_ROWS_NPZ
    point_report = MOD.POINT_DEPTH_ROWS_REPORT
    source = MOD.H / "depth-maximizing-targets-390/run-001"
    source_report = source / "nearest-union-exits.json"
    source_npz = source / "nearest-union-exits.npz"
    assert MOD.sha(candidate) == "814b879cb8f3af988aeaa0540d183222d942e95f5597434b6328f714d35e14d3"
    assert MOD.sha(source_report) == "6b6ded7d9bdbc0c35e636b702432932ee3393a277b3f4a83d9a938eb5eeb3743"
    assert MOD.sha(source_npz) == "a327b7054cd0fb6d96993ac9d207f114a1db13d716723ec97facf866dc5490f0"

    def fake_run(command, log, *, slot=True, timeout=900):
        assert Path(command[1]).resolve() == MOD.POINT_BUILDER.resolve()
        assert command[command.index("--direction-mode") + 1] == "depth-maximizing"
        target_dir = Path(command[command.index("--output-dir") + 1])
        target_dir.mkdir(parents=True)
        shutil.copy2(source_report, target_dir / source_report.name)
        shutil.copy2(source_npz, target_dir / source_npz.name)
        return {"returncode": 0}

    monkeypatch.setattr(MOD, "run", fake_run)
    out = tmp_path / "round"
    out.mkdir()
    npz, report = MOD.run_point_targets(
        None, state, candidate, out, point_depth_rows_npz=point_npz,
        point_depth_rows_report=point_report, matched_375_cache_dir=None)
    assert MOD.read(report)["direction_mode"] == "depth-maximizing"
    assert npz.is_file()


def test_depth_compatible_report_is_rejected_by_391_guard(tmp_path, monkeypatch):
    state_path = MOD.H / "coherent-support-continuation-362/uniform-scale-recovery/backtrack/selected-result.json"
    state = MOD.read(state_path)
    candidate = Path(state["candidate"])
    source = MOD.H / "depth-maximizing-targets-390/run-001"
    source_report = source / "nearest-union-exits.json"
    source_npz = source / "nearest-union-exits.npz"

    def fake_run(command, log, *, slot=True, timeout=900):
        target_dir = Path(command[command.index("--output-dir") + 1])
        target_dir.mkdir(parents=True)
        bad = MOD.read(source_report)
        bad["direction_mode"] = "depth-compatible"
        (target_dir / "nearest-union-exits.json").write_text(json.dumps(bad))
        shutil.copy2(source_npz, target_dir / source_npz.name)
        return {"returncode": 0}

    monkeypatch.setattr(MOD, "run", fake_run)
    out = tmp_path / "round"
    out.mkdir()
    with pytest.raises(ValueError, match="390 point targets"):
        MOD.run_point_targets(
            None, state, candidate, out, point_depth_rows_npz=MOD.POINT_DEPTH_ROWS_NPZ,
            point_depth_rows_report=MOD.POINT_DEPTH_ROWS_REPORT, matched_375_cache_dir=None)


def test_candidate_specific_depth_rows_do_not_claim_322_correction_sidecar():
    state_path = MOD.H / "coherent-support-continuation-362/uniform-scale-recovery/backtrack/selected-result.json"
    candidate = Path(MOD.read(state_path)["candidate"])
    receipt = MOD.validate_point_depth_preflight(
        candidate, MOD.POINT_DEPTH_ROWS_NPZ, MOD.POINT_DEPTH_ROWS_REPORT, MOD.POINT_DEPTH_OWNER)
    correction = receipt["selection_policy_correction"]
    assert correction == {
        "applies": False,
        "path": None,
        "sha256": None,
        "reason": "candidate-specific 357 table; selection is documented in its matched report",
    }
    assert receipt["row_count"] == 7033 and receipt["inside_row_count"] == 486
    assert receipt["outside_row_count"] == 6547 and receipt["bank_count"] == 185


def test_384_eight_audit_uniform_budget_skips_duplicate_local_replay(tmp_path):
    backtrack = tmp_path / "global" / "backtrack"
    backtrack.mkdir(parents=True)
    trial = tmp_path / "trial.npz"
    trial.write_bytes(b"same raw proposal")
    cfg = {
        "uniform_failure_backtracking": True,
        "maximum_exact_audits": 8,
        "trial": str(trial),
    }
    (backtrack.parent / "backtrack-config.json").write_text(json.dumps(cfg))
    for i in range(1, 9):
        pass_dir = backtrack / f"pass-{i:03d}"
        audit = pass_dir / "audit-130"
        audit.mkdir(parents=True)
        candidate = pass_dir / "candidate.npz"
        np.savez(candidate, points=np.array([i], dtype=np.int64))
        digest = MOD.sha(candidate)
        (pass_dir / "result.json").write_text(json.dumps({
            "candidate_sha256": digest, "emitted_points_sha256": digest,
            "native_admission": False, "accepted_physical_steps": 0,
            "exact_self_pairs": 0,
        }))
        (audit / "summary.json").write_text(json.dumps({
            "candidate_sha256": digest, "complete_surface_audit": True,
        }))
    exhausted = backtrack / "pass-009"
    exhausted.mkdir()
    (exhausted / "result.json").write_text(json.dumps({"status": "EXACT_AUDIT_BUDGET_REACHED"}))
    receipt = MOD.exhausted_uniform_global_budget_receipt(backtrack, trial, 8)
    assert receipt["status"] == "SKIP_IDENTICAL_UNIFORM_LOCAL_REPLAY_AFTER_GLOBAL_BUDGET"
    assert receipt["completed_exact_audits"] == 8
    assert receipt["local_fallback_launched"] is False
    assert receipt["learned_global_witnesses_retained"] is True
