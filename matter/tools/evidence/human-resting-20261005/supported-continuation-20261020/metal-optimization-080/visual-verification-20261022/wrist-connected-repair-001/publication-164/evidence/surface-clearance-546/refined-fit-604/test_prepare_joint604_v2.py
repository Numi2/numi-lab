from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("prepare_joint604_v2.py")
spec = importlib.util.spec_from_file_location("prepare_joint604_v2_tested", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_flag_indices_reject_missing_and_duplicates():
    with pytest.raises(ValueError):
        m.exact_flag_index(["python", "runner.py", "--x", "1"], "--y")
    with pytest.raises(ValueError):
        m.exact_flag_index(["python", "runner.py", "--x", "1", "--x", "2"], "--x")
    with pytest.raises(ValueError):
        m.exact_flag_index(["python", "runner.py", "--x", "--y"], "--x")


def test_wrapper_replacement_requires_expected_old_entry(tmp_path):
    good = [str(m.PYTHON), str(m.OLD_564_ENTRY), "--output", str(tmp_path / "fit")]
    replaced = m.replace_entry(good, m.OLD_564_ENTRY, m.ENTRY_603)
    assert replaced[1] == str(m.ENTRY_603)
    assert good[1] == str(m.OLD_564_ENTRY)
    with pytest.raises(ValueError):
        m.replace_entry([str(m.PYTHON), str(m.ENTRY_603)], m.OLD_564_ENTRY, m.ENTRY_603)


def test_face_bundle_checks_target_candidate_and_rows(tmp_path):
    face = tmp_path / "face"
    face.mkdir()
    candidate_sha = "a" * 64
    npz = face / "targets.npz"
    report_path = face / "report.json"
    records = face / "target-records.jsonl"
    recovery_path = face / "recovery-verification.json"
    compatibility = tmp_path / "compat.json"
    np.savez(npz, face_rows=np.array([2, 5], dtype=np.int64),
             target_world_f32_m=np.zeros((2, 3), dtype=np.float32))
    records.write_text("{}\n{}\n")
    report = {"inputs": {"candidate_sha256": candidate_sha},
              "matched_159_complete": True, "banks": 185, "crossing_pairs": 2,
              "passing_targets": 2, "result_rows": 2}
    report_path.write_text(json.dumps(report))
    recovery = {"status": "PASS_CERTIFIED_DIRECTION_TARGETS_ONLY",
                "candidate_sha256": candidate_sha, "exact_endpoint_bank_count": 185,
                "target_count": 2, "artifact_pins": {}}
    recovery_path.write_text(json.dumps(recovery))
    ns, rs, rr = m.sha(npz), m.sha(records), m.sha(report_path)
    fit = {"original_npz_sha256": ns, "targets_npz_sha256": ns,
           "original_records_sha256": rs, "original_report_sha256": rr,
           "recovery_verification_sha256": m.sha(recovery_path),
           "joint_direction_rows": 1, "source_only_exception_rows": 1}
    compat = {"inputs": {"candidate_sha256": candidate_sha},
              "fit_direction_compatibility": fit,
              "source_current_direction_extension": {"all_rows": 2}}
    compatibility.write_text(json.dumps(compat))
    recovery["artifact_pins"] = {
        str(npz): ns, str(records): rs, str(report_path): rr,
    }
    recovery_path.write_text(json.dumps(recovery))
    fit["recovery_verification_sha256"] = m.sha(recovery_path)
    compatibility.write_text(json.dumps(compat))
    assert m.validate_face_bundle(candidate_sha, face, compatibility)["row_count"] == 2
    bad = json.loads(compatibility.read_text())
    bad["inputs"]["candidate_sha256"] = "b" * 64
    compatibility.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="candidate mismatch"):
        m.validate_face_bundle(candidate_sha, face, compatibility)


def test_fit_contract_rejects_duplicate_and_policy_change(tmp_path):
    candidate = tmp_path / "base.npz"
    surface = tmp_path / "159.json"
    matched = tmp_path / "matched.json"
    output = tmp_path / "fit"
    for p in (candidate, surface, matched):
        p.write_bytes(b"x")
    flags = [
        "--base-candidate", str(candidate), "--raw-trial", str(candidate),
        "--self-pairs", "seed.npy", "--area-reference-npz", str(candidate),
        "--point-target-npz", "p.npz", "--point-target-report", "p.json",
        "--face-target-npz", "f.npz", "--face-target-report", "f.json",
        "--point-depth-rows-npz", "d.npz", "--point-depth-rows-report", "d.json",
        "--fixed-boundary-npz", "g.npz", "--fixed-boundary-report", "g.json",
        "--matched-inputs-json-490", str(matched), "--audit159-490", str(surface),
        "--output", str(output), "--trust-mm", "4", "--source-cap-mm", "12",
        "--area-floor", "0.1", "--max-edge-gradient", "0.5",
        "--time-limit-s", "90", "--area-scp-iterations", "2",
        "--immediate-parent-basis", "--allow-soft-target-regression-485",
        "--allow-partial-timeout-zero-fallback-564",
    ]
    m.assert_fit_contract(flags, candidate, surface, matched, output, 4)
    with pytest.raises(ValueError, match="exactly one --base-candidate"):
        m.assert_fit_contract(flags + ["--base-candidate", str(candidate)],
                              candidate, surface, matched, output, 4)
    changed = list(flags)
    changed[changed.index("--max-edge-gradient") + 1] = "0.1"
    with pytest.raises(ValueError, match="policy changed"):
        m.assert_fit_contract(changed, candidate, surface, matched, output, 4)


def test_source_pin_rejects_hash_mismatch(tmp_path):
    p = tmp_path / "source.py"
    p.write_text("source")
    with pytest.raises(ValueError, match="pinned source hash changed"):
        m.pin_file(p, {}, "0" * 64)
