import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
ROOT = D / "D/D/area-blocker-zero-bounds-505"
HELPER = ROOT / "area_blocker_zero_bounds_505.py"
WRAPPER = ROOT / "run_matched_fit_505.py"
REF = D / "D/outward-plane-backtrack-461/run-001/backtrack-373/pass-011/candidate.npz"
CANDIDATE = D / "D/refreshed-connected-continuation-492/run-001/iteration-002/check/backtrack-373/pass-013/candidate.npz"
AUDIT = D / "D/refreshed-connected-continuation-492/run-001/iteration-002/check/backtrack-373/pass-013/audit-159/report.json"
TEMPLATE_CONFIG = D / "D/D/component-lp-decomposition-490/matched_inputs_476.json"
TARGET_ROOT = D / "D/refreshed-connected-continuation-492/run-001/iteration-003/refresh/connected-continuation-prep500/run-001"
FACE_TARGET = TARGET_ROOT / "surface-targets-335/targets.npz"
POINT_TARGET = TARGET_ROOT / "point-targets-390/nearest-union-exits.npz"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


h = load("area_blocker_zero_bounds_505_test", HELPER)
w = load("run_matched_fit_505_test", WRAPPER)


def make_config(tmp_path, *, candidate=CANDIDATE, audit=AUDIT):
    entry = load("matched_490_for_505_fixture", D / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py")
    config = json.loads(TEMPLATE_CONFIG.read_text())
    config["inputs"]["candidate"] = {"path": str(Path(candidate).resolve()), "sha256": sha(candidate)}
    config["inputs"]["audit159"] = {"path": str(Path(audit).resolve()), "sha256": sha(audit)}
    config["inputs"]["face_target_npz"] = {"path": str(FACE_TARGET.resolve()), "sha256": sha(FACE_TARGET)}
    config["inputs"]["point_target_npz"] = {"path": str(POINT_TARGET.resolve()), "sha256": sha(POINT_TARGET)}
    config_path = tmp_path / "matched-current-505.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    actual = {
        "candidate": Path(candidate), "audit159": Path(audit),
        "face_target_npz": FACE_TARGET, "point_target_npz": POINT_TARGET,
    }
    matched = entry.validate_matched_inputs(config_path, actual)
    return entry, config_path, actual, matched


def validate(tmp_path, *, candidate=CANDIDATE, audit=AUDIT):
    _entry, config_path, actual, matched = make_config(tmp_path, candidate=candidate, audit=audit)
    result = h.validate_blocker_candidate(
        actual["candidate"], actual["audit159"], matched_inputs=matched["inputs"],
        matched_config_path=config_path, matched_config_sha256=matched["config_sha256"],
    )
    return result, config_path, actual, matched


def test_current_aef_candidate_passes_config_derived_identity_checks(tmp_path):
    receipt, config_path, _actual, matched = validate(tmp_path)
    assert receipt["candidate_sha256"] == matched["inputs"]["candidate"]["sha256"]
    assert receipt["audit159_sha256"] == matched["inputs"]["audit159"]["sha256"]
    assert receipt["matched_config_sha256"] == sha(config_path)
    assert receipt["reference_candidate_sha256"] == h.REFERENCE_CANDIDATE_SHA256
    assert receipt["noncoordinate_identity_keys"] == 169
    assert receipt["audit159_complete"] and receipt["audit159_bank_count"] == 185
    assert receipt["blocker_rows_are_current_159_crossings"] is False
    assert [x["face_row"] for x in receipt["blocker_face_rows"]] == [32434, 11844, 40017]


def test_only_pinned_blocker_vertex_bounds_change(tmp_path):
    validation, *_ = validate(tmp_path)
    with np.load(CANDIDATE, allow_pickle=False) as z:
        faces = np.asarray(z["source_O_faces"])
    support = np.arange(20889, dtype=np.int64)
    lo = np.full(3 * len(support), -0.4)
    hi = np.full(3 * len(support), 0.4)
    out_lo, out_hi, receipt = h.apply_blocker_zero_bounds(lo, hi, support, faces, validation=validation)
    vertices = sorted({int(v) for row in h.BLOCKER_FACE_ROWS for v in faces[row]})
    expected = np.asarray([3*v+a for v in vertices for a in range(3)], dtype=np.int64)
    changed = np.flatnonzero((out_lo != lo) | (out_hi != hi))
    np.testing.assert_array_equal(changed, expected)
    assert np.all(out_lo[expected] == 0) and np.all(out_hi[expected] == 0)
    assert receipt["nonblocker_bounds_bitwise_unchanged"] is True
    assert receipt["runtime_anchor_or_force_semantics"] is False


def test_candidate_hash_must_come_from_matched_config(tmp_path):
    validation, config_path, actual, matched = validate(tmp_path)
    bad_inputs = dict(matched["inputs"])
    bad_inputs["candidate"] = dict(bad_inputs["candidate"], sha256="0" * 64)
    with pytest.raises(ValueError, match="candidate digest from 490 matched config"):
        h.validate_blocker_candidate(actual["candidate"], actual["audit159"], matched_inputs=bad_inputs,
            matched_config_path=config_path, matched_config_sha256=matched["config_sha256"])


def test_noncoordinate_geometry_identity_is_fail_closed(tmp_path):
    arrays = {}
    with np.load(CANDIDATE, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]).copy() for k in z.files}
    arrays["face_kind"][0] = arrays["face_kind"][0] + 1
    altered_candidate = tmp_path / "altered-noncoordinate.npz"
    np.savez_compressed(altered_candidate, **arrays)
    audit_data = json.loads(AUDIT.read_text())
    audit_data["candidate_sha256"] = sha(altered_candidate)
    altered_audit = tmp_path / "altered-audit.json"
    altered_audit.write_text(json.dumps(audit_data))
    _entry, config_path, actual, matched = make_config(tmp_path, candidate=altered_candidate, audit=altered_audit)
    with pytest.raises(ValueError, match="non-coordinate geometry/provenance identity changed"):
        h.validate_blocker_candidate(actual["candidate"], actual["audit159"], matched_inputs=matched["inputs"],
            matched_config_path=config_path, matched_config_sha256=matched["config_sha256"])


def test_current_crossing_face_is_rejected(tmp_path):
    audit_data = json.loads(AUDIT.read_text())
    audit_data["banks"][0]["result"]["triangle_pairs"] = [[32434, 0]]
    altered = tmp_path / "crossing-audit.json"
    altered.write_text(json.dumps(audit_data))
    _entry, config_path, actual, matched = make_config(tmp_path, audit=altered)
    with pytest.raises(ValueError, match="current 159 crossing face"):
        h.validate_blocker_candidate(actual["candidate"], actual["audit159"], matched_inputs=matched["inputs"],
            matched_config_path=config_path, matched_config_sha256=matched["config_sha256"])


def test_changed_blocker_ancestry_is_rejected(tmp_path, monkeypatch):
    _entry, config_path, actual, matched = make_config(tmp_path)
    monkeypatch.setattr(h, "EXPECTED_BLOCKER_ANCESTRY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source ancestry witness changed"):
        h.validate_blocker_candidate(actual["candidate"], actual["audit159"], matched_inputs=matched["inputs"],
            matched_config_path=config_path, matched_config_sha256=matched["config_sha256"])


def test_runtime_hook_preserves_nonblocker_bounds(tmp_path):
    validation, *_ = validate(tmp_path)
    with np.load(CANDIDATE, allow_pickle=False) as z:
        points_mm = np.asarray(z["points_world_m"], dtype=np.float64) * 1000
        faces = np.asarray(z["source_O_faces"], dtype=np.int64)
    support = np.arange(20889, dtype=np.int64)
    def base(points, source, rows, anchors, trust, cap):
        return np.full(3*len(rows), -0.25), np.full(3*len(rows), 0.25), {"base": True}
    owner = SimpleNamespace(source_safe_trust_bounds=base)
    state = h.install_owner_bounds(owner, validation=validation)
    lo, hi, report = owner.source_safe_trust_bounds(points_mm, points_mm.copy(), support, np.empty(0, dtype=np.int64), 1.0, 12.0)
    assert state["calls"] == 1 and len(state["receipts"]) == 1
    assert report["area_blocker_zero_bounds_505"]["nonblocker_bounds_bitwise_unchanged"] is True
    assert np.all(lo[3*100:3*100+3] == -0.25) and np.all(hi[3*100:3*100+3] == 0.25)
    assert len({v for row in h.BLOCKER_FACE_ROWS for v in faces[row]}) == 9


def test_wrapper_main_uses_actual_490_validation_and_nested_408_receipt(tmp_path, monkeypatch):
    entry, config_path, actual, matched = make_config(tmp_path)
    output_dir = tmp_path / "stubbed-delegate-output"
    argv = [
        w.FLAG, "--matched-inputs-json-490", str(config_path),
        "--audit159-490", str(actual["audit159"]),
        "--base-candidate", str(actual["candidate"]),
        "--face-target-npz", str(actual["face_target_npz"]),
        "--point-target-npz", str(actual["point_target_npz"]),
        "--output", str(output_dir),
    ]
    observed = {}
    fake_458 = SimpleNamespace()
    def fake_install(owner, objective, crossing):
        support = np.arange(20889, dtype=np.int64)
        lo, hi, receipt = owner.source_safe_trust_bounds(
            owner.points_mm, owner.points_mm.copy(), support, np.empty(0, dtype=np.int64), 1.0, 12.0)
        observed["source_box"] = receipt
        return lo, hi, receipt
    fake_458.install_objective = fake_install

    fake_485 = SimpleNamespace()
    fake_485.load = lambda name, path: fake_458 if name == "matched458_for_485" else None
    def fake_485_main():
        observed["fit_argv"] = list(sys.argv[1:])
        owner = SimpleNamespace()
        with np.load(actual["candidate"], allow_pickle=False) as z:
            owner.points_mm = np.asarray(z["points_world_m"], dtype=np.float64) * 1000.0
        owner.source_safe_trust_bounds = lambda p, s, rows, a, t, c: (
            np.full(3*len(rows), -0.25), np.full(3*len(rows), 0.25), {"base": True})
        fake_485.load("matched458_for_485", Path("/fake/458.py")).install_objective(owner, None, None)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(json.dumps({
            "constraints": {"source_safe_box": observed["source_box"]}
        }))
    fake_485.main = fake_485_main
    entry._load_module = lambda path: fake_485

    real_load = w.load_pinned
    def load_for_smoke(name, path):
        if Path(path).resolve() == w.ENTRY_490.resolve():
            return entry
        return real_load(name, path)
    monkeypatch.setattr(w, "load_pinned", load_for_smoke)
    monkeypatch.setattr(sys, "argv", [str(w.__file__), *argv])
    w.main()
    assert observed["fit_argv"] == [
        "--base-candidate", str(actual["candidate"]),
        "--face-target-npz", str(actual["face_target_npz"]),
        "--point-target-npz", str(actual["point_target_npz"]),
        "--output", str(output_dir),
    ]
    report = json.loads((output_dir / "report.json").read_text())
    assert report["area_blocker_zero_bounds_505"]["runtime_hook_receipt"]["nonblocker_bounds_bitwise_unchanged"]
    assert report["area_blocker_zero_bounds_505"]["validated_blocker_inputs"]["matched_config_sha256"] == matched["config_sha256"]
