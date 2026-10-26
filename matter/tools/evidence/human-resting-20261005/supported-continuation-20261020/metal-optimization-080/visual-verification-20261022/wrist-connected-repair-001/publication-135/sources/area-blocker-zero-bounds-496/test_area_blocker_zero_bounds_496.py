import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
ROOT = D / "D/D/area-blocker-zero-bounds-496"
HELPER = ROOT / "area_blocker_zero_bounds_496.py"
WRAPPER = ROOT / "run_matched_fit_496.py"
CANDIDATE = D / "D/outward-plane-backtrack-461/run-001/backtrack-373/pass-011/candidate.npz"
AUDIT = D / "D/outward-plane-backtrack-461/run-001/backtrack-373/pass-011/audit-159/report.json"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

h = load("area_blocker_zero_bounds_496_test", HELPER)
w = load("run_matched_fit_496_test", WRAPPER)


def test_pinned_candidate_audit_and_ancestry_validate():
    receipt = h.validate_blocker_candidate(CANDIDATE, AUDIT)
    assert receipt["blocker_rows_are_current_159_crossings"] is False
    assert receipt["audit159_complete"] and receipt["audit159_bank_count"] == 185
    assert [x["face_row"] for x in receipt["blocker_face_rows"]] == [32434, 11844, 40017]
    assert receipt["blocker_source_ancestry_sha256"] == h.EXPECTED_BLOCKER_ANCESTRY_SHA256


def test_only_pinned_blocker_vertex_bounds_change():
    validation = h.validate_blocker_candidate(CANDIDATE, AUDIT)
    with np.load(CANDIDATE, allow_pickle=False) as z:
        faces = np.asarray(z["source_O_faces"])
    support = np.arange(20889, dtype=np.int64)
    lo = np.full(3 * len(support), -0.4, dtype=np.float64)
    hi = np.full(3 * len(support), 0.4, dtype=np.float64)
    out_lo, out_hi, receipt = h.apply_blocker_zero_bounds(lo, hi, support, faces, validation=validation)
    vertices = sorted({int(v) for row in h.BLOCKER_FACE_ROWS for v in faces[row]})
    expected = np.asarray([3 * v + a for v in vertices for a in range(3)], dtype=np.int64)
    changed = np.flatnonzero((out_lo != lo) | (out_hi != hi))
    np.testing.assert_array_equal(changed, expected)
    assert np.all(out_lo[expected] == 0.0) and np.all(out_hi[expected] == 0.0)
    assert np.array_equal(out_lo[np.setdiff1d(np.arange(len(lo)), expected)], lo[np.setdiff1d(np.arange(len(lo)), expected)])
    assert receipt["nonblocker_bounds_bitwise_unchanged"] is True
    assert receipt["runtime_anchor_or_force_semantics"] is False


def test_wrong_candidate_hash_is_rejected(tmp_path):
    bad = tmp_path / "not-the-pinned-candidate.npz"
    bad.write_bytes(b"not the pinned candidate")
    with pytest.raises(ValueError, match="pinned to a different candidate"):
        h.validate_blocker_candidate(bad, AUDIT)


def test_current_crossing_face_is_rejected(tmp_path, monkeypatch):
    audit = json.loads(AUDIT.read_text())
    audit["banks"][0]["result"] = {"triangle_pairs": [[32434, 0]]}
    bad = tmp_path / "crossing-audit.json"
    bad.write_text(json.dumps(audit))
    monkeypatch.setattr(h, "EXPECTED_AUDIT159_SHA256", h.file_sha256(bad))
    with pytest.raises(ValueError, match="current 159 crossing face"):
        h.validate_blocker_candidate(CANDIDATE, bad)


def test_changed_source_ancestry_is_rejected(monkeypatch):
    monkeypatch.setattr(h, "EXPECTED_BLOCKER_ANCESTRY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source ancestry witness changed"):
        h.validate_blocker_candidate(CANDIDATE, AUDIT)


def test_458_install_callback_patches_408_bounds_without_running_fit():
    validation = h.validate_blocker_candidate(CANDIDATE, AUDIT)
    with np.load(CANDIDATE, allow_pickle=False) as z:
        points_mm = np.asarray(z["points_world_m"], dtype=np.float64) * 1000.0
        faces = np.asarray(z["source_O_faces"], dtype=np.int64)
    support = np.arange(20889, dtype=np.int64)
    anchors = np.asarray([7587], dtype=np.int64)
    def source_safe(points, source, support_rows, anchor_rows, trust, cap):
        assert np.array_equal(points, points_mm)
        n = len(support_rows) * 3
        return np.full(n, -0.25), np.full(n, 0.25), {"source_bound": "base"}
    owner = SimpleNamespace(source_safe_trust_bounds=source_safe)
    def original_install(owner_arg, objective, crossing):
        lo, hi, report = owner_arg.source_safe_trust_bounds(
            points_mm, points_mm.copy(), support, anchors, 1.0, 12.0)
        return lo, hi, report
    wrapped_458 = SimpleNamespace(install_objective=original_install)
    def fake_485_load(name, path):
        assert name == "matched458_for_485"
        return wrapped_458
    fake_485 = SimpleNamespace(load=fake_485_load, RUNNER_485=Path("/fake/458.py"))
    entry = SimpleNamespace(_load_module=lambda path: fake_485, RUNNER_485=Path("/fake/458.py"))
    states = []
    w.patch_485_loader(entry, validation, states, h)
    patched_485 = entry._load_module(entry.RUNNER_485)
    loaded_458 = patched_485.load("matched458_for_485", Path("/fake/458.py"))
    lo, hi, report = loaded_458.install_objective(owner, object(), object())
    assert states[0]["calls"] == 1 and len(states[0]["receipts"]) == 1
    assert report["area_blocker_zero_bounds_496"]["nonblocker_bounds_bitwise_unchanged"] is True
    blocker = set(v for row in h.BLOCKER_FACE_ROWS for v in map(int, faces[row]))
    assert all(np.all(lo[3*n:3*n+3] == 0.0) for n in blocker)
    assert np.all(lo[3*100:3*100+3] == -0.25)
    assert np.all(hi[3*100:3*100+3] == 0.25)

def test_actual_490_matched_preflight_and_source_lock():
    entry = load("matched_490_for_496_preflight", D / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py")
    config_path = D / "D/D/component-lp-decomposition-490/matched_inputs_476.json"
    config = json.loads(config_path.read_text())
    actual = {role: Path(record["path"]) for role, record in config["inputs"].items()}
    checked = entry.validate_matched_inputs(config_path, actual)
    assert checked["inputs"]["candidate"]["sha256"] == h.EXPECTED_CANDIDATE_SHA256
    assert len(entry.validate_source_lock()) >= 10
    blocker = h.validate_blocker_candidate(actual["candidate"], actual["audit159"])
    assert blocker["candidate_sha256"] == checked["inputs"]["candidate"]["sha256"]



def test_wrapper_main_reparses_original_490_argv_and_reads_nested_report(tmp_path, monkeypatch):
    """Exercise real 490.main/extractor with a fake 485 terminal before any fit."""
    entry = load("matched_490_for_496_main_smoke", D / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py")
    config_path = D / "D/D/component-lp-decomposition-490/matched_inputs_476.json"
    config = json.loads(config_path.read_text())
    inputs = {role: Path(rec["path"]) for role, rec in config["inputs"].items()}
    output_dir = tmp_path / "mock-fit"
    argv = [
        w.FLAG,
        "--matched-inputs-json-490", str(config_path),
        "--audit159-490", str(inputs["audit159"]),
        "--base-candidate", str(inputs["candidate"]),
        "--face-target-npz", str(inputs["face_target_npz"]),
        "--point-target-npz", str(inputs["point_target_npz"]),
        "--output", str(output_dir),
    ]
    # Verify preflight parse and the delegate's second parse see the same 490 flags.
    forwarded = w.delegated_490_argv(argv)
    preflight_remaining, preflight_config, preflight_output, preflight_inputs = entry.extract_fit_inputs(forwarded)
    assert preflight_remaining == [
        "--base-candidate", str(inputs["candidate"]),
        "--face-target-npz", str(inputs["face_target_npz"]),
        "--point-target-npz", str(inputs["point_target_npz"]),
        "--output", str(output_dir),
    ]
    assert preflight_config == config_path
    assert preflight_output == output_dir
    assert preflight_inputs["candidate"] == inputs["candidate"]

    observed = {}
    fake_458 = SimpleNamespace()
    def fake_install(owner, objective, crossing):
        support = np.arange(20889, dtype=np.int64)
        anchors = np.empty(0, dtype=np.int64)
        lo, hi, report = owner.source_safe_trust_bounds(
            owner.points_mm, owner.points_mm.copy(), support, anchors, 1.0, 12.0)
        observed["source_report"] = report
        return lo, hi, report
    fake_458.install_objective = fake_install

    fake_485 = SimpleNamespace()
    fake_485.load = lambda name, path: fake_458 if name == "matched458_for_485" else None
    def fake_485_main():
        observed["fit_argv"] = list(sys.argv[1:])
        owner = SimpleNamespace()
        with np.load(inputs["candidate"], allow_pickle=False) as z:
            owner.points_mm = np.asarray(z["points_world_m"], dtype=np.float64) * 1000.0
        def base_bounds(points, source, support, anchors, trust, cap):
            n = 3 * len(support)
            return np.full(n, -0.25), np.full(n, 0.25), {"base": True}
        owner.source_safe_trust_bounds = base_bounds
        fake_485.load("matched458_for_485", Path("/fake/matched458.py")).install_objective(owner, None, None)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(json.dumps({
            "constraints": {"source_safe_box": observed["source_report"]}
        }))
    fake_485.main = fake_485_main
    entry._load_module = lambda path: fake_485

    real_load = w.load_pinned
    def load_for_smoke(name, path):
        if Path(path).resolve() == w.ENTRY_490.resolve():
            return entry
        return real_load(name, path)
    monkeypatch.setattr(w, "load_pinned", load_for_smoke)
    monkeypatch.setattr(sys, "argv", [str(Path(w.__file__)), *argv])
    w.main()

    assert observed["fit_argv"] == [
        "--base-candidate", str(inputs["candidate"]),
        "--face-target-npz", str(inputs["face_target_npz"]),
        "--point-target-npz", str(inputs["point_target_npz"]),
        "--output", str(output_dir),
    ]
    final_report = json.loads((output_dir / "report.json").read_text())
    receipt = final_report["area_blocker_zero_bounds_496"]["runtime_hook_receipt"]
    assert receipt["nonblocker_bounds_bitwise_unchanged"] is True
    assert "source_safe_box" in final_report["constraints"]
