import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170")
SOURCE = BASE / "run_coherent_target_continuation_336.py"
ACTUAL = BASE / "coherent-support-unlock-334/round-001"

def load_module():
    spec = importlib.util.spec_from_file_location("continuation336_actual_adapter", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def test_point_target_consumer_accepts_actual_334_report_without_optional_field(tmp_path, monkeypatch):
    m = load_module()
    state = json.loads((ACTUAL / "input-state.json").read_text())
    candidate = Path(state["candidate"])
    point_summary = Path(state["point_summary"])
    point_report = json.loads(point_summary.read_text())
    assert point_report["inside_nodes"] == 489
    actual_dir = ACTUAL / "point-targets"
    actual_report = actual_dir / "nearest-union-exits.json"
    actual_npz = actual_dir / "nearest-union-exits.npz"
    report = json.loads(actual_report.read_text())
    arrays = np.load(actual_npz, allow_pickle=False)
    assert "source_normal_unavailable_rows" not in report
    assert report["passing_targets"] == 489
    assert report["status"] == "PASS_FIRST_UNION_EXIT_TARGETS_ONLY"
    assert all(np.all(arrays[key]) for key in ("pass_mask", "source_normal_available_mask", "target_exactly_outside_all185"))
    copied = {key: arrays[key].copy() for key in arrays.files}
    out = tmp_path / "mocked-round"
    out.mkdir()
    calls = []

    def fake_run(argv, log, *, slot, timeout=None):
        calls.append((list(argv), Path(log), slot))
        args = {str(argv[i]): argv[i + 1] for i in range(2, len(argv) - 1) if str(argv[i]).startswith("--")}
        assert Path(args["--candidate"]) == candidate
        assert Path(args["--point-summary"]) == point_summary
        assert Path(args["--query-dir"]) == point_summary.parent / "query"
        assert float(args["--clearance-m"]) == m.CONSTRUCTION_CLEARANCE_M
        target_dir = Path(args["--output-dir"])
        target_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(actual_report, target_dir / "nearest-union-exits.json")
        shutil.copy2(actual_npz, target_dir / "nearest-union-exits.npz")

    monkeypatch.setattr(m, "run", fake_run)
    out_npz, out_report = m.run_point_targets(None, state, candidate, out)
    assert len(calls) == 1 and calls[0][2] is True
    assert sha(out_npz) == sha(actual_npz)
    assert sha(out_report) == sha(actual_report)
    assert len(copied["query_rows"]) == 489


def test_initial_334_point_and_335_face_cache_are_reused_only_for_pinned_candidate(tmp_path, monkeypatch):
    m = load_module()
    state = json.loads((ACTUAL / "input-state.json").read_text())
    candidate = Path(state["candidate"])
    surface = Path(state["surface_report"])
    out = tmp_path / "cached-round"
    out.mkdir()
    def forbidden_run(*args, **kwargs):
        raise AssertionError("pinned initial targets should be reused without a subprocess")
    monkeypatch.setattr(m, "run", forbidden_run)
    point_npz, point_report = m.run_point_targets(None, state, candidate, out, m.INITIAL_TARGET_CACHE_DIR)
    face_npz, face_report = m.run_face_targets(candidate, surface, out, cache_dir=m.INITIAL_TARGET_CACHE_DIR)
    assert sha(point_npz) == m.INITIAL_TARGET_CACHE_PINS["point_npz_sha256"]
    assert sha(point_report) == m.INITIAL_TARGET_CACHE_PINS["point_report_sha256"]
    assert sha(face_npz) == m.INITIAL_TARGET_CACHE_PINS["face_npz_sha256"]
    assert sha(face_report) == m.INITIAL_TARGET_CACHE_PINS["face_report_sha256"]
    assert json.loads((out / "point-targets/cache-use.json").read_text())["matched_cache"] is True
    assert json.loads((out / "face-targets/cache-use.json").read_text())["matched_cache"] is True
