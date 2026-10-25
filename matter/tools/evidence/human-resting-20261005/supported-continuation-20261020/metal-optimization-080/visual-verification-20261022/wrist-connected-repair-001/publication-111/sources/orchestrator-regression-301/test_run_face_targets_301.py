from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


RUNNER = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/run_connected_clearance_301.py")
SPEC = importlib.util.spec_from_file_location("connected_clearance_301_regression", RUNNER)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
PAIR_COUNT = 3886


def install_builder_fixture(tmp_path, monkeypatch, *, result_rows):
    candidate = tmp_path / "candidate.npz"
    surface = tmp_path / "surface.json"
    candidate.write_bytes(b"candidate-for-run_face_targets")
    surface.write_text(json.dumps({"complete": True}))
    builder = tmp_path / "mock-builder-303.py"
    builder.write_text("# mock builder bytes\\n")
    monkeypatch.setattr(MOD, "FACE_BUILDER", builder)
    monkeypatch.setattr(MOD, "surface_pair_count", lambda report, *, expected_faces: PAIR_COUNT)
    out = tmp_path / "round-001"
    out.mkdir()
    calls = {"fit": 0, "point": 0}

    def fake_run(command, log, *, slot=True, timeout=900):
        args = list(map(str, command))
        assert args[0] == str(MOD.PYTHON)
        assert args[1] == str(builder)
        assert args[args.index("--direction-mode") + 1] == "outward-nearest"
        destination = Path(args[args.index("--output-dir") + 1])
        assert destination == out / "face-targets"
        assert not destination.exists(), "builder owns creation of an uncached output path"
        assert slot is True
        destination.mkdir()
        np.savez_compressed(destination / "targets.npz", target_rows=np.arange(result_rows, dtype=np.int32))
        report = {
            "runner_sha256": MOD.sha(builder),
            "direction_mode": "outward-nearest",
            "inputs": {
                "candidate_sha256": MOD.sha(candidate),
                "surface_report_sha256": MOD.sha(surface),
            },
            "matched_159_complete": True,
            "failure_counts": {},
            "banks": 185,
            "passing_targets": result_rows,
            "crossing_pairs": PAIR_COUNT,
            "matched_surface_bank_triangle_pairs": PAIR_COUNT,
            "result_rows": result_rows,
        }
        (destination / "report.json").write_text(json.dumps(report))
        return {"returncode": 0}

    def fit_not_reached(*_args, **_kwargs):
        calls["fit"] += 1
        raise AssertionError("fit must not run before full target completeness passes")

    def point_not_reached(*_args, **_kwargs):
        calls["point"] += 1
        raise AssertionError("point preparation must follow face-target completeness")

    monkeypatch.setattr(MOD, "run", fake_run)
    monkeypatch.setattr(MOD, "run_fit_attempt", fit_not_reached)
    monkeypatch.setattr(MOD, "run_point_targets", point_not_reached)
    return candidate, surface, out, calls


def test_301_actual_face_target_path_accepts_complete_3886_report(tmp_path, monkeypatch):
    candidate, surface, out, calls = install_builder_fixture(tmp_path, monkeypatch, result_rows=PAIR_COUNT)
    npz, report = MOD.run_face_targets(candidate, surface, out)
    assert npz.is_file() and report.is_file()
    receipt = json.loads((out / "face-targets" / "completeness-preflight.json").read_text())
    assert receipt["status"] == "PASS_COMPLETE_MATCHED_FACE_TARGETS_ONLY"
    assert receipt["surface_pair_count"] == PAIR_COUNT
    assert calls == {"fit": 0, "point": 0}


def test_301_actual_face_target_path_rejects_3885_before_point_or_fit(tmp_path, monkeypatch):
    candidate, surface, out, calls = install_builder_fixture(tmp_path, monkeypatch, result_rows=PAIR_COUNT - 1)
    with pytest.raises(ValueError, match="passing_targets mismatch"):
        MOD.run_face_targets(candidate, surface, out)
    assert not (out / "face-targets" / "completeness-preflight.json").exists()
    assert calls == {"fit": 0, "point": 0}


def test_301_main_orders_full_face_preflight_before_point_preparation_and_fit():
    tree = ast.parse(RUNNER.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"run_face_targets", "run_point_targets", "run_fit_attempt"}:
                calls.append((node.lineno, node.func.id))
    first_line = {name: min(line for line, call_name in calls if call_name == name)
                  for name in ("run_face_targets", "run_point_targets", "run_fit_attempt")}
    assert first_line["run_face_targets"] < first_line["run_point_targets"] < first_line["run_fit_attempt"]
