from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/run_coherent_clearance_308.py")
SPEC = importlib.util.spec_from_file_location("coherent_clearance_308_tested", SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _main_fixture(tmp_path, monkeypatch, *, attempts, accumulation_sequence):
    candidate = tmp_path / "base-candidate.bin"
    candidate.write_bytes(b"base candidate")
    point_summary = tmp_path / "point-summary.json"
    point_summary.write_text(json.dumps({
        "candidate_sha256": MOD.sha(candidate),
        "complete_surface_audit": True,
        "inside_nodes": 1,
    }))
    surface_report = tmp_path / "surface-report.json"
    surface_report.write_text(json.dumps({"candidate_sha256": MOD.sha(candidate), "complete": True}))
    point_npz = tmp_path / "point-targets.npz"
    point_npz.write_bytes(b"point targets")
    point_report = tmp_path / "point-targets.json"
    point_report.write_text("{}")
    face_npz = tmp_path / "face-targets.npz"
    face_npz.write_bytes(b"face targets")
    face_report = tmp_path / "face-targets.json"
    face_report.write_text("{}")
    trial = tmp_path / "trial.npz"
    trial.write_bytes(b"raw trial")
    state = {
        "candidate": str(candidate),
        "candidate_sha256": MOD.sha(candidate),
        "point_summary": str(point_summary),
        "surface_report": str(surface_report),
        "parent_fit": "parent-fit-fixture",
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    out = tmp_path / "continuation"
    face_rows = np.array([[0, 1, 2]], dtype=np.int64)
    base_pair = (0, 1)
    new_pair = (1, 2)
    monkeypatch.setattr(MOD, "verify_file_pins", lambda: {"mocked": True})
    monkeypatch.setattr(MOD, "validate_state", lambda _path: (state, candidate, face_rows, np.array([], dtype=np.int64)))
    monkeypatch.setattr(MOD, "seed_archive_index", lambda: ({}, {"verified": True}))
    monkeypatch.setattr(MOD, "collect_pairs", lambda *_args, **_kwargs: ({base_pair}, [{"seed": "receipt"}]))
    monkeypatch.setattr(MOD, "surface_pair_count", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(MOD, "require_headroom", lambda _stage: None)
    monkeypatch.setattr(MOD, "load_module", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(MOD, "run_face_targets", lambda *_args, **_kwargs: (face_npz, face_report))
    monkeypatch.setattr(MOD, "run_point_targets", lambda *_args, **_kwargs: (point_npz, point_report))
    fit_calls = []
    monkeypatch.setattr(
        MOD,
        "run_fit_attempt",
        lambda _state, _candidate, pairs, *_args: (fit_calls.append(np.asarray(pairs).copy()) or trial),
    )
    pair_iter = iter(accumulation_sequence)
    accumulation_calls = []

    def fake_accumulate(_faces, _archive_members):
        pairs = set(next(pair_iter))
        accumulation_calls.append(pairs)
        return pairs, [{"pair_source": "mock"}]

    monkeypatch.setattr(MOD, "accumulate_pairs", fake_accumulate)
    backtracker_calls = []

    def fake_backtracker(_state, _trial, outpath, helper, entry, max_audits, max_passes):
        mode = "local" if helper == MOD.LOCAL_HELPER else "global"
        backtracker_calls.append((mode, outpath, entry, max_audits, max_passes))
        return None

    monkeypatch.setattr(MOD, "run_backtracker", fake_backtracker)
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        [
            str(SOURCE), "--state", str(state_path), "--out", str(out),
            "--rounds", "1", "--attempts-per-round", str(attempts),
        ],
    )
    return {
        "state_path": state_path, "out": out, "fit_calls": fit_calls,
        "accumulation_calls": accumulation_calls, "backtracker_calls": backtracker_calls,
        "base_pair": base_pair, "new_pair": new_pair,
    }


def test_new_probe_pairs_skip_local_fallback_then_reach_attempt_two(tmp_path, monkeypatch):
    A, B = (0, 1), (1, 2)
    fixture = _main_fixture(
        tmp_path, monkeypatch, attempts=2,
        accumulation_sequence=[{A}, {A, B}, {A, B}, {A, B}, {A, B}],
    )
    MOD.main()
    assert [pairs.tolist() for pairs in fixture["fit_calls"]] == [[[0, 1]], [[0, 1], [1, 2]]]
    assert [c[0] for c in fixture["backtracker_calls"]] == ["global", "global", "local"]
    first = fixture["out"] / "round-001" / "attempt-001"
    assert json.loads((first / "result.json").read_text())["status"] == "EXACT_SELF_WITNESSES_LEARNED_BEFORE_LOCAL_DAMPING"
    assert not (first / "local-183-fallback").exists()
    assert fixture["backtracker_calls"][-1][1] == fixture["out"] / "round-001" / "attempt-002" / "local-183-fallback"


def test_no_new_probe_pairs_triggers_same_trial_local_fallback(tmp_path, monkeypatch):
    A = (0, 1)
    fixture = _main_fixture(
        tmp_path, monkeypatch, attempts=2,
        accumulation_sequence=[{A}, {A}, {A}],
    )
    MOD.main()
    assert len(fixture["fit_calls"]) == 1
    assert [c[0] for c in fixture["backtracker_calls"]] == ["global", "local"]
    attempt = fixture["out"] / "round-001" / "attempt-001"
    assert json.loads((attempt / "stop-repeated-input.json").read_text())["status"] == "STOP_IDENTICAL_NEXT_LP_INPUT"


def test_new_pairs_on_last_attempt_still_trigger_local_fallback(tmp_path, monkeypatch):
    A, B = (0, 1), (1, 2)
    fixture = _main_fixture(
        tmp_path, monkeypatch, attempts=1,
        accumulation_sequence=[{A}, {A, B}, {A, B}],
    )
    MOD.main()
    assert len(fixture["fit_calls"]) == 1
    assert [c[0] for c in fixture["backtracker_calls"]] == ["global", "local"]
    attempt = fixture["out"] / "round-001" / "attempt-001"
    learned = json.loads((attempt / "learned-self-pairs.json").read_text())
    assert learned["unique_pairs_after_attempt"] == 2


def _fit_fixture(tmp_path, monkeypatch, *, invalid_report=None):
    candidate = tmp_path / "candidate.npz"
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    anchors = np.array([0], dtype=np.int64)
    np.savez_compressed(candidate, points_world_m=points, source_O_faces=faces,
                        typed_coarse_anchor_output_rows_018=anchors)
    point_npz = tmp_path / "point.npz"; point_npz.write_bytes(b"point npz")
    point_report = tmp_path / "point.json"; point_report.write_text("{}")
    face_npz = tmp_path / "face.npz"; face_npz.write_bytes(b"face npz")
    face_report = tmp_path / "face.json"; face_report.write_text("{}")
    solver = tmp_path / "solver.py"; solver.write_text("# solver pin\\n")
    monkeypatch.setattr(MOD, "SOLVER", solver)
    monkeypatch.setattr(MOD, "FIXED_PREFIX", 1)
    monkeypatch.setattr(MOD, "require_headroom", lambda _stage: None)
    pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
    support = np.array([1, 2, 3], dtype=np.int64)
    delta = np.array([[.25, 0., 0.], [0., .5, 0.], [0., 0., .75]], dtype=np.float64)
    out = tmp_path / "attempt"; out.mkdir()
    seen = {}

    def fake_run(command, log, *, slot=True, timeout=900):
        args = list(map(str, command))
        def arg(flag):
            return args[args.index(flag) + 1]
        assert arg("--base-candidate") == str(candidate)
        assert arg("--raw-trial") == str(candidate)
        assert float(arg("--max-edge-gradient")) == .5
        assert float(arg("--slack-tolerance")) == .05
        assert int(arg("--time-limit-s")) == 60
        assert int(arg("--area-scp-iterations")) == 4
        assert int(arg("--fixed-prefix-nodes")) == 1
        assert slot is True and timeout == 900
        proposal = Path(arg("--output"))
        assert proposal == out / "proposal" and not proposal.exists()
        assert Path(arg("--point-target-npz")) == point_npz
        assert Path(arg("--point-target-report")) == point_report
        assert Path(arg("--face-target-npz")) == face_npz
        assert Path(arg("--face-target-report")) == face_report
        saved_pairs = np.load(arg("--self-pairs"), allow_pickle=False)
        np.testing.assert_array_equal(saved_pairs, pairs)
        proposal.mkdir()
        np.savez_compressed(proposal / "raw-lp-candidate.npz",
            nodes_lp_support_rows=support, nodes_lp_delta_mm_f64=delta)
        report = {
            "support": {"threatened_pairs": len(pairs)},
            "constraints": {
                "actual_pair_records": [{"faces": row.tolist()} for row in pairs],
                "edge_gradient": {"enabled": True, "max_component_gradient": .5},
                "edge_gradient_rows": 3,
            },
            "inputs": {
                "base_candidate_sha256": MOD.sha(candidate),
                "self_pairs_sha256": MOD.sha(Path(arg("--self-pairs"))),
                "point_target_npz_sha256": MOD.sha(point_npz),
                "point_target_report_sha256": MOD.sha(point_report),
                "face_target_npz_sha256": MOD.sha(face_npz),
                "face_target_report_sha256": MOD.sha(face_report),
            },
        }
        if invalid_report == "input":
            report["inputs"]["face_target_report_sha256"] = "f" * 64
        elif invalid_report == "pairs":
            report["constraints"]["actual_pair_records"] = [{"faces": [0, 2]}]
        elif invalid_report == "gradient":
            report["constraints"]["edge_gradient"]["enabled"] = False
        (proposal / "report.json").write_text(json.dumps(report))
        seen.update({"args": args, "proposal": proposal})

    monkeypatch.setattr(MOD, "run", fake_run)
    trial = MOD.run_fit_attempt(
        {"parent_fit": "fixture"}, candidate, pairs,
        point_npz, point_report, face_npz, face_report, out,
    )
    return trial, candidate, points, seen, out


def test_fit_attempt_arguments_and_report_are_coherent(tmp_path, monkeypatch):
    trial, candidate, points, seen, out = _fit_fixture(tmp_path, monkeypatch)
    assert trial == out / "trial-halfstep.npz" and trial.is_file()
    assert seen["proposal"] == out / "proposal"
    emitted = np.load(trial, allow_pickle=False)["points_world_m"]
    assert emitted.dtype == np.float32
    assert np.array_equal(emitted[0].view(np.uint32), points[0].view(np.uint32))
    receipt = json.loads((out / "verified-fit-inputs.json").read_text())
    assert receipt["pair_count"] == 2
    assert receipt["edge_gradient"]["max_component_gradient"] == .5
    assert json.loads((out / "trial-provenance.json").read_text())["accepted_physical_steps"] == 0


@pytest.mark.parametrize(
    ("invalid_report", "message"),
    [("input", "different base, exact pairs or target inputs"),
     ("pairs", "pair-record identities"),
     ("gradient", "neighbor-gradient proposal constraint")],
)
def test_fit_report_validator_rejects_wrong_inputs_pairs_or_gradient(tmp_path, monkeypatch, invalid_report, message):
    with pytest.raises(ValueError, match=message):
        _fit_fixture(tmp_path, monkeypatch, invalid_report=invalid_report)


@pytest.mark.parametrize("rounds,attempts,expected", [(1, 1, True), (3, 4, True), (3, 5, False), (4, 1, False), (True, 1, False)])
def test_attempt_limit_contract(rounds, attempts, expected):
    assert MOD.valid_limits(rounds, attempts) is expected
