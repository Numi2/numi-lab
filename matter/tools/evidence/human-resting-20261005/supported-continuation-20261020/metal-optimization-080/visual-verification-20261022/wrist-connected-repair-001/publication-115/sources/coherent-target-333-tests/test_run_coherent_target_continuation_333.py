import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/run_coherent_target_continuation_333.py")


def load_module():
    spec = importlib.util.spec_from_file_location("continuation333_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj))


def point_setup(tmp_path, n=2):
    m = load_module()
    candidate = tmp_path / "candidate.npz"
    np.savez_compressed(candidate, points_world_m=np.zeros((10, 3), dtype=np.float32),
                        source_O_faces=np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32))
    state = {"point_summary": str(tmp_path / "point-summary.json")}
    point = Path(state["point_summary"])
    point.parent.mkdir(parents=True, exist_ok=True)
    dump(point, {"inside_nodes": n})
    query = point.parent / "query"
    query.mkdir()
    for name in ("query-receipt.json", "points.csv", "nodes.csv", "pairs.csv"):
        (query / name).write_text(name)
    out = tmp_path / "round"
    out.mkdir()
    return m, candidate, state, point, query, out


def point_report(candidate, point, query, n, *, candidate_sha=None, point_sha=None,
                 query_sha=None, provenance=True, status="PASS_FIRST_UNION_EXIT_TARGETS_ONLY",
                 passing=None):
    return {
        "status": status,
        "passing_targets": n if passing is None else passing,
        "inputs": {
            "candidate_sha256": sha(candidate) if candidate_sha is None else candidate_sha,
            "point_summary_sha256": sha(point) if point_sha is None else point_sha,
            "query_receipt_sha256": sha(query / "query-receipt.json") if query_sha is None else query_sha,
            "query_points_sha256": sha(query / "points.csv"),
            "query_nodes_sha256": sha(query / "nodes.csv"),
            "query_pairs_sha256": sha(query / "pairs.csv"),
        },
        "source_normal_unavailable_rows": [],
        "query_rows": n, "inside_query_nodes": n, "all_bank_ray_results": n,
        "targets": ([{
            "query_row": 4 + i, "status": "PASS_FIRST_UNION_EXIT",
            "target_world_f32_m": [float(i + 1), float(i + 2), float(i + 3)],
            "proposal_direction_world_unit": [0.0, 0.0, 1.0],
            "source_surface_normal_world_unit": [0.0, 0.0, 1.0],
            "target_distance_m": 0.00025,
            "target_exactly_outside_all185": True,
            "end_outside_all185_by_event_parity": True,
            "exact_endpoint_bank_count": 185,
            "source_normal_provenance": {
                "incident_candidate_face_rows": [0], "unique_incident_face_count": 1,
                "source037_area_sum_m2": 1e-6, "resultant_over_area_sum": 1.0,
                "source_face_area_weights": [{
                    "candidate_face_row": 0, "source037_area_m2": 1e-6,
                    "registered_source_normal_world_unit": [0.0, 0.0, 1.0],
                    "source_ancestry": {"kind": "original_NHSKIN_face", "row": 4},
                }],
            },
        } for i in range(n)] if provenance else [{"query_row": 4 + i, "status": "PASS_FIRST_UNION_EXIT"} for i in range(n)]),
    }


def install_point_runner(monkeypatch, m, candidate, point, query, n, *, report_overrides=None,
                         masks=None, target_overrides=None, output_dir_assertion=True):
    calls = []

    def fake_run(argv, log, *, slot, timeout=None):
        calls.append((argv, Path(log), slot))
        assert Path(argv[1]) == m.POINT_BUILDER
        args = {str(argv[i]): argv[i + 1] for i in range(2, len(argv) - 1) if str(argv[i]).startswith("--")}
        assert Path(args["--candidate"]) == candidate
        assert Path(args["--point-summary"]) == point
        assert Path(args["--query-dir"]) == query
        assert float(args["--clearance-m"]) == pytest.approx(0.00025)
        target_dir = Path(args["--output-dir"])
        if output_dir_assertion:
            assert not target_dir.exists(), "327 must own creation of its output directory"
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = {key: np.ones(n, dtype=bool) for key in (
            "pass_mask", "source_normal_available_mask", "target_exactly_outside_all185")}
        payload.update({
            "query_rows": np.array([4 + i for i in range(n)], dtype=np.int64),
            "target_world_f32_m": np.array([[i + 1, i + 2, i + 3] for i in range(n)], dtype=np.float32).reshape(n, 3),
            "proposal_direction_world_unit": np.tile([0.0, 0.0, 1.0], (n, 1)),
            "source_surface_normal_world_unit": np.tile([0.0, 0.0, 1.0], (n, 1)),
            "target_distance_m": np.full(n, 0.00025, dtype=np.float64),
        })
        if masks:
            payload.update(masks)
        target_npz = target_dir / "nearest-union-exits.npz"
        np.savez_compressed(target_npz, **payload)
        report = point_report(candidate, point, query, n, **(report_overrides or {}))
        if target_overrides:
            report["targets"][0].update(target_overrides)
        report["outputs"] = {"npz_sha256": sha(target_npz)}
        dump(target_dir / "nearest-union-exits.json", report)

    monkeypatch.setattr(m, "run", fake_run)
    return calls


def test_positive_point_targets_call_327_with_matched_inputs_and_margin(tmp_path, monkeypatch):
    m, candidate, state, point, query, out = point_setup(tmp_path)
    calls = install_point_runner(monkeypatch, m, candidate, point, query, 2)
    npz, report = m.run_point_targets(m.D / "unused-185-summary.json", state, candidate, out)
    assert len(calls) == 1 and calls[0][2] is True
    assert npz.is_file() and report.is_file()


@pytest.mark.parametrize("key", ["pass_mask", "source_normal_available_mask", "target_exactly_outside_all185"])
def test_positive_point_targets_reject_false_masks(tmp_path, monkeypatch, key):
    m, candidate, state, point, query, out = point_setup(tmp_path)
    install_point_runner(monkeypatch, m, candidate, point, query, 2,
                         masks={key: np.array([True, False])})
    with pytest.raises(ValueError, match="endpoint/source proof incomplete"):
        m.run_point_targets(m.D / "unused-185-summary.json", state, candidate, out)


@pytest.mark.parametrize("target_override", [
    {"exact_endpoint_bank_count": 184},
    {"target_exactly_outside_all185": False},
    {"end_outside_all185_by_event_parity": False},
])
def test_positive_point_targets_reject_missing_all185_row_proof(tmp_path, monkeypatch, target_override):
    m, candidate, state, point, query, out = point_setup(tmp_path)
    install_point_runner(monkeypatch, m, candidate, point, query, 2, target_overrides=target_override)
    with pytest.raises(ValueError, match="all-185 endpoint proof"):
        m.run_point_targets(m.D / "unused-185-summary.json", state, candidate, out)


def test_positive_point_targets_reject_wrong_mask_shape(tmp_path, monkeypatch):
    m, candidate, state, point, query, out = point_setup(tmp_path)
    install_point_runner(monkeypatch, m, candidate, point, query, 2,
                         masks={"target_exactly_outside_all185": np.ones(1, dtype=bool)})
    with pytest.raises(ValueError, match="endpoint/source proof incomplete"):
        m.run_point_targets(m.D / "unused-185-summary.json", state, candidate, out)


@pytest.mark.parametrize("overrides,match", [
    ({"candidate_sha": "0" * 64}, "different candidate"),
    ({"status": "INCOMPLETE"}, "did not certify every inside row"),
    ({"passing": 1}, "did not certify every inside row"),
])
def test_positive_point_targets_reject_incomplete_or_misbound_report(tmp_path, monkeypatch, overrides, match):
    m, candidate, state, point, query, out = point_setup(tmp_path)
    install_point_runner(monkeypatch, m, candidate, point, query, 2, report_overrides=overrides)
    with pytest.raises(ValueError, match=match):
        m.run_point_targets(m.D / "unused-185-summary.json", state, candidate, out)


@pytest.mark.parametrize("field", ["point_sha", "query_sha"])
def test_positive_point_targets_reject_misbound_summary_or_query_hash(tmp_path, monkeypatch, field):
    m, candidate, state, point, query, out = point_setup(tmp_path)
    install_point_runner(monkeypatch, m, candidate, point, query, 2,
                         report_overrides={field: "0" * 64})
    with pytest.raises(ValueError, match="summary|query|input hash"):
        m.run_point_targets(m.D / "unused-185-summary.json", state, candidate, out)


def test_positive_point_targets_reject_missing_source_ancestry_records(tmp_path, monkeypatch):
    m, candidate, state, point, query, out = point_setup(tmp_path)
    install_point_runner(monkeypatch, m, candidate, point, query, 2,
                         report_overrides={"provenance": False})
    with pytest.raises(ValueError, match="ancestry|provenance"):
        m.run_point_targets(m.D / "unused-185-summary.json", state, candidate, out)


def test_zero_inside_uses_235_empty_target_emitter(tmp_path, monkeypatch):
    m, candidate, state, point, query, out = point_setup(tmp_path, n=0)
    calls = []

    def fake_run(argv, log, *, slot, timeout=None):
        calls.append(argv)
        assert Path(argv[1]) == m.POINT_EMITTER
        args = {str(argv[i]): argv[i + 1] for i in range(2, len(argv) - 1) if str(argv[i]).startswith("--")}
        assert Path(args["--candidate"]) == candidate
        assert Path(args["--summary"]) == point
        assert Path(args["--query-dir"]) == query
        npz = Path(args["--npz"]); report = Path(args["--report"])
        np.savez_compressed(npz, query_rows=np.empty(0,dtype=np.int64),
                            target_world_f32_m=np.empty((0,3),dtype=np.float32),
                            proposal_direction_world_unit=np.empty((0,3),dtype=np.float64),
                            source_surface_normal_world_unit=np.empty((0,3),dtype=np.float64),
                            source_normal_available_mask=np.empty(0,dtype=bool),
                            pass_mask=np.empty(0,dtype=bool))
        dump(report, {"status": "PASS_FIRST_UNION_EXIT_TARGETS_ONLY", "passing_targets": 0,
                      "query_rows": 0, "inside_query_nodes": 0, "all_bank_ray_results": 0,
                      "zero_targets_emitted": True, "targets": [],
                      "inputs": {"candidate_sha256": sha(candidate),
                                 "point_summary_sha256": sha(point),
                                 "query_receipt_sha256": sha(query / "query-receipt.json"),
                                 "query_points_sha256":sha(query / "points.csv"),
                                 "query_nodes_sha256":sha(query / "nodes.csv"),
                                 "query_pairs_sha256":sha(query / "pairs.csv")},
                      "outputs":{"npz_sha256":sha(npz)}})

    monkeypatch.setattr(m, "run", fake_run)
    npz, report = m.run_point_targets(m.D / "unused-185-summary.json", state, candidate, out)
    assert len(calls) == 1
    assert Path(calls[0][1]) == m.POINT_EMITTER
    assert np.load(npz)["pass_mask"].shape == (0,)
    assert report.is_file()


def test_face_targets_keep_source_outward_and_same_clearance(tmp_path, monkeypatch):
    m = load_module()
    candidate = tmp_path / "candidate.npz"; candidate.write_bytes(b"candidate")
    surface = tmp_path / "surface.json"; dump(surface, {"complete_surface_audit": True})
    out = tmp_path / "round"; out.mkdir()
    builder = tmp_path / "builder.py"; builder.write_text("# pinned fake builder\n")
    checker = tmp_path / "checker.py"; checker.write_text("# fake checker\n")
    monkeypatch.setattr(m, "FACE_BUILDER", builder)
    monkeypatch.setattr(m, "FACE_CHECKER", checker)
    monkeypatch.setattr(m, "surface_pair_count", lambda report, expected_faces: 7)
    seen = {}

    def fake_run(argv, log, *, slot, timeout=None):
        seen["argv"] = argv
        assert Path(argv[1]) == builder
        args = {str(argv[i]): argv[i + 1] for i in range(2, len(argv) - 1) if str(argv[i]).startswith("--")}
        assert Path(args["--candidate"]) == candidate
        assert Path(args["--surface-report"]) == surface
        assert args["--direction-mode"] == "outward-nearest"
        assert float(args["--clearance-m"]) == pytest.approx(0.00025)
        td = Path(args["--output-dir"]); assert not td.exists(); td.mkdir()
        (td / "targets.npz").write_bytes(b"targets")
        dump(td / "report.json", {"runner_sha256": sha(builder), "direction_mode": "outward-nearest",
                                   "inputs": {"candidate_sha256": sha(candidate)}})

    def fake_load(path, name):
        assert path == checker
        class Check:
            @staticmethod
            def validate_full_face_target_report(report, **kwargs):
                assert kwargs == {"expected_candidate_sha256": sha(candidate),
                                  "expected_surface_report_sha256": sha(surface),
                                  "expected_surface_pair_count": 7, "expected_banks": 185}
                return {"complete": True}
        return Check

    monkeypatch.setattr(m, "run", fake_run)
    monkeypatch.setattr(m, "load_module", fake_load)
    npz, report = m.run_face_targets(candidate, surface, out)
    assert npz.is_file() and report.is_file()
    assert json.loads((out / "face-targets/completeness-preflight.json").read_text())["complete"] is True


def test_accumulate_pairs_unions_322_309_318_and_current(tmp_path, monkeypatch):
    m = load_module()
    roots = [tmp_path / name for name in ("322", "current", "309", "318")]
    monkeypatch.setattr(m, "SEED_ROOT", roots[0])
    monkeypatch.setattr(m, "CURRENT_OUT", roots[1], raising=False)
    monkeypatch.setattr(m, "EXTRA_SEED_ROOT", roots[2])
    monkeypatch.setattr(m, "ADDITIONAL_SEED_ROOT", roots[3])
    calls = []
    values = [({(0, 1)}, ["r322"]), ({(1, 2)}, ["current"]),
              ({(2, 3)}, ["r309"]), ({(3, 4)}, ["r318"])]

    def fake_collect(root, faces, archive_members, *, allow_missing_local_candidate=False):
        calls.append((root, allow_missing_local_candidate))
        return values[len(calls) - 1]

    monkeypatch.setattr(m, "collect_pairs", fake_collect)
    pairs, receipts = m.accumulate_pairs(np.zeros((5, 3), dtype=np.int64), {})
    assert calls == [(roots[0], False), (roots[1], True), (roots[2], False), (roots[3], False)]
    assert pairs == {(0, 1), (1, 2), (2, 3), (3, 4)}
    assert receipts == ["r322", "r309", "r318", "current"]


def test_support_scoped_unlock_allows_prefix_rows_inside_support_and_keeps_anchor():
    m = load_module()
    base = np.zeros((8, 3), dtype=np.float32)
    trial = base.copy()
    trial[1, 0] = np.nextafter(np.float32(0), np.float32(1))
    receipt = m.validate_fit_scope_arrays(base, trial, np.array([0, 1, 2], dtype=np.int64), np.array([0, 7], dtype=np.int64))
    assert receipt["changed_rows"] == 1
    assert receipt["outside_support_bitwise_unchanged"]
    assert receipt["typed_anchors_bitwise_unchanged"]
    assert m.FIT_FIXED_PREFIX == 0 and m.FIXED_PREFIX == 18430


def test_fit_scope_rejects_outside_support_change():
    m = load_module()
    base = np.zeros((8, 3), dtype=np.float32)
    trial = base.copy(); trial[6, 1] = 1e-7
    with pytest.raises(ValueError, match="outside its explicit LP support"):
        m.validate_fit_scope_arrays(base, trial, np.array([0, 1, 2]), np.array([7]))


def test_fit_scope_rejects_anchor_change():
    m = load_module()
    base = np.zeros((8, 3), dtype=np.float32)
    trial = base.copy(); trial[2, 0] = 1e-7
    with pytest.raises(ValueError, match="typed coarse anchor"):
        m.validate_fit_scope_arrays(base, trial, np.array([1, 2, 3]), np.array([2, 7]))


@pytest.mark.parametrize("support", [np.array([1, 1]), np.array([-1]), np.array([8]), np.array([1.0])])
def test_fit_scope_rejects_malformed_support(support):
    m = load_module()
    points = np.zeros((8, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        m.validate_fit_scope_arrays(points, points, support, np.array([7]))


def test_ensure_fit_scope_checks_anchor_identity_and_surface_topology(tmp_path):
    m = load_module()
    base = tmp_path / "base.npz"; trial = tmp_path / "trial.npz"
    points = np.zeros((6, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    anchors = np.array([0, 5], dtype=np.int64)
    np.savez_compressed(base, points_world_m=points, source_O_faces=faces,
                        typed_coarse_anchor_output_rows_018=anchors)
    moved = points.copy(); moved[1, 2] = 1e-7
    np.savez_compressed(trial, points_world_m=moved, source_O_faces=faces,
                        typed_coarse_anchor_output_rows_018=anchors)
    assert m.ensure_fit_scope(base, trial, np.array([1, 2]))["changed_rows"] == 1
    np.savez_compressed(trial, points_world_m=moved, source_O_faces=faces,
                        typed_coarse_anchor_output_rows_018=np.array([1, 5], dtype=np.int64))
    with pytest.raises(ValueError, match="anchor identity"):
        m.ensure_fit_scope(base, trial, np.array([1, 2]))
    np.savez_compressed(trial, points_world_m=moved, source_O_faces=np.array([[0, 2, 1]], dtype=np.int32),
                        typed_coarse_anchor_output_rows_018=anchors)
    with pytest.raises(ValueError, match="surface topology"):
        m.ensure_fit_scope(base, trial, np.array([1, 2]))
