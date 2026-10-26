import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).with_name("capture_raw_lp_directions546.py")
SPEC = importlib.util.spec_from_file_location("raw_lp_capture546", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def _basis(points, scope, anchors=(0,), controls=(1,)):
    return M.capture_basis_snapshot(
        {"points_world_m": np.asarray(points, dtype=np.float32)},
        np.asarray(scope, dtype=np.int64),
        np.asarray(anchors, dtype=np.int64),
        np.asarray(controls, dtype=np.int64),
    )


def test_solver_success_delta_is_copied_and_original_result_returned():
    points = np.zeros((4, 3), np.float32)
    basis = _basis(points, [0, 1, 2], anchors=[0], controls=[1, 2])
    delta = np.array([0, 0, 0, 0.5, 0, 0, 0, 1.25, 0], np.float64)
    result = {"success": True, "status": 0, "delta": delta, "message": "ok"}
    records = []
    returned = M.capture_successful_solver_result(result, basis, records)
    assert returned is result
    assert records[0]["delta_captured"] is True
    assert records[0]["delta_mm"] is not delta
    assert np.array_equal(records[0]["delta_mm"], delta)
    delta[3] = 99
    assert records[0]["delta_mm"][3] == 0.5
    assert result["delta"][3] == 99


def test_failed_solver_result_never_captures_or_emits_delta():
    result = {"success": False, "status": 1, "delta": np.ones(6), "message": "timeout"}
    records = []
    returned = M.capture_successful_solver_result(result, None, records)
    assert returned is result
    assert len(records) == 1
    assert records[0]["delta_captured"] is False
    assert "delta_mm" not in records[0]


def test_success_without_basis_or_with_wrong_scope_shape_is_not_emitted():
    records = []
    result = {"success": True, "status": 0, "delta": np.zeros(3)}
    M.capture_successful_solver_result(result, None, records)
    assert records[-1]["delta_captured"] is False
    basis = _basis(np.zeros((3, 3), np.float32), [0, 1], anchors=[0], controls=[1])
    M.capture_successful_solver_result(result, basis, records)
    assert records[-1]["status"] == "SUCCESSFUL_SOLVER_RETURN_WITH_MALFORMED_DELTA"
    assert records[-1]["delta_captured"] is False


def test_delta_units_are_mm_and_emission_is_f32():
    base = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=np.float32)
    delta = np.array([0., 0., 0., 1000., -500., 250.])
    points, free, anchor_delta = M.emitted_raw_points(base, np.array([0, 1]), np.array([0]), delta)
    assert anchor_delta == 0.0
    assert points.dtype == np.float32
    assert points[1].tolist() == [2.0, 1.5, 3.25]
    assert np.array_equal(free, [1])


def test_outside_scope_and_anchor_coordinates_remain_bitwise_unchanged():
    base = np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.], [9., 10., 11.]], dtype=np.float32)
    delta = np.array([0., 0., 0., .125, -.25, .5])
    points, free, anchor_delta = M.emitted_raw_points(base, [0, 2], [0], delta)
    assert np.array_equal(points[[0, 1, 3]].view(np.uint32), base[[0, 1, 3]].view(np.uint32))
    assert free.tolist() == [2]
    assert anchor_delta == 0.0


def test_anchor_roundoff_is_measured_but_owner_resets_anchor_exactly():
    base = np.array([[1., 0., 0.], [0., 0., 0.]], np.float32)
    delta = np.array([1.e-6, 0., 0., 0., 0., 0.])
    points, free, anchor_delta = M.emitted_raw_points(base, [0, 1], [0], delta)
    assert anchor_delta == 1.e-6
    assert np.array_equal(points[0].view(np.uint32), base[0].view(np.uint32))
    assert free.tolist() == [1]


def test_emission_matches_pinned_408_owner_including_half_ulp_case():
    owner_path=M.OWNER_408
    spec=importlib.util.spec_from_file_location("owner408_emission_parity", owner_path)
    owner=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=owner
    spec.loader.exec_module(owner)
    base=np.array([[1.,0.,0.],[0.125,2.,-1.],[.75,.5,.25]],dtype=np.float32)
    scope=np.array([0,1,2],dtype=np.int64)
    anchors=np.array([0],dtype=np.int64)
    half_ulp_mm=float(np.spacing(np.float32(1.0)))*500.0
    delta=np.array([[0.,0.,0.],[half_ulp_mm,-half_ulp_mm,0.],[.000059604644775390625,0.,0.]])
    actual,free,anchor_delta=M.emitted_raw_points(base,scope,anchors,delta.reshape(-1))
    expected=owner.emit_raw_lp_points(base,base.astype(np.float64)*1000.0,scope,delta,anchors)
    assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert free.tolist()==[1,2]
    assert anchor_delta==0.0


@pytest.mark.parametrize("scope", [[1, 1], [-1], [3], [[0, 1]]])
def test_invalid_scope_identity_is_rejected(scope):
    base = np.zeros((3, 3), np.float32)
    with pytest.raises(ValueError):
        M.emitted_raw_points(base, scope, [], np.zeros(3 * (len(scope) if np.asarray(scope).ndim == 1 else 1)))


def test_candidate_payload_preserves_base_provenance_arrays():
    base = {"points_world_m": np.zeros((2, 3), np.float32),
            "source_O_faces": np.array([[0, 1, 1]], np.int64),
            "typed_marker": np.array([4, 9], np.int32)}
    points = np.ones((2, 3), np.float32)
    payload = M.candidate_payload_with_raw_points(base, points)
    assert np.array_equal(payload["points_world_m"], points)
    assert np.array_equal(payload["typed_marker"], base["typed_marker"])
    assert np.array_equal(payload["source_O_faces"], base["source_O_faces"])


def test_frozen_fit_hook_source_pins_and_exact_hook_locations():
    source = M._load_original_fit_source()
    assert M.sha(M.FIT) == M.FIT_SHA256
    assert "CAPTURE[\"basis_snapshots\"].append" in source
    assert "original_active_solve=active.solve_by_components" in source
    assert "m.solve_by_components=active.solve_by_components" in source
    assert M.sha(M.OWNER_408) == M.OWNER_408_SHA256
    compile(source, str(M.FIT), "exec")



def test_each_candidate_subdir_has_the_exact_545_projector_report_contract(tmp_path):
    base_path = tmp_path / "base.npz"
    points = np.array([[0., 0., 0.], [1., 2., 3.], [4., 5., 6.]], np.float32)
    np.savez_compressed(base_path, points_world_m=points,
                        source_O_faces=np.array([[0, 1, 2]], np.int64),
                        typed_coarse_anchor_output_rows_018=np.array([0], np.int64))
    base_sha = M.sha(base_path)
    patch_dir = tmp_path / "raw-direction-001"
    patch_dir.mkdir()
    emitted, free, anchor_delta = M.emitted_raw_points(
        points, [0, 1, 2], [0], np.array([0., 0., 0., .5, 0., 0., 0., .25, 0.]))
    assert anchor_delta == 0.0
    payload = M.candidate_payload_with_raw_points(
        {"points_world_m": points, "source_O_faces": np.array([[0, 1, 2]], np.int64)},
        emitted)
    candidate_path = patch_dir / "candidate.npz"
    np.savez_compressed(candidate_path, **payload)
    report_path = patch_dir / "report.json"
    patch_report = M.make_545_patch_report(
        base_path=base_path, base_sha256=base_sha, free_nodes=free,
        candidate_path=candidate_path, delta_mm=np.arange(9, dtype=np.float64),
        solver_call_index=1)
    report_path.write_text(__import__("json").dumps(patch_report))
    # Exercise the same reads performed by the real 545 projector.
    report = __import__("json").loads(report_path.read_text())
    assert M.sha(Path(report["base_candidate"])) == report["base_sha256"]
    z = np.load(candidate_path, allow_pickle=False)
    assert z["points_world_m"].dtype == np.float32
    free_rows = np.asarray(report["free_nodes"], dtype=np.int64)
    anchors = np.load(base_path, allow_pickle=False)["typed_coarse_anchor_output_rows_018"]
    assert not np.intersect1d(free_rows, anchors).size
    assert patch_report["native_admission"] is False
    assert patch_report["accepted_physical_steps"] == 0


def test_allowlisted_entry_choice_and_report_provenance():
    bary=M.fit_entry_provenance("barycentric")
    plane=M.fit_entry_provenance("full-plane")
    assert bary["fit_entry_path"]==str(M.FIT_BARY)
    assert bary["fit_entry_sha256"]==M.FIT_BARY_SHA256
    assert bary["fit_objective_contract"]=="all matched 408 barycentric face targets"
    assert plane["fit_entry_path"]==str(M.FIT_FULL_PLANE)
    assert plane["fit_entry_sha256"]=="9899e952efc24ef26cd82462fbe534d7abe3db7d3176be08c7486d9059c93446"
    assert plane["fit_objective_contract"]=="existing 458 positive-corner plane soft objective"
    with pytest.raises(ValueError, match="fit mode"):
        M.fit_entry_provenance("unknown")


@pytest.mark.parametrize("mode", ["barycentric", "full-plane"])
def test_both_frozen_entries_accept_capture_hooks_and_compile(mode):
    path,digest,_=M.FIT_ENTRIES[mode]
    source=M._load_original_fit_source(path,digest)
    assert M.sha(path)==digest
    assert source.count("original_active_solve=active.solve_by_components")==1
    assert source.count("CAPTURE[\"basis_snapshots\"].append")==1
    assert source.count("m.solve_by_components=active.solve_by_components")==1
    compile(source,str(path),"exec")


def test_final_capture_report_records_selected_full_plane_entry(tmp_path, monkeypatch):
    base_path=tmp_path/"base.npz"
    np.savez_compressed(
        base_path,
        points_world_m=np.zeros((2,3),np.float32),
        source_O_faces=np.array([[0,1,1]],np.int64),
        typed_coarse_anchor_output_rows_018=np.array([0],np.int64),
    )
    monkeypatch.setattr(sys,"argv",["capture-wrapper","--output",str(tmp_path/"fit-output")])
    report_path,report=M._finalize_capture(
        tmp_path/"capture-full-plane",base_path,[],
        {"solver_return_count":0,"basis_snapshots":[]},
        fit_entry_mode="full-plane",
    )
    saved=__import__("json").loads(report_path.read_text())
    assert saved["fit_entry_mode"]=="full-plane"
    assert saved["fit_entry_path"]==str(M.FIT_FULL_PLANE)
    assert saved["fit_entry_sha256"]==M.FIT_FULL_PLANE_SHA256
    assert saved["fit_objective_contract"]=="existing 458 positive-corner plane soft objective"
    assert saved["status"]=="NO_SUCCESSFUL_LP_DELTA_CAPTURED"
    assert saved["native_admission"] is False
    assert report["accepted_physical_steps"]==0
