from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "retessellate_volume_cavities_001.py"
SPEC = importlib.util.spec_from_file_location("volume_cavity", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_all_six_prism_staircases_are_positive_closed_and_equal_volume():
    tri = np.array([0, 1, 2], dtype=np.int64)
    X = np.zeros((3 * MOD.N, 3), dtype=np.float64)
    X[[0, 1, 2]] = [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 0.9, 0.1]]
    X[MOD.N + np.array([0, 1, 2])] = [[0.1, 0.0, 1.1], [1.3, 0.1, 1.1], [0.3, 0.9, 1.2]]
    expected_caps = {
        tuple(sorted((0, 1, 2))),
        tuple(sorted((MOD.N, MOD.N + 1, MOD.N + 2))),
    }
    vols = []
    for perm in itertools.permutations(range(3)):
        raw = MOD.prism_decomposition(tri, 0, perm)
        T = MOD.orient_tets_positive(raw, X)
        assert np.all(MOD.dets(X, T) > 0.0)
        boundary = MOD.oriented_boundary(T, X)
        assert len(boundary) == 8
        cap_keys = {k for k in boundary if len({v // MOD.N for v in k}) == 1}
        assert cap_keys == expected_caps
        cycle = [0, 1, 2]
        for edge_index, u in enumerate(cycle):
            v = cycle[(edge_index + 1) % len(cycle)]
            quad = {u, v, MOD.N + u, MOD.N + v}
            hits = [k for k in boundary if set(k).issubset(quad)]
            assert len(hits) == 2
            assert sum(boundary[k] for k in hits) == 0
        vols.append(float(MOD.dets(X, T).sum() / 6.0))
    assert np.ptp(vols) < 1e-14


def test_frozen_oid_cavity_candidate_preserves_oriented_boundary_and_outside_rows():
    out = HERE / "retessellation-001/volume-cavity-001"
    payload = np.load(out / "oid-cavity-replacement.npz")
    report = json.loads((out / "report.json").read_text())
    old = payload["old_tetrahedra"].astype(np.int64)
    new = payload["candidate_tetrahedra"].astype(np.int64)
    oldF = payload["old_O_faces"].astype(np.int64)
    newF = payload["candidate_O_faces"].astype(np.int64)
    MOD.old_faces_array = oldF
    changed = payload["changed_tet_rows"].astype(np.int64)
    assert np.array_equal(changed, np.unique(changed))
    assert len(changed) == 72
    assert np.array_equal(newF, payload["candidate_I_faces"] - MOD.N)
    assert np.array_equal(newF, payload["candidate_D_faces"] - 2 * MOD.N)
    Xref = payload["reference_points_AP_m"]
    outside = np.setdiff1d(np.arange(len(old)), changed)
    assert np.array_equal(new[outside], old[outside])
    assert np.all(MOD.dets(Xref, new) > 0.0)

    cell_map = json.loads(str(payload["replacement_cell_map_json"]))
    assert len(cell_map) == len(changed)
    assert {r["output_tet_row"] for r in cell_map} == set(changed.tolist())
    assert all(r["inherited_old_tet_row"] == r["output_tet_row"] for r in cell_map)

    for label, face_ids in MOD.PATCHES.items():
        oldpatch = oldF[np.asarray(face_ids, dtype=np.int64)]
        newpatch = newF[np.asarray(face_ids, dtype=np.int64)]
        cycle, _ = MOD.directed_boundary(oldpatch)
        for layer in (0, 1):
            rows, _ = MOD.cavity_cells(old, face_ids, layer)
            old_boundary = MOD.oriented_boundary(old[rows], Xref)
            new_boundary = MOD.oriented_boundary(new[rows], Xref)
            old_side = MOD.side_boundary(old_boundary, cycle, layer)
            assert MOD.side_boundary(new_boundary, cycle, layer) == old_side
            assert len(rows) == 12
            assert report["patches"][label]["layers"][str(layer)]["old_tet_rows"] == rows
            # Exact side faces and their orientations survive; only the two
            # cap triangulations are replaced by the new O/I/D face rows.
            assert set(old_boundary) == set(old_side) | {
                tuple(sorted(map(int, tri + k * MOD.N)))
                for tri in oldpatch for k in (layer, layer + 1)
            }
            assert set(new_boundary) == set(old_side) | {
                tuple(sorted(map(int, tri + k * MOD.N)))
                for tri in newpatch for k in (layer, layer + 1)
            }
            assert MOD.cap_orientation(new_boundary, newpatch, layer) == MOD.cap_orientation(old_boundary, oldpatch, layer)
            assert MOD.cap_orientation(new_boundary, newpatch, layer + 1) == MOD.cap_orientation(old_boundary, oldpatch, layer + 1)


def test_candidate_keeps_native_material_rows_and_reports_current_failure():
    out = HERE / "retessellation-001/volume-cavity-001"
    payload = np.load(out / "oid-cavity-replacement.npz")
    report = json.loads((out / "report.json").read_text())
    oldmat = payload["old_material_ids"]
    newmat = payload["candidate_material_ids"]
    changed = payload["changed_tet_rows"]
    assert np.array_equal(oldmat, newmat)
    outside = np.setdiff1d(np.arange(len(oldmat)), changed)
    assert np.array_equal(payload["old_tetrahedra"][outside], payload["candidate_tetrahedra"][outside])
    gates = report["full_volume_gates"]
    assert gates["reference_candidate_nonpositive"] == 0
    assert gates["current_candidate_nonpositive_with_target_O_fixed_ID"] == 126
    assert gates["outside_cavity_old_connectivity_bitwise_unchanged"] is True

