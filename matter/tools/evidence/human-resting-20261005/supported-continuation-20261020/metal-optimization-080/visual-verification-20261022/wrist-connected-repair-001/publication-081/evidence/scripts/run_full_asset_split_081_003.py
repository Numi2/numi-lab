#!/usr/bin/env python3
"""Full-data geometry-only scalar split regression on frozen 084 mesh."""
from __future__ import annotations
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

D = Path("/Users/n/Work/numi-human-connected-surface-069")
TOOL_DIR = D / "publication-081/canonical/matter/tools"
sys.path.insert(0, str(TOOL_DIR))
from human_skin_fat_layer_reconstruction import (
    _clip_oriented_triangle,
    _pulling_polygon_triangles,
    split_tetrahedra_by_linear_scalar_field,
)

INPUT = D / "whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
OUTDIR = D / "publication-081/evidence/full084-split-003"
NORMAL = np.asarray([0.31, 0.47, 0.826], dtype=np.float64)
OFFSET_M = 1.23e-4
LOCAL_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
ALL_EDGE_PAIRS = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def det_quality(points, tets):
    d = np.empty(len(tets), dtype=np.float64)
    q = np.empty(len(tets), dtype=np.float64)
    for i, tet in enumerate(tets):
        p = points[tet].astype(np.float64)
        d[i] = np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0]))
        longest = max(float(np.linalg.norm(p[a]-p[b])) for a,b in ALL_EDGE_PAIRS)
        q[i] = d[i] / longest**3 if longest > 0 else np.nan
    return d, q


def face_rows(tets):
    counts = collections.Counter()
    oriented = {}
    for ci, tet in enumerate(tets):
        for loc in LOCAL_FACES:
            f = tuple(int(tet[i]) for i in loc)
            key = tuple(sorted(f))
            counts[key] += 1
            oriented.setdefault(key, (ci, f))
    return counts, oriented


def verify_boundary_conformity(points, source_tets, source_phi, out, gids):
    source_counts, source_oriented = face_rows(source_tets)
    if any(c > 2 for c in source_counts.values()):
        raise ValueError("source tetrahedral complex has nonmanifold face incidence")
    expected = set()
    key_to_row = {tuple(map(int, edge)): len(points) + i
                  for i, edge in enumerate(out.intersection_edge_global_ids)}
    def intersection(a, b):
        a, b = int(a), int(b)
        if source_phi[a] == 0.0:
            return a
        if source_phi[b] == 0.0:
            return b
        key = tuple(sorted((int(gids[a]), int(gids[b]))))
        return key_to_row[key]

    for key, incidence in source_counts.items():
        if incidence != 1:
            continue
        _, oriented = source_oriented[key]
        for side in (-1, 1):
            poly = _clip_oriented_triangle(oriented, source_phi, side, intersection)
            if len(poly) >= 3:
                for tri in _pulling_polygon_triangles(poly, out.global_vertex_order_keys):
                    expected.add(tuple(sorted(map(int, tri))))

    output_counts, _ = face_rows(out.tetrahedra)
    if any(c > 2 for c in output_counts.values()):
        raise ValueError("output tetrahedral complex has face incidence greater than two")
    emitted_boundary = {face for face, c in output_counts.items() if c == 1}
    missing = expected - emitted_boundary
    unexpected = emitted_boundary - expected
    return {
        "source_boundary_triangle_count": int(sum(c == 1 for c in source_counts.values())),
        "expected_clipped_boundary_triangle_count": int(len(expected)),
        "emitted_output_boundary_triangle_count": int(len(emitted_boundary)),
        "missing_expected_boundary_triangle_count": int(len(missing)),
        "unexpected_output_boundary_triangle_count": int(len(unexpected)),
        "missing_face_examples": [list(x) for x in sorted(missing)[:12]],
        "unexpected_face_examples": [list(x) for x in sorted(unexpected)[:12]],
        "output_max_face_incidence": int(max(output_counts.values(), default=0)),
        "pass": not missing and not unexpected and max(output_counts.values(), default=0) <= 2,
    }


def main():
    if OUTDIR.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUTDIR}")
    OUTDIR.mkdir(parents=True)
    start = time.time()
    with np.load(INPUT, allow_pickle=False) as z:
        R = np.asarray(z["reference_points_m"], dtype=np.float64)
        X32 = np.asarray(z["current037_points_m_f32"], dtype=np.float32)
        T = np.asarray(z["tetrahedra_zero_based"], dtype=np.int64)
        mats = np.asarray(z["material_ids"], dtype=np.int64) if "material_ids" in z.files else None
        gids = np.asarray(z["output_node_ids"], dtype=np.int64)
        source_input_rows = np.asarray(z["source_input_row"], dtype=np.int64)
        source_cell_uids = np.asarray(z["cell_uids"], dtype=np.int64) if "cell_uids" in z.files else None
        input_keys = list(z.files)
    if R.shape != (7360, 3) or X32.shape != R.shape or T.shape != (32796, 4):
        raise ValueError(f"unexpected compact 084 shapes: R={R.shape} X={X32.shape} T={T.shape}")
    if gids.shape != (len(R),) or source_input_rows.shape != (len(R),):
        raise ValueError("source row identity arrays do not align with compact point rows")
    if len(np.unique(gids)) != len(gids):
        raise ValueError("output node identity labels are not unique")
    if mats is not None and mats.shape != (len(T),):
        raise ValueError("material IDs are misaligned with source tetrahedra")

    n = NORMAL / np.linalg.norm(NORMAL)
    projection = R @ n
    threshold0 = float(np.median(projection) + OFFSET_M)
    threshold = threshold0
    ulp_steps = 0
    phi = projection - threshold
    while np.any(phi == 0.0) and ulp_steps < 32:
        threshold = float(np.nextafter(threshold, np.inf))
        ulp_steps += 1
        phi = projection - threshold
    if np.any(phi == 0.0):
        raise ValueError("could not choose noncoincident deterministic plane threshold")

    split = split_tetrahedra_by_linear_scalar_field(
        R, T, phi, global_vertex_ids=gids, material_ids=mats)
    # Verify that the result is a conforming subdivision of the whole source
    # boundary, not merely a collection of independently valid parent cells.
    conformity = verify_boundary_conformity(R, T, phi, split, gids)
    if not conformity["pass"]:
        raise ValueError(f"whole-boundary conformity failed: {conformity}")

    W = split.interpolation_weights
    current_child64 = np.asarray(W @ X32.astype(np.float64), dtype=np.float64)
    current_child32 = current_child64.astype(np.float32)
    det_current32, q_current32 = det_quality(current_child32, split.tetrahedra)
    ref_det64, _ = det_quality(split.points, split.tetrahedra)
    ref_det32, q_ref32 = det_quality(split.points_float32, split.tetrahedra)
    if np.any(ref_det64 <= 0.0) or np.any(ref_det32 <= 0.0) or np.any(det_current32 <= 0.0):
        raise ValueError("split emitted nonpositive reference or mapped-current cells")

    child_ref_vol = ref_det64 / 6.0
    source_ref_det, _ = det_quality(R, T)
    parent_ref_vol = source_ref_det / 6.0
    ref_parent_volumes = np.bincount(split.parent_cell_rows, weights=child_ref_vol, minlength=len(T))
    ref_parent_abs = np.abs(ref_parent_volumes - parent_ref_vol)
    current_parent_det64, _ = det_quality(X32.astype(np.float64), T)
    current_parent_vol = current_parent_det64 / 6.0
    current_child_det64, _ = det_quality(current_child64, split.tetrahedra)
    if np.any(current_parent_det64 <= 0.0) or np.any(current_child_det64 <= 0.0):
        raise ValueError("source or split current037 Float64 geometry has nonpositive cells")
    current_child_vol64 = current_child_det64 / 6.0
    current_child_sums64 = np.bincount(split.parent_cell_rows, weights=current_child_vol64, minlength=len(T))
    current_parent_abs64 = np.abs(current_child_sums64 - current_parent_vol)
    current_parent_rel64 = current_parent_abs64 / np.maximum(np.abs(current_parent_vol), np.finfo(float).tiny)
    # Float32 vertex rounding perturbs summed child volumes independently.
    # Keep this as a measured rounding diagnostic; positivity remains a strict
    # gate above and the exact mapped-double partition is the conservation gate.
    current_child_vol32 = det_current32.astype(np.float64) / 6.0
    current_child_sums32 = np.bincount(split.parent_cell_rows, weights=current_child_vol32, minlength=len(T))
    current_parent_abs32 = np.abs(current_child_sums32 - current_parent_vol)
    current_parent_rel32 = current_parent_abs32 / np.maximum(np.abs(current_parent_vol), np.finfo(float).tiny)
    if np.max(ref_parent_abs / parent_ref_vol, initial=0.0) > 1e-11:
        raise ValueError("reference child volumes do not sum to their source parents")
    if np.max(current_parent_rel64, initial=0.0) > 1e-11:
        raise ValueError("mapped-current Float64 child volumes do not sum to their source parents")

    scalar_rows = np.asarray(W @ phi).reshape(-1)
    if np.max(np.abs(scalar_rows - split.scalar_values), initial=0.0) > 1e-13 * max(float(np.max(np.abs(phi))), 1.0):
        raise ValueError("sparse interpolation does not reproduce split scalar values")
    if np.max(np.abs(np.asarray(W.sum(axis=1)).ravel() - 1.0), initial=0.0) > 1e-13:
        raise ValueError("interpolation matrix row sums are not one")
    side_scalar = split.scalar_values[split.tetrahedra] * split.side_signs[:, None]
    scalar_tol = 256 * np.finfo(float).eps * float(np.max(np.abs(phi)))
    if np.any(side_scalar < -scalar_tol):
        raise ValueError("output cells cross their assigned scalar half-domain")

    material_volume = {}
    if mats is not None:
        child_mats = split.material_ids
        for mid in np.unique(mats):
            mask0 = mats == mid
            mask1 = child_mats == mid
            before = float(parent_ref_vol[mask0].sum())
            after = float(child_ref_vol[mask1].sum())
            current_before = float(current_parent_vol[mask0].sum())
            current_after = float(current_child_vol64[mask1].sum())
            material_volume[str(int(mid))] = {
                "source_reference_volume_m3": before,
                "child_reference_volume_m3": after,
                "reference_absolute_difference_m3": abs(after-before),
                "reference_relative_difference": abs(after-before) / before,
                "source_current037_volume_m3": current_before,
                "child_current037_volume_m3": current_after,
                "current037_absolute_difference_m3": abs(current_after-current_before),
                "current037_relative_difference": abs(current_after-current_before) / current_before,
            }
    out_npz = OUTDIR / "scalar-plane-split-084.npz"
    np.savez_compressed(
        out_npz,
        points_reference_m=split.points,
        points_reference_f32=split.points_float32,
        points_current037_f32=current_child32,
        tetrahedra=split.tetrahedra,
        parent_cell_rows=split.parent_cell_rows,
        side_signs=split.side_signs,
        material_ids=np.asarray([] if split.material_ids is None else split.material_ids),
        source_global_vertex_ids=gids,
        source_input_row=source_input_rows,
        source_cell_uids=np.asarray([] if source_cell_uids is None else source_cell_uids),
        intersection_edge_global_ids=split.intersection_edge_global_ids,
        scalar_values=split.scalar_values,
        interpolation_weights_data=W.data,
        interpolation_weights_indices=W.indices,
        interpolation_weights_indptr=W.indptr,
        interpolation_weights_shape=np.asarray(W.shape,dtype=np.int64),
    )
    report = {
        "status":"PASS_STATIC_FULL_ASSET_GEOMETRY_REGRESSION",
        "physical_or_native_admission":False,
        "input_npz":str(INPUT),"input_sha256":sha256(INPUT),"input_keys":input_keys,
        "splitter_source":str(TOOL_DIR/"human_skin_fat_layer_reconstruction.py"),
        "splitter_source_sha256":sha256(TOOL_DIR/"human_skin_fat_layer_reconstruction.py"),
        "test_source_sha256":sha256(TOOL_DIR/"test_human_skin_fat_layer_reconstruction.py"),
        "scalar_field_definition":{
            "kind":"nonphysiological deterministic plane used only as a splitter regression",
            "normal_raw":[0.31,0.47,0.826],"normal_unit":n.tolist(),
            "projection_median_m":float(np.median(projection)),"offset_m":OFFSET_M,
            "threshold_m":threshold,"threshold_ulp_steps_to_avoid_exact_zero":ulp_steps,
            "strict_zero_vertex_count":int(np.count_nonzero(phi==0.0)),
            "negative_source_vertex_count":int(np.count_nonzero(phi<0.0)),
            "positive_source_vertex_count":int(np.count_nonzero(phi>0.0)),
            "scalar_min":float(phi.min()),"scalar_max":float(phi.max()),
        },
        "source_counts":{"vertices":len(R),"tetrahedra":len(T),"materials":None if mats is None else {str(int(k)):int(v) for k,v in zip(*np.unique(mats,return_counts=True))}},
        "split_diagnostics":split.diagnostics,
        "whole_boundary_conformity":conformity,
        "reference_volume_partition":{
            "max_parent_absolute_error_m3":float(np.max(ref_parent_abs,initial=0.0)),
            "max_parent_relative_error":float(np.max(ref_parent_abs/parent_ref_vol,initial=0.0)),
            "per_material":material_volume,
        },
        "mapped_current037_float64_partition":{
            "max_parent_absolute_error_m3":float(np.max(current_parent_abs64,initial=0.0)),
            "max_parent_relative_error":float(np.max(current_parent_rel64,initial=0.0)),
            "gate_relative_tolerance":1e-11,
        },
        "mapped_current037_float32_volume_rounding_diagnostic":{
            "max_parent_absolute_difference_m3":float(np.max(current_parent_abs32,initial=0.0)),
            "max_parent_relative_difference":float(np.max(current_parent_rel32,initial=0.0)),
            "worst_parent_row":int(np.argmax(current_parent_rel32)),
            "strict_gate":"all emitted Float32 child determinants positive; exact split partition checked before Float32 rounding",
        },
        "float32_reference_quality":{"nonpositive_count":int(np.count_nonzero(ref_det32<=0)),"min_det_m3_times6":float(ref_det32.min()),"min_det_over_longest_edge_cubed":float(q_ref32.min())},
        "float32_current037_quality":{"nonpositive_count":int(np.count_nonzero(det_current32<=0)),"min_det_m3_times6":float(det_current32.min()),"min_det_over_longest_edge_cubed":float(q_current32.min())},
        "mass_inputs_present":False,
        "output_npz":str(out_npz),"output_sha256":sha256(out_npz),"output_bytes":out_npz.stat().st_size,
        "elapsed_seconds":time.time()-start,
        "scope_limit":"This checks a nonphysiological planar cut of the frozen 084 tetrahedral mesh; it does not construct skin/fat layers, establish tissue semantics, or qualify physics.",
    }
    report_path=OUTDIR/"report.json"
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"output":str(out_npz),"output_sha256":report["output_sha256"],"vertices":split.diagnostics["output_vertex_count"],"tetrahedra":split.diagnostics["output_tetrahedron_count"],"intersections":split.diagnostics["intersection_vertex_count"],"boundary":conformity,"min_reference_q":float(q_ref32.min()),"min_current_q":float(q_current32.min()),"elapsed_seconds":report["elapsed_seconds"]},indent=2))

if __name__ == "__main__":
    main()
