#!/usr/bin/env python3
"""One-face-patch diagnostic for the pinned 30886 fixed-037 degeneracy.

This is a construction proposal only. It does not waive 541 or run full 130/159
checks. Only the existing 402/405 two-face flip operation is considered.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069/D")
OWNER402 = D / "local-two-parent-retriangulation-402/two_parent_patch_flips_402.py"
OWNER405 = D / "local-two-parent-retriangulation-405/cross_seam_search_405.py"
OWNER541 = D / "focused-clearance-541/topology/repair_focused_clearance_541.py"
EXPECTED_402 = "f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764"
EXPECTED_405 = "e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967"
EXPECTED_541 = "26930be05518834f02724948cffa5ff74719f070a4ac4e4917b0965f42a412aa"
EXPECTED_BASE = "5e5db207c5bf4ceecc11553817194bfc49e28d772af2fb9ff123fc481497ea7f"
PARENT_CANDIDATE = Path("/Users/numi/Work/numi-human-connected-surface-069/publication-108/evidence/outward-surface-resume-264/pass-001/candidate.npz")
PARENT_SHA = "5a7239e62cbc0cf461440a48a8c4747d7d733eec630d63cf9e08682009593cc6"
CANONICAL_037 = Path("/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-037-local-refinement-006-prep/run-001/refined-candidate.npz")
CANONICAL_037_SHA = "0853407a1ce7fb9aed2d91a51ea712f8a85dadfcfb4b048e29979453dabab4dc"
WITNESS_NODES = frozenset((18627, 20508, 15020))
WITNESS_DETAIL = 88498
WITNESS_SUBFACE = 88506
MAX_OLD_REFERENCE_CROSS_MM2 = 1.0e-12
MIN_REFERENCE_AREA_GAIN = 1.0e6
MIN_CURRENT_ANGLE_DEG = 5.0
MIN_AREA_RATIO = 0.1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load owner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_candidate(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}


def arrays_bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.asarray(a)
    bb = np.asarray(b)
    return (
        aa.dtype == bb.dtype
        and aa.shape == bb.shape
        and np.ascontiguousarray(aa).tobytes() == np.ascontiguousarray(bb).tobytes()
    )


def cross_norm(points_mm: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = points_mm[np.asarray(tri, dtype=np.int64)]
    return float(np.linalg.norm(np.cross(b - a, c - a)))


def edge_set(faces: np.ndarray) -> set[tuple[int, int]]:
    return {
        (min(int(a), int(b)), max(int(a), int(b)))
        for tri in np.asarray(faces)
        for a, b in zip(tri, np.roll(tri, -1))
    }


def boundary_edges(faces: np.ndarray) -> set[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for tri in np.asarray(faces):
        for a, b in zip(tri, np.roll(tri, -1)):
            counts[(min(int(a), int(b)), max(int(a), int(b)))] += 1
    return {edge for edge, count in counts.items() if count == 1}


def profile_rejections(
    old_reference_cross_mm2: float,
    new_reference_cross_mm2: np.ndarray,
    normal_cos: np.ndarray,
    current_min_angle_deg: np.ndarray,
    area_ratio: np.ndarray,
) -> tuple[list[str], dict]:
    old_area = float(old_reference_cross_mm2)
    new_area = np.asarray(new_reference_cross_mm2, dtype=np.float64)
    cos = np.asarray(normal_cos, dtype=np.float64)
    angle = np.asarray(current_min_angle_deg, dtype=np.float64)
    ratio = np.asarray(area_ratio, dtype=np.float64)
    if not len(new_area) or not (len(new_area) == len(cos) == len(angle) == len(ratio)):
        raise ValueError("profile metrics must be nonempty and have matching lengths")
    gain = float(np.min(new_area) / old_area) if old_area > 0 else float("inf")
    reasons = []
    if not np.isfinite(old_area) or old_area <= 0 or old_area > MAX_OLD_REFERENCE_CROSS_MM2:
        reasons.append("old_reference_not_near_collinear")
    if not np.isfinite(new_area).all() or np.min(new_area) <= 0 or gain < MIN_REFERENCE_AREA_GAIN:
        reasons.append("reference_area_gain_below_1e6")
    if not np.isfinite(cos).all() or np.min(cos) <= 0:
        reasons.append("current_reference_orientation_invalid")
    if not np.isfinite(angle).all() or np.min(angle) < MIN_CURRENT_ANGLE_DEG:
        reasons.append("minimum_current_angle_below_5deg")
    if not np.isfinite(ratio).all() or np.min(ratio) < MIN_AREA_RATIO:
        reasons.append("037_area_ratio_below_0.1")
    metrics = {
        "old_target_reference_cross_norm_mm2": old_area,
        "new_reference_cross_norms_mm2": new_area.tolist(),
        "minimum_reference_area_gain": gain,
        "new_normal_cosines": cos.tolist(),
        "new_current_min_angles_deg": angle.tolist(),
        "new_area_ratios": ratio.tolist(),
    }
    return reasons, metrics


def csr_rows(owner405, candidate: dict, ptr_key: str, values_key: str, n: int):
    return owner405.csr_rows(candidate[ptr_key], candidate[values_key], n)


def check_csr_union(owner405, before: dict, after: dict, pair: tuple[int, int], schemas: list[tuple[str, str]]) -> dict:
    n = len(before["source_O_faces"])
    i, j = pair
    checked = {}
    for ptr_key, values_key in schemas:
        old_rows = csr_rows(owner405, before, ptr_key, values_key, n)
        new_rows = csr_rows(owner405, after, ptr_key, values_key, n)
        expected = sorted(set(old_rows[i]) | set(old_rows[j]))
        if new_rows[i] != expected or new_rows[j] != expected:
            raise ValueError(f"{values_key}: changed rows do not contain exact ancestry union")
        for row in range(n):
            if row not in pair and new_rows[row] != old_rows[row]:
                raise ValueError(f"{values_key}: unrelated row {row} changed")
        if any(v < 0 for v in expected):
            raise ValueError(f"{values_key}: negative ID in ancestry union")
        checked[values_key] = expected
    return checked


def derive_affine_face_barycentrics(current: dict, parent: dict, canonical: dict, owner405) -> tuple[np.ndarray, dict]:
    """Derive face-vertex weights from pinned parent connectivity and immediate Q."""
    parent_faces = np.asarray(parent["source_O_faces"], dtype=np.int64)
    canonical_faces = np.asarray(canonical["source_O_faces"], dtype=np.int64)
    if not np.array_equal(parent_faces, canonical_faces):
        raise ValueError("pinned parent and canonical-037 connectivity differ")
    nparent = len(parent["points_world_m"])
    if len(parent_faces) != len(canonical_faces) or len(canonical["points_world_m"]) != nparent:
        raise ValueError("pinned parent/canonical037 dimensions differ")

    # Verify the archived immediate-parent P map and current Q@P map before using them.
    for suffix in ("indptr", "node_rows", "weights"):
        if not np.array_equal(
            parent["source_prolongation_" + suffix],
            current["prior_input_parent__source_prolongation_" + suffix],
        ):
            raise ValueError(f"current archived parent P {suffix} differs from pinned parent")
    qptr = np.asarray(current["immediate_refinement_prolongation_indptr"], dtype=np.int64)
    qrows = np.asarray(current["immediate_refinement_prolongation_node_rows"], dtype=np.int64)
    qweights = np.asarray(current["immediate_refinement_prolongation_weights"], dtype=np.float64)
    pptr = np.asarray(parent["source_prolongation_indptr"], dtype=np.int64)
    prows = np.asarray(parent["source_prolongation_node_rows"], dtype=np.int64)
    pweights = np.asarray(parent["source_prolongation_weights"], dtype=np.float64)
    gptr = np.asarray(current["source_prolongation_indptr"], dtype=np.int64)
    grows = np.asarray(current["source_prolongation_node_rows"], dtype=np.int64)
    gweights = np.asarray(current["source_prolongation_weights"], dtype=np.float64)
    ncurrent = len(current["points_world_m"])
    for ptr, rows, weights, n, label in (
        (qptr, qrows, qweights, ncurrent, "Q"),
        (pptr, prows, pweights, nparent, "P"),
        (gptr, grows, gweights, ncurrent, "Q@P"),
    ):
        if ptr.shape != (n + 1,) or int(ptr[0]) != 0 or int(ptr[-1]) != len(rows) or len(rows) != len(weights) or np.any(ptr[1:] < ptr[:-1]):
            raise ValueError(f"invalid {label} CSR")
        if np.any(rows < 0) or not np.isfinite(weights).all():
            raise ValueError(f"invalid {label} entries")

    parent_xyz = np.asarray(canonical["points_world_m"], dtype=np.float64)
    current_ref = np.asarray(current["area_reference_037_points_world_m_f64"], dtype=np.float64)
    max_source_error = 0.0
    max_reference_point_error = 0.0
    qmaps = []
    for node in range(ncurrent):
        qa, qb = int(qptr[node]), int(qptr[node + 1])
        qmap = {int(qrows[k]): float(qweights[k]) for k in range(qa, qb)}
        if not qmap or not np.isclose(sum(qmap.values()), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(f"Q row {node} is empty or does not preserve affine weights")
        qmaps.append(qmap)
        point = sum((w * parent_xyz[pn] for pn, w in qmap.items()), np.zeros(3, dtype=np.float64))
        max_reference_point_error = max(max_reference_point_error, float(np.max(np.abs(point - current_ref[node]))))
        composed = {}
        for parent_node, qweight in qmap.items():
            if parent_node < 0 or parent_node >= nparent:
                raise ValueError(f"Q row {node} references invalid parent node {parent_node}")
            pa, pb = int(pptr[parent_node]), int(pptr[parent_node + 1])
            for k in range(pa, pb):
                source_node = int(prows[k])
                composed[source_node] = composed.get(source_node, 0.0) + qweight * float(pweights[k])
        ga, gb = int(gptr[node]), int(gptr[node + 1])
        actual = {int(grows[k]): float(gweights[k]) for k in range(ga, gb)}
        keys = set(composed) | set(actual)
        error = max((abs(composed.get(k, 0.0) - actual.get(k, 0.0)) for k in keys), default=0.0)
        max_source_error = max(max_source_error, error)
    if max_source_error > 1e-12:
        raise ValueError(f"Q@P does not reproduce global source CSR: {max_source_error}")
    if max_reference_point_error > 1e-12:
        raise ValueError(f"Q does not reproduce matched canonical037 points: {max_reference_point_error}")

    parent_ptr = np.asarray(current["refinement_face_parent_ancestry_indptr"], dtype=np.int64)
    parent_rows = np.asarray(current["refinement_face_parent_ancestry_face_rows"], dtype=np.int64)
    scalar_parent = np.asarray(current["refinement_face_parent_face_row"], dtype=np.int64)
    faces = np.asarray(current["source_O_faces"], dtype=np.int64)
    stored = np.asarray(current["refinement_face_parent_barycentric"], dtype=np.float64)
    expected = np.array(stored, copy=True)
    singleton_rows = []
    stale_rows = []
    unmapped_rows = []
    multi_parent_count = 0
    multi_parent_ancestry_length_counts = {}
    max_stale_delta = 0.0
    for face_row, tri in enumerate(faces):
        parent_row = int(scalar_parent[face_row])
        lo, hi = int(parent_ptr[face_row]), int(parent_ptr[face_row + 1])
        ancestry = list(map(int, parent_rows[lo:hi]))
        if parent_row < 0:
            if len(ancestry) <= 1 or not np.isnan(stored[face_row]).all():
                raise ValueError(f"face {face_row}: multi-parent scalar/NaN sentinel disagrees with CSR")
            multi_parent_count += 1
            key = str(len(ancestry))
            multi_parent_ancestry_length_counts[key] = multi_parent_ancestry_length_counts.get(key, 0) + 1
            continue
        if ancestry != [parent_row]:
            raise ValueError(f"face {face_row}: scalar parent disagrees with its CSR ancestry")
        if parent_row >= len(parent_faces):
            raise ValueError(f"face {face_row}: parent face row is out of range")
        parent_tri = list(map(int, parent_faces[parent_row]))
        row_weights = []
        for node in map(int, tri):
            qmap = qmaps[node]
            weights = np.zeros(3, dtype=np.float64)
            for parent_node, weight in qmap.items():
                if abs(weight) <= 1e-15:
                    continue
                if parent_node not in parent_tri:
                    raise ValueError(f"face {face_row}, node {node}: Q support leaves parent face {parent_row}")
                weights[parent_tri.index(parent_node)] += weight
            if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0.0):
                unmapped_rows.append(face_row)
                break
            row_weights.append(weights)
        if len(row_weights) != 3:
            continue
        row_weights = np.asarray(row_weights, dtype=np.float64)
        expected[face_row] = row_weights
        singleton_rows.append(face_row)
        delta = float(np.max(np.abs(stored[face_row] - row_weights))) if np.isfinite(stored[face_row]).all() else float("inf")
        if delta > 1e-12:
            stale_rows.append(face_row)
            max_stale_delta = max(max_stale_delta, delta)
    if unmapped_rows:
        raise ValueError(f"singleton-parent rows have unmapped Q support: {unmapped_rows[:10]}")
    return expected, {
        "parent_candidate_sha256": PARENT_SHA,
        "canonical037_candidate_sha256": CANONICAL_037_SHA,
        "parent_face_count": int(len(parent_faces)),
        "parent_row_27587_nodes": parent_faces[27587].tolist(),
        "parent_P_matches_archived_P_bitwise": True,
        "Q_times_P_max_abs_source_weight_error": max_source_error,
        "Q_to_canonical037_max_abs_point_error_m": max_reference_point_error,
        "singleton_parent_face_count": len(singleton_rows),
        "multi_parent_face_count": multi_parent_count,
        "multi_parent_ancestry_length_counts": multi_parent_ancestry_length_counts,
        "stale_singleton_face_barycentric_count": len(stale_rows),
        "stale_singleton_face_rows": stale_rows,
        "maximum_stale_barycentric_component_delta": max_stale_delta,
        "unmapped_singleton_face_count": len(unmapped_rows),
        "rule": "old node uses exact one-hot parent-face membership; new node uses immediate-Q row support restricted to the pinned parent-face node rows",
        "face_barycentric_values_derived_from_exact_Q": True,
        "coordinate_fit_or_least_squares_used": False,
    }


def repair_singleton_face_barycentrics(current: dict, expected: np.ndarray, audit: dict) -> dict:
    faces = np.asarray(current["source_O_faces"], dtype=np.int64)
    scalar_parent = np.asarray(current["refinement_face_parent_face_row"], dtype=np.int64)
    before = np.array(current["refinement_face_parent_barycentric"], copy=True)
    stale_rows = list(map(int, audit["stale_singleton_face_rows"]))
    current["refinement_face_parent_barycentric"][scalar_parent >= 0] = expected[scalar_parent >= 0]
    after = np.asarray(current["refinement_face_parent_barycentric"])
    remaining = []
    for row in range(len(faces)):
        if int(scalar_parent[row]) >= 0 and not np.array_equal(after[row], expected[row]):
            remaining.append(row)
    if remaining:
        raise ValueError(f"singleton-parent barycentric repair left mismatches: {remaining[:10]}")
    return {
        "changed_rows": stale_rows,
        "changed_row_count": len(stale_rows),
        "rows_rebuilt_from_pinned_parent_and_Q": int(np.sum(scalar_parent >= 0)),
        "remaining_singleton_mismatches": len(remaining),
        "barycentric_values_regenerated_from_pinned_parent_Q": True,
        "coordinate_fit_or_least_squares_used": False,
    }


def witness_face_rows(faces: np.ndarray) -> list[int]:
    return [r for r, tri in enumerate(np.asarray(faces)) if frozenset(map(int, tri)) == WITNESS_NODES]


def build_proposal(
    candidate_path: Path,
    expected_candidate_sha: str = EXPECTED_BASE,
    parent_path: Path = PARENT_CANDIDATE,
    canonical037_path: Path = CANONICAL_037,
) -> tuple[dict, dict]:
    for path, expected, label in (
        (OWNER402, EXPECTED_402, "402"),
        (OWNER405, EXPECTED_405, "405"),
        (OWNER541, EXPECTED_541, "541"),
    ):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"pinned {label} owner SHA mismatch: {actual}")
    candidate_sha = sha256(candidate_path)
    if candidate_sha != expected_candidate_sha:
        raise ValueError(f"base candidate SHA mismatch: {candidate_sha}")

    parent_sha = sha256(parent_path)
    canonical_sha = sha256(canonical037_path)
    if parent_sha != PARENT_SHA or canonical_sha != CANONICAL_037_SHA:
        raise ValueError("pinned refinement parent or canonical-037 SHA mismatch")
    c = load_candidate(candidate_path)
    parent = load_candidate(parent_path)
    canonical = load_candidate(canonical037_path)
    m402 = load_module("flip402_557", OWNER402)
    m405 = load_module("flip405_557", OWNER405)
    m541 = load_module("quality541_557", OWNER541)
    m402.validate_candidate(c)
    original_payload = {k: np.array(v, copy=True) for k, v in c.items()}

    faces = np.asarray(c["source_O_faces"])
    ref_faces = np.asarray(c["area_reference_037_source_O_faces"])
    if not np.array_equal(faces, ref_faces):
        raise ValueError("current and fixed-037 face connectivity differ")
    targets = witness_face_rows(faces)
    if len(targets) != 1:
        raise ValueError(f"expected one witness triangle, found {targets}")
    target = targets[0]
    if (
        int(c["face_kind"][target]) != 2
        or int(c["parent_fine_detail_face_row"][target]) != WITNESS_DETAIL
        or int(c["parent_fine_subface_row"][target]) != WITNESS_SUBFACE
    ):
        raise ValueError("witness face typed source/detail identity changed")

    x_mm = np.asarray(c["points_world_m"], dtype=np.float64) * 1000.0
    r_mm = np.asarray(c["area_reference_037_points_world_m_f64"], dtype=np.float64) * 1000.0
    old_ref_area = cross_norm(r_mm, faces[target])
    if old_ref_area > MAX_OLD_REFERENCE_CROSS_MM2:
        raise ValueError("fixed-037 witness is no longer near-collinear")

    # Repair only stale per-face metadata using the exact pinned Q/P lineage; no positions or faces move here.
    base_expected_bary, base_affine_audit = derive_affine_face_barycentrics(c, parent, canonical, m405)
    base_barycentric_repair = repair_singleton_face_barycentrics(c, base_expected_bary, base_affine_audit)
    base_non_bary_payload_unchanged = all(
        arrays_bitwise_equal(original_payload[k], c[k])
        for k in original_payload
        if k != "refinement_face_parent_barycentric"
    )
    if not base_non_bary_payload_unchanged:
        raise ValueError("base lineage repair changed payload outside per-face affine rows")
    m402.validate_candidate(c)

    # The profile only opens this one explicitly named source-detail group.
    m405.group_of = lambda ds: 0 if ds and set(map(int, ds)).issubset({WITNESS_DETAIL}) else None
    detail_rows = m405.detail_csr(c, {WITNESS_DETAIL})
    locks, outer_locks, between_locks, selected_rows = m405.boundary_locks(
        c, detail_rows, {WITNESS_DETAIL}
    )
    edge_map = m402.edge_map(faces)
    edge_records = []
    proposals = []
    for a, b in zip(faces[target], np.roll(faces[target], -1)):
        edge = (min(int(a), int(b)), max(int(a), int(b)))
        incidence = edge_map.get(edge, [])
        record = {
            "edge": list(edge),
            "incidence": [list(map(int, item)) for item in incidence],
            "locked": edge in locks,
        }
        if len(incidence) != 2:
            record["reason"] = "not_two_face_interior_edge"
        else:
            op, reason = m405.propose_one(c, edge, incidence, locks)
            record["reason"] = reason
            if op is not None:
                proposals.append((edge, op))
                record["operation"] = op
        edge_records.append(record)
    if len(proposals) != 1:
        raise ValueError(f"expected exactly one 402/405 local candidate, found {len(proposals)}")

    edge, op = proposals[0]
    pair = tuple(map(int, op["face_rows"]))
    new_faces = np.asarray(op["new_faces"], dtype=np.int64)
    new_q = m541.face_quality(x_mm, r_mm, new_faces, new_faces)
    new_ref_areas = np.asarray([cross_norm(r_mm, tri) for tri in new_faces])
    reasons, profile_metrics = profile_rejections(
        old_ref_area,
        new_ref_areas,
        new_q["normal_cos"],
        new_q["min_angle_deg"],
        new_q["area_ratio"],
    )
    _legacy_541_result, legacy_541_reason = m541.evaluate_op(c, op, None, x_mm, r_mm)
    proposed = m405.apply_op(c, op)
    post_flip_expected_bary, post_flip_affine_audit = derive_affine_face_barycentrics(proposed, parent, canonical, m405)
    post_flip_barycentric_repair = repair_singleton_face_barycentrics(
        proposed, post_flip_expected_bary, post_flip_affine_audit
    )
    m402.validate_candidate(proposed)

    # Prove this proposal is topological only and preserves the complete patch boundary.
    for key in ("points_world_m", "area_reference_037_points_world_m_f64", "typed_coarse_anchor_output_rows_018"):
        if key in c and c[key].tobytes() != proposed[key].tobytes():
            raise ValueError(f"fixed payload changed: {key}")
    anchor_rows = np.asarray(c["typed_coarse_anchor_output_rows_018"], dtype=np.int64)
    if c["points_world_m"][anchor_rows].tobytes() != proposed["points_world_m"][anchor_rows].tobytes():
        raise ValueError("typed anchor coordinates changed")
    if boundary_edges(faces) != boundary_edges(proposed["source_O_faces"]):
        raise ValueError("global mesh boundary edge set changed")
    old_patch_boundary = {
        edge for edge in edge_set(faces[list(pair)])
        if sum(edge in edge_set([faces[row]]) for row in pair) == 1
    }
    new_patch_boundary = {
        edge for edge in edge_set(proposed["source_O_faces"][list(pair)])
        if sum(edge in edge_set([proposed["source_O_faces"][row]]) for row in pair) == 1
    }
    if old_patch_boundary != new_patch_boundary:
        raise ValueError("two-face patch boundary ring changed")

    ancestry = check_csr_union(
        m405,
        c,
        proposed,
        pair,
        [
            ("face_source_ancestry_indptr", "face_source_ancestry_face_rows"),
            ("face_fine_detail_ancestry_indptr", "face_fine_detail_ancestry_rows"),
            ("face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows"),
            ("refinement_face_parent_ancestry_indptr", "refinement_face_parent_ancestry_face_rows"),
        ],
    )
    parent_union = ancestry["refinement_face_parent_ancestry_face_rows"]
    if len(parent_union) == 1:
        expected_scalar = int(parent_union[0])
        if any(int(proposed["refinement_face_parent_face_row"][r]) != expected_scalar for r in pair):
            raise ValueError("singleton immediate-parent scalar mismatch")
        if not np.isfinite(proposed["refinement_face_parent_barycentric"][list(pair)]).all():
            raise ValueError("singleton immediate-parent barycentrics are not finite")
    else:
        if any(int(proposed["refinement_face_parent_face_row"][r]) != -1 for r in pair):
            raise ValueError("multi-parent scalar sentinel was not set")
        if not np.isnan(proposed["refinement_face_parent_barycentric"][list(pair)]).all():
            raise ValueError("multi-parent barycentric sentinel was not set")

    report = {
        "status": (
            "DIAGNOSTIC_FLIP_REJECTED_BY_REFERENCE_DEGENERACY_PROFILE"
            if reasons
            else "DIAGNOSTIC_REFERENCE_DEGENERACY_REPAIR_PROPOSAL_NOT_FULLY_CHECKED"
        ),
        "candidate_selected": False,
        "native_admission": False,
        "profile": "REFERENCE_DEGENERACY_REPAIR_30886",
        "base_candidate": str(candidate_path),
        "base_candidate_sha256": candidate_sha,
        "output_npz_sha256": None,
        "helper_sha256": sha256(Path(__file__)),
        "owner_sha256": {
            "402": sha256(OWNER402),
            "405": sha256(OWNER405),
            "541": sha256(OWNER541),
        },
        "witness": {
            "face_row": int(target),
            "node_rows": sorted(WITNESS_NODES),
            "face_kind": int(c["face_kind"][target]),
            "detail_face_row": int(c["parent_fine_detail_face_row"][target]),
            "subface_row": int(c["parent_fine_subface_row"][target]),
            "old_fixed_037_cross_norm_mm2": old_ref_area,
        },
        "edge_preflight": edge_records,
        "operation": op,
        "profile_metrics": profile_metrics,
        "profile_rejections": reasons,
        "operation_candidate_emitted": not bool(reasons),
        "legacy_541": {
            "pass": legacy_541_reason == "eligible",
            "reason": legacy_541_reason,
            "policy_note": "Original 541 angle-loss and quality policy is retained and reported; this diagnostic profile does not waive it.",
        },
        "preservation": {
            "base_non_bary_payload_unchanged": base_non_bary_payload_unchanged,
            "coordinates_bitwise_unchanged": True,
            "matched_037_points_bitwise_unchanged": True,
            "typed_anchor_rows_and_coordinates_bitwise_unchanged": True,
            "global_boundary_edges_unchanged": True,
            "two_face_patch_boundary_unchanged": True,
            "csr_unions": ancestry,
            "immediate_parent_union": parent_union,
            "base_affine_lineage_audit": base_affine_audit,
            "base_barycentric_repair": base_barycentric_repair,
            "post_flip_affine_lineage_audit": post_flip_affine_audit,
            "post_flip_barycentric_repair": post_flip_barycentric_repair,
            "changed_face_rows": list(pair),
            "all_other_face_rows_unchanged": bool(
                np.array_equal(
                    np.delete(c["source_O_faces"], list(pair), axis=0),
                    np.delete(proposed["source_O_faces"], list(pair), axis=0),
                )
            ),
        },
        "required_next_checks": [
            "the flip is not emitted because the top-029 min-current-angle profile gate fails",
            "the lineage-repaired base has unchanged topology and coordinates; physical and exact audits remain attached to its pinned input",
            "retain the explicit 541 rejection in any review; this is not physical admission",
        ],
    }
    return proposed, c, report


def write_proposal(
    candidate_path: Path,
    output_dir: Path,
    expected_candidate_sha: str = EXPECTED_BASE,
    parent_path: Path = PARENT_CANDIDATE,
    canonical037_path: Path = CANONICAL_037,
) -> dict:
    proposed, repaired_base, report = build_proposal(
        candidate_path, expected_candidate_sha, parent_path, canonical037_path
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    base_npz = output_dir / "lineage-repaired-base.npz"
    tmp_base = output_dir / ".lineage-repaired-base.npz.tmp"
    with tmp_base.open("wb") as f:
        np.savez_compressed(f, **repaired_base)
    tmp_base.replace(base_npz)
    report["lineage_repaired_base_npz"] = str(base_npz)
    report["lineage_repaired_base_sha256"] = sha256(base_npz)
    if report["operation_candidate_emitted"]:
        out_npz = output_dir / "diagnostic-proposal.npz"
        tmp_npz = output_dir / ".diagnostic-proposal.npz.tmp"
        with tmp_npz.open("wb") as f:
            np.savez_compressed(f, **proposed)
        tmp_npz.replace(out_npz)
        report["output_npz"] = str(out_npz)
        report["output_npz_sha256"] = sha256(out_npz)
    else:
        report["output_npz"] = None
        report["output_npz_sha256"] = None
    out_report = output_dir / "report.json"
    tmp_report = output_dir / ".report.json.tmp"
    tmp_report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    tmp_report.replace(out_report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--expected-candidate-sha256", default=EXPECTED_BASE)
    parser.add_argument("--parent-candidate", type=Path, default=PARENT_CANDIDATE)
    parser.add_argument("--canonical-037-candidate", type=Path, default=CANONICAL_037)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = write_proposal(
        args.candidate, args.output_dir, args.expected_candidate_sha256,
        args.parent_candidate, args.canonical_037_candidate
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
