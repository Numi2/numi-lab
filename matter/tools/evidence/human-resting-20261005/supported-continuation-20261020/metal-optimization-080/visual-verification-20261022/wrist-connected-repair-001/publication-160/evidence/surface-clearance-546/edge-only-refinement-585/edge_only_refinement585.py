#!/usr/bin/env python3
"""Conforming edge-midpoint refinement through the frozen 546/270 lineage path.

This is construction-only. It inserts shared edge midpoints on selected faces,
triangulates each affected parent without a face centroid, and delegates all
candidate metadata, Q@P composition, anchors, and 037-reference handling to
connected_refinement_adapter546. It does not move points or claim clearance.
"""
from __future__ import annotations
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any
import numpy as np

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069")
ADAPTER_PATH = BASE / "D/surface-clearance-546/connected-refinement-adapter546/connected_refinement_adapter546.py"
ADAPTER_SHA256 = "06081873eaf0d5452fbf4beee3d580a0eba97c3a546a9f7e7a06a0c9f548cbc6"
P037_PATH = BASE / "full-connected-clearance-150/run-037-local-refinement-006-prep/refine_conforming_006.py"
P037_SHA256 = "d3a6e0c31e34f2165d853ffaf411ad67faaa1d307a3af1a6c79ba41a129f4f31"
P084_SHA256 = "eab8deda28164964051a6fa6ff346b8d94bc8bede04e896411dfdd02d46594f6"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _edge_incidence(faces: np.ndarray):
    incidence: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for fi, face in enumerate(np.asarray(faces, dtype=np.int64)):
        a, b, c = map(int, face)
        for u, v in ((a, b), (b, c), (c, a)):
            incidence.setdefault(_edge_key(u, v), []).append((fi, u, v))
    return incidence


def _boundary_summary(edges: set[tuple[int, int]]) -> tuple[int, dict[int, int]]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    degree = {v: len(ns) for v, ns in adjacency.items()}
    seen: set[int] = set()
    components = 0
    for start in sorted(adjacency):
        if start in seen:
            continue
        components += 1
        seen.add(start)
        stack = [start]
        while stack:
            current = stack.pop()
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return components, degree


def edge_only_refinement585(points: np.ndarray, faces: np.ndarray,
                            crossed_face_rows: np.ndarray,
                            parent_face_markers: np.ndarray | None = None,
                            selected_edges_override: np.ndarray | None = None) -> dict[str, np.ndarray | int]:
    """Return the existing P037 `p084_affine_refinement` result mapping.

    All edges of each validated crossed face are selected globally. Every face
    incident to any selected edge is retriangulated with the same boundary and
    only original corners plus shared edge midpoints. Thus 1/2/3 split edges
    yield 2/3/4 children. No face centroid or point deletion is used.
    """
    X = np.asarray(points)
    Fraw = np.asarray(faces)
    if X.ndim != 2 or X.shape[1] != 3 or X.dtype.kind != "f" or X.dtype.itemsize not in (4, 8) or not np.isfinite(X).all():
        raise ValueError("points must be finite Float32/Float64 Nx3")
    if Fraw.ndim != 2 or Fraw.shape[1] != 3 or Fraw.dtype.kind not in "iu" or len(Fraw) == 0:
        raise ValueError("faces must be a nonempty integer Mx3 array")
    F = Fraw.astype(np.int64, copy=False)
    npoints, nfaces = len(X), len(F)
    if npoints < 3 or F.min(initial=0) < 0 or F.max(initial=-1) >= npoints:
        raise ValueError("face node lies outside the current point domain")
    if np.any((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 2] == F[:, 0])):
        raise ValueError("input face contains a repeated vertex")
    if len(np.unique(np.sort(F, axis=1), axis=0)) != nfaces:
        raise ValueError("input face table contains duplicate triangles")
    crossed_raw = np.asarray(crossed_face_rows)
    if crossed_raw.ndim != 1 or crossed_raw.dtype.kind not in "iu" or not len(crossed_raw):
        raise ValueError("crossed_face_rows must be a nonempty integer vector")
    crossed = np.unique(crossed_raw.astype(np.int64, copy=False))
    if crossed.min() < 0 or crossed.max() >= nfaces:
        raise ValueError("crossed face row lies outside the current face domain")
    if parent_face_markers is None:
        markers = np.arange(1, nfaces + 1, dtype=np.int64)
    else:
        markers = np.asarray(parent_face_markers)
        if markers.shape != (nfaces,) or markers.dtype.kind not in "iu":
            raise ValueError("parent_face_markers must be an integer M-vector")

    incidence = _edge_incidence(F)
    if any(len(uses) > 2 for uses in incidence.values()):
        raise ValueError("input surface has a nonmanifold edge")
    for edge, uses in incidence.items():
        if len(uses) == 2 and (uses[0][1], uses[0][2]) != (uses[1][2], uses[1][1]):
            raise ValueError(f"input interior edge {edge} has inconsistent directed winding")
    old_boundary = {edge for edge, uses in incidence.items() if len(uses) == 1}
    old_components, old_degrees = _boundary_summary(old_boundary)
    if any(degree != 2 for degree in old_degrees.values()):
        raise ValueError("input boundary is not a union of simple closed loops")

    candidate_edges: set[tuple[int, int]] = set()
    for fi in crossed:
        a, b, c = map(int, F[int(fi)])
        candidate_edges.update((_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)))
    if selected_edges_override is None:
        selected_edges = candidate_edges
    else:
        raw_edges = np.asarray(selected_edges_override)
        if raw_edges.ndim != 2 or raw_edges.shape[1] != 2 or raw_edges.dtype.kind not in "iu" or not len(raw_edges):
            raise ValueError("selected_edges_override must be a nonempty integer Ex2 array")
        selected_edges = set()
        for raw_edge in raw_edges:
            a, b = map(int, raw_edge)
            if a == b or min(a, b) < 0 or max(a, b) >= npoints:
                raise ValueError("selected edge endpoint is invalid")
            edge = _edge_key(a, b)
            if edge not in candidate_edges:
                raise ValueError(f"edge-only pilot selected an edge outside the validated crossed-face set: {edge}")
            selected_edges.add(edge)
        if len(selected_edges) != len(raw_edges):
            raise ValueError("selected edge override contains duplicates")
    affected = sorted({fi for edge in selected_edges for fi, _, _ in incidence[edge]})
    edges_sorted = sorted(selected_edges)
    edge_rows = {edge: npoints + i for i, edge in enumerate(edges_sorted)}

    new_points = []
    for edge in edges_sorted:
        a, b = edge
        midpoint = ((X[a].astype(np.float64) + X[b].astype(np.float64)) * 0.5).astype(X.dtype)
        if not np.isfinite(midpoint).all() or np.array_equal(midpoint, X[a]) or np.array_equal(midpoint, X[b]):
            raise ValueError(f"Float32/Float64 midpoint collapses for selected edge {edge}")
        # Check the emitted point is exactly represented by this affine Q row.
        replay = ((X[a].astype(np.float64) + X[b].astype(np.float64)) * 0.5).astype(X.dtype)
        if not np.array_equal(midpoint, replay):
            raise AssertionError("midpoint does not replay from its immediate Q row")
        new_points.append(midpoint)
    Xout = np.concatenate((X.copy(), np.asarray(new_points, dtype=X.dtype).reshape(-1, 3)), axis=0)
    if not np.array_equal(Xout[:npoints].view(np.uint8), X.view(np.uint8)):
        raise AssertionError("old point prefix changed")

    identity = np.eye(3, dtype=np.float64)
    child_faces: list[tuple[int, int, int]] = []
    child_parent: list[int] = []
    child_bary: list[np.ndarray] = []

    def vertex(i: int, tri: tuple[int, int, int]):
        bary = np.zeros(3, dtype=np.float64)
        bary[i] = 1.0
        return int(tri[i]), bary

    def midpoint(i: int, j: int, tri: tuple[int, int, int]):
        key = _edge_key(tri[i], tri[j])
        bary = np.zeros(3, dtype=np.float64)
        bary[i] = bary[j] = 0.5
        return edge_rows[key], bary

    for fi, tri_raw in enumerate(F):
        tri = tuple(map(int, tri_raw))
        split = [(_edge_key(tri[i], tri[(i + 1) % 3]) in edge_rows) for i in range(3)]
        count = sum(split)
        pieces: list[tuple[tuple[int, np.ndarray], tuple[int, np.ndarray], tuple[int, np.ndarray]]] = []
        if count == 0:
            pieces = [(vertex(0, tri), vertex(1, tri), vertex(2, tri))]
        elif count == 1:
            i = split.index(True)
            A, B, C = i, (i + 1) % 3, (i + 2) % 3
            va, vm, vb, vc = vertex(A, tri), midpoint(A, B, tri), vertex(B, tri), vertex(C, tri)
            pieces = [(va, vm, vc), (vm, vb, vc)]
        elif count == 2:
            starts = [i for i in range(3) if split[i] and split[(i + 1) % 3]]
            if len(starts) != 1:
                raise AssertionError("two-edge split does not have one cyclic start")
            i = starts[0]
            A, B, C = i, (i + 1) % 3, (i + 2) % 3
            va, vb, vc = vertex(A, tri), vertex(B, tri), vertex(C, tri)
            vab, vbc = midpoint(A, B, tri), midpoint(B, C, tri)
            pieces = [(va, vab, vc), (vab, vb, vbc), (vab, vbc, vc)]
        else:
            va, vb, vc = vertex(0, tri), vertex(1, tri), vertex(2, tri)
            vab, vbc, vca = midpoint(0, 1, tri), midpoint(1, 2, tri), midpoint(2, 0, tri)
            pieces = [(va, vab, vca), (vab, vb, vbc), (vca, vbc, vc), (vab, vbc, vca)]
        for p0, p1, p2 in pieces:
            child_faces.append((p0[0], p1[0], p2[0]))
            child_parent.append(fi)
            child_bary.append(np.vstack((p0[1], p1[1], p2[1])))

    Fout = np.asarray(child_faces, dtype=Fraw.dtype)
    parent_rows = np.asarray(child_parent, dtype=np.int64)
    parent_bary = np.asarray(child_bary, dtype=np.float64)
    if Fout.shape != (len(F) + len(affected) + 0, 3):
        # General output count is old faces + one extra face per selected split
        # edge on each face: child count is 1 + its selected-edge count.
        expected = sum(1 + sum(_edge_key(int(face[i]), int(face[(i + 1) % 3])) in edge_rows for i in range(3)) for face in F)
        if Fout.shape != (expected, 3):
            raise AssertionError("edge-only refinement emitted an invalid child count")
    if parent_rows.shape != (len(Fout),) or parent_bary.shape != (len(Fout), 3, 3):
        raise AssertionError("child immediate-parent metadata has an invalid domain")

    old_cross = np.cross(X[F[:, 1]].astype(np.float64) - X[F[:, 0]].astype(np.float64),
                         X[F[:, 2]].astype(np.float64) - X[F[:, 0]].astype(np.float64))
    new_cross = np.cross(Xout[Fout[:, 1]].astype(np.float64) - Xout[Fout[:, 0]].astype(np.float64),
                         Xout[Fout[:, 2]].astype(np.float64) - Xout[Fout[:, 0]].astype(np.float64))
    new_area2 = np.linalg.norm(new_cross, axis=1)
    orientation = np.einsum("ij,ij->i", new_cross, old_cross[parent_rows])
    if not np.isfinite(new_area2).all() or np.any(new_area2 <= 0.0) or np.any(orientation <= 0.0):
        raise ValueError("edge-only refinement produced a degenerate or reversed child face")
    old_area = 0.5 * np.linalg.norm(old_cross, axis=1)
    new_area = 0.5 * new_area2
    area_sum = np.bincount(parent_rows, weights=new_area, minlength=nfaces)
    area_error = np.abs(area_sum - old_area) / np.maximum(old_area, np.finfo(np.float64).tiny)

    new_incidence = _edge_incidence(Fout)
    if any(len(uses) > 2 for uses in new_incidence.values()):
        raise AssertionError("edge-only refinement produced a nonmanifold edge")
    new_boundary = {edge for edge, uses in new_incidence.items() if len(uses) == 1}
    expected_boundary: set[tuple[int, int]] = set()
    for a, b in old_boundary:
        if (a, b) in edge_rows:
            mid = edge_rows[(a, b)]
            expected_boundary.update((_edge_key(a, mid), _edge_key(mid, b)))
        else:
            expected_boundary.add((a, b))
    if new_boundary != expected_boundary:
        raise AssertionError("boundary chain changed beyond conforming selected-edge bisection")
    new_components, new_degrees = _boundary_summary(new_boundary)
    if new_components != old_components or any(degree != 2 for degree in new_degrees.values()):
        raise AssertionError("boundary loop topology changed")

    node_kind = np.concatenate((np.zeros(npoints, dtype=np.int8), np.ones(len(edges_sorted), dtype=np.int8)))
    parent_edge_rows = np.vstack((np.full((npoints, 2), -1, dtype=np.int64), np.asarray(edges_sorted, dtype=np.int64).reshape(-1, 2)))
    parent_edge_weights = np.vstack((np.full((npoints, 2), -1.0, dtype=np.float64), np.full((len(edges_sorted), 2), 0.5, dtype=np.float64)))
    parent_face_indptr = [0] * (npoints + 1)
    parent_face_rows: list[int] = []
    parent_face_bary: list[np.ndarray] = []
    prolong_indptr = list(range(npoints + 1))
    prolong_rows = list(range(npoints))
    prolong_weights = [1.0] * npoints
    for edge in edges_sorted:
        a, b = edge
        for fi, _, _ in sorted(incidence[edge]):
            tri = tuple(map(int, F[fi]))
            bary = np.zeros(3, dtype=np.float64)
            bary[tri.index(a)] = 0.5
            bary[tri.index(b)] = 0.5
            parent_face_rows.append(int(fi))
            parent_face_bary.append(bary)
        parent_face_indptr.append(len(parent_face_rows))
        prolong_rows.extend((a, b))
        prolong_weights.extend((0.5, 0.5))
        prolong_indptr.append(len(prolong_rows))

    result: dict[str, Any] = {
        "points_world_m": Xout,
        "source_O_faces": Fout,
        "parent_face_rows": parent_rows,
        "parent_face_barycentric": parent_bary,
        "selected_edges": np.asarray(edges_sorted, dtype=np.int64).reshape(-1, 2),
        "edge_midpoint_rows": np.asarray([edge_rows[e] for e in edges_sorted], dtype=np.int64),
        "affected_parent_face_rows": np.asarray(affected, dtype=np.int64),
        "face_centroid_rows": np.empty(0, dtype=np.int64),
        "refinement_node_kind": node_kind,
        "refinement_node_parent_edge_rows": parent_edge_rows,
        "refinement_node_parent_edge_weights": parent_edge_weights,
        "refinement_node_parent_face_indptr": np.asarray(parent_face_indptr, dtype=np.int64),
        "refinement_node_parent_face_rows": np.asarray(parent_face_rows, dtype=np.int64),
        "refinement_node_parent_face_barycentric": np.asarray(parent_face_bary, dtype=np.float64).reshape(-1, 3),
        "source_prolongation_indptr": np.asarray(prolong_indptr, dtype=np.int64),
        "source_prolongation_node_rows": np.asarray(prolong_rows, dtype=np.int64),
        "source_prolongation_weights": np.asarray(prolong_weights, dtype=np.float64),
        "old_face_area_m2": old_area,
        "new_face_area_sum_m2": area_sum,
        "boundary_components_before": int(old_components),
        "boundary_components_after": int(new_components),
        "boundary_edges_before": int(len(old_boundary)),
        "boundary_edges_after": int(len(new_boundary)),
        "relative_parent_area_error": area_error,
        "p084_parent_face_markers": markers[parent_rows].copy(),
    }
    _validate_result585(X, Fraw, result)
    return result


def _validate_result585(points: np.ndarray, faces: np.ndarray, result: dict[str, Any]) -> None:
    X, F = np.asarray(points), np.asarray(faces)
    n = len(X)
    ptr = np.asarray(result["source_prolongation_indptr"])
    cols = np.asarray(result["source_prolongation_node_rows"])
    weights = np.asarray(result["source_prolongation_weights"])
    Y = np.asarray(result["points_world_m"])
    if len(ptr) != len(Y) + 1 or ptr[0] != 0 or ptr[-1] != len(cols) or len(cols) != len(weights):
        raise AssertionError("Q CSR dimensions do not match refined point domain")
    for row in range(len(Y)):
        lo, hi = int(ptr[row]), int(ptr[row + 1])
        if hi <= lo or abs(float(weights[lo:hi].sum()) - 1.0) > 1e-14:
            raise AssertionError(f"Q row {row} is not nonempty affine")
        replay = np.sum(X[cols[lo:hi]].astype(np.float64) * weights[lo:hi, None], axis=0).astype(X.dtype)
        if not np.array_equal(replay, Y[row]):
            raise AssertionError(f"Q row {row} does not reproduce the emitted point")
    parent = np.asarray(result["parent_face_rows"], dtype=np.int64)
    bary = np.asarray(result["parent_face_barycentric"], dtype=np.float64)
    children = np.asarray(result["source_O_faces"], dtype=np.int64)
    if bary.shape != (len(children), 3, 3) or not np.allclose(bary.sum(axis=2), 1.0, atol=1e-12, rtol=0):
        raise AssertionError("child-face barycentric map is malformed")
    for row, p in enumerate(parent):
        if not np.allclose(bary[row] @ X[F[p]].astype(np.float64), Y[children[row]].astype(np.float64), atol=2e-7, rtol=0):
            raise AssertionError(f"child {row} does not reconstruct from its immediate parent")
    if len(Y) != n + len(result["selected_edges"]) or np.any(np.asarray(result["face_centroid_rows"])):
        raise AssertionError("edge-only topology added unexpected nodes")


def _load_adapter585():
    actual = sha256_file(ADAPTER_PATH)
    if actual != ADAPTER_SHA256:
        raise ValueError(f"pinned 546 adapter hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("_connected_refinement_adapter546_for585", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned adapter: {ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_refined_arrays585(source: dict[str, np.ndarray], crossing_report: dict[str, Any],
                            area_reference_points_world_m: np.ndarray,
                            area_reference_faces: np.ndarray, *, candidate_sha256: str,
                            crossing_report_sha256: str, area_reference_sha256: str,
                            selected_edges_override: np.ndarray | None = None,
                            selection_policy: dict[str, Any] | None = None):
    """Use frozen 546/270 ancestry and reference handling with edge-only children."""
    adapter = _load_adapter585()
    original_load_owner = adapter._load_owner270
    patched_owner_refs = []

    def load_owner_with_edge_refiner():
        owner = original_load_owner()
        original_load_pinned = owner._load_pinned
        def load_pinned_with_edge_refiner(path, expected_sha, module_name):
            module = original_load_pinned(path, expected_sha, module_name)
            if Path(path).resolve() == Path(owner.REFINER_PATH).resolve():
                if sha256_file(Path(path)) != P037_SHA256:
                    raise ValueError("pinned P037 refiner changed before edge-only hook")
                if getattr(module, "P084_OWNER_SHA", None) != P084_SHA256:
                    raise ValueError("loaded P037 refiner does not name expected P084 owner")
                def edge_only_hook(points, faces, crossed, markers):
                    return edge_only_refinement585(points, faces, crossed, markers,
                                                   selected_edges_override=selected_edges_override)
                module.p084_affine_refinement = edge_only_hook
                # P037 assumes every affected face also receives a centroid.
                # For this edge-only result, suppress only that node-extension
                # list while leaving the original result and 270 report intact.
                original_extend = module.extend_candidate_arrays
                def extend_edge_only(source_arrays, result_arrays, candidate_sha=None, face_report_sha=None):
                    node_only = dict(result_arrays)
                    node_only.update(dict(affected_parent_face_rows=np.empty(0, dtype=np.int64)))
                    return original_extend(source_arrays, node_only, candidate_sha, face_report_sha)
                module.extend_candidate_arrays = extend_edge_only
            return module
        owner._load_pinned = load_pinned_with_edge_refiner
        patched_owner_refs.append((owner, original_load_pinned))
        return owner

    adapter._load_owner270 = load_owner_with_edge_refiner
    try:
        arrays, report = adapter.build_refined_arrays546(
            source, crossing_report, area_reference_points_world_m, area_reference_faces,
            candidate_sha256=candidate_sha256,
            crossing_report_sha256=crossing_report_sha256,
            area_reference_sha256=area_reference_sha256)
    finally:
        adapter._load_owner270 = original_load_owner
        for owner, original_load_pinned in patched_owner_refs:
            owner._load_pinned = original_load_pinned
    arrays = dict(arrays)
    arrays["immediate_refinement_prolongation_semantics"] = np.asarray([
        "rows map refined nodes to the immediate parent using identity rows or shared-edge midpoint rows; no face centroids"
    ])
    arrays["source_prolongation_semantics"] = np.asarray([
        "rows map refined nodes through immediate Q times preserved parent P; new rows are edge midpoints only"
    ])
    arrays["refinement_node_kind_semantics"] = np.asarray([
        "0=preexisting; 1=selected shared-edge midpoint; no centroid kind is emitted"
    ])
    report = dict(report)
    report.update({
        "refinement_method": "conforming edge-only affine midpoint bisection; no face-centroid nodes",
        "edge_only_refinement585_sha256": sha256_file(Path(__file__)),
        "pinned_546_adapter_sha256": ADAPTER_SHA256,
        "pinned_037_refiner_sha256": P037_SHA256,
        "pinned_084_owner_sha256": P084_SHA256,
        "face_centroid_count": 0,
        "added_face_centroids": 0,
        "geometry_moved": False,
        "containment_or_clearance_claim": False,
        "geometry_admitted": False,
        "physical_steps": 0,
        "selected_edge_subset_policy": selection_policy or {"mode": "all_edges_of_validated_crossed_faces"},
    })
    return arrays, report
