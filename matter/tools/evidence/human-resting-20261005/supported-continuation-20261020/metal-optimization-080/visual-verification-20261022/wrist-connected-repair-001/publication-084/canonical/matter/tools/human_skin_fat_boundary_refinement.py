"""Conforming subdivision of oriented skin-boundary triangles at certified samples.

This module owns topology only. It does not find curved midpoint samples. A caller
must supply one source-supported point and its detailed-surface path/face evidence
for every selected edge. The output ancestry identifies the parent coarse O face;
it is construction provenance, not a conservative material-transfer map.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

class BoundaryRefinementError(ValueError):
    """Input boundary, edge samples, or provenance are not safe to subdivide."""

@dataclass(frozen=True)
class BoundaryRefinement:
    points: np.ndarray
    faces: np.ndarray
    child_parent_face_rows: np.ndarray
    child_parent_face_markers: np.ndarray
    selected_edges: np.ndarray
    edge_midpoint_rows: np.ndarray
    midpoint_source_face_rows: tuple[tuple[int, ...], ...]
    midpoint_source_path_vertex_ids: tuple[tuple[int, ...], ...]
    midpoint_source_endpoint_vertex_ids: tuple[tuple[int, int], ...]
    orientation_diagnostics: Mapping[str, object]

def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)

def _validate_source_sample(edge: tuple[int, int], row: int, points: np.ndarray,
                            record: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, int]]:
    # Source-normal reconstruction is a valid O-layer proposal record too. It is
    # distinct from the legacy path-only payload and is verified against the
    # emitted O sample without claiming measured endpoint identity.
    if isinstance(record, Mapping) and record.get("source_kind") == "detailed_skin_source_normal_scalar":
        _validate_source_normal_layer_sample(edge, row, points, record, "O")
        face_rows = tuple(int(v) for v in record["source_face_rows"])
        path_ids = tuple(int(v) for v in record["source_path_vertex_ids"])
        endpoints = tuple(int(v) for v in record.get("source_path_endpoint_vertex_ids", (path_ids[0], path_ids[-1])))
        return face_rows, path_ids, endpoints
    required = {"source_kind", "source_face_rows", "source_path_vertex_ids",
                "edge_endpoint_source_vertex_ids", "sample_point_world_m",
                "ambiguity_count", "minimum_normal_cosine"}
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise BoundaryRefinementError(f"edge {edge}: missing source-sample provenance fields")
    if record["source_kind"] != "detailed_skin_surface_path":
        raise BoundaryRefinementError(f"edge {edge}: midpoint is not certified on detailed skin source path")
    try:
        face_rows = tuple(int(v) for v in record["source_face_rows"])
        path_ids = tuple(int(v) for v in record["source_path_vertex_ids"])
        endpoints = tuple(int(v) for v in record["edge_endpoint_source_vertex_ids"])
        sample = np.asarray(record["sample_point_world_m"], dtype=np.float64)
        ambiguity = int(record["ambiguity_count"])
        normal_cosine = float(record["minimum_normal_cosine"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise BoundaryRefinementError(f"edge {edge}: malformed source-sample provenance") from exc
    if not face_rows or any(v < 0 for v in face_rows) or not path_ids or any(v < 0 for v in path_ids):
        raise BoundaryRefinementError(f"edge {edge}: source face/path identity is empty or invalid")
    if len(path_ids) < 2 or len(set(path_ids)) != len(path_ids):
        raise BoundaryRefinementError(f"edge {edge}: source path must be a simple sequence with at least two vertices")
    if len(endpoints) != 2 or endpoints[0] < 0 or endpoints[1] < 0 or endpoints[0] == endpoints[1]:
        raise BoundaryRefinementError(f"edge {edge}: source edge endpoint identities are invalid")
    if (path_ids[0], path_ids[-1]) not in (endpoints, endpoints[::-1]):
        raise BoundaryRefinementError(f"edge {edge}: source path endpoints do not match the mapped coarse-edge endpoints")
    if sample.shape != (3,) or not np.isfinite(sample).all():
        raise BoundaryRefinementError(f"edge {edge}: source sample point must be finite xyz")
    # The selected vertex sequence itself is the deterministic construction path.
    # Equal shortest paths are allowed only when the deterministic choice rule is
    # retained; it is not evidence that the path is unique.
    if ambiguity < 0 or (ambiguity > 0 and not str(record.get("path_selection_rule", "")).strip()):
        raise BoundaryRefinementError(f"edge {edge}: path tie count lacks a deterministic source-path selection rule")
    if not np.isfinite(normal_cosine):
        raise BoundaryRefinementError(f"edge {edge}: source normal comparison must be finite")
    normal_status = record.get("normal_orientation_status")
    if normal_cosine <= 0.0 and normal_status not in ("reversed", "unresolved"):
        raise BoundaryRefinementError(f"edge {edge}: opposed source normals need explicit orientation status")
    if normal_status not in (None, "aligned", "reversed", "unresolved"):
        raise BoundaryRefinementError(f"edge {edge}: invalid source normal orientation status")
    emitted = np.asarray(points[row], dtype=np.float64)
    if not np.array_equal(sample.astype(points.dtype).astype(np.float64), emitted):
        raise BoundaryRefinementError(f"edge {edge}: sample coordinate differs from emitted midpoint row")
    return face_rows, path_ids, endpoints

def _validate_offset_sample(edge: tuple[int, int], row: int, points: np.ndarray,
                            record: Mapping[str, object], layer: str) -> None:
    required = {"source_kind", "layer", "source_column_ids", "source_column_weights",
                "outer_source_path_vertex_ids", "outer_source_face_rows",
                "outer_midpoint_world_m", "offset_endpoint_a_m", "offset_endpoint_b_m",
                "sample_point_world_m", "reference_field_sha256",
                "source_column_mapping_sha256"}
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise BoundaryRefinementError(f"{layer} edge {edge}: missing registered offset-field provenance")
    if record["source_kind"] != "registered_layer_offset_interpolation" or record["layer"] != layer:
        raise BoundaryRefinementError(f"{layer} edge {edge}: wrong offset provenance kind/layer")
    try:
        columns = tuple(int(v) for v in record["source_column_ids"])
        weights = np.asarray(record["source_column_weights"], dtype=np.float64)
        path = tuple(int(v) for v in record["outer_source_path_vertex_ids"])
        source_faces = tuple(int(v) for v in record["outer_source_face_rows"])
        outer = np.asarray(record["outer_midpoint_world_m"], dtype=np.float64)
        off_a = np.asarray(record["offset_endpoint_a_m"], dtype=np.float64)
        off_b = np.asarray(record["offset_endpoint_b_m"], dtype=np.float64)
        sample = np.asarray(record["sample_point_world_m"], dtype=np.float64)
        hashes = (str(record["reference_field_sha256"]), str(record["source_column_mapping_sha256"]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise BoundaryRefinementError(f"{layer} edge {edge}: malformed offset provenance") from exc
    if len(columns) != 2 or min(columns) < 0 or columns[0] == columns[1]:
        raise BoundaryRefinementError(f"{layer} edge {edge}: source columns must be two distinct nonnegative IDs")
    if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights < 0) or abs(float(weights.sum()) - 1.) > 1e-12:
        raise BoundaryRefinementError(f"{layer} edge {edge}: source column weights must be a finite partition")
    if not path or any(v < 0 for v in path) or not source_faces or any(v < 0 for v in source_faces):
        raise BoundaryRefinementError(f"{layer} edge {edge}: missing outer detailed source path/face ancestry")
    if any(len(v) != 3 or not np.isfinite(v).all() for v in (outer, off_a, off_b, sample)):
        raise BoundaryRefinementError(f"{layer} edge {edge}: offset vectors and points must be finite xyz")
    if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h.lower()) for h in hashes):
        raise BoundaryRefinementError(f"{layer} edge {edge}: source field hashes must be SHA-256 pins")
    expected = outer + weights[0] * off_a + weights[1] * off_b
    if not np.array_equal(expected, sample):
        raise BoundaryRefinementError(f"{layer} edge {edge}: recorded point does not follow the registered offset interpolation")
    emitted = np.asarray(points[row], dtype=np.float64)
    if not np.array_equal(sample.astype(points.dtype).astype(np.float64), emitted):
        raise BoundaryRefinementError(f"{layer} edge {edge}: interpolated sample differs from emitted midpoint row")



def _oriented_boundary_chain(faces: np.ndarray) -> tuple[tuple[int, int], ...]:
    directed: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = map(int, face)
        for u, v in ((a, b), (b, c), (c, a)):
            directed.setdefault(_edge_key(u, v), []).append((u, v))
    return tuple(sorted(uses[0] for uses in directed.values() if len(uses) == 1))


def _validate_source_normal_layer_sample(edge: tuple[int, int], row: int, points: np.ndarray,
                                         record: Mapping[str, object], layer: str) -> None:
    required = {"source_kind", "layer", "sample_point_world_m", "outer_midpoint_world_m",
                "outward_normal", "skin_depth_m", "fat_depth_m", "endpoint_depths_m",
                "source_face_rows", "source_path_vertex_ids", "reference_field_sha256"}
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise BoundaryRefinementError(f"{layer} edge {edge}: missing source-normal scalar provenance")
    if record["source_kind"] != "detailed_skin_source_normal_scalar" or record["layer"] != layer:
        raise BoundaryRefinementError(f"{layer} edge {edge}: wrong source-normal scalar provenance kind/layer")
    try:
        sample = np.asarray(record["sample_point_world_m"], dtype=np.float64)
        outer = np.asarray(record["outer_midpoint_world_m"], dtype=np.float64)
        normal = np.asarray(record["outward_normal"], dtype=np.float64)
        skin = float(record["skin_depth_m"]); fat = float(record["fat_depth_m"])
        endpoint_depths = np.asarray(record["endpoint_depths_m"], dtype=np.float64)
        faces = tuple(int(v) for v in record["source_face_rows"])
        path = tuple(int(v) for v in record["source_path_vertex_ids"])
        pin = str(record["reference_field_sha256"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise BoundaryRefinementError(f"{layer} edge {edge}: malformed source-normal scalar provenance") from exc
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (sample, outer, normal)):
        raise BoundaryRefinementError(f"{layer} edge {edge}: source-normal sample vectors must be finite xyz")
    nlen = float(np.linalg.norm(normal))
    if not np.isfinite(nlen) or abs(nlen - 1.) > 1e-10:
        raise BoundaryRefinementError(f"{layer} edge {edge}: source normal must be unit length")
    if endpoint_depths.shape != (2, 2) or not np.isfinite(endpoint_depths).all():
        raise BoundaryRefinementError(f"{layer} edge {edge}: signed endpoint depths must be finite 2x2")
    if not faces or min(faces) < 0 or len(path) < 2 or min(path) < 0:
        raise BoundaryRefinementError(f"{layer} edge {edge}: source face/path identity is invalid")
    if len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin.lower()):
        raise BoundaryRefinementError(f"{layer} edge {edge}: reference field hash is not SHA-256")
    if not np.isfinite(skin) or not np.isfinite(fat):
        raise BoundaryRefinementError(f"{layer} edge {edge}: scalar midpoint depths must be finite")
    if layer == "O":
        expected = outer
    elif layer == "I":
        expected = outer - skin * normal
    else:
        expected = outer - (skin + fat) * normal
    if points.dtype == np.float32:
        matches_formula = np.array_equal(expected.astype(np.float32), sample.astype(np.float32))
    else:
        matches_formula = np.array_equal(expected, sample)
    if not matches_formula:
        raise BoundaryRefinementError(f"{layer} edge {edge}: sample does not follow signed source-normal scalar field")
    emitted = np.asarray(points[row], dtype=np.float64)
    if not np.array_equal(sample.astype(points.dtype).astype(np.float64), emitted):
        raise BoundaryRefinementError(f"{layer} edge {edge}: source-normal sample differs from emitted midpoint row")


def subdivide_oriented_surface_edges(
    points: np.ndarray,
    faces: np.ndarray,
    edge_midpoint_rows: Mapping[tuple[int, int], int],
    midpoint_provenance: Mapping[tuple[int, int], Mapping[str, object]],
    *, original_point_count: int, parent_face_markers: np.ndarray,
    layer: str = "O", expected_midpoint_rows: Sequence[int] | None = None,
    expected_parent_normals: np.ndarray | None = None,
    collect_orientation_failures: bool = False,
) -> BoundaryRefinement:
    """Split selected manifold interior edges in an oriented triangle surface.

    `points` contains the immutable original prefix followed by one already sampled
    midpoint per selected edge. Every selected edge must have exactly two incident
    O faces with opposite directed use. All incident faces are split in one pass,
    so the shared midpoint is reused and no T-junction is left on that surface.

    For layer O, provenance contains a detailed source path and emitted sample.
    For I/D, provenance certifies registered endpoint offset interpolation. When
    splitting a common O/I/D point table, expected_midpoint_rows names this
    surface's subset of the appended rows; otherwise every appended row must be
    used by this one surface as in the legacy API.
    """
    if layer not in ("O", "I", "D"):
        raise BoundaryRefinementError("layer must be O, I, or D")
    P0 = np.asarray(points)
    F0 = np.asarray(faces)
    if P0.ndim != 2 or P0.shape[1] != 3 or P0.dtype.kind != "f" or not np.isfinite(P0).all():
        raise BoundaryRefinementError("points must be finite floating Nx3")
    if F0.ndim != 2 or F0.shape[1] != 3 or F0.dtype.kind not in "iu" or len(F0) == 0:
        raise BoundaryRefinementError("faces must be a nonempty integer Mx3 array")
    if not 0 < int(original_point_count) <= len(P0):
        raise BoundaryRefinementError("original_point_count is outside points")
    markers = np.asarray(parent_face_markers)
    if markers.shape != (len(F0),) or markers.dtype.kind not in "iu" or np.any(markers <= 0):
        raise BoundaryRefinementError("parent_face_markers must be one positive integer marker per input face")
    if len(np.unique(markers)) != len(markers):
        raise BoundaryRefinementError("parent face markers must be unique")
    expected_normals = None
    if expected_parent_normals is not None:
        expected_normals = np.asarray(expected_parent_normals, dtype=np.float64)
        if expected_normals.shape != (len(F0), 3) or not np.isfinite(expected_normals).all():
            raise BoundaryRefinementError("expected_parent_normals must be finite face-aligned Mx3")
        lengths = np.linalg.norm(expected_normals, axis=1)
        if np.any(lengths <= 1e-30):
            raise BoundaryRefinementError("expected_parent_normals contains a zero face normal")
        expected_normals = expected_normals / lengths[:, None]
    if np.any(F0 < 0) or np.any(F0 >= original_point_count):
        raise BoundaryRefinementError("input face references an invalid/or newly appended point")
    if np.any((F0[:, 0] == F0[:, 1]) | (F0[:, 1] == F0[:, 2]) | (F0[:, 2] == F0[:, 0])):
        raise BoundaryRefinementError("input face contains a repeated vertex")
    if len({tuple(sorted(map(int, f))) for f in F0}) != len(F0):
        raise BoundaryRefinementError("input face table contains duplicate triangles")
    if not edge_midpoint_rows or set(edge_midpoint_rows) != set(midpoint_provenance):
        raise BoundaryRefinementError("selected edges/provenance must be nonempty with identical keys")

    normalized: dict[tuple[int, int], int] = {}
    used_midpoints: set[int] = set()
    for raw_edge, raw_row in edge_midpoint_rows.items():
        try:
            if len(raw_edge) != 2: raise ValueError
            a, b = int(raw_edge[0]), int(raw_edge[1])
            row = int(raw_row)
        except (TypeError, ValueError, IndexError) as exc:
            raise BoundaryRefinementError("edge/midpoint identity is malformed") from exc
        edge = _edge_key(a, b)
        if a == b or edge in normalized:
            raise BoundaryRefinementError(f"duplicate or degenerate selected edge {raw_edge}")
        if edge[0] < 0 or edge[1] >= original_point_count:
            raise BoundaryRefinementError(f"selected edge {edge} is outside original point domain")
        if row < original_point_count or row >= len(P0) or row in used_midpoints:
            raise BoundaryRefinementError(f"edge {edge} midpoint row must be unique and appended")
        normalized[edge] = row
        used_midpoints.add(row)

    incidence: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for fi, face in enumerate(F0):
        a, b, c = map(int, face)
        for u, v in ((a, b), (b, c), (c, a)):
            incidence.setdefault(_edge_key(u, v), []).append((fi, u, v))
    prov_by_edge: dict[tuple[int, int], Mapping[str, object]] = {}
    for raw, record in midpoint_provenance.items():
        edge = _edge_key(int(raw[0]), int(raw[1]))
        if edge in prov_by_edge:
            raise BoundaryRefinementError(f"duplicate provenance for edge {edge}")
        prov_by_edge[edge] = record
    appended = set(range(int(original_point_count), len(P0)))
    expected_rows = appended if expected_midpoint_rows is None else {int(v) for v in expected_midpoint_rows}
    if not expected_rows.issubset(appended) or used_midpoints != expected_rows:
        raise BoundaryRefinementError("appended point rows do not exactly match this layer's midpoint assignment")
    for edge, row in normalized.items():
        uses = incidence.get(edge, [])
        if len(uses) != 2:
            raise BoundaryRefinementError(f"edge {edge} must be an interior two-face edge; got {len(uses)} incident O faces")
        if uses[0][1:] != uses[1][1:][::-1]:
            raise BoundaryRefinementError(f"edge {edge} incident faces do not use opposite orientation")
        if layer == "O":
            _validate_source_sample(edge, row, P0, prov_by_edge[edge])
        else:
            record = prov_by_edge[edge]
            if record.get("source_kind") == "detailed_skin_source_normal_scalar":
                _validate_source_normal_layer_sample(edge, row, P0, record, layer)
            else:
                _validate_offset_sample(edge, row, P0, record, layer)

    face_edges = []
    for f in F0:
        a, b, c = map(int, f)
        face_edges.append((_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)))
    output: list[tuple[int, int, int]] = []
    parent_rows: list[int] = []
    orientation_failures: list[dict[str, object]] = []
    child_cosines: list[float] = []
    child_area2: list[float] = []
    def emit(parent: int, tri: tuple[int, int, int]) -> None:
        coords = np.asarray(P0[list(tri)], dtype=np.float64)
        parent_coords = np.asarray(P0[F0[parent]], dtype=np.float64)
        n0 = np.cross(parent_coords[1] - parent_coords[0], parent_coords[2] - parent_coords[0])
        n1 = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        ref = expected_normals[parent] if expected_normals is not None else n0
        n1len = float(np.linalg.norm(n1)); reflen = float(np.linalg.norm(ref))
        cosine = float(n1 @ ref) / (n1len * reflen) if n1len > 0 and reflen > 0 else float("-inf")
        child_cosines.append(cosine); child_area2.append(n1len)
        if not np.isfinite(n0).all() or not np.isfinite(n1).all() or not np.isfinite(cosine) or n1len <= 0.0 or cosine <= 0.0:
            orientation_failures.append({"parent_face_row": int(parent), "child_triangle": list(map(int, tri)),
                "signed_cosine_to_expected_normal": cosine, "twice_area_m2": n1len})
            if not collect_orientation_failures:
                raise BoundaryRefinementError(f"child triangle from face {parent} is degenerate or orientation-reversed")
        output.append(tri); parent_rows.append(parent)

    for fi, face in enumerate(F0):
        a, b, c = map(int, face)
        eab, ebc, eca = face_edges[fi]
        sab, sbc, sca = eab in normalized, ebc in normalized, eca in normalized
        count = int(sab) + int(sbc) + int(sca)
        if count == 0:
            emit(fi, (a, b, c))
        elif count == 1:
            if sab:
                m = normalized[eab]; tris = ((a, m, c), (m, b, c))
            elif sbc:
                m = normalized[ebc]; tris = ((b, m, a), (m, c, a))
            else:
                m = normalized[eca]; tris = ((c, m, b), (m, a, b))
            for tri in tris: emit(fi, tri)
        elif count == 2:
            if sab and sbc:
                mab, mbc = normalized[eab], normalized[ebc]
                tris = ((mab, b, mbc), (a, mab, c), (mab, mbc, c))
            elif sbc and sca:
                mbc, mca = normalized[ebc], normalized[eca]
                tris = ((mbc, c, mca), (b, mbc, a), (mbc, mca, a))
            else:
                mca, mab = normalized[eca], normalized[eab]
                tris = ((mca, a, mab), (c, mca, b), (mca, mab, b))
            for tri in tris: emit(fi, tri)
        else:
            mab, mbc, mca = normalized[eab], normalized[ebc], normalized[eca]
            for tri in ((a, mab, mca), (b, mbc, mab), (c, mca, mbc), (mab, mbc, mca)):
                emit(fi, tri)

    F1 = np.asarray(output, dtype=F0.dtype)
    parent = np.asarray(parent_rows, dtype=np.int64)
    out_incidence: dict[tuple[int, int], int] = {}
    out_directed: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in F1:
        a, b, c = map(int, face)
        for u, v in ((a, b), (b, c), (c, a)):
            e = _edge_key(u, v); out_incidence[e] = out_incidence.get(e, 0) + 1
            out_directed.setdefault(e, []).append((u, v))
    if _oriented_boundary_chain(F0) != _oriented_boundary_chain(F1):
        raise BoundaryRefinementError("oriented input boundary chain changed during edge subdivision")
    for edge, mid in normalized.items():
        if edge in out_incidence:
            raise BoundaryRefinementError(f"unsplit T-junction remains on edge {edge}")
        for child_edge in (_edge_key(edge[0], mid), _edge_key(mid, edge[1])):
            uses = out_directed.get(child_edge, [])
            if len(uses) != 2:
                raise BoundaryRefinementError(f"replacement edge {child_edge} is not shared by both incident O faces")
            if uses[0] != uses[1][::-1]:
                raise BoundaryRefinementError(f"replacement edge {child_edge} changes the oriented face chain")
    ordered_edges = sorted(normalized)
    source_faces, source_paths, source_endpoints = [], [], []
    for edge in ordered_edges:
        if layer == "O":
            frows, path, endpoints = _validate_source_sample(edge, normalized[edge], P0, prov_by_edge[edge])
        else:
            record = prov_by_edge[edge]
            if record.get("source_kind") == "detailed_skin_source_normal_scalar":
                _validate_source_normal_layer_sample(edge, normalized[edge], P0, record, layer)
                frows = tuple(int(v) for v in record["source_face_rows"])
                path = tuple(int(v) for v in record["source_path_vertex_ids"])
                endpoints = tuple(int(v) for v in record["source_column_ids"])
            else:
                _validate_offset_sample(edge, normalized[edge], P0, record, layer)
                frows = tuple(int(v) for v in record["outer_source_face_rows"])
                path = tuple(int(v) for v in record["outer_source_path_vertex_ids"])
                endpoints = tuple(int(v) for v in record["source_column_ids"])
        source_faces.append(frows); source_paths.append(path); source_endpoints.append(endpoints)
    orientation_diagnostics = {
        "expected_normal_source": "caller supplied per-parent source-oriented normals" if expected_normals is not None else "original parent face normal",
        "collect_orientation_failures": bool(collect_orientation_failures),
        "child_count": int(len(F1)),
        "failed_child_count": int(len(orientation_failures)),
        "failed_children": orientation_failures,
        "minimum_signed_cosine": float(min(child_cosines)) if child_cosines else None,
        "minimum_twice_area_m2": float(min(child_area2)) if child_area2 else None,
        "oriented_boundary_chain_preserved": True,
    }
    return BoundaryRefinement(
        points=P0.copy(), faces=F1, child_parent_face_rows=parent,
        child_parent_face_markers=markers[parent].copy(),
        selected_edges=np.asarray(ordered_edges, dtype=np.int64),
        edge_midpoint_rows=np.asarray([normalized[e] for e in ordered_edges], dtype=np.int64),
        midpoint_source_face_rows=tuple(source_faces),
        midpoint_source_path_vertex_ids=tuple(source_paths),
        midpoint_source_endpoint_vertex_ids=tuple(source_endpoints),
        orientation_diagnostics=orientation_diagnostics,
    )


@dataclass(frozen=True)
class ConnectedBoundaryRefinement:
    """One explicit column-aligned O/I/D surface refinement, topology only."""
    points_by_frame: Mapping[str, np.ndarray]
    faces_by_layer: Mapping[str, np.ndarray]
    child_parent_face_rows_by_layer: Mapping[str, np.ndarray]
    child_parent_face_markers_by_layer: Mapping[str, np.ndarray]
    selected_outer_edges: np.ndarray
    source_column_ids_by_edge: tuple[tuple[int, int], ...]
    source_layer_edges_by_edge: tuple[Mapping[str, tuple[int, int]], ...]
    midpoint_rows_by_layer: Mapping[str, np.ndarray]
    midpoint_source_face_rows_by_edge: tuple[tuple[int, ...], ...]
    midpoint_source_path_vertex_ids_by_edge: tuple[tuple[int, ...], ...]
    outer_path_provenance_by_edge: tuple[Mapping[str, object], ...]
    sample_fractions_from_path_start: np.ndarray
    source_column_weights_by_edge: np.ndarray
    offset_provenance_by_layer: Mapping[str, tuple[Mapping[str, object], ...]]


def subdivide_registered_column_edges(
    points_by_frame: Mapping[str, np.ndarray],
    faces_by_layer: Mapping[str, np.ndarray],
    parent_face_markers_by_layer: Mapping[str, np.ndarray],
    column_layer_rows: Mapping[int, Mapping[str, int]],
    selected_column_edges: Sequence[Mapping[str, object]],
    *, reference_field_sha256: str,
    source_column_mapping_sha256: str,
) -> ConnectedBoundaryRefinement:
    """Split mapped O/I/D column edges using detailed O paths and registered offsets.

    Each selected item supplies ``column_ids=(a,b)``, explicit ``layer_edges``
    for O/I/D, ``sample_fraction`` measured along its detailed O source path, and
    ``outer_source_by_frame`` with one certified detailed-surface path record per
    frame. ``column_layer_rows`` explicitly maps column IDs to point rows for all
    three layers; no contiguous 3*N or layer-offset assumption is made.

    For every frame, the new I and D coordinates are constructed as
    ``O_new + (1-s)*(I_old-O_old)_a + s*(I_old-O_old)_b`` and analogously for D.
    This is a source-registered offset-field interpolation, not a straight chord
    midpoint and not a measured anatomical sample. The caller must later run the
    capacity and full geometry gates. Face markers are inherited from exact
    parent rows, but no conservative material ancestry is produced here.
    """
    layers = ("O", "I", "D")
    if set(points_by_frame) == set() or any(not isinstance(k, str) or not k for k in points_by_frame):
        raise BoundaryRefinementError("at least one named coordinate frame is required")
    if set(faces_by_layer) != set(layers) or set(parent_face_markers_by_layer) != set(layers):
        raise BoundaryRefinementError("faces and parent markers must explicitly contain exactly O/I/D")
    if len(reference_field_sha256) != 64 or any(c not in "0123456789abcdef" for c in reference_field_sha256.lower()):
        raise BoundaryRefinementError("reference_field_sha256 must be pinned")
    if len(source_column_mapping_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_column_mapping_sha256.lower()):
        raise BoundaryRefinementError("source_column_mapping_sha256 must be pinned")
    frames: dict[str, np.ndarray] = {}
    shape = None
    for name, raw in points_by_frame.items():
        points = np.asarray(raw)
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind != "f" or not np.isfinite(points).all():
            raise BoundaryRefinementError(f"frame {name}: points must be finite floating Nx3")
        if shape is None: shape = points.shape
        if points.shape != shape:
            raise BoundaryRefinementError("all coordinate frames must have the same original point domain")
        frames[name] = points
    assert shape is not None
    original_count = shape[0]
    if original_count == 0 or not selected_column_edges:
        raise BoundaryRefinementError("nonempty original point and selected column-edge domains are required")
    if not isinstance(column_layer_rows, Mapping) or not column_layer_rows:
        raise BoundaryRefinementError("explicit source column map is required")
    for column_id, layer_rows in column_layer_rows.items():
        try:
            cid = int(column_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BoundaryRefinementError("source column IDs must be integer keys") from exc
        if cid < 0 or not isinstance(layer_rows, Mapping) or set(layer_rows) != set(layers):
            raise BoundaryRefinementError(f"source column {column_id}: explicit O/I/D row map is required")
        try:
            row_values = [int(layer_rows[layer]) for layer in layers]
        except (TypeError, ValueError, OverflowError) as exc:
            raise BoundaryRefinementError(f"source column {column_id}: malformed O/I/D node row") from exc
        if any(row < 0 or row >= original_count for row in row_values):
            raise BoundaryRefinementError(f"source column {column_id}: node row outside original point domain")

    normalized_columns: dict[tuple[int, int], Mapping[str, object]] = {}
    parsed: list[tuple[tuple[int, int], Mapping[str, object], tuple[int, int], dict[str, tuple[int, int]], float, np.ndarray]] = []
    used_edges_by_layer = {layer: set() for layer in layers}
    for index, item in enumerate(selected_column_edges):
        if not isinstance(item, Mapping):
            raise BoundaryRefinementError(f"selected column edge {index}: record must be a mapping")
        try:
            raw_columns = tuple(int(v) for v in item["column_ids"])
            fraction = float(item["sample_fraction"])
            raw_layer_edges = item["layer_edges"]
            outer_by_frame = item["outer_source_by_frame"]
            detail_endpoints = tuple(int(v) for v in item["column_endpoint_detail_vertex_ids"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise BoundaryRefinementError(f"selected column edge {index}: missing/malformed explicit mapping") from exc
        if len(raw_columns) != 2 or raw_columns[0] < 0 or raw_columns[1] < 0 or raw_columns[0] == raw_columns[1]:
            raise BoundaryRefinementError(f"selected column edge {index}: column IDs must be two distinct nonnegative IDs")
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise BoundaryRefinementError(f"selected column edge {index}: sample_fraction must lie strictly inside (0,1)")
        if len(detail_endpoints) != 2 or min(detail_endpoints) < 0 or detail_endpoints[0] == detail_endpoints[1]:
            raise BoundaryRefinementError(f"selected column edge {index}: detail endpoint IDs must be two distinct nonnegative IDs")
        if raw_columns[0] not in column_layer_rows or raw_columns[1] not in column_layer_rows:
            raise BoundaryRefinementError(f"selected column edge {index}: source column has no explicit O/I/D row map")
        if not isinstance(raw_layer_edges, Mapping) or set(raw_layer_edges) != set(layers):
            raise BoundaryRefinementError(f"selected column edge {index}: explicit O/I/D edge map is required")
        if not isinstance(outer_by_frame, Mapping) or set(outer_by_frame) != set(frames):
            raise BoundaryRefinementError(f"selected column edge {index}: detailed O path evidence must be supplied for every frame")
        layer_edges: dict[str, tuple[int, int]] = {}
        for layer in layers:
            try:
                edge = tuple(int(v) for v in raw_layer_edges[layer])
            except (TypeError, ValueError) as exc:
                raise BoundaryRefinementError(f"selected column edge {index}: malformed {layer} edge") from exc
            if len(edge) != 2 or edge[0] == edge[1]:
                raise BoundaryRefinementError(f"selected column edge {index}: degenerate {layer} edge")
            try:
                mapped = (int(column_layer_rows[raw_columns[0]][layer]), int(column_layer_rows[raw_columns[1]][layer]))
            except (KeyError, TypeError, ValueError) as exc:
                raise BoundaryRefinementError(f"selected column edge {index}: missing explicit {layer} endpoint row") from exc
            if set(edge) != set(mapped):
                raise BoundaryRefinementError(f"selected column edge {index}: {layer} edge does not connect the mapped source columns")
            if any(v < 0 or v >= original_count for v in edge):
                raise BoundaryRefinementError(f"selected column edge {index}: {layer} endpoint outside original point domain")
            key = _edge_key(*edge)
            if key in used_edges_by_layer[layer]:
                raise BoundaryRefinementError(f"selected column edge {index}: duplicate {layer} edge mapping")
            used_edges_by_layer[layer].add(key)
            layer_edges[layer] = edge
        okey = _edge_key(*layer_edges["O"])
        if okey in normalized_columns:
            raise BoundaryRefinementError(f"selected column edge {index}: duplicate O topology edge")
        for frame_name, source in outer_by_frame.items():
            if not isinstance(source, Mapping):
                raise BoundaryRefinementError(f"selected column edge {index}: malformed O path for frame {frame_name}")
            _validate_source_sample(okey, 0, np.asarray([source.get("sample_point_world_m", [np.nan]*3)], dtype=np.float64), source)
        first_outer = outer_by_frame[next(iter(frames))]
        path_endpoints = tuple(int(v) for v in first_outer.get("edge_endpoint_source_vertex_ids", ()))
        if len(path_endpoints) != 2 or set(path_endpoints) != set(detail_endpoints):
            raise BoundaryRefinementError(f"selected column edge {index}: detailed path endpoints do not map to the explicit endpoint columns")
        if path_endpoints == detail_endpoints:
            column_weights = np.asarray([1.0 - fraction, fraction], dtype=np.float64)
        else:
            column_weights = np.asarray([fraction, 1.0 - fraction], dtype=np.float64)
        normalized_columns[okey] = item
        parsed.append((okey, item, raw_columns, layer_edges, fraction, column_weights))
    parsed.sort(key=lambda x: x[0])

    # Allocate one appended point per column-edge per layer, explicitly recorded.
    midpoint_rows_by_edge: dict[tuple[int, int], dict[str, int]] = {}
    appended_by_frame: dict[str, list[np.ndarray]] = {frame: [] for frame in frames}
    provenance_by_layer_edge: dict[str, dict[tuple[int, int], Mapping[str, object]]] = {layer: {} for layer in layers}
    column_ids_by_edge = []
    layer_edges_by_edge = []
    path_faces_by_edge = []
    path_vertices_by_edge = []
    outer_provenance_by_edge = []
    fractions = []
    weights_by_edge = []
    for edge_index, (okey, item, columns, layer_edges, fraction, column_weights) in enumerate(parsed):
        row_by_layer = {layer: original_count + edge_index * 3 + li for li, layer in enumerate(layers)}
        midpoint_rows_by_edge[okey] = row_by_layer
        column_ids_by_edge.append(columns)
        layer_edges_by_edge.append({layer: tuple(layer_edges[layer]) for layer in layers})
        fractions.append(fraction)
        weights_by_edge.append(column_weights.copy())
        source_by_frame = item["outer_source_by_frame"]
        source_ref = source_by_frame[next(iter(frames))]
        # Every O frame carries a complete path certificate; path IDs/faces must be stable across frames.
        path_ids_ref = tuple(int(v) for v in source_ref["source_path_vertex_ids"])
        face_ids_ref = tuple(int(v) for v in source_ref["source_face_rows"])
        path_vertices_by_edge.append(path_ids_ref); path_faces_by_edge.append(face_ids_ref)
        outer_provenance_by_edge.append(dict(source_ref))
        for frame_name, base in frames.items():
            source = source_by_frame[frame_name]
            if tuple(int(v) for v in source["source_path_vertex_ids"]) != path_ids_ref or tuple(int(v) for v in source["source_face_rows"]) != face_ids_ref:
                raise BoundaryRefinementError(f"edge {okey}: source path/face identity changes between coordinate frames")
            sample_o = np.asarray(source["sample_point_world_m"], dtype=np.float64)
            if sample_o.shape != (3,) or not np.isfinite(sample_o).all():
                raise BoundaryRefinementError(f"edge {okey}: O midpoint sample must be finite xyz")
            appended_by_frame[frame_name].append(sample_o.astype(base.dtype))
            weights = column_weights
            for layer in ("I", "D"):
                rows_a = column_layer_rows[columns[0]]
                rows_b = column_layer_rows[columns[1]]
                oa = base[int(rows_a["O"])].astype(np.float64)
                ob = base[int(rows_b["O"])].astype(np.float64)
                la = base[int(rows_a[layer])].astype(np.float64)
                lb = base[int(rows_b[layer])].astype(np.float64)
                offset_a, offset_b = la - oa, lb - ob
                offset_mid = weights[0] * offset_a + weights[1] * offset_b
                sample_layer = sample_o + offset_mid
                appended_by_frame[frame_name].append(sample_layer.astype(base.dtype))
                record = {
                    "source_kind": "registered_layer_offset_interpolation", "layer": layer,
                    "source_column_ids": list(columns), "source_column_weights": weights.tolist(),
                    "outer_source_path_vertex_ids": list(path_ids_ref),
                    "outer_source_face_rows": list(face_ids_ref),
                    "outer_midpoint_world_m": sample_o.tolist(),
                    "offset_endpoint_a_m": offset_a.tolist(), "offset_endpoint_b_m": offset_b.tolist(),
                    "sample_point_world_m": sample_layer.tolist(),
                    "reference_field_sha256": reference_field_sha256.lower(),
                    "source_column_mapping_sha256": source_column_mapping_sha256.lower(),
                }
                provenance_by_layer_edge[layer][layer_edges[layer]] = record
                # Shared edge key is canonicalized later; ensure the explicit edge has matching columns.
        provenance_by_layer_edge["O"][layer_edges["O"]] = source_ref

    points_out: dict[str, np.ndarray] = {}
    for frame_name, base in frames.items():
        appended = np.asarray(appended_by_frame[frame_name], dtype=base.dtype).reshape(-1, 3)
        points_out[frame_name] = np.concatenate((base.copy(), appended), axis=0)

    faces_out: dict[str, np.ndarray] = {}
    parent_rows_out: dict[str, np.ndarray] = {}
    parent_markers_out: dict[str, np.ndarray] = {}
    midpoint_rows_layer: dict[str, np.ndarray] = {}
    layer_topology_by_frame: dict[tuple[str, str], BoundaryRefinement] = {}
    for layer in layers:
        edge_rows = {edge: midpoint_rows_by_edge[okey][layer]
                     for okey, _, _, layer_edges, _, _ in parsed
                     for edge in (_edge_key(*layer_edges[layer]),)}
        expected_rows = [midpoint_rows_by_edge[okey][layer] for okey, *_ in parsed]
        midpoint_rows_layer[layer] = np.asarray(expected_rows, dtype=np.int64)
        for frame_name, frame_points in points_out.items():
            # Match provenance keys to canonical topology edge keys.
            provenance = {_edge_key(*edge): record for edge, record in provenance_by_layer_edge[layer].items()}
            result = subdivide_oriented_surface_edges(
                frame_points, faces_by_layer[layer], edge_rows, provenance,
                original_point_count=original_count,
                parent_face_markers=parent_face_markers_by_layer[layer],
                layer=layer, expected_midpoint_rows=expected_rows)
            layer_topology_by_frame[(layer, frame_name)] = result
        first = layer_topology_by_frame[(layer, next(iter(frames)))]
        for frame_name in frames:
            other = layer_topology_by_frame[(layer, frame_name)]
            if not np.array_equal(first.faces, other.faces) or not np.array_equal(first.child_parent_face_rows, other.child_parent_face_rows):
                raise BoundaryRefinementError(f"{layer}: generated topology differs across coordinate frames")
        faces_out[layer] = first.faces
        parent_rows_out[layer] = first.child_parent_face_rows
        parent_markers_out[layer] = first.child_parent_face_markers

    return ConnectedBoundaryRefinement(
        points_by_frame=points_out, faces_by_layer=faces_out,
        child_parent_face_rows_by_layer=parent_rows_out,
        child_parent_face_markers_by_layer=parent_markers_out,
        selected_outer_edges=np.asarray([edge for edge in sorted(midpoint_rows_by_edge)], dtype=np.int64),
        source_column_ids_by_edge=tuple(column_ids_by_edge),
        source_layer_edges_by_edge=tuple(layer_edges_by_edge),
        midpoint_rows_by_layer=midpoint_rows_layer,
        midpoint_source_face_rows_by_edge=tuple(path_faces_by_edge),
        midpoint_source_path_vertex_ids_by_edge=tuple(path_vertices_by_edge),
        outer_path_provenance_by_edge=tuple(outer_provenance_by_edge),
        sample_fractions_from_path_start=np.asarray(fractions, dtype=np.float64),
        source_column_weights_by_edge=np.asarray(weights_by_edge, dtype=np.float64),
        offset_provenance_by_layer={layer: tuple(provenance_by_layer_edge[layer][edge]
            for edge in provenance_by_layer_edge[layer]) for layer in layers if layer != "O"},
    )

@dataclass(frozen=True)
class AffineSurfaceRefinement:
    """Same-surface affine subdivision with explicit source-face/edge ancestry.

    This is a topology/field-prolongation primitive. It does not reconstruct a
    curved source surface, establish bone clearance, or transfer physical mass.
    """
    points: np.ndarray
    faces: np.ndarray
    child_parent_face_rows: np.ndarray
    child_parent_face_markers: np.ndarray
    child_parent_face_barycentric: np.ndarray
    selected_edges: np.ndarray
    edge_midpoint_rows: np.ndarray
    affected_parent_face_rows: np.ndarray
    face_centroid_rows: np.ndarray
    refinement_node_kind: np.ndarray
    refinement_node_parent_edge_rows: np.ndarray
    refinement_node_parent_edge_weights: np.ndarray
    refinement_node_parent_face_indptr: np.ndarray
    refinement_node_parent_face_rows: np.ndarray
    refinement_node_parent_face_barycentric: np.ndarray
    source_prolongation_indptr: np.ndarray
    source_prolongation_node_rows: np.ndarray
    source_prolongation_weights: np.ndarray
    boundary_components_before: int
    boundary_components_after: int
    boundary_edges_before: int
    boundary_edges_after: int
    relative_parent_area_error: np.ndarray


def _affine_edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    incidence: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for fi, face in enumerate(np.asarray(faces, dtype=np.int64)):
        a, b, c = map(int, face)
        for u, v in ((a, b), (b, c), (c, a)):
            incidence.setdefault(_edge_key(u, v), []).append((fi, u, v))
    return incidence


def _affine_boundary_summary(edges: set[tuple[int, int]]) -> tuple[int, dict[int, int]]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    degree = {v: len(neighbours) for v, neighbours in adjacency.items()}
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
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return components, degree


def subdivide_crossed_faces_affine(
    points: np.ndarray,
    faces: np.ndarray,
    crossed_face_rows: Sequence[int] | np.ndarray,
    *,
    parent_face_markers: np.ndarray | None = None,
) -> AffineSurfaceRefinement:
    """Conformingly refine crossed triangles with shared edge mids and face fans.

    Every edge of every crossed face is bisected once globally. Every face
    incident to a selected edge is then fanned from its affine centroid to its
    ordered boundary polygon, so adjacent faces share midpoint rows and no
    T-junction is introduced. Original vertices and all unaffected faces remain
    unchanged. New rows have explicit edge or parent-face ancestry and a convex
    CSR prolongation from the original point domain.

    Boundary edges may be bisected; their geometric segments and boundary-loop
    count are preserved. A selected edge with more than two incident faces, an
    open/nonmanifold boundary, or inconsistent directed winding is rejected.
    """
    X = np.asarray(points)
    Fraw = np.asarray(faces)
    if X.ndim != 2 or X.shape[1] != 3 or X.dtype.kind != "f" or X.dtype.itemsize not in (4, 8) or not np.isfinite(X).all():
        raise BoundaryRefinementError("points must be finite Float32/Float64 Nx3")
    if Fraw.ndim != 2 or Fraw.shape[1] != 3 or Fraw.dtype.kind not in "iu" or len(Fraw) == 0:
        raise BoundaryRefinementError("faces must be a nonempty integer Mx3 array")
    npoints, nfaces = len(X), len(Fraw)
    F = Fraw.astype(np.int64, copy=False)
    if npoints < 3 or np.any(F < 0) or np.any(F >= npoints):
        raise BoundaryRefinementError("faces reference a point outside the original point domain")
    if np.any((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 2] == F[:, 0])):
        raise BoundaryRefinementError("input face contains a repeated vertex")
    if len(np.unique(np.sort(F, axis=1), axis=0)) != nfaces:
        raise BoundaryRefinementError("input face table contains duplicate triangles")
    if parent_face_markers is None:
        markers = np.arange(1, nfaces + 1, dtype=np.int64)
    else:
        markers = np.asarray(parent_face_markers)
        if markers.shape != (nfaces,) or markers.dtype.kind not in "iu":
            raise BoundaryRefinementError("parent_face_markers must be an integer M-vector")

    crossed_raw = np.asarray(crossed_face_rows)
    if crossed_raw.ndim != 1 or crossed_raw.dtype.kind not in "iu" or len(crossed_raw) == 0:
        raise BoundaryRefinementError("crossed_face_rows must be a nonempty integer vector")
    crossed = np.unique(crossed_raw.astype(np.int64, copy=False))
    if np.any(crossed < 0) or np.any(crossed >= nfaces):
        raise BoundaryRefinementError("crossed face row is outside the input face domain")

    incidence = _affine_edge_incidence(F)
    if any(len(uses) > 2 for uses in incidence.values()):
        raise BoundaryRefinementError("input surface has a nonmanifold edge")
    # A coherent oriented triangle surface must use every interior edge in
    # opposite directions. Validate the whole domain before constructing fans.
    for edge, uses in incidence.items():
        if len(uses) == 2 and (uses[0][1], uses[0][2]) != (uses[1][2], uses[1][1]):
            raise BoundaryRefinementError(f"input interior edge {edge} has inconsistent directed winding")
    old_boundary = {edge for edge, uses in incidence.items() if len(uses) == 1}
    old_components, old_degrees = _affine_boundary_summary(old_boundary)
    if any(degree != 2 for degree in old_degrees.values()):
        raise BoundaryRefinementError("input boundary is not a union of simple closed loops")

    selected_edges: set[tuple[int, int]] = set()
    for fi in crossed:
        a, b, c = map(int, F[int(fi)])
        selected_edges.update((_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)))
    affected = sorted({fi for edge in selected_edges for fi, _, _ in incidence[edge]})
    edges_sorted = sorted(selected_edges)

    edge_midpoint_rows = {edge: npoints + i for i, edge in enumerate(edges_sorted)}
    face_centroid_rows = {fi: npoints + len(edges_sorted) + i for i, fi in enumerate(affected)}
    all_new_points: list[np.ndarray] = []
    node_kind = np.zeros(npoints, dtype=np.int8)
    parent_edge_rows = np.full((npoints, 2), -1, dtype=np.int64)
    parent_edge_weights = np.full((npoints, 2), -1.0, dtype=np.float64)
    parent_face_indptr = [0] * (npoints + 1)
    parent_face_rows: list[int] = []
    parent_face_bary: list[np.ndarray] = []
    prolong_indptr = [0]
    prolong_rows: list[int] = []
    prolong_weights: list[float] = []

    for row in range(npoints):
        prolong_rows.append(row)
        prolong_weights.append(1.0)
        prolong_indptr.append(len(prolong_rows))

    for edge in edges_sorted:
        a, b = edge
        weights = np.array([0.5, 0.5], dtype=np.float64)
        q = np.sum(X[[a, b]].astype(np.float64) * weights[:, None], axis=0).astype(X.dtype)
        if not np.isfinite(q).all() or np.array_equal(q, X[a]) or np.array_equal(q, X[b]):
            raise BoundaryRefinementError(f"Float32/Float64 midpoint collapses for selected edge {edge}")
        all_new_points.append(q)
        node_kind = np.append(node_kind, np.int8(1))
        parent_edge_rows = np.vstack((parent_edge_rows, np.asarray(edge, dtype=np.int64)))
        parent_edge_weights = np.vstack((parent_edge_weights, weights))
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

    for fi in affected:
        tri = tuple(map(int, F[fi]))
        weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
        q = np.sum(X[np.asarray(tri)].astype(np.float64) * weights[:, None], axis=0).astype(X.dtype)
        if not np.isfinite(q).all() or any(np.array_equal(q, X[v]) for v in tri):
            raise BoundaryRefinementError(f"Float32/Float64 centroid collapses for parent face {fi}")
        all_new_points.append(q)
        node_kind = np.append(node_kind, np.int8(2))
        parent_edge_rows = np.vstack((parent_edge_rows, np.asarray((-1, -1), dtype=np.int64)))
        parent_edge_weights = np.vstack((parent_edge_weights, np.full(2, -1.0, dtype=np.float64)))
        bary = weights.copy()
        parent_face_rows.append(int(fi))
        parent_face_bary.append(bary)
        parent_face_indptr.append(len(parent_face_rows))
        prolong_rows.extend(tri)
        prolong_weights.extend((1.0 / 3.0,) * 3)
        prolong_indptr.append(len(prolong_rows))

    Xout = np.concatenate((X.copy(), np.asarray(all_new_points, dtype=X.dtype).reshape(-1, 3)), axis=0)
    if not np.array_equal(Xout[:npoints].view(np.uint8), X.view(np.uint8)):
        raise AssertionError("original point prefix changed")

    affected_set = set(affected)
    child_faces: list[tuple[int, int, int]] = []
    child_parent: list[int] = []
    child_bary: list[np.ndarray] = []
    identity = np.eye(3, dtype=np.float64)
    for fi, raw_tri in enumerate(F):
        tri = tuple(map(int, raw_tri))
        if fi not in affected_set:
            child_faces.append(tri)
            child_parent.append(fi)
            child_bary.append(identity.copy())
            continue
        polygon_rows: list[int] = []
        polygon_bary: list[np.ndarray] = []
        for j in range(3):
            a, b = tri[j], tri[(j + 1) % 3]
            ba = np.zeros(3, dtype=np.float64); ba[j] = 1.0
            polygon_rows.append(a); polygon_bary.append(ba)
            edge = _edge_key(a, b)
            if edge in edge_midpoint_rows:
                bm = np.zeros(3, dtype=np.float64)
                bm[tri.index(edge[0])] = 0.5
                bm[tri.index(edge[1])] = 0.5
                polygon_rows.append(edge_midpoint_rows[edge]); polygon_bary.append(bm)
        center = face_centroid_rows[fi]
        center_bary = np.full(3, 1.0 / 3.0, dtype=np.float64)
        for j, row_a in enumerate(polygon_rows):
            k = (j + 1) % len(polygon_rows)
            row_b = polygon_rows[k]
            child_faces.append((row_a, row_b, center))
            child_parent.append(fi)
            child_bary.append(np.vstack((polygon_bary[j], polygon_bary[k], center_bary)))

    Fout = np.asarray(child_faces, dtype=F.dtype)
    child_parent_rows = np.asarray(child_parent, dtype=np.int64)
    child_markers = markers[child_parent_rows].copy()
    child_parent_bary = np.asarray(child_bary, dtype=np.float64)

    new_incidence = _affine_edge_incidence(Fout)
    if any(len(uses) > 2 for uses in new_incidence.values()):
        raise AssertionError("refinement produced a nonmanifold edge")
    new_boundary = {edge for edge, uses in new_incidence.items() if len(uses) == 1}
    expected_boundary: set[tuple[int, int]] = set()
    for a, b in old_boundary:
        edge = _edge_key(a, b)
        if edge in edge_midpoint_rows:
            mid = edge_midpoint_rows[edge]
            expected_boundary.add(_edge_key(a, mid))
            expected_boundary.add(_edge_key(mid, b))
        else:
            expected_boundary.add(edge)
    if new_boundary != expected_boundary:
        raise AssertionError("boundary chain changed beyond selected-edge bisection")
    new_components, new_degrees = _affine_boundary_summary(new_boundary)
    if new_components != old_components or any(d != 2 for d in new_degrees.values()):
        raise AssertionError("boundary loop topology changed")

    old_cross = np.cross(X[F[:, 1]].astype(np.float64) - X[F[:, 0]].astype(np.float64),
                         X[F[:, 2]].astype(np.float64) - X[F[:, 0]].astype(np.float64))
    new_cross = np.cross(Xout[Fout[:, 1]].astype(np.float64) - Xout[Fout[:, 0]].astype(np.float64),
                         Xout[Fout[:, 2]].astype(np.float64) - Xout[Fout[:, 0]].astype(np.float64))
    new_area2 = np.linalg.norm(new_cross, axis=1)
    if not np.isfinite(new_area2).all() or np.any(new_area2 <= 0.0):
        raise BoundaryRefinementError("Float32/Float64 refinement produced a zero-area child face")
    orientation = np.einsum("ij,ij->i", new_cross, old_cross[child_parent_rows])
    if np.any(orientation <= 0.0):
        raise BoundaryRefinementError("child face winding is not positive relative to its parent")
    old_area = 0.5 * np.linalg.norm(old_cross, axis=1)
    new_area = 0.5 * new_area2
    area_sum = np.bincount(child_parent_rows, weights=new_area, minlength=nfaces)
    area_error = np.abs(area_sum - old_area) / np.maximum(old_area, np.finfo(np.float64).tiny)

    ptr = np.asarray(prolong_indptr, dtype=np.int64)
    cols = np.asarray(prolong_rows, dtype=np.int64)
    weights = np.asarray(prolong_weights, dtype=np.float64)
    if len(ptr) != len(Xout) + 1 or ptr[0] != 0 or ptr[-1] != len(cols) or len(cols) != len(weights):
        raise AssertionError("source prolongation CSR shape mismatch")
    for row in range(len(Xout)):
        start, stop = int(ptr[row]), int(ptr[row + 1])
        if stop <= start or np.any(cols[start:stop] < 0) or np.any(cols[start:stop] >= npoints):
            raise AssertionError("source prolongation has an empty row or invalid source row")
        if len(np.unique(cols[start:stop])) != stop - start or np.any(weights[start:stop] < 0) or abs(float(np.sum(weights[start:stop])) - 1.0) > 1e-14:
            raise AssertionError("source prolongation row is not a convex source map")
        reconstructed = np.sum(X[cols[start:stop]].astype(np.float64) * weights[start:stop, None], axis=0).astype(X.dtype)
        if not np.array_equal(reconstructed, Xout[row]):
            raise AssertionError(f"source prolongation does not reproduce output point row {row}")

    return AffineSurfaceRefinement(
        points=Xout, faces=Fout, child_parent_face_rows=child_parent_rows,
        child_parent_face_markers=child_markers,
        child_parent_face_barycentric=child_parent_bary,
        selected_edges=np.asarray(edges_sorted, dtype=np.int64).reshape(-1, 2),
        edge_midpoint_rows=np.asarray([edge_midpoint_rows[e] for e in edges_sorted], dtype=np.int64),
        affected_parent_face_rows=np.asarray(affected, dtype=np.int64),
        face_centroid_rows=np.asarray([face_centroid_rows[fi] for fi in affected], dtype=np.int64),
        refinement_node_kind=node_kind,
        refinement_node_parent_edge_rows=parent_edge_rows,
        refinement_node_parent_edge_weights=parent_edge_weights,
        refinement_node_parent_face_indptr=np.asarray(parent_face_indptr, dtype=np.int64),
        refinement_node_parent_face_rows=np.asarray(parent_face_rows, dtype=np.int64),
        refinement_node_parent_face_barycentric=np.asarray(parent_face_bary, dtype=np.float64).reshape(-1, 3),
        source_prolongation_indptr=ptr, source_prolongation_node_rows=cols,
        source_prolongation_weights=weights,
        boundary_components_before=old_components, boundary_components_after=new_components,
        boundary_edges_before=len(old_boundary), boundary_edges_after=len(new_boundary),
        relative_parent_area_error=area_error,
    )


def prolong_registered_depth_fields(
    source_depth_fields: np.ndarray,
    source_prolongation_indptr: np.ndarray,
    source_prolongation_node_rows: np.ndarray,
    source_prolongation_weights: np.ndarray,
    *,
    expected_source_field_sha256: str | None = None,
    source_field_sha256: str | None = None,
) -> np.ndarray:
    """Convexly prolong positive registered depth fields using an explicit CSR map.

    Hash arguments are optional as a pair. When present they bind the field
    table to the caller's immutable source pin; a mismatch is rejected. This is
    scalar field ancestry only, not a physical mass or material-transfer map.
    """
    H = np.asarray(source_depth_fields)
    if H.ndim != 2 or H.shape[0] == 0 or H.shape[1] == 0 or H.dtype.kind != "f" or H.dtype.itemsize not in (4, 8):
        raise BoundaryRefinementError("source_depth_fields must be a nonempty floating NxK array")
    if not np.isfinite(H).all() or np.any(H <= 0):
        raise BoundaryRefinementError("source depth fields must be finite and strictly positive")
    if (expected_source_field_sha256 is None) != (source_field_sha256 is None):
        raise BoundaryRefinementError("source field hash pin and observed hash must be provided together")
    if expected_source_field_sha256 is not None:
        valid_hash = lambda h: isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h.lower())
        if not valid_hash(expected_source_field_sha256) or not valid_hash(source_field_sha256):
            raise BoundaryRefinementError("source field hashes must be SHA-256 hex strings")
        if expected_source_field_sha256.lower() != source_field_sha256.lower():
            raise BoundaryRefinementError("source depth field SHA-256 does not match the expected pin")

    ptr = np.asarray(source_prolongation_indptr)
    cols = np.asarray(source_prolongation_node_rows)
    weights = np.asarray(source_prolongation_weights)
    if ptr.ndim != 1 or ptr.dtype.kind not in "iu" or len(ptr) < 2:
        raise BoundaryRefinementError("source prolongation indptr must be an integer vector")
    if cols.ndim != 1 or cols.dtype.kind not in "iu" or weights.ndim != 1 or weights.dtype.kind != "f":
        raise BoundaryRefinementError("source prolongation indices/weights must be vectors")
    ptr = ptr.astype(np.int64, copy=False); cols = cols.astype(np.int64, copy=False)
    weights = weights.astype(np.float64, copy=False)
    if ptr[0] != 0 or np.any(np.diff(ptr) < 0) or ptr[-1] != len(cols) or len(cols) != len(weights):
        raise BoundaryRefinementError("source prolongation CSR pointers are inconsistent")
    if np.any(cols < 0) or np.any(cols >= len(H)) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise BoundaryRefinementError("source prolongation contains invalid columns or weights")
    nout = len(ptr) - 1
    out = np.empty((nout, H.shape[1]), dtype=H.dtype)
    for row in range(nout):
        start, stop = int(ptr[row]), int(ptr[row + 1])
        if stop <= start:
            raise BoundaryRefinementError(f"source prolongation row {row} is empty")
        row_cols = cols[start:stop]
        row_weights = weights[start:stop]
        if len(np.unique(row_cols)) != len(row_cols):
            raise BoundaryRefinementError(f"source prolongation row {row} repeats a source column")
        if abs(float(np.sum(row_weights)) - 1.0) > 1e-14:
            raise BoundaryRefinementError(f"source prolongation row {row} is not a convex partition")
        value = np.sum(H[row_cols].astype(np.float64) * row_weights[:, None], axis=0)
        out[row] = value.astype(H.dtype)
    if not np.isfinite(out).all() or np.any(out <= 0):
        raise BoundaryRefinementError("prolongation produced a nonpositive or nonfinite depth")
    return out
