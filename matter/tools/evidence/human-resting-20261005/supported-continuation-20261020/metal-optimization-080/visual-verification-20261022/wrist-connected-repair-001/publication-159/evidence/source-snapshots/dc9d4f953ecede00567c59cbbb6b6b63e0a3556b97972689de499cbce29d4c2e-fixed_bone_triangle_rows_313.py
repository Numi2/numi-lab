"""Fixed-bone triangle barriers adapted from the existing 026 self-separation owner.

The owner computes rows for two moving triangles. This adapter treats the bone
triangle as fixed, removes its displacement columns, and maps skin-local vertex
columns back to the supplied skin mesh. Rows remain construction constraints;
exact emitted-geometry audits remain the acceptance gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np


_REQUIRED = (
    "skin_face_row",
    "bank_body",
    "stable_bone_id",
    "bank_triangle_stable_id",
    "fixed_triangle_world_mm",
)


def _integer_id(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer ID")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _area_vector(triangle):
    return np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])


def build_fixed_triangle_rows(points_mm, skin_faces, records, pair_owner, margin_fraction=0.25):
    """Return 026-format rows mapped to skin nodes and typed pair records.

    points_mm is an (N,3) finite skin point array in millimetres. skin_faces is
    an integer (F,3) table. Each record contains skin_face_row, bank_body,
    stable_bone_id, bank_triangle_stable_id and fixed_triangle_world_mm (3,3).
    pair_owner is the existing module/object exposing build_pair_rows.

    Returned row nodes are global indices into points_mm; coefficients have
    shape (row-node-count,3), are in world millimetres, and include only skin
    DOFs. The bone triangle displacement is exactly zero. The 026 lower value
    is retained unchanged because its row was evaluated on all six baseline
    points before the bone coefficients were removed. Typed pair records use
    canonical fields skin_face_row, bank_body, stable_bone_id and
    bank_triangle_stable_id, with rows_start/rows_stop indexing the returned
    rows. Nonseparable or baseline-infeasible inputs fail closed in 026.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    raw_faces = np.asarray(skin_faces)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N,3)")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or raw_faces.dtype.kind not in "iu":
        raise ValueError("skin_faces must be an integer array with shape (F,3)")
    faces = raw_faces.astype(np.int64, copy=False)
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(points)):
        raise ValueError("skin_faces contain an out-of-range point index")
    if faces.size and np.any((faces[:, 0] == faces[:, 1]) |
                             (faces[:, 1] == faces[:, 2]) |
                             (faces[:, 2] == faces[:, 0])):
        raise ValueError("skin_faces must have three distinct vertices")
    try:
        margin = float(margin_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("margin_fraction must be finite and in (0,1)") from exc
    if not np.isfinite(margin) or not (0.0 < margin < 1.0):
        raise ValueError("margin_fraction must be finite and in (0,1)")
    build = getattr(pair_owner, "build_pair_rows", None)
    if not callable(build):
        raise ValueError("pair_owner must expose callable build_pair_rows")

    normalized = []
    seen = set()
    try:
        source_records = list(records)
    except TypeError as exc:
        raise ValueError("records must be iterable") from exc
    for record_index, record in enumerate(source_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {record_index} must be a mapping")
        missing = [name for name in _REQUIRED if name not in record]
        if missing:
            raise ValueError(f"record {record_index} missing fields: {missing}")
        face_row = _integer_id(record["skin_face_row"], "skin_face_row")
        body = _integer_id(record["bank_body"], "bank_body")
        bone = _integer_id(record["stable_bone_id"], "stable_bone_id")
        triangle_id = _integer_id(record["bank_triangle_stable_id"], "bank_triangle_stable_id")
        if face_row >= len(faces):
            raise ValueError(f"skin_face_row out of range: {face_row}")
        fixed_triangle = np.asarray(record["fixed_triangle_world_mm"], dtype=np.float64)
        if fixed_triangle.shape != (3, 3) or not np.isfinite(fixed_triangle).all():
            raise ValueError(
                f"fixed_triangle_world_mm must be finite with shape (3,3) for record {record_index}"
            )
        skin_triangle = points[faces[face_row]]
        if float(np.linalg.norm(_area_vector(skin_triangle))) <= 1e-24:
            raise ValueError(f"degenerate skin triangle at face row {face_row}")
        if float(np.linalg.norm(_area_vector(fixed_triangle))) <= 1e-24:
            raise ValueError(f"degenerate fixed bone triangle for record {record_index}")
        key = (face_row, body, bone, triangle_id)
        if key in seen:
            raise ValueError(f"duplicate typed skin-bone pair: {key}")
        seen.add(key)
        normalized.append((key, skin_triangle.copy(), fixed_triangle.copy()))

    normalized.sort(key=lambda item: item[0])
    rows = []
    typed_records = []
    two_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    for key, skin_triangle, fixed_triangle in normalized:
        face_row, body, bone, triangle_id = key
        temporary_points = np.vstack((skin_triangle, fixed_triangle))
        try:
            local_rows, local_records = build(
                temporary_points, two_faces, [(0, 1)], margin_fraction=margin
            )
        except Exception as exc:
            raise ValueError(f"fixed-bone pair {key} rejected by 026 owner: {exc}") from exc
        if len(local_records) != 1:
            raise ValueError(f"026 owner returned {len(local_records)} records for typed pair {key}")
        if not isinstance(local_rows, (list, tuple)):
            raise ValueError(f"026 owner returned malformed rows for typed pair {key}")

        row_start = len(rows)
        omitted_fixed_only = 0
        for local_row in local_rows:
            local_nodes = np.asarray(local_row.get("nodes"), dtype=np.int64)
            coefficients = np.asarray(local_row.get("coefficients"), dtype=np.float64)
            lower = float(local_row.get("lower", np.nan))
            label = str(local_row.get("label", ""))
            if (local_nodes.ndim != 1 or coefficients.shape != (len(local_nodes), 3) or
                    not np.isfinite(coefficients).all() or not np.isfinite(lower) or
                    np.any(local_nodes < 0) or np.any(local_nodes >= 6)):
                raise ValueError(f"026 owner returned malformed row for typed pair {key}")
            mapped = {}
            for local_node, coefficient in zip(local_nodes, coefficients):
                local_node = int(local_node)
                if local_node < 3:
                    global_node = int(faces[face_row, local_node])
                    mapped[global_node] = mapped.get(global_node, np.zeros(3, dtype=np.float64)) + coefficient
            nonzero = [(node, vector) for node, vector in sorted(mapped.items())
                       if float(np.dot(vector, vector)) > 1e-28]
            if not nonzero:
                if lower > 0.0:
                    raise ValueError(
                        f"fixed-only row infeasible for typed pair {key}: lower={lower:.17g} mm"
                    )
                omitted_fixed_only += 1
                continue
            rows.append({
                "nodes": np.asarray([node for node, _ in nonzero], dtype=np.int64),
                "coefficients": np.asarray([vector for _, vector in nonzero], dtype=np.float64),
                "lower": lower,
                "label": f"fixedbone:{face_row}:{body}:{bone}:{triangle_id}:{label}",
            })

        typed_records.append({
            "skin_face_row": face_row,
            "bank_body": body,
            "stable_bone_id": bone,
            "bank_triangle_stable_id": triangle_id,
            "kind": "fixed_bone_triangle",
            "rows_start": row_start,
            "rows_stop": len(rows),
            "omitted_fixed_only_rows": omitted_fixed_only,
            "pair_owner_record": dict(local_records[0]),
        })
    return rows, typed_records
