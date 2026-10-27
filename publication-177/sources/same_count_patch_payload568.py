"""Same-count face-patch application with the 405 lineage merge contract.

This helper only changes face connectivity and face-level ancestry. It does not
move points, add/delete nodes, or certify geometric clearance.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

CSR_FAMILIES = (
    ("face_source_ancestry_indptr", "face_source_ancestry_face_rows"),
    ("face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows"),
    ("refinement_face_parent_ancestry_indptr", "refinement_face_parent_ancestry_face_rows"),
    ("face_fine_detail_ancestry_indptr", "face_fine_detail_ancestry_rows"),
)
SEMANTIC_ARRAYS = {
    "refinement_face_parent_ancestry_semantics",
    "face_fine_detail_ancestry_semantics",
    "refinement_face_parent_barycentric_semantics",
}
MUTATED_KEYS = {
    "source_O_faces",
    "area_reference_037_source_O_faces",
    *(key for pair in CSR_FAMILIES for key in pair),
    *SEMANTIC_ARRAYS,
    "refinement_face_parent_face_row",
    "refinement_face_parent_barycentric",
    "parent_fine_detail_face_row",
    "parent_fine_subface_row",
    "parent_coarse_face_row",
    "parent_source_face_marker",
}


class PatchPayloadError(ValueError):
    pass


def csr_rows(ptr: np.ndarray, vals: np.ndarray, n: int) -> list[list[int]]:
    ptr = np.asarray(ptr)
    vals = np.asarray(vals)
    if ptr.shape != (n + 1,) or ptr.dtype.kind not in "iu":
        raise PatchPayloadError("invalid CSR indptr shape or dtype")
    if vals.ndim != 1 or vals.dtype.kind not in "iu":
        raise PatchPayloadError("invalid CSR values shape or dtype")
    if int(ptr[0]) != 0 or int(ptr[-1]) != len(vals) or np.any(ptr[1:] < ptr[:-1]):
        raise PatchPayloadError("invalid CSR offsets")
    return [list(map(int, vals[int(ptr[i]):int(ptr[i + 1])])) for i in range(n)]


def csr_build(rows: Sequence[Sequence[int]], dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    vals: list[int] = []
    for i, row in enumerate(rows):
        vals.extend(sorted(set(map(int, row))))
        ptr[i + 1] = len(vals)
    return ptr, np.asarray(vals, dtype=dtype)


def _directed_boundary(faces: np.ndarray) -> tuple[tuple[int, int], ...]:
    counts: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tri in np.asarray(faces):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            counts.setdefault(key, []).append((u, v))
    boundary = []
    for edge, directions in counts.items():
        if len(directions) > 2:
            raise PatchPayloadError(f"nonmanifold patch edge {edge}")
        if len(directions) == 2:
            if directions[0] != directions[1][::-1]:
                raise PatchPayloadError(f"inconsistently oriented internal edge {edge}")
        else:
            boundary.append(directions[0])
    return tuple(sorted(boundary))


def _validate_patch(candidate: Mapping[str, np.ndarray], face_rows: Sequence[int],
                    new_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if "source_O_faces" not in candidate or "points_world_m" not in candidate:
        raise PatchPayloadError("candidate is missing points or faces")
    faces = np.asarray(candidate["source_O_faces"])
    points = np.asarray(candidate["points_world_m"])
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise PatchPayloadError("source_O_faces must be integer Fx3")
    if points.ndim != 2 or points.shape[1] != 3:
        raise PatchPayloadError("points_world_m must be Nx3")
    if "area_reference_037_source_O_faces" not in candidate:
        raise PatchPayloadError("missing matched 037 faces")
    if not np.array_equal(faces, np.asarray(candidate["area_reference_037_source_O_faces"])):
        raise PatchPayloadError("037 connectivity must match current faces")

    rows = np.asarray(face_rows)
    repl = np.asarray(new_faces)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(rows) < 2:
        raise PatchPayloadError("face_rows must be an integer vector with at least two rows")
    if len(set(map(int, rows))) != len(rows):
        raise PatchPayloadError("duplicate changed face row")
    if np.any(rows < 0) or np.any(rows >= len(faces)):
        raise PatchPayloadError("changed face row is out of bounds")
    if repl.shape != (len(rows), 3) or repl.dtype.kind not in "iu":
        raise PatchPayloadError("new_faces must be integer kx3 matching face_rows")
    if repl.size and (int(repl.min()) < 0 or int(repl.max()) >= len(points)):
        raise PatchPayloadError("new face vertex is out of bounds")
    if any(len(set(map(int, tri))) != 3 for tri in repl):
        raise PatchPayloadError("new face has repeated vertex indices")
    if len({tuple(sorted(map(int, tri))) for tri in repl}) != len(repl):
        raise PatchPayloadError("duplicate new face")
    old = faces[rows.astype(np.int64)]
    if any(len(set(map(int, tri))) != 3 for tri in old):
        raise PatchPayloadError("old patch contains repeated vertex indices")
    if set(map(int, old.ravel())) != set(map(int, repl.ravel())):
        raise PatchPayloadError("old and new patch use different point sets")
    if _directed_boundary(old) != _directed_boundary(repl):
        raise PatchPayloadError("oriented patch boundary changed")

    if "face_kind" not in candidate:
        raise PatchPayloadError("missing face_kind")
    kinds = np.asarray(candidate["face_kind"])
    if kinds.shape != (len(faces),):
        raise PatchPayloadError("face_kind must have one value per face")
    old_kinds = np.unique(kinds[rows.astype(np.int64)])
    if len(old_kinds) != 1:
        raise PatchPayloadError("mixed face_kind in patch")

    return rows.astype(np.int64), repl.astype(faces.dtype, copy=False), int(old_kinds[0])


def apply_same_count_patch(candidate: Mapping[str, np.ndarray], face_rows: Sequence[int],
                           new_faces: np.ndarray) -> dict[str, np.ndarray]:
    """Replace k faces by k faces and union face ancestry exactly as owner 405.

    The replacement must retain the complete local vertex set and oriented
    boundary. All four face CSR ancestry families are required and unioned
    across the removed patch, then assigned to every replacement row.
    """
    rows, repl, _kind = _validate_patch(candidate, face_rows, new_faces)
    nface = len(candidate["source_O_faces"])
    for ptr_key, val_key in CSR_FAMILIES:
        if ptr_key not in candidate or val_key not in candidate:
            raise PatchPayloadError(f"missing ancestry family {ptr_key}/{val_key}")
        csr_rows(candidate[ptr_key], candidate[val_key], nface)
    required_scalars = (
        "parent_fine_detail_face_row",
        "parent_fine_subface_row",
        "refinement_face_parent_face_row",
        "refinement_face_parent_barycentric",
    )
    missing = [k for k in required_scalars if k not in candidate]
    if missing:
        raise PatchPayloadError(f"missing face parent fields: {missing}")
    for key in required_scalars[:-1]:
        if np.asarray(candidate[key]).shape != (nface,):
            raise PatchPayloadError(f"{key} must have one value per face")
    if np.asarray(candidate["refinement_face_parent_barycentric"]).shape != (nface, 3, 3):
        raise PatchPayloadError("parent barycentrics must be Fx3x3")

    out = {k: np.array(v, copy=True) for k, v in candidate.items()}
    out["source_O_faces"][rows] = repl
    out["area_reference_037_source_O_faces"][rows] = repl

    unions: dict[str, list[int]] = {}
    csr_payloads = {}
    for ptr_key, val_key in CSR_FAMILIES:
        row_lists = csr_rows(candidate[ptr_key], candidate[val_key], nface)
        union = sorted({v for row in rows for v in row_lists[int(row)]})
        unions[val_key] = union
        for row in rows:
            row_lists[int(row)] = union
        ptr, vals = csr_build(row_lists, np.asarray(candidate[val_key]).dtype)
        csr_payloads[ptr_key] = ptr
        csr_payloads[val_key] = vals
    out.update(csr_payloads)

    parents = unions["refinement_face_parent_ancestry_face_rows"]
    details = unions["face_fine_detail_ancestry_rows"]
    subfaces = unions["face_fine_subface_ancestry_rows"]
    out["refinement_face_parent_face_row"][rows] = parents[0] if len(parents) == 1 else -1
    if len(parents) > 1:
        out["refinement_face_parent_barycentric"][rows] = np.nan
    out["parent_fine_detail_face_row"][rows] = details[0] if len(details) == 1 else -1
    out["parent_fine_subface_row"][rows] = subfaces[0] if len(subfaces) == 1 else -1

    for key in ("parent_coarse_face_row", "parent_source_face_marker"):
        if key in out:
            old_values = np.asarray(candidate[key])[rows]
            if not np.all(old_values == old_values[0]):
                out[key][rows] = -1

    out["refinement_face_parent_ancestry_semantics"] = np.asarray([
        "CSR union of immediate parent face rows; singular row -1 and singular barycentrics NaN when multiple parents apply"
    ])
    out["face_fine_detail_ancestry_semantics"] = np.asarray([
        "CSR union of source fine detail face rows; singular row -1 for cross-detail flips"
    ])
    out["refinement_face_parent_barycentric_semantics"] = np.asarray([
        "single-parent affine map only; NaN explicitly marks multi-parent patch flips"
    ])
    return out



def apply_patch(candidate: Mapping[str, np.ndarray], operation: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Compatibility entry point for star search operation dictionaries."""
    if not isinstance(operation, Mapping) or "face_rows" not in operation or "new_faces" not in operation:
        raise PatchPayloadError("operation must provide face_rows and new_faces")
    return apply_same_count_patch(candidate, operation["face_rows"], operation["new_faces"])
