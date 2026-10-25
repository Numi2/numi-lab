"""Offline adapter for the frozen AcceptedFEMSkinTransfer 075 CSV export.

The returned sparse map preserves the owner's per-detail transfer rows. It is
not a geometry, tissue, or physical-admission validator.
"""
from __future__ import annotations
import csv, hashlib, struct
from pathlib import Path
from typing import Mapping
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

class Transfer075Error(ValueError):
    """The pinned transfer export is malformed or fails its source pins."""

PIN_KEYS = frozenset({"transfer_rows", "coarse_reference", "coarse_faces", "detail_payload"})

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _read_csv(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise Transfer075Error(f"{path}: missing required CSV columns")
        rows = list(reader)
    if not rows:
        raise Transfer075Error(f"{path}: empty CSV")
    return rows

def _int(row: Mapping[str, str], key: str, context: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise Transfer075Error(f"{context}: invalid integer field {key}") from exc

def _float(row: Mapping[str, str], key: str, context: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise Transfer075Error(f"{context}: invalid numeric field {key}") from exc
    if not np.isfinite(value):
        raise Transfer075Error(f"{context}: nonfinite field {key}")
    return value

def _decode_detail_faces(path: str | Path) -> tuple[int, np.ndarray]:
    raw = Path(path).read_bytes()
    if len(raw) < 60:
        raise Transfer075Error("detail NHSKIN payload is truncated")
    magic, abi, binding_count, vertex_count, index_count, _reserved = struct.unpack_from("<8s5I", raw)
    if magic != b"NHSKIN1\0" or abi != 5 or binding_count == 0 or vertex_count == 0 or index_count == 0 or index_count % 3:
        raise Transfer075Error("detail payload must be a valid NHSKIN v5 triangle asset")
    vertex_offset = 60 + binding_count * 36
    index_offset = vertex_offset + vertex_count * 56
    weight_offset = index_offset + index_count * 4
    expected_bytes = weight_offset + vertex_count * binding_count * 4
    if len(raw) != expected_bytes:
        raise Transfer075Error("detail NHSKIN payload byte count does not match its header")
    faces = np.frombuffer(raw, dtype="<u4", count=index_count, offset=index_offset).reshape(-1, 3).copy()
    if faces.size and int(faces.max()) >= vertex_count:
        raise Transfer075Error("detail NHSKIN face index is out of range")
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise Transfer075Error("detail NHSKIN contains a repeated face vertex")
    return vertex_count, faces

def load_transfer075(transfer_rows_csv: str | Path, coarse_reference_csv: str | Path,
                     coarse_faces_csv: str | Path, detail_payload: str | Path, *,
                     expected_sha256: Mapping[str, str]
                     ) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Load a hash-pinned 075 export as (W, detail_origin, coarse_origin, faces).

    W has shape (detail vertices, coarse vertices) and Float32 data. Inactive
    detailed vertices have all-zero rows. Origins are emitted CSV world points
    cast to Float32. Returned faces are detailed NHSKIN faces from detail_payload;
    the coarse-face CSV only validates active rows' selected coarse triangle.
    """
    if set(expected_sha256) != PIN_KEYS or any(not expected_sha256[k] for k in PIN_KEYS):
        raise Transfer075Error(f"expected_sha256 must pin exactly {sorted(PIN_KEYS)}")
    paths = {"transfer_rows": Path(transfer_rows_csv), "coarse_reference": Path(coarse_reference_csv),
             "coarse_faces": Path(coarse_faces_csv), "detail_payload": Path(detail_payload)}
    for key, path in paths.items():
        actual = sha256_file(path)
        if actual != expected_sha256[key]:
            raise Transfer075Error(f"{key} SHA-256 mismatch: expected {expected_sha256[key]}, got {actual}")

    transfer_required = {"detail_row", "active", "detail_source_vertex_id", "detail_donor_body",
        "transfer_donor_body", "coarse_vertex0", "coarse_vertex1", "coarse_vertex2",
        "weight0", "weight1", "weight2", "selected_face_row", "selected_face_stable_id",
        "detail_rest_world_x_m", "detail_rest_world_y_m", "detail_rest_world_z_m"}
    coarse_required = {"coarse_row", "world_x_m", "world_y_m", "world_z_m"}
    face_required = {"coarse_face_row", "stable_source_face_id", "vertex0", "vertex1", "vertex2"}
    transfer = _read_csv(paths["transfer_rows"], transfer_required)
    coarse = _read_csv(paths["coarse_reference"], coarse_required)
    coarse_face_rows = _read_csv(paths["coarse_faces"], face_required)
    detail_count, detail_faces = _decode_detail_faces(paths["detail_payload"])
    coarse_count, face_count = len(coarse), len(coarse_face_rows)

    if len(transfer) != detail_count:
        raise Transfer075Error("transfer row count does not match detailed payload vertex count")
    if any(_int(row, "detail_row", f"transfer row {i}") != i for i, row in enumerate(transfer)):
        raise Transfer075Error("detail_row values must be unique contiguous row indices")
    if any(_int(row, "detail_source_vertex_id", f"transfer row {i}") != i for i, row in enumerate(transfer)):
        raise Transfer075Error("075 detail source IDs must preserve payload vertex identity")
    if any(_int(row, "coarse_row", f"coarse row {i}") != i for i, row in enumerate(coarse)):
        raise Transfer075Error("coarse_row values must be unique contiguous row indices")
    if any(_int(row, "coarse_face_row", f"coarse face row {i}") != i for i, row in enumerate(coarse_face_rows)):
        raise Transfer075Error("coarse_face_row values must be unique contiguous row indices")

    detail_origin = np.empty((detail_count, 3), dtype=np.float32)
    for i, row in enumerate(transfer):
        detail_origin[i] = [_float(row, f"detail_rest_world_{axis}_m", f"transfer row {i}") for axis in "xyz"]
    coarse_origin = np.empty((coarse_count, 3), dtype=np.float32)
    for i, row in enumerate(coarse):
        coarse_origin[i] = [_float(row, f"world_{axis}_m", f"coarse row {i}") for axis in "xyz"]
    if not np.isfinite(detail_origin).all() or not np.isfinite(coarse_origin).all():
        raise Transfer075Error("emitted Float32 origin contains nonfinite coordinates")

    coarse_faces = np.empty((face_count, 3), dtype=np.int64)
    stable_ids = np.empty(face_count, dtype=np.int64)
    for i, row in enumerate(coarse_face_rows):
        context = f"coarse face {i}"
        stable_ids[i] = _int(row, "stable_source_face_id", context)
        coarse_faces[i] = [_int(row, f"vertex{j}", context) for j in range(3)]
    if coarse_faces.size and (int(coarse_faces.min()) < 0 or int(coarse_faces.max()) >= coarse_count):
        raise Transfer075Error("coarse face table contains an out-of-range vertex")

    active_rows, active_cols, active_values = [], [], []
    active_mask = np.zeros(detail_count, dtype=bool)
    for i, row in enumerate(transfer):
        context = f"transfer row {i}"
        active = _int(row, "active", context)
        if active not in (0, 1):
            raise Transfer075Error(f"{context}: active must be 0 or 1")
        if not active:
            continue
        active_mask[i] = True
        if _int(row, "detail_donor_body", context) != _int(row, "transfer_donor_body", context):
            raise Transfer075Error(f"{context}: transfer donor differs from detailed source owner")
        face_row = _int(row, "selected_face_row", context)
        if face_row < 0 or face_row >= face_count:
            raise Transfer075Error(f"{context}: selected coarse face row is out of range")
        if _int(row, "selected_face_stable_id", context) != int(stable_ids[face_row]):
            raise Transfer075Error(f"{context}: selected stable face identity does not match the face table")
        cols = [_int(row, f"coarse_vertex{j}", context) for j in range(3)]
        if any(col < 0 or col >= coarse_count for col in cols):
            raise Transfer075Error(f"{context}: active coarse vertex is out of range")
        if any(col not in coarse_faces[face_row] for col in cols):
            raise Transfer075Error(f"{context}: active support is not contained in the selected coarse face")
        weights = np.asarray([_float(row, f"weight{j}", context) for j in range(3)], dtype=np.float32)
        if np.any(weights < 0.0):
            raise Transfer075Error(f"{context}: active transfer weights must be nonnegative")
        if abs(float(weights.astype(np.float64).sum()) - 1.0) > 1.0e-6:
            raise Transfer075Error(f"{context}: active transfer weights do not sum to one")
        active_rows.extend((i, i, i))
        active_cols.extend(cols)
        active_values.extend(weights)

    W = coo_matrix((np.asarray(active_values, dtype=np.float32),
        (np.asarray(active_rows, dtype=np.int64), np.asarray(active_cols, dtype=np.int64))),
        shape=(detail_count, coarse_count), dtype=np.float32).tocsr()
    W.sum_duplicates()
    W.eliminate_zeros()
    W.sort_indices()
    row_sums = np.asarray(W.sum(axis=1), dtype=np.float64).reshape(-1)
    if np.any(np.abs(row_sums[active_mask] - 1.0) > 1.0e-6) or np.any(row_sums[~active_mask] != 0.0):
        raise Transfer075Error("CSR assembly changed active partition or inactive zero-row semantics")
    return W, detail_origin, coarse_origin, detail_faces
