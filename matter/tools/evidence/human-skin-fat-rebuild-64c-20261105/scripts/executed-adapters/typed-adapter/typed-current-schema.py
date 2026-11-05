#!/usr/bin/env python3
"""Versioned current-1020 O/I/D schema adapter for the frozen 433 exporter.

This is a code adapter only. It reads current typed role NPZs without rewriting
them, retains their immediate maps and all three face-ancestry lanes, then
delegates candidate construction/export to the hash-pinned existing 433 owner.
"""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys

VERSION = "current1020-433-typed-role-adapter-1021"
ROLE_ORDER = ("O", "I", "D")
BASE_PATH = pathlib.Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/downstream-004-same-material-current-reference-229/native-domain-217-source-007/compat/export_refined_coarse_registered_volume_433_p1_anchor.py')
BASE_SHA256 = "c9d6b6be03b421073cd5ea1c756b995cb3d5d1b57db99bffbca5e9783acf0b73"
CURRENT_KEYS = (
    "points_world_m", "points_world_m_f32", "faces",
    "source_input_vertex_quotient", "source_input_face_rows",
    "quality_star_point_output_to_input_row", "quality_star_face_output_to_input_row",
    "face_source_layer", "face_source_body", "face_source_bone",
    "face_source_face_id", "face_source_original_id",
    "face_source_identity0", "face_source_identity1",
    "face_ancestry_offsets", "face_ancestry_source_face_rows",
    "face_ancestry_source_original_ids", "face_ancestry_parent_face_rows",
)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _int_array(z, key):
    import numpy as np
    if key not in z.files and key in (
            "quality_star_point_output_to_input_row", "quality_star_face_output_to_input_row"):
        # Joint surface/volume quotients already carry explicit immediate parent
        # rows. Verify their pinned parent and every emitted face before exposing
        # the old quality-star aliases. Geometry and complete typed CSR stay intact.
        required={"typed_parent_candidate_path","typed_parent_candidate_sha256",
                  "source_input_vertex_quotient","source_input_face_rows"}
        if not required.issubset(z.files):
            raise ValueError("joint typed quotient lacks explicit immediate parent maps")
        parent=pathlib.Path(str(np.asarray(z["typed_parent_candidate_path"]).item()))
        digest=str(np.asarray(z["typed_parent_candidate_sha256"]).item())
        if not parent.is_file() or sha256_file(parent)!=digest:
            raise ValueError("joint typed immediate parent pin mismatch")
        with np.load(parent,allow_pickle=False) as p:
            for lane in ("points_world_m","points_world_m_f32"):
                if p[lane].dtype!=z[lane].dtype or p[lane].shape!=z[lane].shape or p[lane].tobytes()!=z[lane].tobytes():
                    raise ValueError("joint typed point row identity differs from its parent")
            quotient=np.asarray(z["source_input_vertex_quotient"])
            rows=np.asarray(z["source_input_face_rows"])
            if quotient.dtype.kind not in "iu" or rows.dtype.kind not in "iu":
                raise ValueError("joint typed parent maps must retain integer IDs")
            if (quotient.shape!=(len(p["points_world_m_f32"]),) or
                rows.shape!=(len(z["faces"]),) or np.any(quotient<0) or
                np.any(quotient>=len(quotient)) or np.any(rows<0) or np.any(rows>=len(p["faces"])) or
                len(np.unique(rows))!=len(rows)):
                raise ValueError("joint typed immediate map domain mismatch")
            if not np.array_equal(quotient[p["faces"][rows]],z["faces"]):
                raise ValueError("joint typed emitted faces do not replay their parent quotient")
        return (np.arange(len(z["points_world_m_f32"]),dtype=np.int64)
                if key=="quality_star_point_output_to_input_row" else rows.astype(np.int64))
    raw = np.asarray(z[key])
    if raw.dtype.kind not in "iu":
        raise ValueError(f"current typed field {key} must retain integer IDs")
    return raw.astype(np.int64, copy=False)


def _validate_current_ancestry(z, face_count):
    import numpy as np
    offsets = _int_array(z, "face_ancestry_offsets")
    source_rows = _int_array(z, "face_ancestry_source_face_rows")
    source_ids = _int_array(z, "face_ancestry_source_original_ids")
    parent_rows = _int_array(z, "face_ancestry_parent_face_rows")
    if (offsets.shape != (face_count + 1,) or offsets[0] != 0 or
            np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(source_rows) or
            source_ids.shape != source_rows.shape or parent_rows.shape != source_rows.shape):
        raise ValueError("current typed face ancestry CSR has inconsistent offsets or lanes")
    return offsets.copy(), source_rows.copy(), source_ids.copy(), parent_rows.copy()


def load_current_typed_role_asset(role, path, expected_sha256):
    """Load current O/I/D fields in the role-asset shape consumed by 433.

    No old microedge CSR is synthesized. Current source-input and quality-star
    maps are retained with their own names; complete face ancestry remains in
    three aligned lanes.
    """
    import numpy as np
    role = str(role)
    path = pathlib.Path(path).resolve()
    if role not in ROLE_ORDER:
        raise ValueError("current typed role must be O, I, or D")
    if len(str(expected_sha256)) != 64 or sha256_file(path) != str(expected_sha256):
        raise ValueError(f"current typed {role} role asset hash pin mismatch")
    with np.load(path, allow_pickle=False) as z:
        missing = set(CURRENT_KEYS) - set(z.files) - {"quality_star_point_output_to_input_row", "quality_star_face_output_to_input_row"}
        if missing:
            raise ValueError("current typed role asset lacks fields: " + ",".join(sorted(missing)))
        points64 = np.asarray(z["points_world_m"])
        points_f32 = np.asarray(z["points_world_m_f32"])
        if (points64.dtype.kind != "f" or points64.dtype.itemsize != 8 or
                points_f32.dtype != np.dtype(np.float32) or
                points64.ndim != 2 or points64.shape[1] != 3 or
                points_f32.shape != points64.shape or len(points64) == 0 or
                not np.isfinite(points64).all() or not np.isfinite(points_f32).all() or
                not np.array_equal(points64.astype(np.float32), points_f32) or
                not np.array_equal(points64, points_f32.astype(np.float64))):
            raise ValueError("current typed role coordinates are not an exact aligned F32 emission")
        points_f32 = points_f32.copy()
        faces_raw = np.asarray(z["faces"])
        if faces_raw.dtype.kind not in "iu":
            raise ValueError("current typed faces must retain integer point IDs")
        faces = faces_raw.astype(np.int64, copy=False)
        if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or
                np.any(faces < 0) or np.any(faces >= len(points_f32)) or
                np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 1] == faces[:, 2]) or
                np.any(faces[:, 2] == faces[:, 0])):
            raise ValueError(f"current typed {role} face row domain is invalid")
        tri = points_f32[faces]
        if (np.any(np.all(tri[:, 0] == tri[:, 1], axis=1)) or
                np.any(np.all(tri[:, 1] == tri[:, 2], axis=1)) or
                np.any(np.all(tri[:, 2] == tri[:, 0], axis=1))):
            raise ValueError(f"current typed {role} contains a Float32-collapsed face")
        faces = faces.copy()
        count = len(faces)
        face_fields = {}
        for key in ("face_source_layer", "face_source_body", "face_source_bone",
                    "face_source_face_id", "face_source_original_id",
                    "face_source_identity0", "face_source_identity1"):
            value = _int_array(z, key)
            if value.shape != (count,):
                raise ValueError(f"current typed {role} {key} row domain mismatch")
            face_fields[key] = value.copy()

        source_vertex_quotient = _int_array(z, "source_input_vertex_quotient")
        if (source_vertex_quotient.ndim != 1 or
                np.any(source_vertex_quotient < -1) or
                np.any(source_vertex_quotient >= len(points_f32))):
            raise ValueError("current source-input vertex quotient is outside the output point domain")
        source_face_rows = _int_array(z, "source_input_face_rows")
        if source_face_rows.shape != (count,) or np.any(source_face_rows < -1):
            raise ValueError("current source-input face rows do not align with output faces")
        quality_point_rows = _int_array(z, "quality_star_point_output_to_input_row")
        if (quality_point_rows.shape != (len(points_f32),) or
                np.any(quality_point_rows < -1)):
            raise ValueError("current quality-star point map does not align with output points")
        quality_face_rows = _int_array(z, "quality_star_face_output_to_input_row")
        if quality_face_rows.shape != (count,) or np.any(quality_face_rows < -1):
            raise ValueError("current quality-star face map does not align with output faces")

        anc_offsets, anc_source_rows, anc_source_ids, anc_parent_rows = _validate_current_ancestry(z, count)
        ancestry = [anc_source_rows[anc_offsets[i]:anc_offsets[i + 1]].tolist()
                    for i in range(count)]
        ancestry_ids = [anc_source_ids[anc_offsets[i]:anc_offsets[i + 1]].tolist()
                        for i in range(count)]
        ancestry_parents = [anc_parent_rows[anc_offsets[i]:anc_offsets[i + 1]].tolist()
                            for i in range(count)]

        result = {
            "role": role, "path": str(path), "sha256": str(expected_sha256),
            "points_world_m_f32": points_f32, "faces": faces,
            **face_fields,
            "source_rows": source_face_rows.copy(),
            "source_input_vertex_quotient": source_vertex_quotient.copy(),
            "source_input_face_rows": source_face_rows.copy(),
            "quality_star_point_output_to_input_row": quality_point_rows.copy(),
            "quality_star_face_output_to_input_row": quality_face_rows.copy(),
            "face_ancestry_offsets": anc_offsets,
            "face_ancestry_source_face_rows": anc_source_rows,
            "face_ancestry_source_original_ids": anc_source_ids,
            "face_ancestry_parent_face_rows": anc_parent_rows,
            "ancestry": ancestry,
            "ancestry_original_ids": ancestry_ids,
            "ancestry_parent_rows": ancestry_parents,
            "typed_face_ancestry_offsets": anc_offsets.copy(),
            "typed_face_ancestry_source_face_rows": anc_source_rows.copy(),
            "typed_face_ancestry_source_original_ids": anc_source_ids.copy(),
            "typed_face_ancestry_parent_face_rows": anc_parent_rows.copy(),
            "typed_face_ancestry_fields": [
                "face_ancestry_source_face_rows",
                "face_ancestry_source_original_ids",
                "face_ancestry_parent_face_rows",
            ],
            "raw_npz_keys": list(z.files),
        }
    return result


def load_pinned_433_owner():
    if not BASE_PATH.is_file() or sha256_file(BASE_PATH) != BASE_SHA256:
        raise ValueError("frozen 433 owner path/hash changed")
    spec = importlib.util.spec_from_file_location("frozen_433_current_schema_adapter_1021", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(str(BASE_PATH))
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    owner.load_current_typed_role_asset = load_current_typed_role_asset
    return owner


def main() -> int:
    owner = load_pinned_433_owner()
    return int(owner.main())


if __name__ == "__main__":
    raise SystemExit(main())

