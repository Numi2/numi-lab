#!/usr/bin/env python3
"""Emit typed fixed bone triangles introduced by complete 159 surface audits.

This is a construction-only constraint payload. It does not change the mesh,
physics state, source mass, or native acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANK_DIR = E / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
BANK_EVIDENCE = E
BANK_HELPER = D / "publication-084/canonical/matter/tools/human_skin_fat_bone_audit.py"
AUDIT_159 = D / "audit_skin_bone_surfaces_159.py"
EXACT_159 = E / "collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
OUTPUT_SCHEMA = "numi.fixed_skin_bone_barriers.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def load_candidate(path: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    path = Path(path)
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as z:
        if "points_world_m" not in z or "source_O_faces" not in z:
            raise ValueError(f"candidate lacks points_world_m/source_O_faces: {path}")
        points = np.asarray(z["points_world_m"])
        faces = np.asarray(z["source_O_faces"])
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32:
        raise ValueError("candidate points_world_m must be the emitted N x 3 Float32 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("candidate source_O_faces must be an integer M x 3 array")
    if not np.isfinite(points).all() or not len(faces):
        raise ValueError("candidate surface is empty or has non-finite points")
    f64 = faces.astype(np.int64, copy=False)
    if int(f64.min()) < 0 or int(f64.max()) >= len(points):
        raise ValueError("candidate face index is outside point domain")
    if np.any(f64[:, 0] == f64[:, 1]) or np.any(f64[:, 1] == f64[:, 2]) or np.any(f64[:, 2] == f64[:, 0]):
        raise ValueError("candidate has repeated vertex indices in a surface face")
    return points, faces, digest, path.as_posix()


def same_face_array(a: np.ndarray, b: np.ndarray) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes(order="C") == y.tobytes(order="C")


def candidate_path_for_report(report_path: Path, expected_sha: str) -> Path:
    """Find the 159 runner's sibling candidate and bind it to the report hash."""
    report_path = Path(report_path)
    candidates = []
    for parent in (report_path.parent.parent, report_path.parent, report_path.parent.parent.parent):
        p = parent / "candidate.npz"
        if p.is_file() and p not in candidates:
            candidates.append(p)
    for p in candidates:
        if sha256_file(p) == expected_sha:
            return p
    raise ValueError(f"cannot find a sibling candidate matching 159 report SHA: {report_path}")


def _exact_int(value: object, what: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} must be an integer, not bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{what} is not an exact integer")


def _hex_digest(value: object, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{what} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{what} is not hexadecimal") from exc
    return value.lower()


def validate_audit_report(
    report: Mapping[str, object],
    *,
    candidate_sha256: str,
    point_count: int,
    face_count: int,
    expected_bank_keys: Iterable[tuple[int, int]],
    expected_script_sha256: str,
    expected_predicate_sha256: str,
) -> set[tuple[int, int, int, int]]:
    """Validate a complete 159 report and return typed (body,bone,face,triID) pairs."""
    if report.get("status") != "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT":
        raise ValueError("159 audit is not complete")
    if report.get("complete") is not True:
        raise ValueError("159 report complete flag is false")
    if report.get("candidate_sha256") != candidate_sha256:
        raise ValueError("159 candidate SHA does not match its candidate bytes")
    if _exact_int(report.get("surface_point_count", -1), "159 point count") != point_count:
        raise ValueError("159 point count differs from the bound candidate")
    if _exact_int(report.get("surface_face_count", -1), "159 face count") != face_count:
        raise ValueError("159 face count differs from the bound candidate")
    if report.get("script_sha256") != expected_script_sha256:
        raise ValueError("159 audit script SHA differs from the pinned owner")
    if report.get("exact_predicate_sha256") != expected_predicate_sha256:
        raise ValueError("159 exact predicate SHA differs from the pinned owner")

    expected_rows = tuple((_exact_int(body, "expected body"), _exact_int(bone, "expected stable bone ID")) for body, bone in expected_bank_keys)
    expected = set(expected_rows)
    if len(expected) != len(expected_rows):
        raise ValueError("duplicate expected typed bank identity")
    banks = report.get("banks")
    if not isinstance(banks, list):
        raise ValueError("159 typed-bank rows are missing")
    if _exact_int(report.get("bank_count", -1), "159 bank count") != len(expected) or _exact_int(report.get("expected_bank_count", -1), "159 expected bank count") != len(expected):
        raise ValueError("159 report does not cover the complete expected bank count")
    seen: set[tuple[int, int]] = set()
    pairs: set[tuple[int, int, int, int]] = set()
    for row in banks:
        if not isinstance(row, dict):
            raise ValueError("malformed typed bank row")
        key = (_exact_int(row["body"], "159 bank body"), _exact_int(row["stable_bone_id"], "159 stable bone ID"))
        if key not in expected or key in seen:
            raise ValueError(f"missing, unexpected, or duplicate typed bank: {key}")
        seen.add(key)
        if _exact_int(row.get("skin_faces", -1), "159 skin face count") != face_count:
            raise ValueError(f"typed bank {key} has a different skin face domain")
        result = row.get("result")
        if result is None:
            if _exact_int(row.get("skin_aabb_active", -1), "159 skin AABB count") != 0 or _exact_int(row.get("bank_aabb_active", -1), "159 bank AABB count") != 0:
                raise ValueError(f"typed bank {key} has a null result without a complete AABB exclusion")
            continue
        if not isinstance(result, dict):
            raise ValueError(f"malformed exact result for typed bank {key}")
        raw_pairs = result.get("triangle_pairs")
        if not isinstance(raw_pairs, list):
            raise ValueError(f"exact triangle-pair list is missing for typed bank {key}")
        if _exact_int(result.get("count", -1), "159 exact pair count") != len(raw_pairs):
            raise ValueError(f"exact pair count/list mismatch for typed bank {key}")
        local_seen: set[tuple[int, int]] = set()
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"malformed exact pair in typed bank {key}")
            face, stable_tri = pair
            face = _exact_int(face, "159 skin face row")
            stable_tri = _exact_int(stable_tri, "159 stable triangle ID")
            if face < 0 or face >= face_count or stable_tri < 0:
                raise ValueError(f"pair identity out of range in typed bank {key}")
            if (face, stable_tri) in local_seen:
                raise ValueError(f"duplicate exact pair in typed bank {key}")
            local_seen.add((face, stable_tri))
            pairs.add((key[0], key[1], face, stable_tri))
    if seen != expected:
        raise ValueError(f"159 audit omitted typed banks: {sorted(expected - seen)[:5]}")
    return pairs


def introduced_pairs(
    base_pairs: set[tuple[int, int, int, int]],
    trial_pair_sets: Iterable[set[tuple[int, int, int, int]]],
) -> tuple[tuple[int, int, int, int], ...]:
    trial_union: set[tuple[int, int, int, int]] = set()
    for pairs in trial_pair_sets:
        trial_union.update(pairs)
    return tuple(sorted(trial_union - base_pairs))


def world_triangle_mm(
    local_triangle_m: np.ndarray,
    world_to_local: np.ndarray,
    com_offset_local_m: np.ndarray,
    body_position_world_m: np.ndarray,
) -> np.ndarray:
    """Invert q=M(world-position)-offset; preserve Float64 computation."""
    tri = np.asarray(local_triangle_m, dtype=np.float64)
    matrix = np.asarray(world_to_local, dtype=np.float64)
    offset = np.asarray(com_offset_local_m, dtype=np.float64)
    position = np.asarray(body_position_world_m, dtype=np.float64)
    if tri.shape != (3, 3) or matrix.shape != (3, 3) or offset.shape != (3,) or position.shape != (3,):
        raise ValueError("invalid local triangle or body affine shape")
    if not all(np.isfinite(x).all() for x in (tri, matrix, offset, position)):
        raise ValueError("non-finite local triangle or body affine")
    det = float(np.linalg.det(matrix))
    if not np.isfinite(det) or det == 0.0:
        raise ValueError("singular body world-to-local matrix")
    world = np.linalg.solve(matrix, (tri + offset).T).T + position
    world_mm = world * 1000.0
    if not np.isfinite(world_mm).all():
        raise ValueError("non-finite transformed world triangle")
    return world_mm


def typed_triangle_lookup(triangles: np.ndarray, ids: np.ndarray) -> dict[tuple[int, int, int], np.ndarray]:
    tri = np.asarray(triangles, dtype=np.float64)
    ident = np.asarray(ids)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or ident.ndim != 2 or ident.shape[1] < 4 or len(tri) != len(ident):
        raise ValueError("invalid pinned bank triangle/identity arrays")
    if not np.isfinite(tri).all() or not np.isfinite(ident).all():
        raise ValueError("non-finite bank triangle/identity")
    lookup: dict[tuple[int, int, int], np.ndarray] = {}
    for row, (triangle, identity) in enumerate(zip(tri, ident)):
        body, bone, stable_tri = int(identity[0]), int(identity[1]), int(identity[3])
        if float(identity[0]) != body or float(identity[1]) != bone or float(identity[3]) != stable_tri:
            raise ValueError(f"non-integer typed bank identity at row {row}")
        key = (body, bone, stable_tri)
        if key in lookup:
            raise ValueError(f"duplicate stable triangle identity {key}")
        lookup[key] = triangle
    return lookup


def make_barrier_arrays(
    pairs: Iterable[tuple[int, int, int, int]],
    triangles_by_identity: Mapping[tuple[int, int, int], np.ndarray],
    transforms_by_body: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    typed = tuple(sorted(set(tuple(map(int, p)) for p in pairs)))
    face_rows, bodies, bones, tri_ids, fixed = [], [], [], [], []
    for body, bone, face, stable_tri in typed:
        key = (body, bone, stable_tri)
        if key not in triangles_by_identity:
            raise ValueError(f"exact pair names missing typed stable triangle: {key}")
        if body not in transforms_by_body:
            raise ValueError(f"missing pinned body transform: {body}")
        matrix, offset, position = transforms_by_body[body]
        world = world_triangle_mm(triangles_by_identity[key], matrix, offset, position)
        face_rows.append(face)
        bodies.append(body)
        bones.append(bone)
        tri_ids.append(stable_tri)
        fixed.append(world)
    return {
        "skin_face_rows": np.asarray(face_rows, dtype=np.int64),
        "bank_body": np.asarray(bodies, dtype=np.int64),
        "stable_bone_id": np.asarray(bones, dtype=np.int64),
        "bank_triangle_stable_id": np.asarray(tri_ids, dtype=np.int64),
        "fixed_triangles_world_mm": np.asarray(fixed, dtype=np.float64).reshape(-1, 3, 3),
    }


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict:
    with Path(path).open() as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidate", required=True, type=Path)
    parser.add_argument("--base-surface-report", required=True, type=Path)
    parser.add_argument("--trial-surface-report", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.trial_surface_report:
        parser.error("at least one --trial-surface-report is required")

    base_points, base_faces, base_sha, base_path = load_candidate(args.base_candidate)
    face_hash = sha256_array(base_faces)
    point_hash = sha256_array(base_points)
    face_count, point_count = len(base_faces), len(base_points)
    bank_helper_sha = sha256_file(BANK_HELPER)
    audit_script_sha = sha256_file(AUDIT_159)
    predicate_sha = sha256_file(EXACT_159)

    helper = _module("human_skin_fat_bone_audit_314", BANK_HELPER)
    triangles, ids, _gids, _bodies, matrices, offsets, positions, bankkeys, pins, _unused = helper.read_bank_inputs(BANK_DIR, BANK_EVIDENCE)
    expected_keys = tuple((int(row[0]), int(row[1])) for row in np.asarray(bankkeys))
    expected_keys_set = set(expected_keys)
    if len(expected_keys) != 185 or len(expected_keys_set) != 185:
        raise ValueError("pinned 039 input is not exactly 185 unique typed banks")
    lookup = typed_triangle_lookup(triangles, ids)
    transforms = {int(body): (matrices[int(body)], offsets[int(body)], positions[int(body)]) for body in matrices}

    reports = [args.base_surface_report, *args.trial_surface_report]
    audit_data = []
    pair_sets = []
    for index, report_path in enumerate(reports):
        report = _load_json(report_path)
        if index == 0:
            candidate_path = Path(args.base_candidate)
            candidate_sha = base_sha
            points, faces = base_points, base_faces
        else:
            candidate_sha = _hex_digest(report.get("candidate_sha256"), "report candidate SHA")
            candidate_path = candidate_path_for_report(report_path, candidate_sha)
            points, faces, actual_sha, _ = load_candidate(candidate_path)
            if actual_sha != candidate_sha:
                raise ValueError(f"trial candidate SHA mismatch: {candidate_path}")
            if points.shape != base_points.shape or faces.shape != base_faces.shape or not same_face_array(faces, base_faces):
                raise ValueError(f"trial surface topology differs from base: {candidate_path}")
        pairs = validate_audit_report(
            report,
            candidate_sha256=candidate_sha,
            point_count=point_count,
            face_count=face_count,
            expected_bank_keys=expected_keys,
            expected_script_sha256=audit_script_sha,
            expected_predicate_sha256=predicate_sha,
        )
        pair_sets.append(pairs)
        audit_data.append({
            "role": "base" if index == 0 else "trial",
            "report_path": Path(report_path).as_posix(),
            "report_sha256": sha256_file(report_path),
            "candidate_path": candidate_path.as_posix(),
            "candidate_sha256": candidate_sha,
            "point_count": point_count,
            "face_count": face_count,
            "candidate_points_world_m_array_sha256": sha256_array(points),
            "source_O_faces_array_sha256": sha256_array(faces),
            "introduced_or_base_pair_count": len(pairs),
        })

    introduced = introduced_pairs(pair_sets[0], pair_sets[1:])
    arrays = make_barrier_arrays(introduced, lookup, transforms)
    if len(arrays["fixed_triangles_world_mm"]) != len(introduced):
        raise AssertionError("barrier arrays lost typed pairs")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    npz_path = out / "introduced-skin-bone-barriers.npz"
    np.savez_compressed(npz_path, **arrays)
    report = {
        "status": "COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT",
        "schema": OUTPUT_SCHEMA,
        "base_candidate_path": base_path,
        "base_candidate_sha256": base_sha,
        "candidate_points_world_m_array_sha256": point_hash,
        "skin_faces_array_sha256": face_hash,
        "point_count": point_count,
        "face_count": face_count,
        "candidate_points_world_m_shape": list(base_points.shape),
        "skin_faces_shape": list(base_faces.shape),
        "topology_contract": "every trial candidate was located from its 159 report sibling, SHA-verified, and source_O_faces byte-array equal to the base",
        "bank_count": 185,
        "expected_bank_count": 185,
        "audit_script_path": AUDIT_159.as_posix(),
        "audit_script_sha256": audit_script_sha,
        "exact_predicate_path": EXACT_159.as_posix(),
        "exact_predicate_sha256": predicate_sha,
        "bank_reader_path": BANK_HELPER.as_posix(),
        "bank_reader_sha256": bank_helper_sha,
        "bank_source_pins": pins,
        "bank_source_paths": {name: (BANK_DIR / name).as_posix() for name in pins},
        "base_pair_count": len(pair_sets[0]),
        "trial_union_pair_count": len(set().union(*pair_sets[1:])),
        "introduced_pair_count": len(introduced),
        "typed_pairs": [
            {"skin_face_row": int(face), "bank_body": int(body), "stable_bone_id": int(bone),
             "bank_triangle_stable_id": int(tri)}
            for body, bone, face, tri in introduced
        ],
        "fixed_triangles_units": "mm",
        "fixed_triangles_frame": "world",
        "fixed_triangle_transform": "solve(M, (local_triangle_m + com_offset_local_m).T).T + body_position_world_m, then multiply by 1000; pinned raw emitted Float32 triangle values promoted to Float64; no extra Float32 cast",
        "inputs": audit_data,
        "output_npz": npz_path.name,
        "output_npz_sha256": sha256_file(npz_path),
        "geometry_or_native_admission": False,
        "limitations": [
            "This payload adds hard fixed-triangle witnesses for construction-only line searches.",
            "It is not a physics state, mass transfer, tissue deformation, native runtime install, or physical acceptance result.",
        ],
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "base_candidate_sha256", "introduced_pair_count", "output_npz_sha256")}))


if __name__ == "__main__":
    main()
