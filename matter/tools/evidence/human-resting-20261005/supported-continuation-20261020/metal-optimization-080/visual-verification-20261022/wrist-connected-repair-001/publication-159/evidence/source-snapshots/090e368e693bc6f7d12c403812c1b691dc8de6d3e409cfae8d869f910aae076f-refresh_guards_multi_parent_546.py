#!/usr/bin/env python3
"""Isolated 546 shim: frozen 541 refresh with strict registered CSR unions."""
from __future__ import annotations
import hashlib
import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D"
REFRESH541 = D / "focused-clearance-541/constraints/refresh_guards_541.py"
REFRESH541_SHA = "293e4da49a10f07420d1da7c512823dcb182dca246e411c3775d18866b27a241"
OWNER402 = D / "local-two-parent-retriangulation-402/two_parent_patch_flips_402.py"
OWNER402_SHA = "f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _csr(candidate: dict, prefix: str, ptr_key: str, vals_key: str, n: int,
         allowed_ids: set[int] | None = None, upper: int | None = None,
         allow_empty: bool = False) -> tuple[list[set[int]], np.ndarray, np.ndarray]:
    if ptr_key not in candidate or vals_key not in candidate:
        raise ValueError(f"missing {prefix} ancestry CSR")
    ptr = np.asarray(candidate[ptr_key])
    vals = np.asarray(candidate[vals_key])
    if ptr.dtype.kind not in "iu" or vals.dtype.kind not in "iu":
        raise ValueError(f"{prefix} CSR pointer and IDs must be integer arrays")
    ptr = ptr.astype(np.int64, copy=False)
    if ptr.shape != (n + 1,) or int(ptr[0]) != 0 or int(ptr[-1]) != len(vals):
        raise ValueError(f"{prefix} CSR shape or endpoints invalid")
    if np.any(ptr[1:] < ptr[:-1]):
        raise ValueError(f"{prefix} CSR pointers are not monotone")
    ids = vals.astype(np.int64, copy=False)
    if len(ids) and np.any(ids < 0):
        raise ValueError(f"{prefix} CSR has negative registered IDs")
    if upper is not None and len(ids) and int(ids.max()) >= upper:
        raise ValueError(f"{prefix} CSR ID outside registered row range [0,{upper})")
    if allowed_ids is not None and any(int(v) not in allowed_ids for v in ids):
        bad = next(int(v) for v in ids if int(v) not in allowed_ids)
        raise ValueError(f"{prefix} CSR ID {bad} absent from pinned registered parent domain")
    rows = []
    for i in range(n):
        row = ids[int(ptr[i]):int(ptr[i + 1])]
        if not allow_empty and len(row) == 0:
            raise ValueError(f"{prefix} CSR row {i} is empty")
        if len(row) and (len(np.unique(row)) != len(row) or np.any(row[1:] <= row[:-1])):
            raise ValueError(f"{prefix} CSR row {i} must be sorted and unique")
        rows.append(set(map(int, row)))
    return rows, ptr, ids


def _semantic_value(candidate: dict, key: str) -> str:
    if key not in candidate:
        raise ValueError(f"missing provenance semantics {key}")
    return " ".join(map(str, np.asarray(candidate[key]).reshape(-1).tolist()))


def _semantic(candidate: dict, key: str, required: str) -> None:
    value = _semantic_value(candidate, key)
    if required not in value:
        raise ValueError(f"unexpected {key} semantics: {value!r}")


def _parent_lineage_mode(candidate: dict) -> str:
    parent = _semantic_value(candidate, "refinement_face_parent_ancestry_semantics")
    bary = _semantic_value(candidate, "refinement_face_parent_barycentric_semantics")
    legacy = ("CSR union of immediate parent face rows" in parent and
              "NaN explicitly marks multi-parent" in bary)
    affine = ("CSR singleton immediate input-face row for each affine child" in parent and
              bary == "output child vertices in barycentric coordinates of pre-refinement parent O face")
    if legacy:
        return "legacy_parent_union_with_nan_multi_parent"
    if affine:
        return "affine_child_to_immediate_input_face"
    raise ValueError(f"unsupported or inconsistent refinement parent semantics: {parent!r}; {bary!r}")


def _validate_affine_parent_maps(candidate: dict, parent_candidate: dict,
                                 parent_rows: list[set[int]], bary: np.ndarray) -> dict:
    n = len(parent_rows)
    if any(len(row) != 1 for row in parent_rows):
        raise ValueError("affine-child contract requires exactly one immediate input face per child")
    current_points = np.asarray(candidate["points_world_m"])
    current_faces = np.asarray(candidate["source_O_faces"], dtype=np.int64)
    current_reference = np.asarray(candidate.get("area_reference_037_points_world_m_f64"), dtype=np.float64)
    current_reference_faces = np.asarray(candidate.get("area_reference_037_source_O_faces"), dtype=np.int64)
    parent_reference = np.asarray(parent_candidate.get("area_reference_037_points_world_m_f64"), dtype=np.float64)
    parent_faces = np.asarray(parent_candidate["source_O_faces"], dtype=np.int64)
    parent_reference_faces = np.asarray(parent_candidate.get("area_reference_037_source_O_faces"), dtype=np.int64)
    if current_points.dtype != np.float32 or current_points.shape[1:] != (3,) or not np.isfinite(current_points).all():
        raise ValueError("current child coordinates must be finite emitted Float32 N by 3")
    if current_faces.shape != (n, 3) or np.any(current_faces < 0) or np.any(current_faces >= len(current_points)):
        raise ValueError("affine child face domain is malformed")
    if current_reference.shape != current_points.shape or not np.isfinite(current_reference).all():
        raise ValueError("current immutable area-reference points are missing or malformed")
    if current_reference_faces.shape != current_faces.shape or not np.array_equal(current_reference_faces, current_faces):
        raise ValueError("current area-reference face topology differs from the emitted child topology")
    if parent_reference.ndim != 2 or parent_reference.shape[1] != 3 or not np.isfinite(parent_reference).all():
        raise ValueError("immediate-parent immutable area-reference points are missing or malformed")
    if (parent_faces.ndim != 2 or parent_faces.shape[1] != 3 or
            parent_reference_faces.shape != parent_faces.shape or
            not np.array_equal(parent_reference_faces, parent_faces) or
            np.any(parent_faces < 0) or np.any(parent_faces >= len(parent_reference))):
        raise ValueError("immediate-parent area-reference face domain is malformed")
    if (not np.isfinite(bary).all() or np.any(bary < -2e-8) or np.any(bary > 1.0 + 2e-8)
            or np.max(np.abs(bary.sum(axis=2) - 1.0), initial=0.0) > 2e-8):
        raise ValueError("affine parent barycentric map is nonfinite or outside the simplex")
    ids = np.fromiter((next(iter(row)) for row in parent_rows), dtype=np.int64, count=n)
    parent_triangles = parent_reference[parent_reference_faces[ids]]
    reconstructed = np.einsum("fij,fjk->fik", bary, parent_triangles)
    actual = current_reference[current_reference_faces]
    max_error = float(np.max(np.abs(reconstructed - actual), initial=0.0))
    reference_scale = max(1.0, float(np.max(np.abs(parent_reference), initial=0.0)))
    tolerance = max(64.0 * np.finfo(np.float64).eps * reference_scale, 1e-12)
    if max_error > tolerance:
        raise ValueError(f"affine parent barycentrics do not reconstruct immutable area-reference child vertices: {max_error} > {tolerance}")

    old_ptr_key = "refinement_face_parent_ancestry_indptr"
    old_rows_key = "refinement_face_parent_ancestry_face_rows"
    archive_ptr_key = "prior_input_parent_546__input_refinement_face_parent_ancestry_indptr"
    archive_rows_key = "prior_input_parent_546__input_refinement_face_parent_ancestry_face_rows"
    has_parent_csr = old_ptr_key in parent_candidate and old_rows_key in parent_candidate
    has_archive_csr = archive_ptr_key in candidate and archive_rows_key in candidate
    if has_parent_csr != has_archive_csr:
        raise ValueError("archived pre-refinement parent union CSR is incomplete")
    if has_parent_csr:
        if (not np.array_equal(np.asarray(candidate[archive_ptr_key]), np.asarray(parent_candidate[old_ptr_key])) or
                not np.array_equal(np.asarray(candidate[archive_rows_key]), np.asarray(parent_candidate[old_rows_key]))):
            raise ValueError("archived pre-refinement parent union differs from immediate input")
        old_sem = "refinement_face_parent_ancestry_semantics"
        archive_sem = "prior_input_parent_546__input_refinement_face_parent_ancestry_semantics"
        if old_sem in parent_candidate:
            if archive_sem not in candidate or not np.array_equal(np.asarray(candidate[archive_sem]), np.asarray(parent_candidate[old_sem])):
                raise ValueError("archived pre-refinement parent semantics differ from immediate input")
    return {"max_parent_reconstruction_error_m": max_error,
            "parent_reconstruction_tolerance_m": tolerance,
            "reconstruction_domain": "immutable area_reference_037 points and topology",
            "current_geometry_affine_reconstructed": False,
            "current_geometry_qualification": "independent source/trust/area/exact audits",
            "archived_input_parent_union_preserved": bool(has_parent_csr)}


def validate_ancestry(candidate: dict, parent_candidate: dict, surface: dict) -> dict:
    """Validate lineage domains and all parent unions without scalarizing them."""
    b = load_module("two_parent_patch_flips_402_validator_546", OWNER402)
    if sha(OWNER402) != OWNER402_SHA:
        raise ValueError("pinned 402 validator hash mismatch")
    b.validate_candidate(candidate)
    faces = np.asarray(candidate["source_O_faces"])
    n = len(faces)
    parent_faces = np.asarray(parent_candidate["source_O_faces"])
    if parent_faces.ndim != 2 or parent_faces.shape[1] != 3:
        raise ValueError("pinned immediate-parent face domain malformed")
    kind = np.asarray(candidate["face_kind"])
    if np.any(~np.isin(kind, [0, 1, 2])):
        raise ValueError("face_kind includes an unregistered source kind")
    _semantic(candidate, "parent_fine_detail_face_row_semantics", "original NHSKIN face row")
    _semantic(candidate, "parent_fine_subface_row_semantics", "run-026 fine_subfaces row")
    _semantic(candidate, "face_fine_detail_ancestry_semantics", "CSR union of source fine detail face rows")
    parent_mode = _parent_lineage_mode(candidate)

    pfaces, pptr, pvals = _csr(candidate, "immediate-parent face", "refinement_face_parent_ancestry_indptr",
        "refinement_face_parent_ancestry_face_rows", n, upper=len(parent_faces))
    src_parent = np.asarray(parent_candidate.get("face_source_ancestry_face_rows", ()))
    sub_parent = np.asarray(parent_candidate.get("face_fine_subface_ancestry_rows", ()))
    det_parent = np.asarray(parent_candidate.get("face_fine_detail_ancestry_rows", ()))
    src_pool = set(map(int, src_parent))
    sub_pool = set(map(int, sub_parent))
    det_pool = set(map(int, det_parent))
    if not src_pool or not sub_pool or not det_pool:
        raise ValueError("pinned parent lacks registered source/detail/subface ID domains")
    src, _, src_ids = _csr(candidate, "source-face", "face_source_ancestry_indptr",
        "face_source_ancestry_face_rows", n, allowed_ids=src_pool)
    sub, _, sub_ids = _csr(candidate, "fine-subface", "face_fine_subface_ancestry_indptr",
        "face_fine_subface_ancestry_rows", n, allowed_ids=sub_pool, allow_empty=True)
    detail, _, detail_ids = _csr(candidate, "fine-detail", "face_fine_detail_ancestry_indptr",
        "face_fine_detail_ancestry_rows", n, allowed_ids=det_pool, allow_empty=True)

    fine_scalar = np.asarray(candidate["parent_fine_detail_face_row"], dtype=np.int64)
    sub_scalar = np.asarray(candidate["parent_fine_subface_row"], dtype=np.int64)
    source_marker = np.asarray(candidate["parent_source_face_marker"], dtype=np.int64)
    parent_scalar = np.asarray(candidate["refinement_face_parent_face_row"], dtype=np.int64)
    bary = np.asarray(candidate["refinement_face_parent_barycentric"], dtype=np.float64)
    if bary.shape != (n, 3, 3):
        raise ValueError("immediate-parent barycentric map has invalid shape")
    for i in range(n):
        want_marker = next(iter(src[i])) if len(src[i]) == 1 else -1
        if int(source_marker[i]) != want_marker:
            raise ValueError(f"source marker does not match source CSR union at face {i}")
        want_sub = next(iter(sub[i])) if len(sub[i]) == 1 else -1
        if int(sub_scalar[i]) != want_sub:
            raise ValueError(f"fine-subface scalar does not match CSR union at face {i}")
        want_detail = next(iter(detail[i])) if len(detail[i]) == 1 else -1
        if int(kind[i]) == 2:
            if not detail[i] <= src[i]:
                raise ValueError(f"fine-detail IDs are not contained by source-face union at face {i}")
            if int(fine_scalar[i]) != want_detail:
                raise ValueError(f"fine-detail scalar does not match CSR union at face {i}")
        elif detail[i] or int(fine_scalar[i]) != -1:
            raise ValueError(f"non-fine-detail source kind has fine-detail ancestry at face {i}")

        if not pfaces[i]:
            raise ValueError(f"immediate-parent union is empty at face {i}")
        if parent_mode == "affine_child_to_immediate_input_face":
            if len(pfaces[i]) != 1:
                raise ValueError(f"affine child must have one immediate parent at face {i}")
            if int(parent_scalar[i]) != next(iter(pfaces[i])):
                raise ValueError(f"affine immediate-parent scalar differs from CSR at face {i}")
        else:
            want_parent = next(iter(pfaces[i])) if len(pfaces[i]) == 1 else -1
            if int(parent_scalar[i]) != want_parent:
                raise ValueError(f"immediate-parent scalar does not match parent CSR union at face {i}")
            if len(pfaces[i]) == 1:
                row = bary[i]
                if (not np.isfinite(row).all() or np.any(row < -2e-8) or np.any(row > 1 + 2e-8)
                        or np.max(np.abs(row.sum(axis=1) - 1.0)) > 2e-8):
                    raise ValueError(f"single-parent affine map invalid at face {i}")
            elif not np.isnan(bary[i]).all():
                raise ValueError(f"multi-parent face {i} must retain explicit NaN affine map")

    affine_map_receipt = (_validate_affine_parent_maps(candidate, parent_candidate, pfaces, bary)
                          if parent_mode == "affine_child_to_immediate_input_face" else None)

    crossed = sorted({int(f) for bank in surface.get("banks", [])
        for f, _t in ((bank.get("result") or {}).get("triangle_pairs") or [])})
    if any(f < 0 or f >= n for f in crossed):
        raise ValueError("exact surface report contains out-of-range crossing face row")
    for f in crossed:
        if int(kind[f]) != 2 or not detail[f]:
            raise ValueError(f"crossing face {f} lacks registered fine-detail ancestry")
        if not detail[f] <= src[f]:
            raise ValueError(f"crossing face {f} detail union escapes source-face union")
    return {
        "parent_candidate_path": str(parent_candidate.get("candidate_path", "provided")),
        "parent_face_count": len(parent_faces), "current_face_count": n,
        "refinement_parent_csr_rows": len(pvals),
        "refinement_parent_contract_mode": parent_mode,
        "affine_parent_map_validation": affine_map_receipt,
        "multi_parent_current_face_count": int(np.count_nonzero(np.diff(pptr) > 1)),
        "crossing_face_count": len(crossed),
        "crossing_faces_with_multi_parent_csr": sum(len(pfaces[f]) > 1 for f in crossed),
        "crossing_faces_with_multi_source_union": sum(len(src[f]) > 1 for f in crossed),
        "crossing_faces_with_multi_detail_union": sum(len(detail[f]) > 1 for f in crossed),
        "crossing_face_kind_counts": {"2": len(crossed)},
        "crossing_fine_detail_ancestry_unique_and_matching": False,
        "crossing_fine_detail_ancestry_union_validated": True,
        "parent_indices_in_range": True,
        "registered_source_id_count": len(src_pool),
        "registered_subface_id_count": len(sub_pool),
        "registered_detail_id_count": len(det_pool),
        "source_csr_value_count": len(src_ids), "subface_csr_value_count": len(sub_ids),
        "detail_csr_value_count": len(detail_ids),
        "all_scalar_summaries_match_csr_unions": True,
        "construction_only": True, "native_admission": False,
    }


def main() -> None:
    if sha(REFRESH541) != REFRESH541_SHA:
        raise SystemExit("frozen 541 refresh hash mismatch")
    refresh = load_module("frozen_refresh541_with_union_validator_546", REFRESH541)
    refresh.validate_ancestry = validate_ancestry
    refresh.main()


if __name__ == "__main__":
    main()
