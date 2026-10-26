#!/usr/bin/env python3
"""Pinned 546 wrapper extending 500 only for explicit kind-2 source-face unions."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D"
OWNER500 = D / "source-outward-proposal-500/build_outward_nearest_union_targets_500.py"
OWNER500_SHA = "454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf"
EPS = np.finfo(np.float64).eps


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_target_builder_500_extension_546", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned 500 owner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def csr_row(candidate: dict, prefix: str, face_row: int, face_count: int,
            ptr_name: str, values_name: str) -> tuple[list[int], int]:
    if ptr_name not in candidate or values_name not in candidate:
        raise ValueError(f"missing {prefix} ancestry CSR")
    ptr = np.asarray(candidate[ptr_name])
    vals = np.asarray(candidate[values_name])
    if ptr.dtype.kind not in "iu" or vals.dtype.kind not in "iu":
        raise ValueError(f"{prefix} ancestry CSR IDs/pointers are not integers")
    if ptr.shape != (face_count + 1,) or int(ptr[0]) != 0 or int(ptr[-1]) != len(vals) or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError(f"{prefix} ancestry CSR is malformed")
    if face_row < 0 or face_row >= face_count:
        raise ValueError(f"{prefix} face row outside candidate")
    lo, hi = int(ptr[face_row]), int(ptr[face_row + 1])
    row = np.asarray(vals[lo:hi], dtype=np.int64)
    if len(row) == 0 or np.any(row < 0) or len(np.unique(row)) != len(row) or np.any(row[1:] <= row[:-1]):
        raise ValueError(f"{prefix} ancestry row must contain sorted unique nonnegative registered IDs")
    return list(map(int, row)), len(vals)


def multi_parent_source_outward_normal(face_row: int, candidate: dict, detail: dict,
                                       orientation: dict) -> tuple[np.ndarray, dict]:
    """Resolve a kind-2 multi-source normal as an area-weighted oriented union."""
    face_row = int(face_row)
    kind = np.asarray(candidate["face_kind"])
    if kind.ndim != 1 or not 0 <= face_row < len(kind) or int(kind[face_row]) != 2:
        raise ValueError("multi-source normal applies only to an in-range kind-2 face")
    count = len(kind)
    fine_scalar = np.asarray(candidate["parent_fine_detail_face_row"])
    source_marker = np.asarray(candidate["parent_source_face_marker"])
    coarse_scalar = np.asarray(candidate["parent_coarse_face_row"])
    sub_scalar = np.asarray(candidate["parent_fine_subface_row"])
    if any(a.shape != (count,) or a.dtype.kind not in "iu" for a in (fine_scalar, source_marker, coarse_scalar, sub_scalar)):
        raise ValueError("typed parent scalar arrays malformed")
    source_ids, source_nvals = csr_row(candidate, "source-face", face_row, count,
        "face_source_ancestry_indptr", "face_source_ancestry_face_rows")
    detail_ids, detail_nvals = csr_row(candidate, "fine-detail", face_row, count,
        "face_fine_detail_ancestry_indptr", "face_fine_detail_ancestry_rows")
    subface_ids, subface_nvals = csr_row(candidate, "fine-subface", face_row, count,
        "face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows")
    if len(detail_ids) < 2:
        raise ValueError("multi-source extension requires at least two detail ancestry parents")
    # The original scalar fields are uniqueness summaries; -1 explicitly marks
    # unions. A multi-source face may never be assigned a synthetic single parent.
    if int(fine_scalar[face_row]) != -1 or int(source_marker[face_row]) != -1 or int(sub_scalar[face_row]) != -1:
        raise ValueError("multi-source face must retain -1 scalar sentinels")
    if int(coarse_scalar[face_row]) != -1:
        raise ValueError("kind-2 fine-source union unexpectedly has a coarse scalar parent")
    if set(detail_ids) != set(source_ids):
        raise ValueError("fine-detail union and registered source-face union disagree")
    selected_subfaces = np.asarray(candidate["selected_fine_subface_rows"])
    if selected_subfaces.dtype.kind not in "iu" or selected_subfaces.ndim != 1:
        raise ValueError("selected fine-subface registered ID set missing or malformed")
    if not set(subface_ids).issubset(set(map(int, selected_subfaces))):
        raise ValueError("fine-subface ancestry ID outside registered selected subface domain")
    if len(subface_ids) < 2:
        raise ValueError("multi-source detail face lacks matching multi-parent subface CSR")

    if "face_fine_detail_ancestry_semantics" not in candidate:
        raise ValueError("missing fine-detail ancestry semantics")
    semantics = " ".join(map(str, np.asarray(candidate["face_fine_detail_ancestry_semantics"]).reshape(-1).tolist()))
    if "CSR union of source fine detail face rows" not in semantics:
        raise ValueError("fine-detail CSR semantics do not match registered source union")
    if "parent_fine_detail_face_row_semantics" not in candidate:
        raise ValueError("missing scalar parent semantics")
    scalar_semantics = " ".join(map(str, np.asarray(candidate["parent_fine_detail_face_row_semantics"]).reshape(-1).tolist()))
    if "-1 for multi-source ancestry" not in scalar_semantics:
        raise ValueError("parent scalar does not declare multi-source sentinel")

    fine_faces = np.asarray(detail["source_surface_faces_detail_rows"])
    fine_points = np.asarray(detail["detail_baseline_world_m"], dtype=np.float64)
    if fine_faces.ndim != 2 or fine_faces.shape[1] != 3 or fine_faces.dtype.kind not in "iu":
        raise ValueError("pinned source detail triangles malformed")
    if fine_points.ndim != 2 or fine_points.shape[1] != 3 or not np.isfinite(fine_points).all():
        raise ValueError("pinned source detail points malformed")
    if any(parent >= len(fine_faces) for parent in detail_ids):
        raise ValueError("fine-detail parent ID outside pinned source triangle range")
    sign = int(orientation.get("fine_outward_sign", 0))
    if sign not in (-1, 1):
        raise ValueError("pinned fine-source outward sign is invalid")
    unit_normals = []
    areas = []
    for parent in detail_ids:
        vertices = np.asarray(fine_faces[parent], dtype=np.int64)
        if np.any(vertices < 0) or np.any(vertices >= len(fine_points)):
            raise ValueError(f"source triangle {parent} vertex index outside pinned detail point range")
        tri = fine_points[vertices]
        if tri.shape != (3, 3) or not np.isfinite(tri).all():
            raise ValueError(f"source triangle {parent} is nonfinite or malformed")
        raw = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        twice_area = float(np.linalg.norm(raw))
        if not math.isfinite(twice_area) or twice_area <= 0.0:
            raise ValueError(f"source triangle {parent} is degenerate")
        unit_normals.append((sign / twice_area) * raw)
        areas.append(0.5 * twice_area)
    normals = np.asarray(unit_normals, dtype=np.float64)
    areas_a = np.asarray(areas, dtype=np.float64)
    total_area = float(areas_a.sum())
    weighted = np.sum(normals * areas_a[:, None], axis=0)
    resultant = float(np.linalg.norm(weighted))
    if not math.isfinite(total_area) or total_area <= 0 or not math.isfinite(resultant) or resultant <= 64 * EPS * total_area:
        raise ValueError("area-weighted source normals cancel or are numerically unresolved")
    outward = weighted / resultant
    dots = normals @ outward
    if not np.isfinite(dots).all() or np.any(dots <= 64 * EPS):
        raise ValueError("registered source normals do not lie in one strict open hemisphere")
    weights = areas_a / total_area
    provenance = {
        "kind": "area_weighted_multi_source_registered_NHSKIN_face_union",
        "registered_source_face_rows": source_ids,
        "registered_fine_detail_face_rows": detail_ids,
        "registered_fine_subface_rows": subface_ids,
        "source_triangle_area_m2": areas_a.tolist(),
        "area_weights": weights.tolist(),
        "source_outward_normals_world_unit": normals.tolist(),
        "source_normal_dots_with_weighted_union": dots.tolist(),
        "weighted_union_resultant_area_m2": resultant,
        "total_source_area_m2": total_area,
        "minimum_open_hemisphere_dot": float(dots.min()),
        "source_ids_range": [min(detail_ids), max(detail_ids)],
        "detail_face_row_domain_count": int(len(fine_faces)),
        "source_csr_value_count": source_nvals,
        "detail_csr_value_count": detail_nvals,
        "subface_csr_value_count": subface_nvals,
        "normal_policy": "sum all orientation-signed unit source normals weighted by original source triangle area; reject cancellation or any source normal with nonpositive/near-zero dot against the result",
    }
    return outward, provenance


def install_extension(owner):
    original = owner.registered_source_outward_face_normal

    def patched(face_row, candidate, detail, base, orientation):
        i = int(face_row)
        kind = np.asarray(candidate["face_kind"])
        if i < 0 or i >= len(kind):
            return original(face_row, candidate, detail, base, orientation)
        if int(kind[i]) != 2:
            return original(face_row, candidate, detail, base, orientation)
        # Leave the old single-parent implementation byte-for-byte in use.
        ptr_key = "face_fine_detail_ancestry_indptr"
        vals_key = "face_fine_detail_ancestry_rows"
        available = set(candidate.files) if hasattr(candidate, "files") else set(candidate)
        if ptr_key not in available or vals_key not in available:
            return original(face_row, candidate, detail, base, orientation)
        ids, _ = csr_row(candidate, "fine-detail", i, len(kind), ptr_key, vals_key)
        scalar = int(np.asarray(candidate["parent_fine_detail_face_row"])[i])
        if len(ids) <= 1:
            return original(face_row, candidate, detail, base, orientation)
        if scalar != -1:
            raise ValueError("multi-source detail CSR conflicts with non-sentinel scalar parent")
        return multi_parent_source_outward_normal(i, candidate, detail, orientation)

    owner.registered_source_outward_face_normal = patched
    return original


def main(argv=None):
    if sha(OWNER500) != OWNER500_SHA:
        raise SystemExit("pinned 500 target builder hash mismatch")
    owner = load_module(OWNER500)
    original = install_extension(owner)
    owner.main(argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--surface-report", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args, _ = ap.parse_known_args(argv)
    out = args.output_dir.resolve()
    report_path = out / "report.json"
    records_path = out / "target-records.jsonl"
    targets_path = out / "targets.npz"
    if not report_path.is_file() or not records_path.is_file() or not targets_path.is_file():
        raise RuntimeError("pinned 500 builder did not produce its complete target artifacts")
    report = json.loads(report_path.read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    multi = [r for r in records if r.get("source_normal_provenance", {}).get("kind") == "area_weighted_multi_source_registered_NHSKIN_face_union"]
    pass_count = int(report.get("passing_targets", -1))
    pair_count = int(report.get("crossing_pairs", -2))
    report["multi_parent_source_normal_extension"] = {
        "path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve()),
        "pinned_500_owner_path": str(OWNER500), "pinned_500_owner_sha256": sha(OWNER500),
        "method": "all registered fine-source triangles, area-weighted oriented normal sum",
        "single_parent_helper_unchanged": True,
        "multi_parent_records": len(multi), "multi_parent_face_rows": sorted({int(r["face_row"]) for r in multi}),
        "multi_parent_source_ids_recorded_per_target": True,
        "strict_open_hemisphere_required": True, "cancellation_rejected": True,
        "exact_certified_union_ray_and_all_185_endpoint_check": "pinned 500 implementation unchanged",
        "coverage_pass": pass_count == pair_count,
    }
    report["multi_parent_source_normal_extension"]["target_records_sha256"] = sha(records_path)
    report["multi_parent_source_normal_extension"]["targets_npz_sha256"] = sha(targets_path)
    report["multi_parent_source_normal_extension"]["candidate_sha256"] = sha(args.candidate)
    report["multi_parent_source_normal_extension"]["surface_report_sha256"] = sha(args.surface_report)
    report["multi_parent_source_normal_extension"]["owner_single_parent_function_sha256"] = OWNER500_SHA
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    final = {"report": str(report_path), "report_sha256": sha(report_path),
        "candidate_sha256": sha(args.candidate), "surface_report_sha256": sha(args.surface_report),
        "targets_npz": str(targets_path), "targets_npz_sha256": sha(targets_path),
        "target_records": str(records_path), "target_records_sha256": sha(records_path),
        "passing_targets": pass_count, "crossing_pairs": pair_count,
        "multi_parent_faces": report["multi_parent_source_normal_extension"]["multi_parent_face_rows"]}
    (out / "multi-parent-extension-receipt.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2))
    if pass_count != pair_count:
        raise SystemExit(f"target coverage incomplete: {pass_count}/{pair_count}; records preserved")


if __name__ == "__main__":
    main()
