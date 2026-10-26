#!/usr/bin/env python3
"""Refresh same-topology fixed-bone guard identities and 313 rows.

The output is a construction-only subset: prior valid guards plus typed pairs
resolved from the previous to current complete 159 reports, minus every
current crossing. The complete current report remains authoritative.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
P314 = D / "publication-113/sources/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"
P313 = D / "publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"
P401 = D / "D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"
P026 = D / "publication-101/evidence/204/source/fit/self_separation_026.py"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def typed_pairs_from_report(report: dict) -> set[tuple[int, int, int, int]]:
    """Return (body, stable bone id, skin face row, stable triangle id)."""
    result: set[tuple[int, int, int, int]] = set()
    for bank in report.get("banks", []):
        body = int(bank["body"])
        bone = int(bank["stable_bone_id"])
        sub = bank.get("result") or {}
        for face, triangle in sub.get("triangle_pairs", []) or []:
            key = (body, bone, int(face), int(triangle))
            if min(key) < 0:
                raise ValueError(f"negative typed pair identity: {key}")
            if key in result:
                raise ValueError(f"duplicate typed pair identity: {key}")
            result.add(key)
    return result


def validate_report_binding(report: dict, candidate_sha: str,
                            points: np.ndarray, faces: np.ndarray,
                            expected_bank_count: int = 185) -> set[tuple[int, int, int, int]]:
    if (report.get("complete") is not True
            or int(report.get("bank_count", -1)) != expected_bank_count
            or int(report.get("expected_bank_count", -1)) != expected_bank_count
            or report.get("candidate_sha256") != candidate_sha
            or int(report.get("surface_point_count", -1)) != len(points)
            or int(report.get("surface_face_count", -1)) != len(faces)):
        raise ValueError("complete surface report does not bind to the supplied candidate")
    banks = report.get("banks")
    bankkeys = [(int(x["body"]), int(x["stable_bone_id"])) for x in banks or []]
    if len(bankkeys) != expected_bank_count or len(set(bankkeys)) != expected_bank_count:
        raise ValueError(f"surface report lacks {expected_bank_count} unique typed banks")
    return typed_pairs_from_report(report)


def require_same_topology(base_faces: np.ndarray, current_faces: np.ndarray,
                          current_reference_faces: np.ndarray) -> None:
    def same(a: np.ndarray, b: np.ndarray) -> bool:
        a, b = np.asarray(a), np.asarray(b)
        return (a.dtype == b.dtype and a.shape == b.shape
                and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())
    if not same(base_faces, current_faces):
        raise ValueError("candidate topology changed; same-topology guard reuse is invalid")
    if not same(current_faces, current_reference_faces):
        raise ValueError("current faces do not match embedded 037 reference faces")


def compute_guard_identity_sets(previous_pairs: Iterable[tuple[int, int, int, int]],
                                current_pairs: Iterable[tuple[int, int, int, int]],
                                prior_guards: Iterable[tuple[int, int, int, int]]) -> dict[str, set[tuple[int, int, int, int]]]:
    previous = set(previous_pairs)
    current = set(current_pairs)
    prior = set(prior_guards)
    introduced = current - previous
    resolved = previous - current
    proposed = prior | resolved
    removed_current = proposed & current
    retained = proposed - current
    if introduced & retained or removed_current & retained or removed_current | retained != proposed:
        raise ValueError("current-pair filtering partition failed")
    if not resolved.issubset(retained):
        raise ValueError("resolved prior crossings were not retained as guards")
    return {
        "previous": previous, "current": current, "prior": prior,
        "introduced": introduced, "resolved": resolved, "proposed": proposed,
        "removed_current": removed_current, "retained": retained,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-candidate", type=Path, required=True)
    parser.add_argument("--previous-surface-report", type=Path, required=True)
    parser.add_argument("--prior-guard-report", type=Path, required=True)
    parser.add_argument("--prior-guard-npz", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--surface-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    previous_candidate = args.previous_candidate.resolve()
    previous_report_path = args.previous_surface_report.resolve()
    prior_guard_report_path = args.prior_guard_report.resolve()
    prior_guard_npz_path = args.prior_guard_npz.resolve()
    candidate = args.candidate.resolve()
    surface_report_path = args.surface_report.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    for source in (P314, P313, P401, P026):
        if not source.is_file():
            raise FileNotFoundError(source)

    prev_hash, cur_hash = sha(previous_candidate), sha(candidate)
    previous_report = json.loads(previous_report_path.read_text())
    current_report = json.loads(surface_report_path.read_text())
    prev_arrays, cur_arrays = load_npz(previous_candidate), load_npz(candidate)
    prev_points, prev_faces = prev_arrays["points_world_m"], prev_arrays["source_O_faces"]
    points, faces = cur_arrays["points_world_m"], cur_arrays["source_O_faces"]
    if (points.dtype != np.float32 or prev_points.dtype != np.float32
            or points.ndim != 2 or points.shape[1] != 3
            or prev_points.shape != points.shape or not np.isfinite(points).all()
            or not np.isfinite(prev_points).all()):
        raise ValueError("same-topology candidates require finite matching Float32 point domains")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("candidate source_O_faces must be a nonempty Mx3 array")
    current_reference = cur_arrays.get("area_reference_037_source_O_faces")
    if current_reference is None:
        raise ValueError("candidate is missing matched-037 source-face reference")
    require_same_topology(prev_faces, faces, current_reference)
    previous_pairs = validate_report_binding(previous_report, prev_hash, prev_points, prev_faces)
    current_pairs = validate_report_binding(current_report, cur_hash, points, faces)

    prior_report = json.loads(prior_guard_report_path.read_text())
    prior_npz = load_npz(prior_guard_npz_path)
    phase = import_file("fixed_guard_phase401", P401)
    fit = import_file("fixed_guard_fit026", P026)
    exporter = import_file("fixed_guard_export314", P314)
    rows_owner = import_file("fixed_guard_rows313", P313)
    old_records, old_receipt = phase.validate_fixed_boundary_payload(
        prior_npz, prior_report, candidate_sha256=prev_hash,
        candidate_points=prev_points, candidate_faces=prev_faces,
        npz_path=prior_guard_npz_path)
    prior_set: set[tuple[int, int, int, int]] = set()
    for face, body, bone, triangle in zip(
            prior_npz["skin_face_rows"], prior_npz["bank_body"],
            prior_npz["stable_bone_id"], prior_npz["bank_triangle_stable_id"]):
        key = (int(body), int(bone), int(face), int(triangle))
        if key in prior_set:
            raise ValueError(f"duplicate prior guard identity: {key}")
        prior_set.add(key)
    if len(prior_set) != len(old_records):
        raise ValueError("prior guard report/array identity counts disagree")

    identity = compute_guard_identity_sets(previous_pairs, current_pairs, prior_set)
    bank = import_file("fixed_guard_bank_reader", exporter.BANK_HELPER)
    triangles, triangle_ids, _g, _b, matrices, offsets, positions, bank_keys, bank_pins, _ = (
        bank.read_bank_inputs(exporter.BANK_DIR, exporter.BANK_EVIDENCE))
    expected_bank_keys = tuple((int(x[0]), int(x[1])) for x in np.asarray(bank_keys))
    if len(expected_bank_keys) != 185 or len(set(expected_bank_keys)) != 185:
        raise ValueError("source bank inventory is not 185 unique typed banks")
    validated_current = exporter.validate_audit_report(
        current_report, candidate_sha256=cur_hash, point_count=len(points),
        face_count=len(faces), expected_bank_keys=expected_bank_keys,
        expected_script_sha256=sha(exporter.AUDIT_159),
        expected_predicate_sha256=sha(exporter.EXACT_159))
    if validated_current != current_pairs:
        raise ValueError("validated current audit identities differ from report identities")

    transforms = {int(body): (matrices[int(body)], offsets[int(body)], positions[int(body)])
                  for body in matrices}
    arrays = exporter.make_barrier_arrays(
        sorted(identity["retained"]), exporter.typed_triangle_lookup(triangles, triangle_ids), transforms)
    order = sorted(range(len(arrays["skin_face_rows"])),
                   key=lambda i: (int(arrays["skin_face_rows"][i]), int(arrays["bank_body"][i]),
                                  int(arrays["stable_bone_id"][i]), int(arrays["bank_triangle_stable_id"][i])))
    arrays = {key: np.asarray(value)[order].copy() for key, value in arrays.items()}
    records = [{
        "skin_face_row": int(arrays["skin_face_rows"][i]),
        "bank_body": int(arrays["bank_body"][i]),
        "stable_bone_id": int(arrays["stable_bone_id"][i]),
        "bank_triangle_stable_id": int(arrays["bank_triangle_stable_id"][i]),
        "fixed_triangle_world_mm": arrays["fixed_triangles_world_mm"][i],
    } for i in range(len(order))]
    points_mm = points.astype(np.float64) * 1000.0
    barrier_rows, typed = phase.build_fixed_boundary_barriers(
        points_mm, faces, records, fit, rows_owner, margin_fraction=0.25)
    if len(barrier_rows) != 6 * len(identity["retained"]) or len(typed) != len(identity["retained"]):
        raise ValueError("313 row count mismatch")

    out.mkdir(parents=True, exist_ok=False)
    npz_path = out / "introduced-skin-bone-barriers.npz"
    np.savez_compressed(npz_path, **arrays)
    retained = sorted(identity["retained"])
    report = {
        "status": "COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT",
        "schema": "numi.fixed_skin_bone_barriers.v1",
        "filtered_builder_path": str(Path(__file__).resolve()),
        "filtered_builder_sha256": sha(__file__),
        "base_candidate_path": str(candidate), "base_candidate_sha256": cur_hash,
        "candidate_points_world_m_array_sha256": array_sha(points),
        "skin_faces_array_sha256": array_sha(faces),
        "candidate_points_world_m_shape": list(points.shape), "skin_faces_shape": list(faces.shape),
        "point_count": len(points), "face_count": len(faces),
        "bank_count": 185, "expected_bank_count": 185, "bank_source_pins": bank_pins,
        "bank_source_paths": {name: (exporter.BANK_DIR / name).as_posix() for name in bank_pins},
        "exact_predicate_path": str(exporter.EXACT_159), "exact_predicate_sha256": sha(exporter.EXACT_159),
        "audit_script_path": str(exporter.AUDIT_159), "audit_script_sha256": sha(exporter.AUDIT_159),
        "bank_reader_path": str(exporter.BANK_HELPER), "bank_reader_sha256": sha(exporter.BANK_HELPER),
        "fixed_triangles_units": "mm", "fixed_triangles_frame": "world",
        "output_npz": npz_path.name, "output_npz_sha256": sha(npz_path),
        "introduced_pair_count": len(retained),
        "typed_pairs": [{"skin_face_row": f, "bank_body": body, "stable_bone_id": bone,
                         "bank_triangle_stable_id": tri}
                        for body, bone, f, tri in retained],
        "geometry_or_native_admission": False,
        "inputs": [
            {"role": "base", "candidate_path": str(candidate), "candidate_sha256": cur_hash,
             "report_path": str(surface_report_path), "report_sha256": sha(surface_report_path),
             "pair_count": len(current_pairs)},
            {"role": "previous_complete_surface", "candidate_path": str(previous_candidate),
             "candidate_sha256": prev_hash, "report_path": str(previous_report_path),
             "report_sha256": sha(previous_report_path), "pair_count": len(previous_pairs)},
            {"role": "prior_guard_subset", "candidate_path": str(previous_candidate),
             "candidate_sha256": prev_hash, "report_path": str(prior_guard_report_path),
             "report_sha256": sha(prior_guard_report_path), "npz_path": str(prior_guard_npz_path),
             "npz_sha256": sha(prior_guard_npz_path), "pair_count": len(prior_set)},
        ],
        "construction_scope": {
            "selection_scope": "PRIOR_VALID_GUARDS_UNION_PREVIOUS_MINUS_CURRENT_RESOLVED_TYPED_PAIRS_MINUS_CURRENT_COMPLETE_159",
            "prior_guard_pairs": len(prior_set), "previous_complete_pair_count": len(previous_pairs),
            "current_complete_pair_count": len(current_pairs), "introduced_current_pairs": len(identity["introduced"]),
            "resolved_pairs_added_as_guards": len(identity["resolved"]),
            "resolved_pairs_retained": len(identity["resolved"] & identity["retained"]),
            "prior_guards_filtered_as_current": len(prior_set & current_pairs),
            "union_pairs_before_filter": len(identity["proposed"]),
            "current_pairs_removed_from_union": len(identity["removed_current"]),
            "retained_guard_pairs": len(retained), "313_rows": len(barrier_rows),
            "current_complete_185_bank_audit_validated": True,
            "same_topology_bytewise": True, "current_matches_embedded_037_faces": True,
            "interpretation": "Incremental construction guard subset only. Complete current 159 report remains authoritative. Current crossings are excluded; absence from this subset does not establish clearance.",
        },
        "introduced_pairs_previous_to_current": [
            {"bank_body": body, "stable_bone_id": bone, "skin_face_row": face,
             "bank_triangle_stable_id": tri, "current_crossing": True, "constraint_emitted": False}
            for body, bone, face, tri in sorted(identity["introduced"])],
        "resolved_pairs_previous_to_current": [
            {"bank_body": body, "stable_bone_id": bone, "skin_face_row": face,
             "bank_triangle_stable_id": tri, "current_crossing": False,
             "constraint_emitted": ((body, bone, face, tri) in identity["retained"])}
            for body, bone, face, tri in sorted(identity["resolved"])],
        "removed_current_pairs": [
            {"bank_body": body, "stable_bone_id": bone, "skin_face_row": face,
             "bank_triangle_stable_id": tri,
             "source": "prior_guard" if (body, bone, face, tri) in prior_set else "resolved_pair",
             "reason": "PRESENT_IN_COMPLETE_CURRENT_159"}
            for body, bone, face, tri in sorted(identity["removed_current"])],
        "limitations": ["Incremental construction guard subset; complete current 159 report remains authoritative.",
                        "Absence from this subset does not establish clearance.",
                        "No native, physical, or anatomy qualification."],
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    # Re-read and exercise the exact production payload validator plus 313 builder.
    returned, receipt = phase.validate_fixed_boundary_payload(
        load_npz(npz_path), report, candidate_sha256=cur_hash,
        candidate_points=points, candidate_faces=faces, npz_path=npz_path)
    rebuilt, typed_rebuilt = phase.build_fixed_boundary_barriers(
        points_mm, faces, returned, fit, rows_owner, margin_fraction=0.25)
    zero_feasible = all(float(row["lower"]) <= 2e-8 for row in rebuilt)
    if (not zero_feasible or len(rebuilt) != len(barrier_rows)
            or len(typed_rebuilt) != len(typed)):
        raise ValueError("401/313 zero-baseline payload validation failed")
    validation = {
        "status": "PASS_401_FIXED_BOUNDARY_PAYLOAD_AND_313_ROW_VALIDATION",
        "candidate_sha256": cur_hash, "npz_sha256": sha(npz_path),
        "report_sha256": sha(report_path), "filtered_guard_pairs": len(returned),
        "313_rows": len(rebuilt), "all_rows_zero_feasible": zero_feasible,
        "receipt": receipt,
    }
    validation_path = out / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS", "candidate": str(candidate), "candidate_sha256": cur_hash,
        "output_npz": str(npz_path), "output_npz_sha256": sha(npz_path),
        "report": str(report_path), "report_sha256": sha(report_path),
        "validation": str(validation_path), "validation_sha256": sha(validation_path),
        "prior_guards": len(prior_set), "introduced": len(identity["introduced"]),
        "resolved_added": len(identity["resolved"] & identity["retained"]),
        "current": len(current_pairs), "filtered": len(identity["removed_current"]),
        "retained": len(returned), "rows": len(rebuilt),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
