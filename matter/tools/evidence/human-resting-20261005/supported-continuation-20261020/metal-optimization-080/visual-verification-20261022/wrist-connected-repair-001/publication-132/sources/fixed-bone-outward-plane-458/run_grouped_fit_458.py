"""Run pinned 408 fit with matched current fixed-bone outward-plane targets.

This wrapper is opt-in and construction-only. It preserves the frozen 408
validation and point-target block, replaces its face objective with local
fixed-bone outward-plane targets, and adds nonpositive corner bounds to the
existing hard barrier list. Complete exact post-fit audits remain mandatory.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
OWNER = D / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
HELPER = Path(__file__).with_name("fixed_bone_outward_plane_458.py")
EMITTER314 = D / "D/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"
AUDIT159 = D / "audit_skin_bone_surfaces_159.py"
OWNER_SHA = "5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
EMITTER314_SHA = "3422f30a61db0724d234c487d70fc3d74237a802261c2f9a72af9e15843a1c78"
AUDIT159_SHA = "de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1"
EXPECTED_CANDIDATE_SHA = "1ac004ef6f6646ce4720360c50b7fd9855336a4967c3e97b199cf9bc0350049a"
EXPECTED_AUDIT_REPORT_SHA = "55e6e4ffc5b296f8793de9f65060e533290021e09b55d15213221c996091737b"
EXPECTED_FACE_TARGET_SHA = "e7c5ffe5e37637f3207026e5f8d8e2be492e5810794ad07f814b33f2a5797cd0"
EXPECTED_POINT_TARGET_SHA = "c20370c4a856c9a975000eee800c9c96512c6765473022ee84311e870f7dbe0c"
BANK_WINDING_POLICY = "all triangle edges occur twice with opposite orientation and centered signed volume is positive for every implicated typed bank"

_STATE = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arg_value(name, required=True):
    argv = sys.argv[1:]
    if name not in argv:
        if required:
            raise ValueError(f"missing required fit argument {name}")
        return None
    index = argv.index(name) + 1
    if index >= len(argv):
        raise ValueError(f"missing value after {name}")
    return Path(argv[index])


def certify_implicated_bank_winding(triangles_local_m, identities, keys):
    """Check exact-coordinate closed oriented edge incidence and positive volume."""
    certificates = []
    ids = np.asarray(identities)
    tris = np.asarray(triangles_local_m, dtype=np.float64)
    for body, bone in sorted(keys):
        mask = (ids[:, 0].astype(np.int64) == int(body)) & (ids[:, 1].astype(np.int64) == int(bone))
        bank = tris[mask]
        if not len(bank):
            raise ValueError(f"crossed typed bank has no exact emitted triangles: {(body, bone)}")
        edges = defaultdict(lambda: [0, 0])
        for tri in bank:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                ka, kb = tuple(map(float, a)), tuple(map(float, b))
                key = (ka, kb) if ka < kb else (kb, ka)
                edges[key][0] += 1
                edges[key][1] += 1 if ka < kb else -1
        bad = sum(count != 2 or orientation_sum != 0 for count, orientation_sum in edges.values())
        if bad:
            raise ValueError(f"typed bank {(body, bone)} failed closed opposite-edge winding check: {bad}")
        centered = bank - bank.mean(axis=(0, 1), keepdims=True)
        signed_volume_m3 = float(np.einsum(
            "ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2])
        ).sum() / 6.0)
        if not np.isfinite(signed_volume_m3) or signed_volume_m3 <= 0:
            raise ValueError(f"typed bank {(body, bone)} is not positive-volume outward wound")
        certificates.append({
            "body": int(body), "bone_id": int(bone),
            "triangle_count": int(len(bank)), "undirected_edge_count": int(len(edges)),
            "bad_oriented_edge_count": int(bad),
            "centered_signed_volume_m3": signed_volume_m3,
            "closed_opposite_edge_incidence": True,
            "positive_volume_orientation": True,
        })
    return certificates


def prepare_crossing_inputs(base_path, audit_path):
    if sha(OWNER) != OWNER_SHA or sha(EMITTER314) != EMITTER314_SHA:
        raise ValueError("pinned 408/314 owner source hash mismatch")
    if sha(AUDIT159) != AUDIT159_SHA:
        raise ValueError("pinned 159 audit source hash mismatch")
    if sha(HELPER) != "056c8e60dad54f92a9c3ab69498dbfac0dc2c0152be272eadea2a58a3b6ea0e0":
        raise ValueError("458 fixed-bone row helper changed after review")
    emitter = load("fixed_boundary_emitter_314_for_458", EMITTER314)
    points, faces, candidate_digest, candidate_path = emitter.load_candidate(base_path)
    expected_candidate = os.environ.get("NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458",
                                        EXPECTED_CANDIDATE_SHA)
    if candidate_digest != expected_candidate:
        raise ValueError("current fit candidate does not match the pinned 456/457 input")
    report_digest = sha(audit_path)
    expected_report = os.environ.get("NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458",
                                     EXPECTED_AUDIT_REPORT_SHA)
    if report_digest != expected_report:
        raise ValueError("current 159 report SHA differs from the matched 456/457 input pin")
    report = json.loads(Path(audit_path).read_text())
    bank_helper = load("human_skin_fat_bone_audit_for_458", emitter.BANK_HELPER)
    (triangles_local, ids, _gids, _bodies, matrices, offsets, positions,
     bankkeys, bank_pins, _unused) = bank_helper.read_bank_inputs(
        emitter.BANK_DIR, emitter.BANK_EVIDENCE)
    expected_keys = tuple((int(r[0]), int(r[1])) for r in np.asarray(bankkeys))
    typed_pairs = emitter.validate_audit_report(
        report,
        candidate_sha256=candidate_digest,
        point_count=len(points),
        face_count=len(faces),
        expected_bank_keys=expected_keys,
        expected_script_sha256=AUDIT159_SHA,
        expected_predicate_sha256=emitter.sha256_file(emitter.EXACT_159),
    )
    implicated_keys = {(int(body), int(bone)) for body, bone, _face, _tri in typed_pairs}
    winding = certify_implicated_bank_winding(triangles_local, ids, implicated_keys)
    triangle_lookup = emitter.typed_triangle_lookup(triangles_local, ids)
    transforms = {
        int(body): (matrices[int(body)], offsets[int(body)], positions[int(body)])
        for body in matrices
    }
    fixed = emitter.make_barrier_arrays(typed_pairs, triangle_lookup, transforms)
    expected_order = sorted(typed_pairs)
    actual_order = [
        (int(body), int(bone), int(face), int(triangle))
        for face, body, bone, triangle in zip(
            fixed["skin_face_rows"], fixed["bank_body"],
            fixed["stable_bone_id"], fixed["bank_triangle_stable_id"])
    ]
    if actual_order != expected_order:
        raise ValueError("314 exact-world triangle order differs from the complete 159 typed pair set")
    if len(actual_order) != len(typed_pairs) or not np.isfinite(fixed["fixed_triangles_world_mm"]).all():
        raise ValueError("314 did not materialize all current fixed triangles in world millimetres")
    pairs = [
        {"face_row": int(face), "body": int(body), "bone_id": int(bone),
         "triangle_id": int(triangle)}
        for face, body, bone, triangle in zip(
            fixed["skin_face_rows"], fixed["bank_body"],
            fixed["stable_bone_id"], fixed["bank_triangle_stable_id"])
    ]
    bone_triangles = {
        (int(body), int(bone), int(triangle)): tri.copy()
        for body, bone, triangle, tri in zip(
            fixed["bank_body"], fixed["stable_bone_id"],
            fixed["bank_triangle_stable_id"], fixed["fixed_triangles_world_mm"])
    }
    pair_identity_array = np.asarray(expected_order, dtype=np.int64).reshape(-1, 4)
    receipt = {
        "schema": "numi.fixed_bone_outward_plane_458.inputs.v1",
        "classification": "matched current 159 crossing identities and fixed exact 039 bone triangles; construction only",
        "candidate_path": str(candidate_path), "candidate_sha256": candidate_digest,
        "audit159_path": str(Path(audit_path)), "audit159_sha256": report_digest,
        "audit159_script_sha256": AUDIT159_SHA,
        "exact_predicate_path": str(emitter.EXACT_159),
        "exact_predicate_sha256": emitter.sha256_file(emitter.EXACT_159),
        "fixed_triangle_emitter_path": str(EMITTER314),
        "fixed_triangle_emitter_sha256": EMITTER314_SHA,
        "bank_reader_path": str(emitter.BANK_HELPER),
        "bank_reader_sha256": emitter.sha256_file(emitter.BANK_HELPER),
        "bank_export_paths": {name: str(emitter.BANK_DIR / name) for name in bank_pins},
        "bank_export_pins": bank_pins,
        "bank_inventory_count": int(len(expected_keys)),
        "complete_159_pair_count": int(len(typed_pairs)),
        "implicated_bank_count": int(len(implicated_keys)),
        "pair_identity_array_sha256": array_sha(pair_identity_array),
        "fixed_triangles_world_mm_array_sha256": array_sha(fixed["fixed_triangles_world_mm"]),
        "fixed_triangles_frame": "world",
        "fixed_triangles_units": "mm",
        "fixed_triangle_transform": "pinned 314 inverse body affine; actual 039 Float32 triangle values promoted to Float64 before world transform",
        "bank_outward_winding_certificate": {
            "policy": BANK_WINDING_POLICY,
            "verified_implicated_banks": len(winding),
            "all_passed": True,
            "banks": winding,
        },
        "construction_only": True,
        "native_admission": False,
    }
    return {
        "points_mm": points.astype(np.float64) * 1000.0,
        "faces": faces.astype(np.int64),
        "pairs": pairs,
        "bone_triangles_mm": bone_triangles,
        "pair_identity_array": pair_identity_array,
        "fixed_triangles_world_mm": fixed["fixed_triangles_world_mm"],
        "receipt": receipt,
    }


def _pair_set_from_face_targets(face_targets):
    rows = np.asarray(face_targets["face_rows"], dtype=np.int64)
    archive = face_targets.get("face_npz", {})
    def arr(name):
        out = np.asarray(archive.get(name))
        if out.shape != (len(rows),):
            raise ValueError(f"matched 335 archive identity array {name} is malformed")
        return out.astype(np.int64, copy=False)
    bodies, bones, triangles = arr("bank_body"), arr("bank_stable_bone_id"), arr("bank_triangle_stable_id")
    return {(int(face), int(body), int(bone), int(tri))
            for face, body, bone, tri in zip(rows, bodies, bones, triangles)}


def install_objective(owner, objective, crossing):
    original_builder = owner.build_clearance_rows
    original_combine = owner.combine_barrier_rows
    state = {"hard_rows": [], "receipt": None, "objective": None}

    def replacement(points_mm, faces, support, anchors, point_targets, face_targets, bary_owner):
        # Execute and validate the exact pinned 408 builder first; retain its
        # point row prefix byte-for-byte, including weights and metadata.
        old_C, old_h, old_w, old_meta = original_builder(
            points_mm, faces, support, anchors, point_targets, face_targets, bary_owner)
        point_count = len(np.asarray(point_targets["point_rows"], dtype=np.int64))
        if (old_C.shape[0] != point_count + len(np.asarray(face_targets["face_rows"]))
                or len(old_h) != old_C.shape[0] or len(old_w) != old_C.shape[0]
                or len(old_meta) != old_C.shape[0]):
            raise ValueError("frozen 408 builder returned unexpected row counts")
        expected = {(int(face), int(body), int(bone), int(triangle))
                    for body, bone, face, triangle in crossing["pair_identity_array"]}
        prepared = _pair_set_from_face_targets(face_targets)
        if prepared != expected:
            raise ValueError(
                f"matched 335 face identities do not cover current complete 159 pair set: "
                f"missing={len(expected - prepared)}, extra={len(prepared - expected)}"
            )
        result = objective.build_fixed_bone_outward_plane_rows(
            points_mm, faces, support, anchors, crossing["pairs"],
            crossing["bone_triangles_mm"],
            total_weight=(0.5 if point_count else 1.0),
        )
        diagnostics = result["diagnostics"]
        if diagnostics["input_pair_count"] != len(expected) or diagnostics["lower_bound_row_count"] != 3 * len(expected):
            raise ValueError("458 did not produce exactly three lower-bound rows for every current 159 pair")
        if diagnostics["anchored_positive_lower_count"]:
            raise ValueError(
                f"current exact bone-plane targets include {diagnostics['anchored_positive_lower_count']} "
                "positive anchored corner lower bounds"
            )
        if diagnostics["soft_row_count"] <= 0 or diagnostics["soft_pair_count"] <= 0:
            raise ValueError("458 produced no movable positive corner targets")
        if point_count:
            point_w = np.asarray(old_w[:point_count], dtype=np.float64)
            if not np.isclose(float(point_w.sum()), 0.5, rtol=0.0, atol=1e-10):
                raise ValueError("408 point weights do not retain the intended 0.5 objective mass")
        point_C = sparse.csr_matrix(old_C[:point_count], dtype=np.float64)
        point_h = np.asarray(old_h[:point_count], dtype=np.float64).copy()
        point_w = np.asarray(old_w[:point_count], dtype=np.float64).copy()
        point_meta = [dict(item) for item in old_meta[:point_count]]

        anchors_set = set(map(int, np.asarray(anchors, dtype=np.int64)))
        hard_rows = []
        for record in result["guard_metadata"]:
            lower = float(record["lower_mm"])
            node = int(record["node_row"])
            # Positive deficits belong to the normalized LP slack objective.
            # Existing nonpositive corners stay outside the plane under every
            # feasible hard row, preventing the fit from moving them back in.
            if lower > 0.0 or node in anchors_set:
                continue
            hard_rows.append({
                "nodes": np.asarray([node], dtype=np.int64),
                "coefficients": np.asarray([record["normal_world_unit"]], dtype=np.float64),
                "lower": lower,
                "kind": "fixed_bone_outward_plane_nonregression_458",
                "typed_ids": np.asarray([
                    int(record["face_row"]), int(record["body"]), int(record["bone_id"]),
                    int(record["triangle_id"]), int(record["corner_index"]),
                ], dtype=np.int64),
            })
        if len(hard_rows) <= 0:
            raise ValueError("458 emitted no nonpositive-corner hard non-regression rows")
        state["hard_rows"] = hard_rows

        face_C = sparse.csr_matrix(result["C"], dtype=np.float64)
        face_h = np.asarray(result["h"], dtype=np.float64).copy()
        face_w = np.asarray(result["weights"], dtype=np.float64).copy()
        face_meta = []
        for item in result["metadata"]:
            row = dict(item)
            row["kind"] = "face"
            row["row_type"] = "fixed_bone_outward_plane_positive_lp_target_458"
            row["face_row"] = int(row["face_row"])
            row["bank_body"] = int(row["body"])
            row["bank_stable_bone_id"] = int(row["bone_id"])
            row["source_index"] = int(row["pair_index"])
            face_meta.append(row)
        if point_count and not np.isclose(float(face_w.sum()), 0.5, rtol=0.0, atol=1e-10):
            raise ValueError("458 face targets do not retain the intended 0.5 objective mass")
        C = sparse.vstack((point_C, face_C), format="csr")
        h = np.concatenate((point_h, face_h))
        weights = np.concatenate((point_w, face_w))
        state["objective"] = {
            "point_row_count": int(point_count),
            "point_weight_mass": float(point_w.sum()),
            "positive_bone_plane_target_rows": int(len(face_h)),
            "bone_plane_target_pair_count": int(diagnostics["soft_pair_count"]),
            "bone_plane_target_weight_mass": float(face_w.sum()),
            "nonpositive_corner_hard_rows": int(len(hard_rows)),
            "all_current_crossing_pairs_loaded": int(diagnostics["input_pair_count"]),
            "three_corner_lower_bounds_per_pair": int(diagnostics["lower_bound_row_count"]),
            "anchored_positive_lower_count": int(diagnostics["anchored_positive_lower_count"]),
            "skipped_or_zero_target_pairs": int(diagnostics["input_pair_count"] - diagnostics["soft_pair_count"]),
            "objective_semantics": "408 normalized LP slack on positive fixed-bone plane deficits; hard nonregression for baseline-clear corners",
            "clearance_claim": False,
            "native_admission": False,
        }
        return C, h, weights, point_meta + face_meta

    def combine(self, fixed_boundary_rows=()):
        rows = original_combine(self, fixed_boundary_rows)
        rows.extend(state["hard_rows"])
        return rows

    owner.build_clearance_rows = replacement
    owner.combine_barrier_rows = combine
    return state


def main():
    # Keep the original 408 CLI exactly; all variant inputs are env-pinned.
    base_path = arg_value("--base-candidate")
    audit_path_env = os.environ.get("NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458")
    audit_path = Path(audit_path_env) if audit_path_env else base_path.parent / "audit-159/report.json"
    face_target_path = arg_value("--face-target-npz")
    point_target_path = arg_value("--point-target-npz")
    if sha(face_target_path) != EXPECTED_FACE_TARGET_SHA:
        raise ValueError("matched 457 335 face-target NPZ SHA mismatch")
    if sha(point_target_path) != EXPECTED_POINT_TARGET_SHA:
        raise ValueError("matched 457 390 point-target NPZ SHA mismatch")
    crossing = prepare_crossing_inputs(base_path, audit_path)
    objective = load("fixed_bone_outward_plane_objective_458", HELPER)
    owner = load("fitter408_for_outward_plane_458", OWNER)
    state = install_objective(owner, objective, crossing)

    out_path = arg_value("--output")
    receipt = dict(crossing["receipt"])
    receipt["fixed_plane_target_helper_path"] = str(HELPER)
    receipt["fixed_plane_target_helper_sha256"] = sha(HELPER)
    receipt["matched_335_face_target_path"] = str(face_target_path)
    receipt["matched_335_face_target_sha256"] = sha(face_target_path)
    receipt["matched_390_point_target_path"] = str(point_target_path)
    receipt["matched_390_point_target_sha256"] = sha(point_target_path)
    receipt["argv"] = sys.argv[1:]
    sidecar = out_path.with_name(out_path.name + ".sat-crossing-inputs-458.npz")
    sidecar_json = out_path.with_name(out_path.name + ".sat-crossing-inputs-458.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.exists() or sidecar_json.exists():
        raise FileExistsError("refusing to overwrite SAT crossing input receipt")
    np.savez_compressed(
        sidecar,
        pair_identity_body_bone_face_triangle=crossing["pair_identity_array"],
        skin_face_rows=np.asarray([p["face_row"] for p in crossing["pairs"]], dtype=np.int64),
        bank_body=np.asarray([p["body"] for p in crossing["pairs"]], dtype=np.int64),
        stable_bone_id=np.asarray([p["bone_id"] for p in crossing["pairs"]], dtype=np.int64),
        bank_triangle_stable_id=np.asarray([p["triangle_id"] for p in crossing["pairs"]], dtype=np.int64),
        fixed_triangles_world_mm=crossing["fixed_triangles_world_mm"],
    )
    receipt["sidecar_npz_path"] = str(sidecar)
    receipt["sidecar_npz_sha256"] = sha(sidecar)
    sidecar_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["sidecar_json_path"] = str(sidecar_json)
    receipt["sidecar_json_sha256"] = sha(sidecar_json)

    pins = {
        str(path): sha(path)
        for path in (Path(__file__), OWNER, HELPER, EMITTER314,
                     D / "audit_skin_bone_surfaces_159.py",
                     D / "publication-084/canonical/matter/tools/human_skin_fat_bone_audit.py",
                     face_target_path, point_target_path, base_path, audit_path, sidecar, sidecar_json)
    }
    sys.path.insert(0, str(OWNER.parent))
    owner.main()
    report_path = out_path / "report.json"
    report = json.loads(report_path.read_text())
    report["fixed_bone_outward_plane_objective_458"] = {
        "name": "current_complete_159 fixed bone outward plane corner targets with zero-regression bounds",
        "inputs_receipt_path": str(sidecar_json),
        "inputs_receipt_sha256": sha(sidecar_json),
        "input_pair_npz_path": str(sidecar),
        "input_pair_npz_sha256": sha(sidecar),
        "row_counts": state["objective"],
        "bank_outward_winding_certificate": receipt["bank_outward_winding_certificate"],
        "positive_corner_targets_are_LP_slack_targets": True,
        "negative_or_zero_corner_rows_are_hard_nonregression_rows": True,
        "each_met_pair_plane_separates_that_complete_triangle_pair": True,
        "global_surface_clearance_certified": False,
        "native_admission": False,
        "pins": pins,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if any(sha(Path(path)) != digest for path, digest in pins.items()):
        raise RuntimeError("source or matched input changed during 458 run")


if __name__ == "__main__":
    main()

