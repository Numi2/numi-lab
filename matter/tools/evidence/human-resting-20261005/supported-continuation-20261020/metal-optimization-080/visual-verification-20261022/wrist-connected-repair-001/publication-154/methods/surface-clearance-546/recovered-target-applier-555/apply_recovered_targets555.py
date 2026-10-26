from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

# Bound to the one reviewed 555 postprocess recovery. Its report deliberately
# retains the original incomplete status.
EXPECTED_RECOVERY_SHA256 = "06ae6ce653c297b5c0f9a61fa9e50fd43a30f9df7dff36a3f1b4a0354f75a335"
EXPECTED_RECOVERY_SCRIPT_SHA256 = "ccb95c57b9a6eee139752f7c8e11ad7c829d6b9ea127a1bed49cfced1fe7cb77"
EXPECTED_HELPER_SHA256 = "431f4af0b1308a1c8d90f22cf9be65e5dd13b97491ba6ddfd3b4c269bf56e5e5"
EXPECTED_SOURCE303_SHA256 = "c8fe5b7be2efb2c6867a2cde178ebc35280814c37f90eec5aea47e964fc17a1f"
EXPECTED_ORIENTATION_SHA256 = "e43908188f134182635ac36a9a1602abad1918df374a04f3f7cc9eb1812d7497"
CORE_ARRAYS = (
    "query_rows",
    "target_world_f32_m",
    "proposal_direction_world_unit",
    "depth_covector_world_unit",
    "direction_compatibility_mask",
    "pass_mask",
    "target_distance_m",
    "target_exactly_outside_all185",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_recovered_targets(target_dir, expected_recovery_sha256=EXPECTED_RECOVERY_SHA256):
    """Validate the pinned postprocess recovery without rewriting its status."""
    target_dir = Path(target_dir)
    report_path = target_dir / "nearest-union-exits.json"
    npz_path = target_dir / "nearest-union-exits.npz"
    receipt_path = target_dir / "postprocess-recovery-555.json"
    original_dir = target_dir / "preserved-unannotated-original"
    original_report = original_dir / "nearest-union-exits.original.json"
    original_npz = original_dir / "nearest-union-exits.original.npz"

    require(receipt_path.is_file(), "missing postprocess recovery receipt")
    require(sha(receipt_path) == expected_recovery_sha256 == EXPECTED_RECOVERY_SHA256,
            "recovery receipt pin mismatch")
    receipt = json.loads(receipt_path.read_text())
    report = json.loads(report_path.read_text())
    require(receipt.get("schema") == "numi.human.current_parent_normal_target_postprocess_recovery_555.v1",
            "unexpected recovery receipt schema")
    require(receipt.get("status") == "PASS_POSTPROCESS_ONLY_RECOVERY",
            "recovery receipt did not pass")
    require(receipt.get("native_admission") is False and receipt.get("accepted_physical_steps") == 0,
            "recovery receipt claims admission or physical steps")
    require(receipt.get("classification") ==
            "reannotation of completed exact endpoint outputs; no new endpoint query or geometry operation",
            "recovery receipt classification mismatch")
    require(receipt.get("recovered_report_sha256") == sha(report_path),
            "recovered report hash does not match receipt")
    require(receipt.get("recovered_npz_sha256") == sha(npz_path),
            "recovered NPZ hash does not match receipt")
    require(receipt.get("source_report_sha256") == sha(original_report) ==
            receipt.get("preserved_original_outputs", {}).get("report_sha256"),
            "preserved original report hash mismatch")
    require(receipt.get("source_npz_sha256") == sha(original_npz) ==
            receipt.get("preserved_original_outputs", {}).get("npz_sha256"),
            "preserved original NPZ hash mismatch")
    require(receipt.get("recovery_script_sha256") == EXPECTED_RECOVERY_SCRIPT_SHA256,
            "recovery script pin mismatch")
    source_pins = receipt.get("source_owner_hashes", {})
    require(source_pins.get("helper555") == EXPECTED_HELPER_SHA256 and
            source_pins.get("source303") == EXPECTED_SOURCE303_SHA256 and
            source_pins.get("orientation_report") == EXPECTED_ORIENTATION_SHA256,
            "555 source owner pin mismatch")

    require(report.get("status") == "INCOMPLETE_CURRENT_PARENT_NORMAL_OUTPUT_ANNOTATION",
            "original incomplete status must remain explicit")
    require(report.get("output_annotation_status") == "PASS_POSTPROCESS_ONLY_RECOVERY",
            "report lacks successful postprocess annotation")
    require(report.get("output_annotation_error") ==
            "ValueError('current-parent direction has no pinned orientation sign')",
            "unexpected original annotation failure")
    recovery = report.get("postprocess_recovery", {})
    require(recovery.get("geometry_or_endpoint_recomputed") is False and
            recovery.get("original_endpoint_rows_and_all185_statuses_preserved") is True and
            recovery.get("proposal_directions_verified_exactly_against_immediate_parent_normal_recomputation") is True,
            "report recovery facts do not match")
    require(report.get("registered_source_normals_are_reference_only") is True,
            "source reference and proposal direction are not distinguished")
    require(report.get("inputs", {}).get("source303_sha256") == EXPECTED_SOURCE303_SHA256 and
            report.get("inputs", {}).get("source_orientation_report_sha256") == EXPECTED_ORIENTATION_SHA256,
            "report source orientation pins mismatch")
    require(report.get("outputs", {}).get("npz_sha256") == sha(npz_path),
            "report NPZ pin mismatch")
    require(report.get("outputs", {}).get("current_parent_normal_postprocessor_sha256") ==
            EXPECTED_RECOVERY_SCRIPT_SHA256,
            "report recovery tool pin mismatch")

    original = json.loads(original_report.read_text())
    require(original.get("status") == "INCOMPLETE_CURRENT_PARENT_NORMAL_OUTPUT_ANNOTATION" and
            original.get("output_annotation_status") == "FAILED",
            "preserved report is not the failed original")
    require(original.get("outputs", {}).get("npz_sha256") == sha(original_npz),
            "preserved original output pin mismatch")
    preserved = receipt.get("preserved_original_outputs", {})
    require(preserved.get("reported_original_report_sha256") == sha(original_report) and
            preserved.get("reported_original_npz_sha256") == sha(original_npz),
            "preserved original report does not bind preserved files")

    old_targets = original.get("targets", [])
    targets = report.get("targets", [])
    count = receipt.get("target_count")
    require(count == 1248 and len(old_targets) == len(targets) == count,
            "endpoint target count mismatch")
    require(report.get("query_rows") == count and report.get("passing_targets") == count and
            report.get("all_bank_ray_results") == count and report.get("inside_node_bank_pairs") == count,
            "report counts are incomplete")
    for old, new in zip(old_targets, targets):
        require(old.get("query_row") == new.get("query_row") and
                old.get("target_world_f32_m") == new.get("target_world_f32_m") and
                old.get("status") == "PASS_FIRST_UNION_EXIT" == new.get("status") and
                old.get("exact_endpoint_bank_count") == 185 == new.get("exact_endpoint_bank_count") and
                old.get("target_exactly_outside_all185") is True and
                new.get("target_exactly_outside_all185") is True and
                old.get("end_outside_all185_by_event_parity") is True and
                new.get("end_outside_all185_by_event_parity") is True,
                "an original endpoint or its all-185 status changed")

    with np.load(original_npz, allow_pickle=False) as z:
        old_arrays = {k: z[k] for k in z.files}
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    require(all(k in old_arrays and k in arrays for k in CORE_ARRAYS), "endpoint arrays missing")
    for key in CORE_ARRAYS:
        require(old_arrays[key].dtype == arrays[key].dtype and
                old_arrays[key].shape == arrays[key].shape and
                old_arrays[key].tobytes() == arrays[key].tobytes(),
                f"endpoint array changed during recovery: {key}")
    require(np.all(arrays["pass_mask"]) and np.all(arrays["target_exactly_outside_all185"]),
            "one or more endpoint rows are not certified")
    require("current_parent_normal_world_unit" in arrays and
            np.all(arrays["current_parent_normal_available_mask"]) and
            np.array_equal(arrays["proposal_direction_world_unit"],
                           arrays["current_parent_normal_world_unit"]),
            "proposal directions do not match recovered current-parent normals")
    require(np.all(np.isfinite(arrays["target_world_f32_m"])),
            "nonfinite recovered target coordinate")
    return {
        "report": report,
        "receipt": receipt,
        "npz": arrays,
        "report_path": report_path,
        "npz_path": npz_path,
        "receipt_sha256": sha(receipt_path),
    }


def apply(target_dir, baseline_state, output_dir):
    target_dir = Path(target_dir)
    baseline_state = Path(baseline_state)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cert = verify_recovered_targets(target_dir)
    state = json.loads(baseline_state.read_text())
    base = Path(state["candidate"])
    require(sha(base) == state["candidate_sha256"] and
            sha(state["point_summary"]) == state["point_summary_sha256"],
            "baseline pins mismatch")
    report = cert["report"]
    require(report.get("inputs", {}).get("candidate_sha256") == state["candidate_sha256"] and
            report.get("inputs", {}).get("point_summary_sha256") == state["point_summary_sha256"],
            "recovered target inputs do not match baseline state")
    with np.load(base, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    target = cert["npz"]
    rows = target["query_rows"]
    point_summary = json.loads(Path(state["point_summary"]).read_text())
    expected = sorted({int(w["output_node"]) for w in point_summary["inside_witnesses"]})
    require(rows.dtype.kind in "iu" and sorted(rows.tolist()) == expected and
            len(rows) == len(set(rows.tolist())),
            "inside target identity mismatch")
    old_count = json.loads(base.with_name("report.json").read_text())["old_node_count"]
    require(np.all(rows >= old_count), "old node targeted")
    coords = payload["points_world_m"]
    result = coords.copy()
    if target["target_world_f32_m"].dtype != np.float32 or target["target_world_f32_m"].shape != (len(rows), 3):
        raise ValueError("target coordinate domain")
    result[rows] = target["target_world_f32_m"]
    keep = np.setdiff1d(np.arange(len(coords)), rows)
    require(result[keep].tobytes() == coords[keep].tobytes() and
            result[:old_count].tobytes() == coords[:old_count].tobytes(),
            "non-target point changed")
    anchors = payload["typed_coarse_anchor_output_rows_018"]
    require(result[anchors].tobytes() == coords[anchors].tobytes(), "anchor changed")
    payload["points_world_m"] = result
    output_dir.mkdir(parents=True)
    candidate = output_dir / "candidate.npz"
    np.savez_compressed(candidate, **payload)
    source_paths = [Path(__file__), baseline_state, base, Path(state["point_summary"]),
                    cert["report_path"], cert["npz_path"],
                    target_dir / "postprocess-recovery-555.json"]
    result_report = dict(
        status="CONSTRUCTION_ENDPOINT_PROPOSAL_WRITTEN_UNADMITTED",
        candidate=str(candidate), candidate_sha256=sha(candidate),
        baseline=str(base), baseline_sha256=sha(base),
        target_rows=rows.tolist(), target_count=len(rows),
        target_recovery_receipt_sha256=cert["receipt_sha256"],
        target_report_status_preserved=report["status"],
        target_acceptance_basis="verified recovery of completed PASS_FIRST_UNION_EXIT rows; original report status remains incomplete",
        old_point_prefix_bitwise_unchanged=True,
        all_nonpoint_payload_bitwise_unchanged=True,
        maximum_new_node_step_mm=float(np.linalg.norm(
            result[rows].astype(float) - coords[rows].astype(float), axis=1).max() * 1000),
        direction_mode=report["direction_mode"],
        native_admission=False, accepted_physical_steps=0,
        full_geometry_checks_required=True,
        source_pins={str(p): sha(p) for p in source_paths})
    (output_dir / "report.json").write_text(json.dumps(result_report, indent=2, sort_keys=True) + "\n")
    return result_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-state", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-recovery-sha256", default=EXPECTED_RECOVERY_SHA256)
    args = parser.parse_args()
    if args.expected_recovery_sha256 != EXPECTED_RECOVERY_SHA256:
        raise ValueError("only the reviewed recovery receipt is enabled")
    verify_recovered_targets(args.target_dir, args.expected_recovery_sha256)
    result = apply(args.target_dir, args.baseline_state, args.output_dir)
    print(json.dumps({k: result[k] for k in (
        "status", "target_count", "maximum_new_node_step_mm",
        "target_recovery_receipt_sha256", "target_report_status_preserved")}, sort_keys=True))


if __name__ == "__main__":
    main()
