"""Replay saved old-topology coordinates over a checked topology-only candidate.

This is a coordinate/payload merge only. It transfers no geometry certificate or
physical admission; the output requires fresh exact checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FACE_DEPENDENT_580_FIELDS = frozenset({
    # Surface connectivity and fixed 037 face connectivity.
    "source_O_faces",
    "area_reference_037_source_O_faces",
    # Face-to-source and fine-detail/subface CSR ancestry.
    "face_source_ancestry_indptr",
    "face_source_ancestry_face_rows",
    "face_fine_detail_ancestry_indptr",
    "face_fine_detail_ancestry_rows",
    "face_fine_subface_ancestry_indptr",
    "face_fine_subface_ancestry_rows",
    # Immediate parent-face ancestry.
    "refinement_face_parent_ancestry_indptr",
    "refinement_face_parent_ancestry_face_rows",
    "refinement_face_parent_face_row",
    "refinement_face_parent_barycentric",
    # Fine-detail/subface and source parent markers changed by this retriangulation.
    "parent_fine_detail_face_row",
    "parent_fine_subface_row",
    "parent_source_face_marker",
})

EXPECTED_580_CHANGED_FIELDS = FACE_DEPENDENT_580_FIELDS


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return (
        isinstance(a, np.ndarray)
        and isinstance(b, np.ndarray)
        and a.dtype == b.dtype
        and a.shape == b.shape
        and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    )


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_state_documents(
    state: dict[str, Any],
    candidate_sha: str,
    point_summary: dict[str, Any],
    surface_report: dict[str, Any],
    final_report: dict[str, Any],
) -> dict[str, int]:
    """Validate candidate-bound receipts without re-running any geometry audit."""
    _need(state.get("candidate_sha256") == candidate_sha, "state candidate hash mismatch")
    _need(point_summary.get("candidate_sha256") == candidate_sha, "point summary candidate hash mismatch")
    _need(surface_report.get("candidate_sha256") == candidate_sha, "surface report candidate hash mismatch")
    _need(final_report.get("candidate_sha256") == candidate_sha, "final report candidate hash mismatch")

    _need(point_summary.get("status") == "OUTER_NODE_CLEARANCE_DIAGNOSTIC", "point summary status mismatch")
    _need(point_summary.get("complete_surface_audit") is True, "point summary lacks complete surface audit")
    _need(point_summary.get("inside_node_bank_pairs") == 0, "candidate has contained point-bank pairs")
    _need(point_summary.get("inside_nodes") == 0, "candidate has contained nodes")
    _need(point_summary.get("accepted_physical_steps") == 0, "point summary claims accepted physical steps")
    _need(point_summary.get("geometry_admitted") is False, "point summary claims geometry admission")

    _need(surface_report.get("status") == "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT", "surface report status mismatch")
    _need(surface_report.get("complete") is True, "surface report is incomplete")
    _need(surface_report.get("bank_count") == 185, "surface report bank count is not 185")
    _need(surface_report.get("expected_bank_count") == 185, "surface report expected bank count is not 185")
    _need(surface_report.get("geometry_or_native_admission") is False, "surface report claims admission")

    _need(final_report.get("status") == "COMPLETE_EXACT_CONSTRUCTION_CHECKS", "final report is not a completed construction check")
    _need(final_report.get("self_pairs") == 0, "candidate has self-intersection pairs")
    _need(final_report.get("contained_pairs") == 0, "final report has contained point-bank pairs")
    _need(final_report.get("complete_banks") == 185, "final report bank count is not 185")
    _need(final_report.get("accepted_physical_steps") == 0, "final report claims accepted physical steps")
    _need(final_report.get("native_admission") is False, "final report claims native admission")
    _need(state.get("accepted_physical_steps") == 0, "state claims accepted physical steps")
    _need(state.get("native_admission") is False, "state claims native admission")

    if "point_summary" in final_report:
        _need(Path(final_report["point_summary"]).resolve() == Path(state["point_summary"]).resolve(),
             "final report points at a different point summary")
    if "surface_report" in final_report:
        _need(Path(final_report["surface_report"]).resolve() == Path(state["surface_report"]).resolve(),
             "final report points at a different surface report")

    return {
        "surface_pairs": int(final_report["surface_pairs"]),
        "inside_pairs": int(final_report["contained_pairs"]),
        "self_pairs": int(final_report["self_pairs"]),
        "complete_banks": int(final_report["complete_banks"]),
    }


def read_state(state_path: Path) -> dict[str, Any]:
    state_path = Path(state_path).resolve(strict=True)
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    for key in ("candidate", "candidate_sha256", "point_summary", "point_summary_sha256",
                "surface_report", "surface_report_sha256"):
        _need(key in state, f"state missing {key}")

    candidate_path = Path(state["candidate"]).resolve(strict=True)
    candidate_sha = sha256(candidate_path)
    _need(candidate_sha == state["candidate_sha256"], "candidate hash does not match state")

    point_path = Path(state["point_summary"]).resolve(strict=True)
    surface_path = Path(state["surface_report"]).resolve(strict=True)
    _need(sha256(point_path) == state["point_summary_sha256"], "point summary hash does not match state")
    _need(sha256(surface_path) == state["surface_report_sha256"], "surface report hash does not match state")

    point_summary = json.loads(point_path.read_text())
    surface_report = json.loads(surface_path.read_text())
    final_path = state_path.parent / "report.json"
    _need(final_path.is_file(), "final construction report missing beside state")
    final_report = json.loads(final_path.read_text())
    counts = validate_state_documents(state, candidate_sha, point_summary, surface_report, final_report)

    return {
        "state": state,
        "state_path": str(state_path),
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "point_summary_path": str(point_path),
        "point_summary_sha256": sha256(point_path),
        "surface_report_path": str(surface_path),
        "surface_report_sha256": sha256(surface_path),
        "final_report_path": str(final_path.resolve()),
        "final_report_sha256": sha256(final_path),
        "counts": counts,
    }


def load_payload(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(bundle["candidate_path"], allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def merge(
    current_topology: dict[str, np.ndarray],
    saved_baseline: dict[str, np.ndarray],
    coordinate_proposal: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Copy current topology payload and replace only its Float32 point coordinates."""
    if set(current_topology) != set(saved_baseline) or set(current_topology) != set(coordinate_proposal):
        raise ValueError("payload inventory mismatch")
    points_key = "points_world_m"
    for label, payload in (("current topology", current_topology),
                           ("saved baseline", saved_baseline),
                           ("coordinate proposal", coordinate_proposal)):
        if points_key not in payload or payload[points_key].dtype != np.float32 or payload[points_key].ndim != 2 or payload[points_key].shape[1] != 3:
            raise ValueError(f"{label} Float32 point field invalid")

    if not same(current_topology[points_key], saved_baseline[points_key]):
        raise ValueError("current topology coordinates differ from saved baseline coordinates")
    if current_topology[points_key].shape != coordinate_proposal[points_key].shape:
        raise ValueError("coordinate proposal point domain mismatch")

    changed_face_fields = []
    for key in current_topology:
        if key == points_key:
            continue
        if not same(saved_baseline[key], coordinate_proposal[key]):
            raise ValueError("saved coordinate proposal changed nonpoint field: " + key)
        if not same(current_topology[key], saved_baseline[key]):
            if key not in FACE_DEPENDENT_580_FIELDS:
                raise ValueError("current topology changed non-face field: " + key)
            changed_face_fields.append(key)

    if "source_O_faces" not in changed_face_fields:
        raise ValueError("current topology does not contain the expected retriangulation")

    anchors_key = "typed_coarse_anchor_output_rows_018"
    if anchors_key not in current_topology:
        raise ValueError("typed anchor rows missing")
    anchors = current_topology[anchors_key]
    if anchors.ndim != 1 or anchors.dtype.kind not in "iu" or len(anchors) != 237:
        raise ValueError("typed anchor rows are invalid")
    if len(np.unique(anchors)) != len(anchors) or np.any(anchors < 0) or np.any(anchors >= len(current_topology[points_key])):
        raise ValueError("typed anchor rows are duplicate or out of range")
    if not same(anchors, saved_baseline[anchors_key]) or not same(anchors, coordinate_proposal[anchors_key]):
        raise ValueError("typed anchor identity changed")
    if not same(coordinate_proposal[points_key][anchors], saved_baseline[points_key][anchors]):
        raise ValueError("coordinate proposal moved typed anchors")

    merged = {key: np.array(value, copy=True) for key, value in current_topology.items()}
    merged[points_key] = coordinate_proposal[points_key].copy()

    if set(merged) != set(current_topology):
        raise AssertionError("output inventory changed")
    for key in current_topology:
        if key != points_key and not same(merged[key], current_topology[key]):
            raise AssertionError("output changed current topology payload: " + key)
    return merged, sorted(changed_face_fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-topology-state", type=Path, required=True)
    parser.add_argument("--saved-baseline-state", type=Path, required=True)
    parser.add_argument("--coordinate-proposal-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bundles = [
        read_state(args.current_topology_state),
        read_state(args.saved_baseline_state),
        read_state(args.coordinate_proposal_state),
    ]
    payloads = [load_payload(bundle) for bundle in bundles]
    result, changed_face_fields = merge(*payloads)
    _need(not args.out.exists(), "output directory already exists")

    helper_path = Path(__file__).resolve()
    pins: dict[str, str] = {str(helper_path): sha256(helper_path)}
    for bundle in bundles:
        for key in ("state_path", "candidate_path", "point_summary_path", "surface_report_path", "final_report_path"):
            p = Path(bundle[key])
            pins[str(p)] = sha256(p)

    # Detect changes after loading/validation before creating output.
    for path, expected in pins.items():
        if sha256(Path(path)) != expected:
            raise ValueError("input changed during merge: " + path)

    args.out.mkdir(parents=True, exist_ok=False)
    target = args.out / "candidate.npz"
    np.savez_compressed(target, **result)
    report = {
        "status": "SAVED_COORDINATE_FIELD_REPLAY_PENDING_FRESH_GEOMETRY_CHECKS",
        "source_pins": pins,
        "current_topology_state": bundles[0]["state_path"],
        "current_topology_candidate_sha256": bundles[0]["candidate_sha256"],
        "current_topology_surface_pairs": bundles[0]["counts"]["surface_pairs"],
        "saved_baseline_state": bundles[1]["state_path"],
        "saved_baseline_candidate_sha256": bundles[1]["candidate_sha256"],
        "saved_baseline_surface_pairs": bundles[1]["counts"]["surface_pairs"],
        "coordinate_proposal_state": bundles[2]["state_path"],
        "coordinate_proposal_candidate_sha256": bundles[2]["candidate_sha256"],
        "coordinate_proposal_surface_pairs": bundles[2]["counts"]["surface_pairs"],
        "face_dependent_fields_changed_by_current_topology": changed_face_fields,
        "candidate": str(target),
        "candidate_sha256": sha256(target),
        "node_count": int(result["points_world_m"].shape[0]),
        "changed_nodes_vs_current_topology": int(np.any(
            result["points_world_m"] != payloads[0]["points_world_m"], axis=1).sum()),
        "maximum_coordinate_delta_mm": float(np.linalg.norm(
            result["points_world_m"].astype(np.float64) -
            payloads[0]["points_world_m"].astype(np.float64), axis=1).max() * 1000.0),
        "all_current_topology_nonpoint_fields_bitwise_preserved": True,
        "typed_anchor_count": 237,
        "typed_anchors_bitwise_unchanged": True,
        "constraints_reused": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
        "fresh_complete_185_bank_and_self_checks_required": True,
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
