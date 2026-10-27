from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = "numi.human.joint-refined-construction-preparation595.v1"

def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path: Path | str) -> dict:
    v = json.loads(Path(path).read_text())
    if not isinstance(v, dict):
        raise ValueError(f"expected object JSON: {path}")
    return v

def _get(arrays: dict, key: str):
    if key not in arrays:
        raise ValueError(f"candidate missing required array {key}")
    return np.asarray(arrays[key])

def validate_joint_refined_inputs(*, state: dict, point_summary: dict,
                                  surface_report: dict, arrays: dict,
                                  candidate_sha256: str, point_target_report: dict,
                                  point_targets: dict, point_target_sha256: str,
                                  point_depth_report: dict, depth_npz: dict,
                                  depth_sha256: str, prior_context: dict,
                                  prior_candidate_sha256: str,
                                  ancestry_parent_sha256: str) -> dict:
    """Strict candidate-bound validation for the explicit contained-baseline joint branch."""
    if state.get("candidate_sha256") != candidate_sha256:
        raise ValueError("state candidate hash mismatch")
    if point_summary.get("candidate_sha256") != candidate_sha256:
        raise ValueError("point summary is not candidate-bound")
    if surface_report.get("candidate_sha256") != candidate_sha256:
        raise ValueError("surface report is not candidate-bound")
    if point_summary.get("complete_surface_audit") is not True:
        raise ValueError("joint refined preparation requires complete 130 point audit")
    if int(point_summary.get("exact_PLC_surface_pairs", -1)) != 0:
        raise ValueError("joint refined preparation requires zero exact self pairs")
    if surface_report.get("complete") is not True or int(surface_report.get("bank_count", -1)) != 185 or int(surface_report.get("expected_bank_count", -1)) != 185:
        raise ValueError("joint refined preparation requires complete exact 185-bank surface report")
    inside = int(point_summary.get("inside_node_bank_pairs", -1))
    if inside <= 0:
        raise ValueError("this explicit joint adapter is only for a checked contained refined baseline")
    if state.get("continuation_mode") != "point_continuation" or state.get("temporary_unselected_construction") is not True or state.get("candidate_selected") is not False:
        raise ValueError("contained baseline must remain explicitly temporary and unselected")
    witnesses = point_summary.get("inside_witnesses")
    if not isinstance(witnesses, list) or len(witnesses) != inside:
        raise ValueError("contained witness rows do not match reported inside-node/bank count")
    rows = np.asarray(point_targets.get("query_rows"), dtype=np.int64)
    target = np.asarray(point_targets.get("target_world_f32_m"), dtype=np.float32)
    if rows.ndim != 1 or target.shape != (len(rows), 3) or len(rows) != inside:
        raise ValueError("point target rows do not cover exactly the current inside witnesses")
    witness_rows = sorted(int(w["output_node"]) for w in witnesses)
    if rows.tolist() != witness_rows:
        raise ValueError("point target IDs differ from exact contained witness output-node IDs")
    if not np.array_equal(rows, np.unique(rows)):
        raise ValueError("point target rows must be sorted and unique")
    npoints = len(np.asarray(_get(arrays, "points_world_m")))
    if np.any(rows < 0) or np.any(rows >= npoints):
        raise ValueError("point target row outside refined candidate")
    suffix = int(point_target_report.get("outputs", {}).get("parent_prefix_node_count", npoints - len(rows)))
    if np.any(rows < suffix):
        raise ValueError("joint-refined point targets must be new midpoint suffix rows")
    tr_inputs = point_target_report.get("inputs", {})
    if tr_inputs.get("candidate_sha256") != candidate_sha256 or tr_inputs.get("point_summary_sha256") != state.get("point_summary_sha256"):
        raise ValueError("point target report pins differ from current candidate/130 summary")
    if int(point_target_report.get("query_rows", -1)) != len(rows) or int(point_target_report.get("passing_targets", -1)) != len(rows):
        raise ValueError("point target report does not certify every contained node")
    for name in ("pass_mask", "target_exactly_outside_all185"):
        v = np.asarray(point_targets.get(name), dtype=bool)
        if v.shape != (len(rows),) or not np.all(v):
            raise ValueError(f"point target {name} is not true for every row")
    for name in ("target_distance_m", "proposal_direction_world_unit", "source_surface_normal_world_unit"):
        v = np.asarray(point_targets.get(name))
        if not np.isfinite(v).all():
            raise ValueError(f"point target {name} contains nonfinite values")
    depth_covector = np.asarray(point_targets.get("depth_covector_world_unit"))
    if depth_covector.shape != (len(rows), 3):
        raise ValueError("point target depth covector has the wrong shape")
    if point_target_report.get("direction_mode") == "per-row-certified-source-or-depth-selection":
        row_proofs = point_target_report.get("per_row_selected_parent_receipts")
        if not isinstance(row_proofs, list) or [int(x.get("query_row", -1)) for x in row_proofs] != rows.tolist():
            raise ValueError("mixed point targets lack ordered per-row parent provenance")
        for i, proof in enumerate(row_proofs):
            if proof.get("origin") == "depth-maximizing" and not np.isfinite(depth_covector[i]).all():
                raise ValueError(f"depth-maximizing row {rows[i]} lacks its finite depth covector")
            if proof.get("origin") not in ("depth-maximizing", "source-outward"):
                raise ValueError(f"mixed point row {rows[i]} has an unknown direction origin")
    elif not np.isfinite(depth_covector).all():
        raise ValueError("point target depth covector contains nonfinite values")
    anchors = np.asarray(_get(arrays, "typed_coarse_anchor_output_rows_018"), dtype=np.int64)
    if np.intersect1d(rows, anchors).size:
        raise ValueError("contained midpoint point targets overlap typed anchors")
    if point_target_report.get("status") not in ("PASS_FIRST_UNION_EXIT_TARGETS_ONLY", "PASS_ALL_INSIDE_TARGETS_OUTSIDE_ALL185"):
        raise ValueError("point target report status is not an accepted exact union-exit status")
    if point_depth_report.get("candidate_sha256") != candidate_sha256 or sha_depth_mismatch(point_depth_report, depth_sha256):
        raise ValueError("point-depth rows are not pinned to the refined candidate")
    if point_depth_report.get("complete_bank_accounting_verified") is not True or int(point_depth_report.get("bank_count", -1)) != 185:
        raise ValueError("point-depth rows lack complete 185-bank accounting")
    depth_rows = np.asarray(depth_npz.get("node_rows"), dtype=np.int64)
    if depth_rows.ndim != 1 or len(depth_rows) != int(point_depth_report.get("row_count", -1)):
        raise ValueError("point-depth row payload and receipt disagree")
    if int(point_depth_report.get("inside_row_count", -1)) != inside:
        raise ValueError("point-depth report inside row count differs from exact contained witnesses")
    if int(point_depth_report.get("outside_row_count", -1)) <= 0:
        raise ValueError("point-depth report has no outside rows")
    if prior_context.get("candidate_sha256") != prior_candidate_sha256 or prior_context.get("candidate_sha256") != ancestry_parent_sha256:
        raise ValueError("prior seed/guard context and ancestry parent do not share one exact parent")
    return {
        "status": "PASS_JOINT_REFINED_CONSTRUCTION_INPUTS",
        "candidate_sha256": candidate_sha256,
        "inside_node_bank_pairs": inside,
        "point_target_rows": rows.tolist(),
        "point_target_rows_sha256": hashlib.sha256(np.ascontiguousarray(rows).view(np.uint8)).hexdigest(),
        "point_targets_exactly_outside_all185": True,
        "point_depth_row_count": len(depth_rows),
        "point_depth_inside_rows": inside,
        "point_depth_outside_rows": int(point_depth_report["outside_row_count"]),
        "self_pairs": 0,
        "exact_surface_banks": 185,
        "prior_context_candidate_sha256": prior_candidate_sha256,
        "ancestry_parent_candidate_sha256": ancestry_parent_sha256,
        "temporary_unselected_construction": True,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }

def sha_depth_mismatch(report: dict, actual_npz_sha256: str) -> bool:
    expected = report.get("output_npz_sha256") or report.get("rows_npz_sha256")
    return expected != actual_npz_sha256

def replace_cli_value(argv: list[str], flag: str, value: str | Path) -> None:
    if argv.count(flag) != 1:
        raise ValueError(f"expected one {flag} in fit template")
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise ValueError(f"missing value after {flag}")
    argv[i + 1] = str(value)

def fit_argv_from_template(template: list[str], *, candidate: Path, audit159: Path,
                           point_target_npz: Path, point_target_report: Path,
                           face_target_npz: Path, face_target_report: Path,
                           point_depth_npz: Path, point_depth_report: Path,
                           self_pairs: Path, fixed_guard_npz: Path,
                           fixed_guard_report: Path, matched_inputs: Path,
                           output: Path, trust_mm: float = 4.0) -> list[str]:
    argv = list(template)
    replacements = {
        "--base-candidate": candidate, "--raw-trial": candidate,
        "--self-pairs": self_pairs, "--area-reference-npz": candidate,
        "--point-target-npz": point_target_npz, "--point-target-report": point_target_report,
        "--face-target-npz": face_target_npz, "--face-target-report": face_target_report,
        "--point-depth-rows-npz": point_depth_npz, "--point-depth-rows-report": point_depth_report,
        "--fixed-boundary-npz": fixed_guard_npz, "--fixed-boundary-report": fixed_guard_report,
        "--matched-inputs-json-490": matched_inputs, "--audit159-490": audit159,
        "--output": output,
    }
    for flag,value in replacements.items(): replace_cli_value(argv,flag,value)
    replace_cli_value(argv,"--trust-mm",str(float(trust_mm)))
    if float(argv[argv.index("--source-cap-mm")+1]) != 12.0:
        raise ValueError("source cap must remain 12 mm")
    if float(argv[argv.index("--area-floor")+1]) != 0.1:
        raise ValueError("fixed 037 area floor must remain 0.1")
    if float(argv[argv.index("--max-edge-gradient")+1]) != 0.5:
        raise ValueError("maximum edge gradient must remain 0.5")
    if "--immediate-parent-basis" not in argv:
        raise ValueError("fit must retain the matched immediate-parent basis")
    if "--allow-soft-target-regression-485" not in argv:
        raise ValueError("fit template must retain pinned joint row policy")
    return [str(x) for x in argv]

