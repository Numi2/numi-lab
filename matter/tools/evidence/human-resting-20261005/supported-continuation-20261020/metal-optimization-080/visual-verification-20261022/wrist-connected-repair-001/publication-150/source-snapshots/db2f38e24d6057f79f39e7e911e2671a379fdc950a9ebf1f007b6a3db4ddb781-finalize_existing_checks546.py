"""Consolidate fresh arithmetic with hash-matched exact owners.

Default mode enforces the strict surface-clearance policy.  The explicitly
requested point-continuation mode is temporary construction evidence only.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D"
C = D / "contained-clearance-545/run_candidate_checks545.py"
MERIT_OWNER = D / "joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py"
MERIT_OWNER_SHA256 = "9d8c4d7bc8397a07192b395f0ed42cb8cf60f15ef00cfe7fd9f48eeb17133dba"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_pins(pins):
    mismatches = [(str(path), expected, sha(path)) for path, expected in pins.items()
                  if sha(path) != expected]
    if mismatches:
        raise ValueError("pinned input/source changed: " + repr(mismatches))


def validate_trust_mm(value):
    if isinstance(value, bool):
        raise ValueError("trust-mm must be a finite positive number")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("trust-mm must be a finite positive number") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("trust-mm must be a finite positive number")
    return value


def make_parser():
    parser = argparse.ArgumentParser()
    for key in ("candidate", "baseline-state", "point-summary", "surface-report", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--point-continuation", action="store_true",
                        help="apply the pinned P175 point-depth continuation policy; output stays unselected")
    parser.add_argument("--trust-mm", type=float, default=1.0,
                        help="construction step trust radius forwarded to the existing 545 arithmetic checker")
    return parser


def arithmetic_command(candidate, baseline_state, output_dir, trust_mm):
    trust_mm = validate_trust_mm(trust_mm)
    return [sys.executable, str(C), "--candidate", str(candidate),
            "--baseline-state", str(baseline_state), "--output-dir", str(output_dir),
            "--arithmetic-only", "--trust-mm", format(trust_mm, ".17g")]


def _required_bool(report, key):
    value = report.get(key)
    if type(value) is not bool:
        raise ValueError("P175 merit lacks boolean " + key)
    return value


def assess_point_continuation(owner, before_points, after_points, before_surface, after_surface):
    """Apply the existing 524/P175 acceptance predicates without changing merit."""
    point = owner.assess_point_clearance_construction_progress(before_points, after_points)
    joint = owner.assess_joint_clearance_construction_progress(
        before_points, after_points, before_surface, after_surface)
    depth = owner.assess_construction_depth_progress(
        before_points, after_points, before_surface, after_surface)
    point_depth_ok = (_required_bool(point, "maximum_depth_nonincreasing")
                      and _required_bool(point, "squared_depth_nonincreasing"))
    progress = (_required_bool(joint, "construction_progress")
                or _required_bool(depth, "construction_progress"))
    return {
        "point_merit": point,
        "joint_merit": joint,
        "depth_merit": depth,
        "point_depth_nonincrease": point_depth_ok,
        "joint_or_depth_construction_progress": progress,
        "accepted": point_depth_ok and progress,
    }


def validate_arithmetic_report(report, requested_trust_mm, expected_anchor_rows):
    """Validate the field names and gates emitted by the pinned 545 checker."""
    if report.get("status") != "ARITHMETIC_PASS" or report.get("arithmetic_pass") is not True:
        raise ValueError("545 checker did not certify arithmetic")
    if report.get("construction_trust_mm") != requested_trust_mm:
        raise ValueError("545 checker did not record requested trust radius")
    if report.get("full_f32_area_gate", {}).get("pass") is not True:
        raise ValueError("full Float32 area gate is absent or failed")
    anchors = report.get("anchor_rows")
    if type(anchors) is not int or anchors != expected_anchor_rows:
        raise ValueError("arithmetic anchor count differs from candidate")
    step = report.get("maximum_step_mm")
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step):
        raise ValueError("545 arithmetic report lacks finite maximum_step_mm")
    if step > requested_trust_mm + 1e-6:
        raise ValueError("construction step exceeds requested trust radius")
    # This is the actual 545 schema field. Do not accept a guessed alias.
    source_distance = report.get("maximum_source_distance_mm")
    if (isinstance(source_distance, bool) or not isinstance(source_distance, (int, float))
            or not math.isfinite(source_distance)):
        raise ValueError("545 arithmetic report lacks finite maximum_source_distance_mm")
    if source_distance > 12.0 + 1e-6:
        raise ValueError("source displacement cap failed")
    return {"maximum_step_mm": float(step),
            "maximum_source_distance_mm": float(source_distance),
            "anchor_rows": anchors,
            "full_f32_area_gate": report["full_f32_area_gate"]}


def _checked_point_report(report, candidate_hash, candidate):
    if report.get("candidate_sha256") != candidate_hash:
        raise ValueError("point report candidate hash mismatch")
    if report.get("complete_surface_audit") is not True:
        raise ValueError("point report is incomplete")
    if type(report.get("exact_PLC_surface_pairs")) is not int or report["exact_PLC_surface_pairs"] != 0:
        raise ValueError("exact self-pair count is not zero")
    inside = report.get("inside_node_bank_pairs")
    if type(inside) is not int or inside < 0:
        raise ValueError("invalid inside-node bank pair count")
    z = np.load(candidate, allow_pickle=False)
    if report.get("points") != len(z["points_world_m"]) or report.get("faces") != len(z["source_O_faces"]):
        raise ValueError("point report topology differs from candidate")
    return inside


def _checked_self_pairs(path):
    pairs = np.load(path, allow_pickle=False)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != 0:
        raise ValueError("exact self-pair array is malformed or nonempty")


def _baseline_inputs(base_state_path, state):
    required = ("candidate", "point_summary", "surface_report")
    for key in required:
        if not isinstance(state.get(key), str) or not isinstance(state.get(key + "_sha256"), str):
            raise ValueError("baseline state missing pinned " + key)
    paths = {key: Path(state[key]) for key in required}
    for key, path in paths.items():
        if sha(path) != state[key + "_sha256"]:
            raise ValueError("baseline " + key + " hash mismatch")
    return paths


def main(argv=None):
    args = make_parser().parse_args(argv)
    trust_mm = validate_trust_mm(args.trust_mm)
    args.out.mkdir(parents=True, exist_ok=False)

    checker = load_module("checks545_for546", C)
    if sha(MERIT_OWNER) != MERIT_OWNER_SHA256:
        raise ValueError("pinned P175 merit owner changed")
    merit_owner = load_module("merit175_for546", MERIT_OWNER)

    base = read(args.baseline_state)
    base_paths = _baseline_inputs(args.baseline_state, base)
    base_hash = base["candidate_sha256"]
    base_points = read(base_paths["point_summary"])
    base_surface = read(base_paths["surface_report"])
    base_inside = _checked_point_report(base_points, base_hash, base_paths["candidate"])
    _checked_self_pairs(base_paths["point_summary"].parent / "surface-self-pairs.npy")
    base_surface_count, base_bank_counts = checker.validate_banks(base_surface, base_hash)

    current_self_pairs = args.point_summary.parent / "surface-self-pairs.npy"
    paths = [args.candidate, args.baseline_state, args.point_summary, args.surface_report,
             current_self_pairs, base_paths["candidate"], base_paths["point_summary"],
             base_paths["surface_report"], base_paths["point_summary"].parent / "surface-self-pairs.npy",
             C, MERIT_OWNER, Path(__file__)]
    pins = {Path(path): sha(path) for path in paths}

    arithmetic_dir = args.out / "arithmetic"
    command = arithmetic_command(args.candidate, args.baseline_state, arithmetic_dir, trust_mm)
    with (args.out / "arithmetic.log").open("x") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    arithmetic = read(arithmetic_dir / "report.json")
    if completed.returncode != 0:
        raise ValueError("existing 545 arithmetic/source/trust/area/anchor checks failed")
    candidate_arrays = np.load(args.candidate, allow_pickle=False)
    arithmetic_evidence = validate_arithmetic_report(
        arithmetic, trust_mm, len(candidate_arrays["typed_coarse_anchor_output_rows_018"]))

    candidate_hash = sha(args.candidate)
    current_points = read(args.point_summary)
    current_surface = read(args.surface_report)
    inside = _checked_point_report(current_points, candidate_hash, args.candidate)
    _checked_self_pairs(current_self_pairs)
    surface_count, bank_counts = checker.validate_banks(current_surface, candidate_hash)

    point_assessment = None
    if args.point_continuation:
        point_assessment = assess_point_continuation(
            merit_owner, base_points, current_points, base_surface, current_surface)
        if not point_assessment["accepted"]:
            raise ValueError("P175 point-depth/nonregression construction policy rejected candidate")
        status = "POINT_CONTINUATION_CONSTRUCTION_ONLY"
        construction_progress = True
    else:
        if inside != 0:
            raise ValueError("zero containment must be preserved")
        if surface_count >= base_surface_count:
            raise ValueError("not a strict surface reduction")
        status = "COMPLETE_EXACT_CONSTRUCTION_CHECKS"
        construction_progress = True

    verify_pins(pins)
    report = dict(
        status=status,
        candidate_path=str(args.candidate),
        candidate_sha256=candidate_hash,
        baseline_state=str(args.baseline_state),
        baseline_candidate_sha256=base_hash,
        point_summary=str(args.point_summary),
        surface_report=str(args.surface_report),
        point_pairs=inside,
        contained_pairs=inside,
        surface_pairs=surface_count,
        self_pairs=0,
        surface_delta=surface_count - base_surface_count,
        contained_delta=inside - base_inside,
        complete_banks=185,
        per_bank_deltas={key: bank_counts[key] - base_bank_counts[key]
                          for key in bank_counts if bank_counts[key] != base_bank_counts[key]},
        construction_trust_mm=trust_mm,
        construction_progress=construction_progress,
        exact_audits_reused_without_reexecution=True,
        exact_reuse_pins={str(path): expected for path, expected in pins.items()},
        arithmetic_argv=command,
        arithmetic_sha256=sha(arithmetic_dir / "report.json"),
        arithmetic_status=arithmetic["status"],
        maximum_step_mm=arithmetic_evidence["maximum_step_mm"],
        maximum_source_distance_mm=arithmetic_evidence["maximum_source_distance_mm"],
        full_f32_area_gate=arithmetic_evidence["full_f32_area_gate"],
        anchor_rows=arithmetic_evidence["anchor_rows"],
        native_admission=False,
        accepted_physical_steps=0,
    )
    if point_assessment is not None:
        report.update(
            continuation_mode="point_continuation",
            temporary_unselected_construction=True,
            candidate_selected=False,
            point_merit=point_assessment["point_merit"],
            joint_merit=point_assessment["joint_merit"],
            depth_merit=point_assessment["depth_merit"],
            point_depth_nonincrease=point_assessment["point_depth_nonincrease"],
            joint_or_depth_construction_progress=point_assessment["joint_or_depth_construction_progress"],
        )
    else:
        report.update(continuation_mode="strict_surface", temporary_unselected_construction=False)
    write(args.out / "report.json", report)
    state = {
        "candidate": str(args.candidate),
        "candidate_sha256": candidate_hash,
        "point_summary": str(args.point_summary),
        "point_summary_sha256": sha(args.point_summary),
        "surface_report": str(args.surface_report),
        "surface_report_sha256": sha(args.surface_report),
        "continuation_mode": report["continuation_mode"],
        "temporary_unselected_construction": report["temporary_unselected_construction"],
        "candidate_selected": report.get("candidate_selected", False),
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    write(args.out / "state.json", state)
    print(json.dumps({key: report[key] for key in
                      ("status", "surface_pairs", "contained_pairs", "self_pairs",
                       "construction_progress", "continuation_mode", "candidate_sha256")}))


if __name__ == "__main__":
    main()
