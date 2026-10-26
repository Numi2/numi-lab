"""Prepare, fit, backtrack, and finalize one same-topology surface-clearance round.

This orchestrator is inert until run. Each fit is slot-wrapped; the existing
545 checked backtracker is called directly because it manages its own child
execution/slot use. Finalization reuses its selected exact audits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D/surface-clearance-546"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
SLOT = Path("/tmp/numi-human-geometry-slot.py")
PREPARE = D / "prepare_surface546.py"
FIT_WRAPPER = D / "run_surface_only_full_coordinate_fit546.py"
FIT_TEMPLATE = D / "round-001/fit-argv.json"
MATCHED_TEMPLATE = D / "round-001/matched-inputs.json"
BACKTRACKER = ROOT / "D/contained-clearance-545/run_checked_batch_context545.py"
FINALIZER = D / "finalize_existing_checks546.py"
TEMPLATE_ROUND = D / "round-001"

def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def write_json(path: Path | str, value: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def empty_point_archive() -> dict[str, np.ndarray]:
    return {
        "query_rows": np.empty((0,), dtype=np.int64),
        "target_world_f32_m": np.empty((0, 3), dtype=np.float32),
        "proposal_direction_world_unit": np.empty((0, 3), dtype=np.float64),
        "source_surface_normal_world_unit": np.empty((0, 3), dtype=np.float64),
        "depth_covector_world_unit": np.empty((0, 3), dtype=np.float64),
        "direction_compatibility_mask": np.empty((0,), dtype=bool),
        "source_normal_available_mask": np.empty((0,), dtype=bool),
        "pass_mask": np.empty((0,), dtype=bool),
        "target_distance_m": np.empty((0,), dtype=np.float64),
        "target_exactly_outside_all185": np.empty((0,), dtype=bool),
    }

def write_empty_point_targets(directory: Path, candidate: Path, point_summary: Path,
                              depth_report_path: Path) -> tuple[Path, Path]:
    summary = read_json(point_summary)
    depth = read_json(depth_report_path)
    candidate_sha = sha(candidate)
    if summary.get("candidate_sha256") != candidate_sha:
        raise ValueError("empty-point source summary does not match current candidate")
    if summary.get("complete_surface_audit") is not True:
        raise ValueError("empty-point source summary is incomplete")
    if int(summary.get("exact_PLC_surface_pairs", -1)) != 0:
        raise ValueError("current candidate has self intersections")
    if int(summary.get("inside_node_bank_pairs", -1)) != 0:
        raise ValueError("current candidate is not zero-contained")
    if depth.get("candidate_sha256") != candidate_sha:
        raise ValueError("depth report is not candidate matched")
    if int(depth.get("inside_row_count", -1)) != 0 or int(depth.get("outside_row_count", 0)) <= 0:
        raise ValueError("surface-only continuation requires all depth rows outside")
    directory.mkdir(parents=True, exist_ok=False)
    npz = directory / "nearest-union-exits.npz"
    report_path = directory / "nearest-union-exits.json"
    if npz.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite empty point-target inputs")
    np.savez_compressed(npz, **empty_point_archive())
    report = {
        "schema": "numi.human.point-first-union-exits.depth-maximizing-390.v1",
        "status": "PASS_EMPTY_POINT_TARGETS_ZERO_CONTAINMENT",
        "classification": "Empty construction objective from exact zero-contained baseline; no ray targets requested or physical steps",
        "inputs": {
            "candidate_sha256": candidate_sha,
            "point_summary_sha256": sha(point_summary),
            "depth_report_sha256": sha(depth_report_path),
        },
        "passing_targets": 0,
        "query_rows": 0,
        "all_bank_ray_results": 0,
        "inside_node_bank_pairs": 0,
        "inside_query_nodes": 0,
        "physical_steps": 0,
        "targets": [],
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    write_json(report_path, report)
    return npz, report_path

def matched_inputs_from_template(template_path: Path, output_path: Path,
                                 inputs: dict[str, Path]) -> dict:
    template = read_json(template_path)
    roles = {"candidate", "audit159", "face_target_npz", "point_target_npz"}
    if template.get("schema") != "numi.human.matched_inputs_component_fit_475.v1":
        raise ValueError("unexpected matched-input template schema")
    if set(template.get("inputs", {})) != roles or set(inputs) != roles:
        raise ValueError("matched-input roles do not match the pinned 490 contract")
    result = dict(template)
    result["inputs"] = {
        role: {"path": str(Path(path).resolve()), "sha256": sha(path)}
        for role, path in sorted(inputs.items())
    }
    write_json(output_path, result)
    return result

def replace_cli_value(argv: list[str], flag: str, value: Path | str) -> None:
    if argv.count(flag) != 1:
        raise ValueError(f"expected exactly one {flag} in fit template")
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise ValueError(f"missing value after {flag}")
    argv[i + 1] = str(value)

def fit_argv_from_template(template_path: Path, *, candidate: Path, audit159: Path,
                           context: dict, prep: Path, point_npz: Path, point_report: Path,
                           matched: Path, output: Path) -> list[str]:
    argv = json.loads(template_path.read_text())
    if not isinstance(argv, list) or not argv or Path(argv[0]) != PYTHON:
        raise ValueError("fit argv template has an unexpected interpreter")
    replacements = {
        "--base-candidate": candidate,
        "--raw-trial": candidate,
        "--self-pairs": Path(context["seed_pairs_path"]),
        "--area-reference-npz": candidate,
        "--point-target-npz": point_npz,
        "--point-target-report": point_report,
        "--face-target-npz": prep / "surface-targets/targets.npz",
        "--face-target-report": prep / "surface-targets/report.json",
        "--point-depth-rows-npz": prep / "depth/point-depth-rows.npz",
        "--point-depth-rows-report": prep / "depth/365-compatible-report.json",
        "--fixed-boundary-npz": Path(context["guard_npz"]),
        "--fixed-boundary-report": Path(context["guard_report"]),
        "--output": output,
        "--matched-inputs-json-490": matched,
        "--audit159-490": audit159,
    }
    for flag, value in replacements.items():
        replace_cli_value(argv, flag, value)
    if Path(argv[1]) != FIT_WRAPPER:
        raise ValueError("fit template does not use the pinned 546 surface-only wrapper")
    return [str(x) for x in argv]

def run_stage(argv: list[str | Path], stem: Path, *, slot: bool,
              source_paths: tuple[Path, ...] = (), timeout: float | None = None) -> dict:
    args = [str(x) for x in argv]
    actual = [str(PYTHON), str(SLOT), *args] if slot else args
    launch = stem.with_suffix(".launch.json")
    execution = stem.with_suffix(".execution.json")
    log_path = stem.with_suffix(".log")
    if any(p.exists() for p in (launch, execution, log_path)):
        raise FileExistsError(f"stage evidence already exists for {stem}")
    pins = {str(p.resolve()): sha(p) for p in source_paths if p.is_file()}
    write_json(launch, {"argv": actual, "slot_wrapped": slot, "source_pins": pins,
                        "native_admission": False, "accepted_physical_steps": 0})
    started = time.monotonic()
    with log_path.open("x") as log:
        result = subprocess.run(actual, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    unchanged = all(sha(path) == digest for path, digest in pins.items())
    status = {
        "returncode": int(result.returncode),
        "seconds": time.monotonic() - started,
        "source_pins_unchanged": unchanged,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    write_json(execution, status)
    if result.returncode != 0 or not unchanged:
        raise RuntimeError(f"stage failed: {stem}; see {log_path}")
    return status

def validate_prepared_baseline(prep_dir: Path, state_path: Path) -> None:
    prepared = read_json(prep_dir / "prepared.json")
    if (prepared.get("status") != "FRESH_CONSTRAINTS_DEPTH_SURFACE_TARGETS_PREPARED"
            or prepared.get("native_admission") is not False
            or prepared.get("accepted_physical_steps") != 0):
        raise ValueError("prepared input is incomplete or outside construction scope")
    if sha(prepared["state"]) != sha(state_path):
        raise ValueError("prepared baseline state differs from requested state")
    state = read_json(state_path)
    baseline = read_json(prep_dir / "baseline.json")
    context = read_json(prep_dir / "constraint-context.json")
    for role in ("candidate", "point_summary", "surface_report"):
        if baseline[role + "_sha256"] != state[role + "_sha256"]:
            raise ValueError("prepared baseline role mismatch: " + role)
    if context["candidate_sha256"] != state["candidate_sha256"]:
        raise ValueError("prepared context candidate mismatch")
    for role in ("seed_report", "guard_npz", "guard_report", "refresh_report"):
        if context.get(role + "_sha256") is not None and sha(context[role]) != context[role + "_sha256"]:
            raise ValueError("prepared context input changed: " + role)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--prior-context", type=Path, required=True)
    ap.add_argument("--prior-failed-trial", type=Path, required=True)
    ap.add_argument("--prior-failed-audit130", type=Path, required=True)
    ap.add_argument("--round-dir", type=Path, required=True)
    ap.add_argument("--fit-wrapper", type=Path, default=FIT_WRAPPER)
    ap.add_argument("--prior-surface-report", type=Path)
    ap.add_argument("--prepared-dir", type=Path, help="Resume a completed candidate-matched preparation without rerunning it")
    args = ap.parse_args()
    fit_wrapper = args.fit_wrapper.resolve()
    round_dir = args.round_dir.resolve()
    round_dir.mkdir(parents=True, exist_ok=False)
    state = read_json(args.state)
    for key in ("candidate", "point_summary", "surface_report"):
        p = Path(state[key])
        if sha(p) != state[key + "_sha256"]:
            raise ValueError(f"current state hash mismatch: {key}")
    base = Path(state["candidate"])
    prior = read_json(args.prior_context)
    for key in ("candidate", "guard_npz", "guard_report", "seed_pairs_path"):
        if key not in prior or not Path(prior[key]).is_file():
            raise ValueError(f"prior context missing usable {key}")
    # This version is deliberately same-topology only. The prep owner verifies
    # face identity before mapping guards and old witness rows.
    with np.load(base, allow_pickle=False) as current, np.load(prior["candidate"], allow_pickle=False) as parent:
        if not np.array_equal(current["source_O_faces"], parent["source_O_faces"]) and args.prior_surface_report is None:
            raise ValueError("topology changed; use the separate 546 topology-seed remapper")
    orchestrator_path = Path(__file__).resolve()
    owner_paths = (orchestrator_path, PREPARE, fit_wrapper, D / "refresh_surface_constraints546.py", D / "refresh_guards_multi_parent_546.py",
                   D / "surface_objective_546.py",
                   ROOT / "D/contained-clearance-545/component_lp_active545.py",
                   ROOT / "D/contained-clearance-545/run_checked_batch_context545.py",
                   ROOT / "D/contained-clearance-545/run_candidate_checks545.py",
                   FINALIZER)
    owner_pins = {str(p.resolve()): sha(p) for p in owner_paths}
    prep_dir = args.prepared_dir.resolve() if args.prepared_dir else round_dir / "prepare"
    prepare_cmd = [PYTHON, PREPARE, "--state", args.state, "--out", prep_dir,
                   "--prior-context", args.prior_context,
                   "--seed-source-candidate", args.prior_failed_trial,
                   "--seed-source-audit130", args.prior_failed_audit130]
    if args.prior_surface_report is not None:
        prepare_cmd.extend(["--prior-surface-report", args.prior_surface_report])
    if args.prepared_dir is None:
        run_stage(prepare_cmd, round_dir / "prepare-stage", slot=False,
                  source_paths=owner_paths, timeout=1800)
    else:
        validate_prepared_baseline(prep_dir, args.state)
        write_json(round_dir / "prepare-reuse.json", {
            "prepared_dir": str(prep_dir), "prepared_report_sha256": sha(prep_dir / "prepared.json"),
            "state_sha256": sha(args.state), "context_sha256": sha(prep_dir / "constraint-context.json"),
            "native_admission": False, "accepted_physical_steps": 0})
    prepared = read_json(prep_dir / "prepared.json")
    if prepared.get("status") != "FRESH_CONSTRAINTS_DEPTH_SURFACE_TARGETS_PREPARED":
        raise ValueError("surface preparation did not complete")
    context = read_json(prep_dir / "constraint-context.json")
    depth_report = prep_dir / "depth/365-compatible-report.json"
    fit_input_dir = round_dir / "fit-inputs" if args.prepared_dir else prep_dir
    point_npz, point_report = write_empty_point_targets(
        fit_input_dir / "point-targets", base, Path(state["point_summary"]), depth_report
    )
    matched_path = fit_input_dir / "matched-inputs.json"
    matched_inputs_from_template(
        MATCHED_TEMPLATE, matched_path,
        {"candidate": base, "audit159": Path(state["surface_report"]),
         "face_target_npz": prep_dir / "surface-targets/targets.npz",
         "point_target_npz": point_npz},
    )
    fit490 = ROOT / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"
    spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location(
        "fit490_preflight_546_continue", fit490
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load matched-input validator")
    module = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_matched_inputs(matched_path, {
        "candidate": base, "audit159": Path(state["surface_report"]),
        "face_target_npz": prep_dir / "surface-targets/targets.npz",
        "point_target_npz": point_npz,
    })
    module.validate_source_lock()
    fit_dir = round_dir / "fit"
    fit_argv = fit_argv_from_template(
        FIT_TEMPLATE, candidate=base, audit159=Path(state["surface_report"]),
        context=context, prep=prep_dir, point_npz=point_npz,
        point_report=point_report, matched=matched_path, output=fit_dir,
    )
    fit_argv[1] = str(fit_wrapper)
    (round_dir / "fit-argv.json").write_text(json.dumps(fit_argv, indent=2) + "\n")
    run_stage(fit_argv, round_dir / "fit-stage", slot=True,
              source_paths=owner_paths, timeout=1800)
    fit_report = read_json(fit_dir / "report.json")
    if fit_report.get("status") != "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
        raise ValueError("fit did not emit a construction proposal")
    backtrack_dir = round_dir / "backtrack"
    run_stage([PYTHON, BACKTRACKER, args.state, fit_dir, prep_dir / "constraint-context.json", backtrack_dir],
              round_dir / "backtrack-stage", slot=False,
              source_paths=owner_paths, timeout=3600)
    selected_path = backtrack_dir / "backtrack-373/selected-result.json"
    selected = read_json(selected_path)
    for role in ("candidate", "point_summary", "surface_report"):
        if role not in selected or not Path(selected[role]).is_file():
            raise ValueError(f"checked backtracker selected result lacks {role}")
    final_dir = round_dir / "final-check"
    run_stage([PYTHON, FINALIZER, "--candidate", selected["candidate"],
               "--baseline-state", args.state, "--point-summary", selected["point_summary"],
               "--surface-report", selected["surface_report"], "--out", final_dir],
              round_dir / "final-check-stage", slot=False,
              source_paths=owner_paths, timeout=600)
    final_state = read_json(final_dir / "state.json")
    final_report = read_json(final_dir / "report.json")
    write_json(round_dir / "state.json", final_state)
    before = read_json(state["surface_report"])
    before_pairs = sum(int((bank.get("result") or {}).get("count", 0)) for bank in before["banks"])
    progress = {
        "status": final_report["status"],
        "before_surface_pairs": before_pairs,
        "after_surface_pairs": int(final_report["surface_pairs"]),
        "surface_pair_delta": int(final_report["surface_delta"]),
        "contained_pairs": int(final_report["contained_pairs"]),
        "self_pairs": int(final_report["self_pairs"]),
        "candidate_sha256": final_state["candidate_sha256"],
        "candidate": final_state["candidate"],
        "state": str(round_dir / "state.json"),
        "construction_progress": bool(final_report.get("construction_progress")),
        "exact_audits_reused_without_reexecution": True,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    write_json(round_dir / "progress.json", progress)
    if any(sha(path) != digest for path, digest in owner_pins.items()):
        raise RuntimeError("a pinned round owner changed during orchestration")
    write_json(round_dir / "orchestration.json", {
        "schema": "numi.human.surface_clearance_round_546.v1",
        "status": "COMPLETE",
        "input_state": str(args.state),
        "input_state_sha256": sha(args.state),
        "prior_context": str(args.prior_context),
        "prepared_dir_reused": str(prep_dir) if args.prepared_dir else None,
        "prior_failed_trial": str(args.prior_failed_trial),
        "prior_failed_audit130": str(args.prior_failed_audit130),
        "owner_pins": owner_pins,
        "progress": progress,
        "fit_uses_geometry_slot": True,
        "backtracker_outer_slot": False,
        "final_checks_reuse_selected_exact_audits": True,
        "topology_change_supported": args.prior_surface_report is not None,
        "native_admission": False,
        "accepted_physical_steps": 0,
    })
    print(json.dumps(progress, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
