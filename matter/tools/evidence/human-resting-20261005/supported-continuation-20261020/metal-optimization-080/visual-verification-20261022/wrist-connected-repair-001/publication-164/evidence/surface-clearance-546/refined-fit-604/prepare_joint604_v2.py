#!/usr/bin/env python3
"""Prepare the matched 603 full-coordinate joint fit without running it.

This consumes already checked geometry, topology guard, face-target, point-target,
and depth-target artifacts. It never refreshes geometry/targets or launches a fit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069/D")
N = D / "surface-clearance-546"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
HERE = Path(__file__).resolve().parent

HELPER_595 = N / "joint-refined-595/joint_refined595.py"
HELPER_595_SHA = "409d5f6990195373346749f3cb43cdbc55c54afd1fcd2f84c6b30815523b1b4c"
OWNER = N / "continue_surface_round546.py"
OWNER_SHA = "b2fb1407bcbcf8d44cb7138ff2cb9dc6ce43f9a6fcddb64ed6624df672123d17"
FIT_TEMPLATE = N / "connected-continuation-584/run-001/round-001/emitted-fit/fit-argv.json"
FIT_TEMPLATE_SHA = "5eba3df5c353d0ba4f13ca6d83e426be991b6875e0d18748476d8336606b950f"
MATCHED_TEMPLATE = N / "connected-continuation-584/run-001/round-001/emitted-fit/matched-inputs.json"
MATCHED_TEMPLATE_SHA = "28ef8e617b2bd8160d9dc1cf0a1522079f55aa4a4485c4e97b9e0d0d158cd601"
OLD_564_ENTRY = N / "partial-timeout-fallback564/frozen-v2/run_joint_full_coordinate_fit564.py"
OLD_564_ENTRY_SHA = "bce28ca873ba9fdc75427e16443fdf047587742cfbef1f83e48a31df28caa4d9"
ENTRY_603 = N / "component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
ENTRY_603_SHA = "38c3dfa9ddd961250111c87f56d6184075bde43be07c46e09d239f057b06d692"
HELPER_603 = N / "component-scp-area-backtrack-603/frozen-v1/component_scp_backtracking603.py"
HELPER_603_SHA = "f02aed5e462779e85a5d4df46a38a68bcf7eb3972c9dbf565830c4919b387be6"
JOINT_560 = N / "component-scp-area-backtrack-603/frozen-v1/joint_objective_560.py"
JOINT_560_SHA = "09ce0c4f47e69142de6ce394576edd3ef0ee6f4b946824a476f5425919c492c8"
PARTIAL_564 = N / "component-scp-area-backtrack-603/frozen-v1/component_lp_partial_timeout564.py"
PARTIAL_564_SHA = "a24aa58178394e25d7b38c7c35a36c93eaf109a666ace7773c0ca13d33250818"
EXPECTED_PINNED_FILES = {
    HELPER_595: HELPER_595_SHA, OWNER: OWNER_SHA,
    FIT_TEMPLATE: FIT_TEMPLATE_SHA, MATCHED_TEMPLATE: MATCHED_TEMPLATE_SHA,
    OLD_564_ENTRY: OLD_564_ENTRY_SHA, ENTRY_603: ENTRY_603_SHA,
    HELPER_603: HELPER_603_SHA, JOINT_560: JOINT_560_SHA, PARTIAL_564: PARTIAL_564_SHA,
}


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path | str) -> dict:
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pin_file(path: Path, pins: dict[str, str], expected: str | None = None) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"pinned source hash changed: {path}")
    key = str(path)
    old = pins.get(key)
    if old is not None and old != actual:
        raise ValueError(f"conflicting source hash for {path}")
    pins[key] = actual
    return actual


def exact_flag_index(argv: list[str], flag: str) -> int:
    if argv.count(flag) != 1:
        raise ValueError(f"expected exactly one {flag}")
    i = argv.index(flag)
    if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
        raise ValueError(f"missing value after {flag}")
    return i


def replace_entry(argv: list[str], expected_old: Path, new_entry: Path) -> list[str]:
    if len(argv) < 2 or Path(argv[0]) != PYTHON:
        raise ValueError("unexpected fit interpreter/argv shape")
    if Path(argv[1]) != expected_old:
        raise ValueError("fit template does not use the pinned 564-v2 entry")
    if not new_entry.is_file():
        raise FileNotFoundError(new_entry)
    result = list(argv)
    result[1] = str(new_entry)
    return result


def assert_fit_contract(argv: list[str], candidate: Path, surface: Path,
                        matched_inputs: Path, output: Path, trust_mm: float) -> None:
    value_flags = (
        "--base-candidate", "--raw-trial", "--self-pairs", "--area-reference-npz",
        "--point-target-npz", "--point-target-report", "--face-target-npz",
        "--face-target-report", "--point-depth-rows-npz", "--point-depth-rows-report",
        "--fixed-boundary-npz", "--fixed-boundary-report", "--matched-inputs-json-490",
        "--audit159-490", "--output", "--trust-mm", "--source-cap-mm",
        "--area-floor", "--max-edge-gradient", "--time-limit-s", "--area-scp-iterations",
    )
    for flag in value_flags:
        exact_flag_index(argv, flag)
    for flag in ("--immediate-parent-basis", "--allow-soft-target-regression-485",
                 "--allow-partial-timeout-zero-fallback-564"):
        if argv.count(flag) != 1:
            raise ValueError(f"expected exactly one preserved fit policy flag {flag}")
    expected_paths = {
        "--base-candidate": candidate, "--raw-trial": candidate,
        "--area-reference-npz": candidate, "--audit159-490": surface,
        "--matched-inputs-json-490": matched_inputs, "--output": output,
    }
    for flag, expected in expected_paths.items():
        actual = Path(argv[exact_flag_index(argv, flag) + 1])
        if actual != Path(expected):
            raise ValueError(f"fit argument mismatch for {flag}")
    expected_values = {
        "--trust-mm": float(trust_mm), "--source-cap-mm": 12.0, "--area-floor": 0.1,
        "--max-edge-gradient": 0.5, "--time-limit-s": 90.0, "--area-scp-iterations": 2.0,
    }
    for flag, expected in expected_values.items():
        actual = float(argv[exact_flag_index(argv, flag) + 1])
        if actual != expected:
            raise ValueError(f"fit policy changed for {flag}: {actual} != {expected}")


def validate_face_bundle(candidate_sha: str, face_dir: Path,
                         compatibility_path: Path, sha_fn=sha) -> dict:
    target_npz = face_dir / "targets.npz"
    target_report_path = face_dir / "report.json"
    records_path = face_dir / "target-records.jsonl"
    recovery_path = face_dir / "recovery-verification.json"
    for path in (target_npz, target_report_path, records_path, recovery_path, compatibility_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report, recovery, compat = map(read_json, (target_report_path, recovery_path, compatibility_path))
    count = int(report.get("crossing_pairs", -1))
    if report.get("inputs", {}).get("candidate_sha256") != candidate_sha:
        raise ValueError("face targets do not match current refined candidate")
    if report.get("matched_159_complete") is not True or int(report.get("banks", -1)) != 185:
        raise ValueError("face targets lack complete 185-bank matching")
    if count <= 0 or int(report.get("passing_targets", -1)) != count or int(report.get("result_rows", -1)) != count:
        raise ValueError("face target report row counts disagree")
    fit = compat.get("fit_direction_compatibility", {})
    if compat.get("inputs", {}).get("candidate_sha256") != candidate_sha:
        raise ValueError("face-fit compatibility report candidate mismatch")
    if fit.get("original_npz_sha256") != sha_fn(target_npz) or fit.get("targets_npz_sha256") != sha_fn(target_npz):
        raise ValueError("face-fit compatibility target NPZ pin mismatch")
    if fit.get("original_records_sha256") != sha_fn(records_path):
        raise ValueError("face-fit compatibility record pin mismatch")
    if fit.get("original_report_sha256") != sha_fn(target_report_path):
        raise ValueError("face-fit compatibility report pin mismatch")
    if fit.get("recovery_verification_sha256") != sha_fn(recovery_path):
        raise ValueError("face-fit compatibility recovery pin mismatch")
    extension = compat.get("source_current_direction_extension", {})
    if (int(extension.get("all_rows", -1)) != count
            or int(fit.get("joint_direction_rows", -1)) + int(fit.get("source_only_exception_rows", 0)) != count):
        raise ValueError("face-fit compatibility direction/exception row count mismatch")
    if recovery.get("status") != "PASS_CERTIFIED_DIRECTION_TARGETS_ONLY":
        raise ValueError("face target recovery receipt did not pass")
    if recovery.get("candidate_sha256") != candidate_sha or int(recovery.get("exact_endpoint_bank_count", -1)) != 185:
        raise ValueError("face recovery receipt is not candidate-bound to all185")
    if int(recovery.get("target_count", -1)) != count:
        raise ValueError("face recovery target count mismatch")
    for path, expected in recovery.get("artifact_pins", {}).items():
        if not Path(path).is_file() or sha_fn(path) != expected:
            raise ValueError(f"face recovery artifact pin mismatch: {path}")
    with np.load(target_npz, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    rows, targets = np.asarray(arrays.get("face_rows")), np.asarray(arrays.get("target_world_f32_m"))
    if rows.shape != (count,) or targets.shape != (count, 3) or not np.isfinite(targets).all():
        raise ValueError("face target NPZ rows are incomplete or malformed")
    return {"face_target_npz": target_npz, "face_target_report": compatibility_path,
            "face_source_report": target_report_path, "records": records_path,
            "recovery": recovery_path, "row_count": count, "report": report,
            "compatibility": compat}


def validate_current_constraints(geometry_prep: Path, state_path: Path, state: dict,
                                 candidate_sha: str, parent_context_path: Path,
                                 parent_context: dict, sha_fn=sha) -> tuple[dict, Path]:
    prep = read_json(geometry_prep / "prepared.json")
    if prep.get("status") != "CURRENT_TOPOLOGY_GUARDS_AND_FACE_TARGETS_PREPARED":
        raise ValueError("geometry preparation is not the checked 603 preparation")
    if prep.get("accepted_physical_steps") != 0 or prep.get("native_admission") is not False:
        raise ValueError("geometry preparation claims an unpermitted physical/native step")
    if Path(prep.get("state", "")).resolve() != state_path.resolve() or prep.get("state_sha256") != sha_fn(state_path):
        raise ValueError("geometry preparation state path/hash mismatch")
    if Path(prep.get("parent_context", "")).resolve() != parent_context_path.resolve():
        raise ValueError("geometry preparation was not built from the supplied 599 parent context")
    if prep.get("parent_context_sha256") != sha_fn(parent_context_path):
        raise ValueError("supplied parent context hash differs from geometry preparation")
    parent_candidate = Path(parent_context.get("candidate", ""))
    if (not parent_candidate.is_file() or parent_context.get("candidate_sha256") != sha_fn(parent_candidate)
            or parent_context.get("candidate_sha256") != state.get("original_baseline_candidate_sha256")
            or parent_candidate.resolve() != Path(state.get("original_baseline_candidate", "")).resolve()):
        raise ValueError("599 parent context does not pin the 603 immediate input candidate")
    ref_path = Path(state.get("refinement_report", ""))
    if not ref_path.is_file() or sha_fn(ref_path) != state.get("refinement_report_sha256"):
        raise ValueError("603 refinement report hash mismatch")
    ref = read_json(ref_path)
    if (ref.get("status") != "REFINEMENT_CONSTRUCTION_READY"
            or ref.get("input_candidate_sha256") != parent_context.get("candidate_sha256")
            or ref.get("output_candidate_sha256") != candidate_sha):
        raise ValueError("603 report does not prove the supplied 599 immediate-parent lineage")
    refresh_path = Path(prep.get("refresh_report", ""))
    if not refresh_path.is_file() or sha_fn(refresh_path) != prep.get("refresh_report_sha256"):
        raise ValueError("603 guard refresh report pin mismatch")
    ctx = read_json(refresh_path)
    if (ctx.get("status") != "PASS_TOPOLOGY_AWARE_GUARD_AND_SEED_REFRESH"
            or ctx.get("candidate_sha256") != candidate_sha
            or Path(ctx.get("candidate_path", "")).resolve() != Path(state["candidate"]).resolve()):
        raise ValueError("603 refreshed guard/seed context is not bound to its candidate")
    if int(ctx.get("current_exact_self_pairs", -1)) != 0:
        raise ValueError("603 current candidate is not self-intersection free")
    if int(ctx.get("current_exact_bone_crossing_pairs", -1)) <= 0:
        raise ValueError("603 face targets do not correspond to a crossing baseline")
    if ctx.get("fixed_guard_payload_contains_no_current_crossings") is not True or ctx.get("fixed_313_zero_step_certified") is not True:
        raise ValueError("603 fixed bone guards lack zero-step/current-crossing certification")
    for path_key, hash_key in (
        ("seed_pairs_path", "seed_pairs_sha256"), ("seed_report_path", "seed_report_sha256"),
        ("fixed_guard_npz_path", "fixed_guard_npz_sha256"),
        ("fixed_guard_report_path", "fixed_guard_report_sha256"),
        ("fixed_313_rows_path", "fixed_313_rows_sha256"),
        ("fixed_guard_validation_path", "fixed_guard_validation_sha256"),
    ):
        path = Path(ctx[path_key])
        if not path.is_file() or sha_fn(path) != ctx[hash_key]:
            raise ValueError(f"603 refreshed constraint pin mismatch: {path_key}")
    seed = read_json(ctx["seed_report_path"])
    if (seed.get("status") != "PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY"
            or seed.get("candidate_sha256") != candidate_sha or seed.get("zero_step_feasible") is not True):
        raise ValueError("603 self seed report is not 431-compatible or zero-step certified")
    guard = read_json(ctx["fixed_guard_report_path"])
    if guard.get("base_candidate_sha256") != candidate_sha or guard.get("output_npz_sha256") != ctx["fixed_guard_npz_sha256"]:
        raise ValueError("603 fixed guard payload/report mismatch")
    return ctx, refresh_path


def build_preparation(args) -> dict:
    if args.out.exists():
        raise FileExistsError(args.out)
    for path, expected in EXPECTED_PINNED_FILES.items():
        if sha(path) != expected:
            raise ValueError(f"pinned owner/template changed: {path}")
    for path in (args.state, args.parent_context, args.point_target_npz, args.point_target_report,
                 args.depth_npz, args.depth_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.geometry_prep.is_dir():
        raise FileNotFoundError(args.geometry_prep)
    state = read_json(args.state)
    candidate = Path(state["candidate"])
    for key in ("candidate", "point_summary", "surface_report", "refinement_report"):
        p = Path(state[key])
        if not p.is_file() or sha(p) != state.get(key + "_sha256"):
            raise ValueError(f"state pin mismatch for {key}")
    candidate_sha = sha(candidate)
    parent_context = read_json(args.parent_context)
    ctx, refresh_path = validate_current_constraints(
        args.geometry_prep, args.state, state, candidate_sha, args.parent_context, parent_context)
    point_summary = read_json(state["point_summary"])
    surface_report = read_json(state["surface_report"])
    with np.load(candidate, allow_pickle=False) as z:
        candidate_arrays = {k: np.asarray(z[k]).copy() for k in z.files}
    with np.load(args.point_target_npz, allow_pickle=False) as z:
        point_targets = {k: np.asarray(z[k]).copy() for k in z.files}
    point_report = read_json(args.point_target_report)
    correction = point_report.get("emitted_direction_correction_606_v2")
    if not isinstance(correction, dict):
        raise ValueError("point target report must be the corrected 606 emitted-F32 direction receipt")
    if correction.get("candidate_sha256") != candidate_sha or correction.get("target_npz_sha256") != sha(args.point_target_npz):
        raise ValueError("606 direction correction does not match candidate/target NPZ")
    with np.load(args.depth_npz, allow_pickle=False) as z:
        depth_arrays = {k: np.asarray(z[k]).copy() for k in z.files}
    depth_report = read_json(args.depth_report)
    prior_candidate_sha = sha(parent_context["candidate"])
    helper = load_module("joint_refined595_for_604_v2", HELPER_595)
    preflight = helper.validate_joint_refined_inputs(
        state=state, point_summary=point_summary, surface_report=surface_report,
        arrays=candidate_arrays, candidate_sha256=candidate_sha,
        point_target_report=point_report, point_targets=point_targets,
        point_target_sha256=sha(args.point_target_npz), point_depth_report=depth_report,
        depth_npz=depth_arrays, depth_sha256=sha(args.depth_npz), prior_context=parent_context,
        prior_candidate_sha256=prior_candidate_sha, ancestry_parent_sha256=prior_candidate_sha)
    geometry_receipt = read_json(args.geometry_prep / "prepared.json")
    face_dir = Path(geometry_receipt["face_targets"])
    compatibility_path = Path(geometry_receipt["face_compatibility"])
    if compatibility_path.resolve() != (args.geometry_prep / "face-fit-compatibility.json").resolve():
        raise ValueError("603 face compatibility is not the pinned geometry-prep output")
    if sha(compatibility_path) != geometry_receipt.get("face_compatibility_sha256"):
        raise ValueError("603 face compatibility report hash mismatch")
    face = validate_face_bundle(candidate_sha, face_dir, compatibility_path)
    if int(face["row_count"]) != int(read_json(face_dir / "report.json")["crossing_pairs"]):
        raise ValueError("603 face target receipt row count mismatch")

    owner = load_module("continue_surface_round546_for_604_v2", OWNER)
    template_matched = read_json(MATCHED_TEMPLATE)
    if template_matched.get("schema") != "numi.human.matched_inputs_component_fit_475.v1":
        raise ValueError("unexpected 584 matched-490 template schema")
    pins: dict[str, str] = {}
    paths_to_pin: set[Path] = set(EXPECTED_PINNED_FILES)
    paths_to_pin.update({
        args.state, args.geometry_prep / "prepared.json", candidate,
        Path(state["point_summary"]), Path(state["surface_report"]), Path(state["refinement_report"]),
        Path(state["point_summary"]).parent / "query/query-receipt.json",
        args.geometry_prep / "face-fit-compatibility.json",
        face_dir / "targets.npz", face_dir / "report.json", face_dir / "target-records.jsonl",
        face_dir / "recovery-verification.json", refresh_path, args.parent_context,
        Path(parent_context["candidate"]), Path(parent_context["surface_report"]),
        Path(parent_context["guard_npz"]), Path(parent_context["guard_report"]),
        Path(parent_context["seed_pairs_path"]), Path(parent_context["seed_report"]),
        args.point_target_npz, args.point_target_report, args.depth_npz, args.depth_report,
        Path(ctx["seed_pairs_path"]), Path(ctx["seed_report_path"]),
        Path(ctx["fixed_guard_npz_path"]), Path(ctx["fixed_guard_report_path"]),
        Path(ctx["fixed_313_rows_path"]), Path(ctx["fixed_guard_validation_path"]),
        FIT_TEMPLATE, MATCHED_TEMPLATE, HELPER_603, JOINT_560, PARTIAL_564, ENTRY_603,
        OWNER, HELPER_595, OLD_564_ENTRY,
    })
    for section in ("inputs", "source_pins"):
        for record in template_matched.get(section, {}).values():
            if isinstance(record, dict) and "path" in record:
                p = Path(record["path"])
                if not p.is_file() or sha(p) != record.get("sha256"):
                    raise ValueError(f"584 matched-template source pin mismatch: {p}")
                paths_to_pin.add(p)
    for path in paths_to_pin:
        pin_file(path, pins, EXPECTED_PINNED_FILES.get(path))
    argv_template = json.loads(FIT_TEMPLATE.read_text())
    if not isinstance(argv_template, list):
        raise ValueError("584 fit template must be an argv list")
    for token in argv_template:
        p = Path(token)
        if p.is_absolute() and p.is_file():
            pin_file(p, pins)

    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    matched_path = args.out / "matched-inputs.json"
    owner.matched_inputs_from_template(
        MATCHED_TEMPLATE, matched_path,
        {"candidate": candidate, "audit159": Path(state["surface_report"]),
         "face_target_npz": face["face_target_npz"], "point_target_npz": args.point_target_npz})
    argv_helper = load_module("joint_refined595_argv_604_v2", HELPER_595)
    fit_argv = argv_helper.fit_argv_from_template(
        argv_template, candidate=candidate, audit159=Path(state["surface_report"]),
        point_target_npz=args.point_target_npz, point_target_report=args.point_target_report,
        face_target_npz=face["face_target_npz"], face_target_report=face["face_target_report"],
        point_depth_npz=args.depth_npz, point_depth_report=args.depth_report,
        self_pairs=Path(ctx["seed_pairs_path"]), fixed_guard_npz=Path(ctx["fixed_guard_npz_path"]),
        fixed_guard_report=Path(ctx["fixed_guard_report_path"]), matched_inputs=matched_path,
        output=args.out / "fit", trust_mm=4.0)
    fit_argv = replace_entry(fit_argv, OLD_564_ENTRY, ENTRY_603)
    assert_fit_contract(fit_argv, candidate, Path(state["surface_report"]), matched_path,
                        args.out / "fit", 4.0)
    entry_text = ENTRY_603.read_text()
    if "contextlib.redirect_stdout(sink)" not in entry_text or '"report.json"' not in entry_text:
        raise ValueError("603 entry must suppress duplicated stdout and preserve complete report.json output")
    for token in fit_argv:
        p = Path(token)
        if p.is_absolute() and p.is_file():
            pin_file(p, pins)
    argv_path = args.out / "fit-argv.json"
    owner.write_json(argv_path, fit_argv)
    source_pins = dict(sorted(pins.items()))
    receipt = {
        "schema": "numi.human.joint-refined-fit-preparation604.v2",
        "status": "MATCHED_JOINT_REFINED_FIT_PREPARED_UNLAUNCHED",
        "state": str(args.state.resolve()), "state_sha256": sha(args.state),
        "candidate": str(candidate), "candidate_sha256": candidate_sha,
        "geometry_prep": str(args.geometry_prep.resolve()),
        "geometry_prep_sha256": sha(args.geometry_prep / "prepared.json"),
        "parent_context": str(args.parent_context.resolve()),
        "parent_context_sha256": sha(args.parent_context),
        "immediate_parent_candidate_sha256": prior_candidate_sha,
        "current_constraint_report": str(refresh_path), "current_constraint_report_sha256": sha(refresh_path),
        "face_target_npz": str(face["face_target_npz"]), "face_target_npz_sha256": sha(face["face_target_npz"]),
        "face_source_report": str(face["face_source_report"]), "face_source_report_sha256": sha(face["face_source_report"]),
        "face_fit_compatibility": str(compatibility_path), "face_fit_compatibility_sha256": sha(compatibility_path),
        "face_recovery_verification": str(face["recovery"]), "face_recovery_verification_sha256": sha(face["recovery"]),
        "face_target_rows": face["row_count"],
        "point_target_npz": str(args.point_target_npz.resolve()), "point_target_npz_sha256": sha(args.point_target_npz),
        "point_target_report": str(args.point_target_report.resolve()), "point_target_report_sha256": sha(args.point_target_report),
        "point_target_rows": preflight["point_target_rows"],
        "depth_npz": str(args.depth_npz.resolve()), "depth_npz_sha256": sha(args.depth_npz),
        "depth_report": str(args.depth_report.resolve()), "depth_report_sha256": sha(args.depth_report),
        "seed_pairs": str(ctx["seed_pairs_path"]), "seed_report": str(ctx["seed_report_path"]),
        "fixed_guard_npz": str(ctx["fixed_guard_npz_path"]), "fixed_guard_report": str(ctx["fixed_guard_report_path"]),
        "matched_inputs": str(matched_path), "matched_inputs_sha256": sha(matched_path),
        "fit_argv": str(argv_path), "fit_argv_sha256": sha(argv_path),
        "fit_entry": str(ENTRY_603), "fit_entry_sha256": sha(ENTRY_603),
        "fit_entry_stdout_policy": "duplicate stdout redirected to devnull; complete report.json remains written by the pinned fit owner",
        "joint_objective_rows": "existing matched-584 point/face row families retained by frozen 560 adapter; no objective reweighting",
        "fit_not_run": True, "accepted_physical_steps": 0, "native_admission": False,
        "preflight": preflight, "source_pins": source_pins,
    }
    owner.write_json(args.out / "prepared.json", receipt)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--geometry-prep", type=Path, required=True)
    ap.add_argument("--point-target-npz", type=Path, required=True)
    ap.add_argument("--point-target-report", type=Path, required=True)
    ap.add_argument("--depth-npz", type=Path, required=True)
    ap.add_argument("--depth-report", type=Path, required=True)
    ap.add_argument("--parent-context", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = build_preparation(args)
    print(json.dumps({"status": result["status"], "out": str(args.out.resolve()),
                      "fit_argv": result["fit_argv"], "fit_not_run": True}, sort_keys=True))


if __name__ == "__main__":
    main()
