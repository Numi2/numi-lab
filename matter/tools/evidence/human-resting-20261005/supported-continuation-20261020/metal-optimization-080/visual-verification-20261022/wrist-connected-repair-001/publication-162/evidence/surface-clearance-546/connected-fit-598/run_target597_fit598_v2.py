"""Run frozen 564 full-coordinate fit with independently checked 597 targets.

This emits an unadmitted construction proposal. Fresh 546 point/depth/self/fixed
constraints remain matched to the current candidate; only the face target role
is supplied by 597, with its full-coverage and recovery receipts verified.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
N = ROOT / "D/surface-clearance-546"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
FIT583 = N / "matched-connected-fit-583/run_matched_fit583_v2.py"
FIT564 = N / "partial-timeout-fallback564/frozen-v2/run_joint_full_coordinate_fit564.py"
COMPAT_HELPER = N / "connected-fit-598/target-compatibility-001/compatibility_report598_v2.py"
TARGET597 = N / "source-current-targets-597/build_joint_outward_targets597_v2.py"
VERIFY597 = N / "source-current-targets-597/verify_recovered_targets597.py"
REQUIRED = ("face_rows", "bank_body", "bank_stable_bone_id",
            "bank_triangle_stable_id", "target_world_f32_m",
            "target_exactly_outside_all185", "proposal_direction_world_unit")
MODE = "source-current-outward-with-source-fold-exceptions"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def validate_targets(report, arrays, receipt, *, candidate_sha, surface_sha,
                     report_path, npz_path, records_path):
    if receipt.get("status") != "PASS_CERTIFIED_DIRECTION_TARGETS_ONLY":
        raise ValueError("597 recovered target receipt is not a certified pass")
    receipt_pins = receipt.get("artifact_pins", {})
    for path in (report_path, npz_path, records_path):
        if receipt_pins.get(str(path)) != sha(path):
            raise ValueError(f"597 recovery receipt pin mismatch: {path}")
    if receipt.get("candidate_sha256") != candidate_sha or receipt.get("state_sha256") is None:
        raise ValueError("597 recovery receipt belongs to another candidate/state")
    if report.get("direction_mode") != MODE:
        raise ValueError("597 direction policy does not match reviewed mode")
    ext = report.get("source_current_direction_extension", {})
    inp = report.get("inputs", {})
    if inp.get("candidate_sha256") != candidate_sha or inp.get("surface_report_sha256") != surface_sha:
        raise ValueError("597 target source candidate/surface report mismatch")
    n = int(report.get("crossing_pairs", -1))
    joint = int(report.get("joint_outward_direction_count", -1))
    exceptions = int(report.get("source_only_fold_exception_count", -1))
    if joint < 0 or exceptions < 0 or joint + exceptions != n:
        raise ValueError("597 joint/exception row accounting is incomplete")
    if (report.get("matched_159_complete") is not True or int(report.get("banks", -1)) != 185
        or int(report.get("matched_surface_bank_triangle_pairs", -1)) != n
        or int(report.get("passing_targets", -1)) != n
        or int(report.get("nearest_boundary_certified_interval_targets", -1)) != n
        or int(report.get("inside_samples_certified_for_implicated_bank", -1)) != n
        or int(ext.get("all_rows", -1)) != n or int(ext.get("passing_rows", -1)) != n
        or int(receipt.get("exact_endpoint_bank_count", -1)) != 185
        or int(receipt.get("target_count", -1)) != n):
        raise ValueError("597 does not cover and certify every current crossing pair")
    for key in REQUIRED:
        if key not in arrays:
            raise ValueError(f"597 targets missing field {key}")
    if len(arrays["face_rows"]) != n:
        raise ValueError("597 target count differs from current crossing count")
    for key in ("bank_body", "bank_stable_bone_id", "bank_triangle_stable_id"):
        if arrays[key].shape != (n,):
            raise ValueError(f"bad identity field shape: {key}")
    for key in ("target_world_f32_m", "proposal_direction_world_unit"):
        if arrays[key].shape != (n, 3) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"bad vector field: {key}")
    if arrays["target_exactly_outside_all185"].shape != (n,) or not arrays["target_exactly_outside_all185"].all():
        raise ValueError("not all 597 endpoints are certified outside all185")
    return {
        "crossing_pairs": n, "target_rows": n, "joint_source_current_rows": int(report.get("joint_outward_direction_count", -1)),
        "source_only_fold_exceptions": int(report.get("source_only_fold_exception_count", -1)),
        "endpoint_banks": 185, "direction_mode": MODE,
    }

def load_compatibility_report(compat_path, target_npz, target_records, recovery_path):
    compat_path, target_npz, target_records, recovery_path = map(
        Path, (compat_path, target_npz, target_records, recovery_path))
    adapted = read(compat_path)
    info = adapted.get("fit_direction_compatibility")
    if (not isinstance(info, dict) or adapted.get("direction_mode") != "outward-nearest"
            or info.get("selector_route") != "outward-nearest"
            or info.get("nearest_optimality_claim") is not False
            or info.get("operative_direction_mode") != MODE):
        raise ValueError("598 compatibility report lacks the explicit legacy-selector disclosure")
    if info.get("adapter_path") != str(COMPAT_HELPER.resolve()) or info.get("adapter_sha256") != sha(COMPAT_HELPER):
        raise ValueError("compatibility adapter path/hash mismatch")
    original_path = Path(info.get("original_report_path", ""))
    if not original_path.is_file() or sha(original_path) != info.get("original_report_sha256"):
        raise ValueError("original 597 target report pin mismatch")
    recovery = read(recovery_path)
    for key, path in (("original_npz_sha256", target_npz),
                      ("original_records_sha256", target_records),
                      ("recovery_verification_sha256", recovery_path)):
        if info.get(key) != sha(path):
            raise ValueError(f"compatibility source pin mismatch: {key}")
    if info.get("recovery_verification_path") != str(recovery_path.resolve()):
        raise ValueError("compatibility report names another recovery receipt")
    original = read(original_path)
    restored = copy.deepcopy(adapted)
    del restored["fit_direction_compatibility"]
    restored["direction_mode"] = original.get("direction_mode")
    if restored != original:
        raise ValueError("compatibility report changed fields beyond selector routing/disclosure")
    if original.get("direction_mode") != MODE or original.get("direction_method") != info.get("operative_direction_method"):
        raise ValueError("compatibility report does not preserve the operative 597 direction method")
    return original, original_path, info

def run(state_path, prepared, target_report_path, target_npz, recovery_path,
        target_records, output, trust_mm=4.0, edge_gradient=0.5):
    state_path, prepared = Path(state_path), Path(prepared)
    target_report_path, target_npz = Path(target_report_path), Path(target_npz)
    recovery_path, target_records, output = Path(recovery_path), Path(target_records), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(trust_mm) or trust_mm <= 0 or not np.isfinite(edge_gradient) or edge_gradient <= 0:
        raise ValueError("trust and edge gradient must be positive finite values")
    state = read(state_path)
    for role in ("candidate", "point_summary", "surface_report"):
        if sha(state[role]) != state[role + "_sha256"]:
            raise ValueError(f"state pin mismatch: {role}")
    base = Path(state["candidate"])
    context_path = prepared / "constraint-context.json"
    context = read(context_path)
    prep_report = read(prepared / "prepared.json")
    if prep_report.get("status") != "FRESH_CONSTRAINTS_DEPTH_SURFACE_TARGETS_PREPARED":
        raise ValueError("fresh 546 preparation did not complete")
    if context.get("candidate_sha256") != sha(base) or context.get("candidate") != str(base):
        raise ValueError("fresh 546 context is not matched to current base")
    recovery = read(recovery_path)
    with np.load(target_npz, allow_pickle=False) as z:
        arrays = {k: np.array(z[k], copy=True) for k in z.files}
    target_report, original_target_report_path, compat_info = load_compatibility_report(
        target_report_path, target_npz, target_records, recovery_path)
    coverage = validate_targets(target_report, arrays, recovery,
        candidate_sha=sha(base), surface_sha=sha(state["surface_report"]),
        report_path=original_target_report_path, npz_path=target_npz, records_path=target_records)
    if recovery.get("state_sha256") != sha(state_path):
        raise ValueError("597 recovery receipt was verified against a different state file")

    m = load(N / "continue_surface_round546.py", "continue546_598")
    point_dir = prepared / "point-targets"
    point_npz = point_dir / "nearest-union-exits.npz"
    point_report = point_dir / "nearest-union-exits.json"
    if point_npz.is_file() and point_report.is_file():
        point_meta = read(point_report)
    else:
        if point_dir.exists():
            if any(point_dir.iterdir()):
                raise ValueError("partial point-target directory exists; preserving it for diagnosis")
            point_dir.rmdir()
        point_npz, point_report = m.write_empty_point_targets(
            point_dir, base, Path(state["point_summary"]),
            prepared / "depth/365-compatible-report.json")
        point_meta = read(point_report)
    if point_meta.get("status") != "PASS_EMPTY_POINT_TARGETS_ZERO_CONTAINMENT":
        raise ValueError("1585 baseline must have a fresh empty point target")
    if point_meta.get("inputs", {}).get("candidate_sha256") != sha(base):
        raise ValueError("empty point target is stale")
    output.mkdir(parents=True)
    matched = output / "matched-inputs.json"
    m.matched_inputs_from_template(
        m.MATCHED_TEMPLATE, matched,
        {"candidate": base, "audit159": Path(state["surface_report"]),
         "face_target_npz": target_npz, "point_target_npz": point_npz})
    argv = m.fit_argv_from_template(
        m.FIT_TEMPLATE, candidate=base, audit159=Path(state["surface_report"]),
        context=context, prep=prepared, point_npz=point_npz,
        point_report=point_report, matched=matched, output=output / "fit")
    argv[1] = str(FIT564)
    for flag, value in (("--face-target-npz", target_npz),
                        ("--face-target-report", target_report_path),
                        ("--trust-mm", str(trust_mm)),
                        ("--max-edge-gradient", str(edge_gradient)),
                        ("--source-cap-mm", "12"), ("--time-limit-s", "90")):
        m.replace_cli_value(argv, flag, value)
    if "--allow-partial-timeout-zero-fallback-564" not in argv:
        argv.append("--allow-partial-timeout-zero-fallback-564")
    m.write_json(output / "fit-policy.json", {
        "fit_owner": str(FIT564), "fit_owner_sha256": sha(FIT564),
        "base_candidate": str(base), "base_candidate_sha256": sha(base),
        "state": str(state_path), "state_sha256": sha(state_path),
        "prepared_context": str(context_path), "prepared_context_sha256": sha(context_path),
        "face_target_report": str(target_report_path), "face_target_report_sha256": sha(target_report_path),
        "original_face_target_report": str(original_target_report_path), "original_face_target_report_sha256": sha(original_target_report_path),
        "compatibility_disclosure": compat_info,
        "face_target_npz": str(target_npz), "face_target_npz_sha256": sha(target_npz),
        "recovery_verification": str(recovery_path), "recovery_verification_sha256": sha(recovery_path),
        "target_coverage": coverage, "trust_mm": trust_mm, "max_edge_gradient": edge_gradient,
        "source_cap_mm": 12.0, "area_floor": 0.1,
        "native_admission": False, "accepted_physical_steps": 0})
    (output / "fit-argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    sources = (Path(__file__), FIT583, FIT564, COMPAT_HELPER, TARGET597, VERIFY597, state_path, base, context_path,
        Path(context["guard_npz"]), Path(context["guard_report"]),
        Path(context["seed_pairs_path"]), Path(context["seed_report"]),
        target_report_path, original_target_report_path, target_npz, target_records, recovery_path, point_npz, point_report, matched)
    m.run_stage(argv, output / "fit-stage", slot=True, source_paths=sources, timeout=420)
    report_path = output / "fit/report.json"
    report = read(report_path)
    if report.get("inputs", {}).get("base_candidate_sha256") != sha(base):
        raise ValueError("564 report base hash mismatch")
    load(FIT583, "fit583_validation_for598")._validate_matched_report(report, matched, sha(base))
    result = {"status": report.get("status"), "fit_report": str(report_path),
        "fit_report_sha256": sha(report_path), "candidate": report.get("candidate_path"),
        "candidate_sha256": report.get("candidate_sha256"), "target_coverage": coverage,
        "matched_inputs": str(matched), "matched_inputs_sha256": sha(matched),
        "native_admission": False, "accepted_physical_steps": 0}
    m.write_json(output / "598-result.json", result)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--face-target-report", type=Path, required=True)
    p.add_argument("--face-target-npz", type=Path, required=True)
    p.add_argument("--target-records", type=Path, required=True)
    p.add_argument("--recovery-verification", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trust-mm", type=float, default=4.0)
    p.add_argument("--max-edge-gradient", type=float, default=0.5)
    a = p.parse_args()
    print(json.dumps(run(a.state, a.prepared_dir, a.face_target_report, a.face_target_npz,
        a.recovery_verification, a.target_records, a.out, a.trust_mm, a.max_edge_gradient),
        sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
