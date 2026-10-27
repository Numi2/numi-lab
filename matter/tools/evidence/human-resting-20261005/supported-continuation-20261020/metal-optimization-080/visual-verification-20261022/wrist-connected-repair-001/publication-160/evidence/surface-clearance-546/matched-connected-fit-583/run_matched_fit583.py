"""Matched connected-fit continuation with explicit endpoint selection.

The fit remains a construction proposal. Existing 582/545/552 owners gate any
retained target; this helper does not add or replace physical acceptance.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
N = ROOT / "D/surface-clearance-546"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
FIT564 = N / "partial-timeout-fallback564/frozen-v2/run_joint_full_coordinate_fit564.py"
RETENTION582 = N / "local-witness-retention-582/retain_local_witnesses582.py"
OWNER552 = N / "retain_surface_components552.py"
CHECK545 = ROOT / "D/contained-clearance-545/run_candidate_checks545.py"
FINALIZE546 = N / "finalize_existing_checks546.py"
PRIOR_CONTEXT = N / "round-033-565-refresh/constraints/constraint-context.json"
SEED_TRIAL = N / "round-026-guard-cuts556/raw-directions/raw-direction-001/candidate.npz"
SEED_AUDIT130 = N / "round-026-guard-cuts556/raw-self-audit130/summary.json"
TARGET_POLICIES = ("emitted", "raw-last")


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path | str) -> dict:
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def same(a, b) -> bool:
    aa = np.ascontiguousarray(a)
    bb = np.ascontiguousarray(b)
    return aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes() == bb.tobytes()


def _fit_dir(path: Path) -> Path:
    return path / "fit" if (path / "fit/report.json").is_file() else path


def _prepared_dir(path: Path) -> Path:
    return path / "prepare" if (path / "prepare/prepared.json").is_file() else path


def _validate_matched_report(fit_report: dict, matched_path: Path, base_sha: str) -> None:
    if fit_report.get("inputs", {}).get("base_candidate_sha256") != base_sha:
        raise ValueError("fit report is bound to a different baseline candidate")
    recorded = fit_report.get("matched_component_fit_490", {}).get("matched_inputs", {}).get("inputs")
    current = read_json(matched_path).get("inputs", {})
    if not isinstance(recorded, dict) or set(recorded) != set(current):
        raise ValueError("fit report lacks the matched 490 input-role receipt")
    for role, expected in current.items():
        actual = recorded.get(role, {})
        if actual.get("sha256") != expected.get("sha256"):
            raise ValueError(f"fit matched input differs from prepared role {role}")


FIT_RECEIPT_PREFIXES = (
    "local_witness_cluster_222_", "local_witness_cluster_230_",
    "local_witness_cluster_315_", "local_witness_cluster_354_",
)


def _validate_target_payload(base_path: Path, target_path: Path, *,
                             allow_fit_receipt_updates: bool = False) -> None:
    with np.load(base_path, allow_pickle=False) as z:
        base = {k: np.array(z[k], copy=True) for k in z.files}
    with np.load(target_path, allow_pickle=False) as z:
        target = {k: np.array(z[k], copy=True) for k in z.files}
    if set(base) != set(target) or "points_world_m" not in base:
        raise ValueError("fit target archive differs from baseline schema")
    for key in base:
        if key == "points_world_m":
            if target[key].shape != base[key].shape or target[key].dtype != base[key].dtype:
                raise ValueError("target point field shape/dtype mismatch")
        elif not same(base[key], target[key]):
            if allow_fit_receipt_updates and key.startswith(FIT_RECEIPT_PREFIXES):
                continue
            raise ValueError(f"fit target changed nonpoint payload {key}")


def choose_target(base_path: Path, fit_dir: Path, policy: str) -> dict:
    if policy not in TARGET_POLICIES:
        raise ValueError(f"target policy must be one of {TARGET_POLICIES}")
    report_path = fit_dir / "report.json"
    fit_report = read_json(report_path)
    base_sha = sha(base_path)
    if fit_report.get("inputs", {}).get("base_candidate_sha256") != base_sha:
        raise ValueError("fit base hash does not match requested state")
    successful = (
        fit_report.get("status") == "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED"
        and bool(fit_report.get("solver", {}).get("success"))
    )
    emitted_path = Path(fit_report.get("candidate_path", fit_dir / "candidate.npz"))
    emitted_ok = successful and emitted_path.is_file() and sha(emitted_path) == fit_report.get("candidate_sha256")
    if emitted_ok:
        _validate_target_payload(base_path, emitted_path, allow_fit_receipt_updates=True)
    capture_report_path = fit_dir / "raw-joint-capture-563/report.json"
    capture = read_json(capture_report_path) if capture_report_path.is_file() else {}
    returns = capture.get("solver_returns", [])
    raw = None
    if returns:
        if (capture.get("base_candidate") != str(base_path.resolve())
                or capture.get("base_candidate_sha256") != base_sha):
            raise ValueError("raw capture summary is not pinned to this baseline")
        record = returns[-1]
        raw_path = Path(record["candidate_path"])
        proposal_report = Path(record["proposal_report_path"])
        if (not raw_path.is_file() or sha(raw_path) != record.get("candidate_sha256")
                or not proposal_report.is_file()
                or sha(proposal_report) != record.get("proposal_report_sha256")):
            raise ValueError("last captured raw direction file/report pin mismatch")
        _validate_target_payload(base_path, raw_path)
        raw = {"path": str(raw_path), "sha256": sha(raw_path),
               "capture_record": record, "capture_report": str(capture_report_path),
               "capture_report_sha256": sha(capture_report_path)}

    if policy == "raw-last":
        if raw is None:
            raise ValueError("raw-last requested but no captured direction exists")
        selected = raw["path"]
        reason = "explicit raw-last policy; still unadmitted and sent through 582/545/552"
    elif emitted_ok:
        selected = str(emitted_path)
        reason = "successful emitted fitter endpoint selected before any pre-area-alpha raw direction"
    elif raw is not None:
        selected = raw["path"]
        reason = "fit did not emit a successful endpoint; captured raw direction is fallback only"
    else:
        raise ValueError("neither successful fit endpoint nor captured raw fallback is available")
    return {
        "policy": policy, "selected_target": selected, "selected_target_sha256": sha(selected),
        "selection_reason": reason, "fit_dir": str(fit_dir.resolve()),
        "fit_report": str(report_path), "fit_report_sha256": sha(report_path),
        "fit_status": fit_report.get("status"), "fit_success": successful,
        "emitted_endpoint_available": emitted_ok, "emitted_endpoint": str(emitted_path),
        "emitted_endpoint_sha256": sha(emitted_path) if emitted_path.is_file() else None,
        "raw_last_available": raw is not None, "raw_last": raw,
        "native_admission": False, "accepted_physical_steps": 0,
    }


def _load_continue():
    spec = importlib.util.spec_from_file_location("continue_surface_round546_for583", N / "continue_surface_round546.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load existing 546 orchestration helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_inputs(state_path: Path, output: Path, prepared_from: Path | None,
                   prior_context: Path, seed_trial: Path, seed_audit: Path):
    m = _load_continue()
    state = m.read_json(state_path)
    base = Path(state["candidate"])
    for role in ("candidate", "point_summary", "surface_report"):
        if sha(state[role]) != state[role + "_sha256"]:
            raise ValueError(f"state pin mismatch for {role}")
    if prepared_from is None:
        prep = output / "prepare"
        prep_owner = N / "prepare_surface546.py"
        m.run_stage(
            [str(PYTHON), prep_owner, "--state", state_path, "--out", prep,
             "--prior-context", prior_context, "--seed-source-candidate", seed_trial,
             "--seed-source-audit130", seed_audit],
            output / "prepare-stage", slot=False,
            source_paths=(Path(__file__), prep_owner, state_path, prior_context, seed_trial, seed_audit),
            timeout=900)
    else:
        prep = _prepared_dir(prepared_from)
    m.validate_prepared_baseline(prep, state_path)
    context = m.read_json(prep / "constraint-context.json")

    # Keep the exact prepared zero-point target report/arrays when available.
    point_dir = prep / "point-targets"
    point_npz = point_dir / "nearest-union-exits.npz"
    point_report = point_dir / "nearest-union-exits.json"
    if not point_npz.is_file() or not point_report.is_file():
        point_npz, point_report = m.write_empty_point_targets(
            point_dir, base, Path(state["point_summary"]), prep / "depth/365-compatible-report.json")
    point_meta = m.read_json(point_report)
    if point_meta.get("inputs", {}).get("candidate_sha256") != state["candidate_sha256"]:
        raise ValueError("prepared empty point target does not match current baseline")
    if point_meta.get("status") != "PASS_EMPTY_POINT_TARGETS_ZERO_CONTAINMENT":
        raise ValueError("surface-only continuation requires the pinned empty point target")

    matched = output / "matched-inputs.json"
    m.matched_inputs_from_template(
        m.MATCHED_TEMPLATE, matched,
        {"candidate": base, "audit159": Path(state["surface_report"]),
         "face_target_npz": prep / "surface-targets/targets.npz",
         "point_target_npz": point_npz})
    return m, state, base, prep, context, point_npz, point_report, matched


def _run_fresh_fit(m, state, base, prep, context, point_npz, point_report, matched,
                   output, trust_mm, fit_edge_gradient):
    argv = m.fit_argv_from_template(
        m.FIT_TEMPLATE, candidate=base, audit159=Path(state["surface_report"]),
        context=context, prep=prep, point_npz=point_npz, point_report=point_report,
        matched=matched, output=output / "fit")
    argv[1] = str(FIT564)
    for flag, value in (("--trust-mm", str(trust_mm)),
                        ("--max-edge-gradient", str(fit_edge_gradient)),
                        ("--source-cap-mm", "12"), ("--time-limit-s", "90")):
        m.replace_cli_value(argv, flag, value)
    if "--allow-partial-timeout-zero-fallback-564" not in argv:
        argv.append("--allow-partial-timeout-zero-fallback-564")
    (output / "fit-argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    m.write_json(output / "sensitivity-policy.json", {
        "fit_trust_mm": trust_mm, "fit_max_edge_gradient": fit_edge_gradient,
        "source_cap_mm": 12, "area_floor": 0.1, "max_total_solver_wall_seconds": 120,
        "retention_is_separate_construction_screen": True,
        "native_admission": False, "accepted_physical_steps": 0})
    sources = (FIT564, FIT564.with_name("component_lp_partial_timeout564.py"),
               FIT564.with_name("joint_objective_560.py"), state["surface_report"],
               matched, point_npz, point_report, base,
               Path(context["guard_npz"]), Path(context["guard_report"]),
               Path(context["seed_pairs_path"]))
    fit_failed = False
    try:
        m.run_stage(argv, output / "fit-stage", slot=True, source_paths=sources, timeout=360)
    except RuntimeError:
        fit_failed = True
        if not (output / "fit/raw-joint-capture-563/report.json").is_file():
            raise
    return fit_failed


def run_retention(m, state_path, base, fit_dir, selection, prepared_dir, output,
                  retention_edge_gradient):
    target = Path(selection["selected_target"])
    retention = output / "local-retention"
    sources = (Path(__file__), RETENTION582, CHECK545, OWNER552, FINALIZE546,
               state_path, base, target, Path(selection["fit_report"]),
               prepared_dir / "constraint-context.json",
               Path(read_json(prepared_dir / "constraint-context.json")["seed_pairs_path"]))
    try:
        m.run_stage(
            [str(PYTHON), RETENTION582, "--state", state_path, "--target", target,
             "--out", retention, "--max-edge-gradient", str(retention_edge_gradient),
             "--trust-mm", "4"],
            output / "582-retention-stage", slot=True, source_paths=sources, timeout=900)
    except RuntimeError:
        report = retention / "report.json"
        if report.is_file():
            return {"status": read_json(report).get("status"), "report": str(report),
                    "candidate": None, "full_check": None}
        raise
    candidate = retention / "candidate.npz"
    report = read_json(retention / "report.json")
    if not candidate.is_file() or sha(candidate) != report.get("candidate_sha256"):
        raise ValueError("582 did not produce a report-pinned retained candidate")

    full = retention / "full-check"
    m.run_stage(
        [str(PYTHON), CHECK545, "--candidate", candidate,
         "--baseline-state", state_path, "--trust-mm", "4", "--output-dir", full],
        output / "local-full-check-stage", slot=False, source_paths=sources, timeout=900)
    checked = read_json(full / "report.json")
    if checked.get("construction_progress") and checked.get("contained_pairs") == 0 and checked.get("self_pairs") == 0:
        m.run_stage(
            [str(PYTHON), FINALIZE546, "--candidate", candidate, "--baseline-state", state_path,
             "--point-summary", full / "audit-130/summary.json",
             "--surface-report", full / "audit-159/report.json",
             "--out", retention / "final", "--trust-mm", "4"],
            output / "local-finalize-stage", slot=False, source_paths=sources, timeout=120)
        return {"status": checked.get("status"), "candidate": str(candidate),
                "candidate_sha256": sha(candidate), "full_check": str(full / "report.json"),
                "final_state": str(retention / "final/state.json"),
                "final_state_sha256": sha(retention / "final/state.json")}

    if checked.get("status") != "COMPLETE_EXACT_CONSTRUCTION_CHECKS" or checked.get("self_pairs") != 0:
        return {"status": checked.get("status"), "candidate": str(candidate),
                "candidate_sha256": sha(candidate), "full_check": str(full / "report.json")}

    regions = output / "retained-regions"
    m.run_stage(
        [str(PYTHON), OWNER552, "--state", state_path, "--proposal", retention,
         "--full-check", full, "--context", prepared_dir / "constraint-context.json",
         "--output", regions],
        output / "regions-stage", slot=False, source_paths=sources, timeout=120)
    region_report = read_json(regions / "report.json")
    baseline_surface = read_json(read_json(state_path)["surface_report"])
    base_count = sum(int((bank.get("result") or {}).get("count", 0))
                     for bank in baseline_surface["banks"])
    if int(region_report["predicted_surface_pairs"]) >= base_count:
        return {"status": "NO_PREDICTED_REGION_PROGRESS",
                "predicted_surface_pairs": int(region_report["predicted_surface_pairs"]),
                "baseline_surface_pairs": base_count}
    region_candidate = regions / "candidate.npz"
    region_check = regions / "full-check"
    m.run_stage(
        [str(PYTHON), CHECK545, "--candidate", region_candidate,
         "--baseline-state", state_path, "--trust-mm", "4", "--output-dir", region_check],
        output / "regions-full-check-stage", slot=False, source_paths=sources, timeout=900)
    result = read_json(region_check / "report.json")
    summary = {"status": result.get("status"), "surface_pairs": result.get("surface_pairs"),
               "contained_pairs": result.get("contained_pairs"), "self_pairs": result.get("self_pairs"),
               "candidate": str(region_candidate), "candidate_sha256": sha(region_candidate),
               "full_check": str(region_check / "report.json")}
    if result.get("construction_progress") and result.get("contained_pairs") == 0 and result.get("self_pairs") == 0:
        m.run_stage(
            [str(PYTHON), FINALIZE546, "--candidate", region_candidate,
             "--baseline-state", state_path, "--point-summary", region_check / "audit-130/summary.json",
             "--surface-report", region_check / "audit-159/report.json",
             "--out", regions / "final", "--trust-mm", "4"],
            output / "regions-finalize-stage", slot=False, source_paths=sources, timeout=120)
        summary["final_state"] = str(regions / "final/state.json")
        summary["final_state_sha256"] = sha(summary["final_state"])
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prepared-from", type=Path)
    p.add_argument("--fit-from", type=Path)
    p.add_argument("--prior-context", type=Path, default=PRIOR_CONTEXT)
    p.add_argument("--seed-source-candidate", type=Path, default=SEED_TRIAL)
    p.add_argument("--seed-source-audit130", type=Path, default=SEED_AUDIT130)
    p.add_argument("--trust-mm", type=float, default=1.0)
    p.add_argument("--fit-edge-gradient", type=float, default=0.1)
    p.add_argument("--retention-edge-gradient", type=float, default=0.5)
    p.add_argument("--target-policy", choices=TARGET_POLICIES, default="emitted")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if not np.isfinite(a.trust_mm) or a.trust_mm <= 0:
        raise ValueError("--trust-mm must be positive")
    if not np.isfinite(a.fit_edge_gradient) or a.fit_edge_gradient <= 0:
        raise ValueError("--fit-edge-gradient must be positive")
    if not np.isfinite(a.retention_edge_gradient) or a.retention_edge_gradient <= 0:
        raise ValueError("--retention-edge-gradient must be positive")
    a.out.mkdir(parents=True)

    prepared_from = _prepared_dir(a.prepared_from) if a.prepared_from else None
    m, state, base, prep, context, point_npz, point_report, matched = prepare_inputs(
        a.state, a.out, prepared_from, a.prior_context, a.seed_source_candidate,
        a.seed_source_audit130)
    if a.fit_from:
        fit_dir = _fit_dir(a.fit_from)
        fit_failed = False
    else:
        fit_failed = _run_fresh_fit(m, state, base, prep, context, point_npz,
                                    point_report, matched, a.out, a.trust_mm,
                                    a.fit_edge_gradient)
        fit_dir = a.out / "fit"

    fit_report = read_json(fit_dir / "report.json")
    _validate_matched_report(fit_report, matched, sha(base))
    if fit_report.get("status") == "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
        edge = fit_report.get("constraints", {}).get("edge_gradient", {}).get("max_component_gradient")
        trust = fit_report.get("constraints", {}).get("trust_mm")
        if edge is not None and not np.isclose(float(edge), a.fit_edge_gradient):
            raise ValueError("fit-from edge-gradient policy does not match request")
        if trust is not None and not np.isclose(float(trust), a.trust_mm):
            raise ValueError("fit-from trust policy does not match request")
    selection = choose_target(base, fit_dir, a.target_policy)
    selection.update({
        "fit_was_run_by_583": not bool(a.fit_from),
        "fit_process_failed": fit_failed,
        "prepared_dir": str(prep.resolve()),
        "prepared_context": str(prep / "constraint-context.json"),
        "matched_inputs": str(matched.resolve()),
        "fit_trust_mm_requested": a.trust_mm,
        "fit_edge_gradient_requested": a.fit_edge_gradient,
        "retention_edge_gradient": a.retention_edge_gradient,
        "physical_retention_trust_mm": 4.0,
    })
    m.write_json(a.out / "target-selection.json", selection)
    result = run_retention(m, a.state, base, fit_dir, selection, prep, a.out,
                           a.retention_edge_gradient)
    result.update({"target_selection": str(a.out / "target-selection.json"),
                   "native_admission": False, "accepted_physical_steps": 0})
    m.write_json(a.out / "583-result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
