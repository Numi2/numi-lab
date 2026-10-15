#!/usr/bin/env python3
"""Read-only audit for the primary-policy CGLS snapshot emitted by source143.

This checks native layout/status/row masking and primary-policy trace
semantics. When supplied, it compares the captured action on native x and r
against a separately retained rich operator capture. The action comparison is
directional only, not a whole-matrix identity or constitutive proof.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any
import numpy as np

MAGIC = 0x534C4743
HEADER_WORDS = 8
SEGMENT_COUNT = 12
VECTOR_NAMES = ("x", "r", "s", "p", "Ax", "b", "mask")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def align16(value: int) -> int:
    return (value + 15) & ~15


def parse_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_WORDS * 4:
        raise ValueError("CGLS snapshot is shorter than its eight-word header")
    magic, version, env_count, nodes_per_env, unknown_count, cap, nsegments, reserved = struct.unpack_from("<8I", raw)
    if magic != MAGIC or version != 1 or nsegments != SEGMENT_COUNT or reserved != 0:
        raise ValueError(f"unsupported CGLS header: {(magic, version, nsegments, reserved)}")
    if env_count == 0 or nodes_per_env == 0 or unknown_count != 2 * env_count * nodes_per_env:
        raise ValueError("CGLS header violates U = 2 * E * N")
    if cap not in (32, 64):
        raise ValueError(f"unsupported CGLS cap {cap}")
    specs: list[tuple[str, str, int, tuple[int, ...]]] = []
    for name in VECTOR_NAMES:
        specs.append((name, "<f4", 4 * unknown_count, (unknown_count, 4)))
    specs.extend([
        ("row_environment", "<u4", unknown_count, (unknown_count,)),
        ("scalars", "<f4", 4 * env_count, (env_count, 4)),
        ("control", "<u4", 4 * env_count, (env_count, 4)),
        ("errors", "<u4", env_count, (env_count,)),
        ("selected", "<u4", env_count, (env_count,)),
    ])
    arrays: dict[str, np.ndarray] = {}
    segments: list[dict[str, int | str]] = []
    offset = HEADER_WORDS * 4
    for name, dtype, count, shape in specs:
        start = align16(offset)
        if start > len(raw) or any(raw[offset:start]):
            raise ValueError(f"nonzero/truncated alignment padding before {name}")
        byte_count = count * 4
        end = start + byte_count
        if end > len(raw):
            raise ValueError(f"truncated CGLS segment {name}")
        arrays[name] = np.frombuffer(raw, dtype=np.dtype(dtype), count=count, offset=start).reshape(shape).copy()
        segments.append({"name": name, "offset": start, "bytes": byte_count})
        offset = end
    if offset != len(raw):
        raise ValueError(f"unexpected trailing bytes: expected {offset}, got {len(raw)}")
    return ({"magic_hex": f"0x{magic:08x}", "version": version,
             "environment_count": env_count, "nodes_per_environment": nodes_per_env,
             "unknown_count": unknown_count, "iteration_cap": cap,
             "segment_count": nsegments, "file_bytes": len(raw), "sha256": sha(path),
             "segments": segments}, arrays)


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        return {"shape_match": False, "left_shape": list(aa.shape), "right_shape": list(bb.shape)}
    d = aa - bb
    if not np.isfinite(d).all():
        return {"shape_match": True, "finite": False,
                "nonfinite_count": int(d.size - np.count_nonzero(np.isfinite(d)))}
    dn, bn = float(np.linalg.norm(d)), float(np.linalg.norm(bb))
    return {"shape_match": True, "finite": True, "difference_l2": dn,
            "reference_l2": bn, "relative_l2": dn / bn if bn else None,
            "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
            "threshold": None, "meaning": "descriptive action comparison; no noise-floor or qualification gate"}


def compare_float32_words(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Compare values after canonical FP32 conversion, including signed-zero bits."""
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.shape != bb.shape:
        return {"shape_match": False, "left_shape": list(aa.shape), "right_shape": list(bb.shape)}
    aw, bw = aa.view(np.uint32), bb.view(np.uint32)
    mismatch = np.flatnonzero(aw != bw)
    return {
        "shape_match": True,
        "left_count": int(aa.size),
        "bitwise_word_match": bool(mismatch.size == 0),
        "mismatch_count": int(mismatch.size),
        "first_mismatch_index": int(mismatch[0]) if mismatch.size else None,
        "first_left_word_hex": f"0x{int(aw[mismatch[0]]):08x}" if mismatch.size else None,
        "first_right_word_hex": f"0x{int(bw[mismatch[0]]):08x}" if mismatch.size else None,
        "numeric_max_abs_difference_after_fp32_cast": float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64)))) if aa.size else 0.0,
        "meaning": "FP32-cast word comparison only; source JSON may not preserve signed-zero spelling, so this is reported rather than treated as an unconditional capture-integrity gate",
    }


def raw_zero(a: np.ndarray) -> bool:
    return bool(np.all(np.asarray(a).view(np.uint32) == 0))


def existing_action_gate(result: dict[str, Any]) -> dict[str, Any]:
    relative = result.get("relative_l2")
    reference_norm = result.get("reference_l2")
    resolved = (isinstance(relative, (int, float)) and np.isfinite(relative) and
                isinstance(reference_norm, (int, float)) and np.isfinite(reference_norm) and
                float(reference_norm) > 0.0)
    return {"threshold_relative_l2": 0.01,
            "applicable_for_nonzero_finite_reference_action": bool(resolved),
            "relative_error": relative,
            "absolute_difference_l2": result.get("difference_l2"),
            "reference_action_l2": reference_norm,
            "within_existing_1_percent_gate": bool(resolved and relative <= 0.01),
            "meaning": "applies the existing 1% assembled-action gate to this bound nonzero vector action; it is not a whole-matrix or solver certificate and adds no absolute noise-floor claim"}


def load_audit_module(snapshot: Path):
    path = snapshot / "matter/tools/cardboard_cached_operator_audit.py"
    if not path.is_file():
        raise ValueError(f"missing source-bound operator audit module {path}")
    spec = importlib.util.spec_from_file_location("source_bound_cardboard_operator_audit", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import source-bound operator audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def verify_prediction_inputs(module, prediction_path: Path, prediction: dict,
                             snapshot: Path) -> dict[str, Any]:
    if Path(prediction.get("source_snapshot", "")).resolve() != snapshot.resolve():
        raise ValueError("prediction source_snapshot does not equal retained source snapshot")
    inputs = prediction.get("inputs_sha256")
    if not isinstance(inputs, dict):
        raise ValueError("prediction lacks source/input hash map")
    runtime_value = prediction.get("runtime_dir")
    runtime = Path(runtime_value) if isinstance(runtime_value, str) and runtime_value else None
    checked, missing = 0, []
    for relative, expected in inputs.items():
        candidate, _ = module._manifest_file(relative, snapshot, runtime, prediction)
        if candidate is None or not candidate.is_file():
            missing.append(relative)
            continue
        if module.sha256(candidate) != expected:
            raise ValueError(f"prediction input hash mismatch: {relative}")
        checked += 1
    if missing:
        raise ValueError(f"prediction inputs unavailable: {missing[:8]}")
    return {"input_count": len(inputs), "verified_count": checked,
            "all_prediction_inputs_verified": checked == len(inputs)}


def parse_markers(stderr: Path, prefix: str) -> list[dict]:
    out = []
    for line in stderr.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            out.append(json.loads(line.split("=", 1)[1]))
    return out


def _indexed_records(records: list[dict], root: int, lo: int, hi: int,
                     direction_kind: str | None = None) -> dict[int, dict]:
    chosen = [r for r in records if int(r.get("root", -1)) == root and
              lo <= int(r.get("newton_iteration", -1)) < hi and
              (direction_kind is None or r.get("direction_kind") == direction_kind)]
    out: dict[int, dict] = {}
    duplicates: list[int] = []
    for record in chosen:
        it = int(record["newton_iteration"])
        if it in out:
            duplicates.append(it)
        out[it] = record
    if duplicates:
        raise ValueError(f"duplicate records for root {root}: {sorted(set(duplicates))}")
    return out


def _selected_environment(record: dict, environment: int) -> dict | None:
    values = record.get("environments")
    return values[environment] if isinstance(values, list) and environment < len(values) else None


def _fallback_is_inert(fallback: dict, direction: dict | None,
                       primary_newton: dict | None, environment: int) -> tuple[bool, dict]:
    row = _selected_environment(fallback, environment)
    primary_row = _selected_environment(primary_newton, environment) if primary_newton else None
    marker_rows = direction.get("environments") if isinstance(direction, dict) else None
    marker_row = marker_rows[environment] if isinstance(marker_rows, list) and environment < len(marker_rows) else None
    candidate_norm = marker_row.get("candidate_l2") if isinstance(marker_row, dict) else None
    prior_accepted = isinstance(primary_row, dict) and primary_row.get("accepted") is True
    prior_noop = isinstance(primary_row, dict) and primary_row.get("certified_noop") is True
    role = row.get("last_trial_status_role") if isinstance(row, dict) else None
    cgls_marker_valid = isinstance(direction, dict) and \
        direction.get("direction_kind") == "truncated_unregularized_CGLS" and \
        direction.get("original_newton_certificate_applies_to_direction") is False
    inert = bool(
        cgls_marker_valid and
        fallback.get("direction_kind") in ("base_cgls", "base_cauchy") and
        isinstance(candidate_norm, (int, float)) and float(candidate_norm) == 0.0 and
        (prior_accepted or prior_noop) and
        role in ("unused_tail_after_selected_accept", "no_private_trial_certified_noop")
    )
    detail = {
        "newton_iteration": fallback.get("newton_iteration"),
        "direction_kind": fallback.get("direction_kind"),
        "selected_environment_candidate_l2": candidate_norm,
        "fallback_direction_marker_valid": cgls_marker_valid,
        "prior_newton_accepted": prior_accepted,
        "prior_newton_certified_noop": prior_noop,
        "fallback_last_trial_status_role": role,
        "classification": "inert observation: no CGLS vector installed and the prior Newton step already selected/no-op" if inert else
                          "active or unresolved fallback observation; retained and blocks exact pretarget-equivalence admission",
    }
    return inert, detail


def summarize_pretarget_trace(run_records: list[dict], baseline_records: list[dict],
                              run_directions: list[dict], baseline_directions: list[dict],
                              root: int, target_iteration: int,
                              environment: int) -> dict[str, Any]:
    actual = _indexed_records(run_records, root, 0, target_iteration, "newton")
    baseline = _indexed_records(baseline_records, root, 0, target_iteration, "newton")
    mismatches = [i for i in range(target_iteration)
                  if i not in actual or i not in baseline or actual.get(i) != baseline.get(i)]

    direction_by_iteration = _indexed_records(run_directions, root, 0, target_iteration)
    inert, active = [], []
    for record in run_records:
        iteration = int(record.get("newton_iteration", -1))
        if int(record.get("root", -1)) != root or not (0 <= iteration < target_iteration):
            continue
        if record.get("direction_kind") == "newton":
            continue
        is_inert, detail = _fallback_is_inert(
            record, direction_by_iteration.get(iteration), actual.get(iteration), environment)
        (inert if is_inert else active).append(detail)

    baseline_extra = [r for r in baseline_records if int(r.get("root", -1)) == root and
                      0 <= int(r.get("newton_iteration", -1)) < target_iteration and
                      r.get("direction_kind") != "newton"]
    canonical_actual = [json.dumps(actual[i], sort_keys=True, separators=(",", ":"), allow_nan=False)
                        for i in sorted(actual)]
    canonical_baseline = [json.dumps(baseline[i], sort_keys=True, separators=(",", ":"), allow_nan=False)
                          for i in sorted(baseline)]
    missing_actual = [i for i in range(target_iteration) if i not in actual]
    missing_baseline = [i for i in range(target_iteration) if i not in baseline]
    exact_newton = bool(not mismatches and not missing_actual and not missing_baseline and
                        len(actual) == target_iteration and len(baseline) == target_iteration)
    return {
        "classification": "compares exactly one ordinary Newton Armijo record per index [0,target); only semantically inert root:all fallback observations are omitted from that comparison and listed below",
        "root": root, "target_newton_iteration_exclusive": target_iteration,
        "environment_used_to_classify_fallback": environment,
        "actual_newton_record_count": len(actual), "baseline_newton_record_count": len(baseline),
        "expected_newton_record_count": target_iteration,
        "missing_actual_iterations": missing_actual, "missing_baseline_iterations": missing_baseline,
        "mismatch_iterations": mismatches,
        "inert_fallback_observations_discarded_from_newton_comparison": inert,
        "active_or_unresolved_fallback_observations": active,
        "baseline_non_newton_pretarget_record_count": len(baseline_extra),
        "exact_pretarget_newton_trace_parity": exact_newton,
        "pretarget_trace_parity_with_inert_fallbacks_discarded": bool(exact_newton and not active and not baseline_extra),
        "actual_canonical_newton_record_sha256": hashlib.sha256("\n".join(canonical_actual).encode()).hexdigest(),
        "baseline_canonical_newton_record_sha256": hashlib.sha256("\n".join(canonical_baseline).encode()).hexdigest(),
    }


def summarize_coupled_capture_scope(stderr: Path, root: int, iteration: int) -> dict[str, Any]:
    skipped = parse_markers(stderr, "fem_coupled_capture_skipped=")
    captured = parse_markers(stderr, "fem_coupled_capture=")
    selected_skips = [m for m in skipped if int(m.get("root", -1)) == root and
                      int(m.get("iteration", -1)) == iteration]
    selected_captures = [m for m in captured if int(m.get("root", -1)) == root and
                         int(m.get("iteration", -1)) == iteration]
    return {
        "root": root, "newton_iteration": iteration,
        "completed_markers": selected_captures,
        "skipped_markers": selected_skips,
        "classification": ("completed rich operator capture marker present" if selected_captures else
                            "runtime emitted a bounded skip marker" if selected_skips else
                            "no completed or skipped rich capture marker; absence is not interpreted as success"),
    }


def summarize_primary_armijo_row(direction: dict, armijo: dict,
                                 environment: int) -> dict[str, Any]:
    direction_rows = direction.get("environments", [])
    eligible_row = direction_rows[environment] if isinstance(direction_rows, list) and environment < len(direction_rows) else None
    if not isinstance(eligible_row, dict) or not isinstance(eligible_row.get("primary_policy_eligible"), bool):
        raise ValueError("primary direction marker lacks a boolean eligibility value for the selected environment")
    eligible = eligible_row["primary_policy_eligible"]
    row = _selected_environment(armijo, environment)
    if armijo.get("direction_kind") != "base_cgls_primary_policy":
        raise ValueError("post-start Armijo record does not carry the primary-policy trace label")
    if not isinstance(row, dict):
        raise ValueError("selected environment missing from primary Armijo record")
    noop = row.get("certified_noop") is True
    accepted = row.get("accepted") is True
    if eligible:
        actual_direction = "CGLS primary direction selected; original Newton certificate does not transfer"
        certificate_transfer = False
    else:
        actual_direction = "original Newton direction retained; the primary-policy label names the selected policy scope, not a CGLS direction"
        certificate_transfer = None
    if accepted and noop:
        effect = "certified_noop_no_state_change"
    elif accepted:
        effect = "accepted_state_update"
    else:
        effect = "rejected_or_blocked_no_accepted_update"
    return {
        "newton_iteration": int(armijo["newton_iteration"]),
        "policy_trace_label": armijo.get("direction_kind"),
        "primary_policy_eligible": bool(eligible),
        "cgls_direction_selected": bool(eligible),
        "actual_direction_classification": actual_direction,
        "original_newton_certificate_transferred_to_CGLS": certificate_transfer,
        "armijo_outcome": effect,
        "accepted": accepted,
        "certified_noop": noop,
        "selected_alpha": row.get("selected_alpha"),
        "trial_alpha": row.get("trial_alpha"),
        "base_l2": row.get("base_l2"),
        "trial_l2": row.get("trial_l2"),
        "trial_outcome": row.get("trial_outcome"),
        "base_health": row.get("base_health"),
        "solver_status_code": row.get("solver_status_code"),
        "pretrial_status": row.get("pretrial_status"),
    }


def summarize_later_primary_trials(records: list[dict], directions: list[dict],
                                   root: int, start: int,
                                   environment: int,
                                   iteration_cap: int) -> dict[str, Any]:
    armijo_by_iteration = _indexed_records(records, root, start, 1 << 30,
                                           "base_cgls_primary_policy")
    direction_by_iteration: dict[int, dict] = {}
    duplicate_directions = []
    for direction in directions:
        if int(direction.get("root", -1)) != root or int(direction.get("newton_iteration", -1)) < start:
            continue
        if direction.get("selection_policy") != "base_cgls_primary_policy":
            continue
        iteration = int(direction["newton_iteration"])
        if iteration in direction_by_iteration:
            duplicate_directions.append(iteration)
        direction_by_iteration[iteration] = direction
    if duplicate_directions:
        raise ValueError(f"duplicate primary direction markers: {sorted(set(duplicate_directions))}")
    if not armijo_by_iteration:
        raise ValueError("no primary-policy Armijo records found at/after selected start")
    non_primary_armijo = [r for r in records
        if int(r.get("root", -1)) == root and
        int(r.get("newton_iteration", -1)) >= start and
        r.get("direction_kind") != "base_cgls_primary_policy"]
    last = max(armijo_by_iteration)
    expected = list(range(start, last + 1))
    missing_armijo = [it for it in expected if it not in armijo_by_iteration]
    missing_direction = [it for it in expected if it not in direction_by_iteration]
    extra_direction = [it for it in direction_by_iteration if it > last]
    rows = []
    for iteration in expected:
        if iteration not in armijo_by_iteration or iteration not in direction_by_iteration:
            continue
        direction = direction_by_iteration[iteration]
        if int(direction.get("start_newton_inclusive", -1)) != start:
            raise ValueError(f"primary direction marker at {iteration} names a different inclusive start")
        if direction.get("direction_kind") != "truncated_unregularized_CGLS" or \
                direction.get("original_newton_certificate_applies_to_direction") is not False or \
                int(direction.get("iteration_budget", -1)) != iteration_cap:
            raise ValueError(f"primary direction marker at {iteration} has unexpected CGLS/certificate/cap metadata")
        rows.append(summarize_primary_armijo_row(
            direction, armijo_by_iteration[iteration], environment))
    return {
        "root": root, "start_newton_inclusive": start,
        "last_observed_newton_iteration": last,
        "armijo_record_count": len(armijo_by_iteration),
        "primary_direction_marker_count": len(direction_by_iteration),
        "missing_armijo_iterations": missing_armijo,
        "missing_primary_direction_iterations": missing_direction,
        "direction_markers_after_last_armijo_iteration": sorted(extra_direction),
        "unexpected_non_primary_armijo_records": [
            {"newton_iteration": r.get("newton_iteration"), "direction_kind": r.get("direction_kind")}
            for r in non_primary_armijo],
        "per_iteration": rows,
        "complete_observed_primary_trace": bool(
            not missing_armijo and not missing_direction and not extra_direction and
            not non_primary_armijo and len(rows) == len(expected)),
        "classification": "per-environment eligibility is used to identify the actual direction; the global primary-policy trace label alone is not treated as proof that CGLS replaced Newton",
    }


def bind_run(run_dir: Path, source_snapshot: Path, module):
    prediction_path = run_dir / "prediction.json"
    receipt_path = run_dir / "receipt.json"
    stderr_path = run_dir / "stderr.log"
    for p in (prediction_path, receipt_path, stderr_path):
        if not p.is_file():
            raise ValueError(f"run is incomplete; missing {p}")
    pred = json.loads(prediction_path.read_text())
    rec = json.loads(receipt_path.read_text())
    if rec.get("schema") != "numi.cardboard.native-run-receipt.v2":
        raise ValueError(f"{run_dir} has unsupported or missing native-run receipt schema")
    if rec.get("execution_started") is not True:
        raise ValueError(f"{run_dir} receipt does not attest that execution started")
    if rec.get("source_inputs_unchanged") is not True:
        raise ValueError(f"{run_dir} receipt does not attest unchanged inputs")
    if rec.get("changed_inputs") != []:
        raise ValueError(f"{run_dir} receipt lists changed inputs")
    if rec.get("timed_out") is True:
        raise ValueError(f"{run_dir} receipt says the process timed out")
    if not isinstance(rec.get("exit_code"), int) or \
            not isinstance(rec.get("finished_at_utc"), str) or not rec["finished_at_utc"] or \
            not isinstance(rec.get("wall_seconds"), (int, float)) or \
            not np.isfinite(rec["wall_seconds"]) or rec["wall_seconds"] < 0.0 or \
            not isinstance(rec.get("termination_errors"), list) or rec["termination_errors"]:
        raise ValueError(f"{run_dir} receipt does not establish a normally ended process")
    checked = verify_prediction_inputs(module, prediction_path, pred, source_snapshot)
    return pred, rec, stderr_path, checked


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    cgls_binary = args.binary.resolve()
    source_snapshot = args.source_snapshot.resolve()
    for p in (cgls_binary, run_dir / "output/compiled.nmatterpack", run_dir / "output/manifest.json"):
        if not p.is_file():
            raise ValueError(f"required native result input missing: {p}")
    source_module, source_module_path = load_audit_module(source_snapshot)
    pred, rec, stderr_path, input_checks = bind_run(run_dir, source_snapshot, source_module)
    snapshot_header, arrays = parse_snapshot(cgls_binary)
    env_count, N, U = (snapshot_header[k] for k in ("environment_count", "nodes_per_environment", "unknown_count"))
    run_environment = pred.get("environment")
    if not isinstance(run_environment, dict):
        raise ValueError("prediction has no sealed environment map")
    if snapshot_header["iteration_cap"] != int(run_environment.get("NM_FEM_BASE_CGLS_ITERATIONS", -1)):
        raise ValueError("snapshot CGLS cap disagrees with sealed environment setting")
    if snapshot_header["iteration_cap"] not in (32, 64):
        raise ValueError("native CGLS snapshot cap is outside the supported bounded values")

    primary_after = run_environment.get("NM_FEM_BASE_CGLS_PRIMARY_AFTER")
    if not isinstance(primary_after, str) or re.fullmatch(r"0|[1-9][0-9]*", primary_after) is None:
        raise ValueError("prediction lacks a canonical primary-policy Newton start")
    primary_start = int(primary_after)
    if primary_start > 256:
        raise ValueError("primary-policy start exceeds the authored 0..256 range")
    capture_iteration_text = run_environment.get("NM_FEM_BASE_CGLS_CAPTURE_NEWTON")
    if not isinstance(capture_iteration_text, str) or re.fullmatch(r"0|[1-9][0-9]*", capture_iteration_text) is None:
        raise ValueError("prediction has invalid CGLS capture selector")
    iteration = int(capture_iteration_text)
    if iteration != primary_start:
        raise ValueError("CGLS snapshot iteration and primary-policy start differ")
    if primary_start != 48 or snapshot_header["iteration_cap"] != 64:
        raise ValueError("this audit package is bound to the source143 Newton-48, CGLS-64 experiment")
    capture_path = run_environment.get("NM_FEM_BASE_CGLS_CAPTURE_PATH")
    if not isinstance(capture_path, str) or Path(capture_path).resolve() != cgls_binary:
        raise ValueError("CGLS binary path does not match the sealed capture-path environment")

    cgls_captures = parse_markers(stderr_path, "fem_base_cgls_capture=")
    cgls_directions = parse_markers(stderr_path, "fem_base_cgls_direction=")
    selector = run_environment.get("NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR")
    if not isinstance(selector, str) or re.fullmatch(r"(0|[1-9][0-9]*):all", selector) is None:
        raise ValueError("primary policy requires a canonical root:all Cauchy selector")
    expected_root = int(selector.split(":", 1)[0])
    if selector != "1:all":
        raise ValueError("this audit package is bound to control-step/root selector 1:all")
    if run_environment.get("NM_FEM_TRUE_RESIDUAL_ROOT") != str(expected_root) or \
            run_environment.get("NM_FEM_TRUE_RESIDUAL_CERTIFY") != "1":
        raise ValueError("run is not configured for true-residual certification on the selected root")
    if "NM_FEM_TRUE_RESIDUAL_NEWTON" in run_environment or \
            "NM_FEM_TRUE_RESIDUAL_SELECTOR" in run_environment:
        raise ValueError("Newton-specific true-residual selector narrows the required all-Newton certificate scope")
    cgls_enabled = parse_markers(stderr_path, "fem_base_cgls_enabled=")
    cgls_rejected = parse_markers(stderr_path, "fem_base_cgls_rejected=")
    if len(cgls_enabled) != 1 or cgls_enabled[0].get("root") != expected_root or \
            cgls_enabled[0].get("iterations") != snapshot_header["iteration_cap"] or \
            cgls_enabled[0].get("operator_changed") is not False or \
            cgls_enabled[0].get("linear_solve_claim") is not False or cgls_rejected:
        raise ValueError("CGLS enable marker disagrees with the snapshot or bounded policy contract")
    all_newton_enabled = parse_markers(stderr_path, "fem_base_cauchy_all_newton_enabled=")
    if len(all_newton_enabled) != 1 or all_newton_enabled[0].get("root") != expected_root or \
            all_newton_enabled[0].get("scope") != "all_newton_iterations_on_one_control_step" or \
            all_newton_enabled[0].get("maximum_trace_records") != 512 or \
            all_newton_enabled[0].get("physical_operator_changed") is not False:
        raise ValueError("root:all Cauchy/CGLS trace scope was not uniquely enabled as bounded")
    policy_enabled = parse_markers(stderr_path, "fem_base_cgls_primary_policy_enabled=")
    policy_rejected = parse_markers(stderr_path, "fem_base_cgls_primary_policy_rejected=")
    if len(policy_enabled) != 1 or policy_rejected:
        raise ValueError("primary-policy configuration was not uniquely enabled or emitted a rejection marker")
    if policy_enabled[0].get("root") != expected_root or \
            policy_enabled[0].get("start_newton_inclusive") != primary_start or \
            policy_enabled[0].get("selection") != "healthy_base_above_noop_bound_with_original_linear_certificate" or \
            policy_enabled[0].get("armijo_fallback_allowed") is not False or \
            policy_enabled[0].get("operator_changed") is not False:
        raise ValueError("primary-policy enable marker disagrees with the sealed selection contract")
    matching = [m for m in cgls_captures if m.get("newton_iteration") == iteration]
    matching_dir = [m for m in cgls_directions if m.get("newton_iteration") == iteration]
    if len(matching) != 1 or len(matching_dir) != 1:
        raise ValueError("expected exactly one CGLS capture and direction marker for configured Newton")
    cm, dm = matching[0], matching_dir[0]
    root = int(cm.get("root", -1))
    if dm.get("root") != root:
        raise ValueError("CGLS capture and direction marker roots differ")
    if root != expected_root:
        raise ValueError("CGLS root differs from the sealed selector")
    if cm.get("complete") is not True or cm.get("physical_state_changed") is not False:
        raise ValueError("CGLS capture is incomplete or claims physical mutation")
    if int(cm.get("bytes", -1)) != cgls_binary.stat().st_size or int(cm.get("expected_bytes", -1)) != cgls_binary.stat().st_size:
        raise ValueError("CGLS marker byte counts do not match snapshot")
    if dm.get("direction_kind") != "truncated_unregularized_CGLS" or \
            dm.get("original_newton_certificate_applies_to_direction") is not False or \
            dm.get("selection_policy") != "base_cgls_primary_policy" or \
            dm.get("start_newton_inclusive") != primary_start:
        raise ValueError("direction marker has unexpected kind or transfers prior certificate")
    if int(dm.get("iteration_budget", -1)) != snapshot_header["iteration_cap"]:
        raise ValueError("direction marker cap disagrees with snapshot")

    compiled_pack = run_dir / "output/compiled.nmatterpack"
    run_manifest = json.loads((run_dir / "output/manifest.json").read_text())
    run_world = int(run_manifest["compiled_world_fingerprint"])
    run_world_hex = f"{run_world:016x}"
    reference = None
    reference_free_nodes = None
    action_checks: dict[str, Any] = {"available": False,
        "reason": "no independently bound same-world/root/Newton rich operator capture supplied"}
    if args.reference_run_dir is not None:
        ref_dir = args.reference_run_dir.resolve()
        ref_snapshot_pred = json.loads((ref_dir / "prediction.json").read_text())
        ref_snapshot = Path(ref_snapshot_pred["source_snapshot"]).resolve()
        ref_module, ref_module_path = load_audit_module(ref_snapshot)
        ref_pred, ref_rec, ref_stderr, ref_inputs = bind_run(ref_dir, ref_snapshot, ref_module)
        ref_capture = ref_dir / "operator-capture"
        header, nodes, tets, integrity = ref_module.capture_records(ref_capture)
        ref_env = args.reference_environment
        if ref_env is None:
            ref_env = args.operator_environment
        ref_bound = ref_module.bind_inputs(ref_capture, ref_dir / "prediction.json",
            ref_dir / "receipt.json", ref_stderr, ref_env, source_snapshot_override=ref_snapshot)
        ref_audit = ref_module.validate_and_assemble(header, nodes, tets, integrity, ref_env)
        mass, dt2K, A, _, meta = ref_audit
        ref_manifest = json.loads((ref_dir / "output/manifest.json").read_text())
        ref_world_hex = f"{int(ref_manifest['compiled_world_fingerprint']):016x}"
        pack_equal = sha(compiled_pack) == sha(ref_dir / "output/compiled.nmatterpack")
        same_world = run_world_hex == ref_world_hex == header.get("world_fingerprint_hex")
        same_dims = (env_count == int(header["environment_count"]) and
                     N == int(header["node_count_per_environment"]) and
                     int(header["tetrahedron_count_per_environment"]) == int(ref_manifest["mesh"]["tetrahedra"]))
        same_selector = (int(header["root"]) == root and int(header["newton_iteration"]) == iteration)
        ref_capture_bound = ref_bound.get("capture_run_fingerprints_match") is True
        if not (pack_equal and same_world and same_dims and same_selector and ref_capture_bound):
            action_checks = {"available": False,
                "reason": "reference capture binding does not match CGLS compiled world/dimensions/root/Newton",
                "compiled_pack_hash_equal": pack_equal, "world_fingerprint_equal": same_world,
                "dimensions_equal": same_dims, "root_newton_equal": same_selector,
                "capture_run_fingerprints_match": ref_capture_bound}
        else:
            env = args.operator_environment
            if env != args.reference_environment and args.reference_environment is not None:
                raise ValueError("operator and reference environments must match")
            if env >= env_count:
                raise ValueError("reference environment outside native snapshot")
            free_nodes = np.asarray(meta["free_nodes"], dtype=np.int64)
            reference_free_nodes = free_nodes
            rows = env * N + free_nodes
            x = arrays["x"][rows, :3].astype(np.float64).reshape(-1)
            r = arrays["r"][rows, :3].astype(np.float64).reshape(-1)
            b = arrays["b"][rows, :3].astype(np.float64).reshape(-1)
            native_ax = arrays["Ax"][rows, :3].astype(np.float64).reshape(-1)
            native_s = arrays["s"][rows, :3].astype(np.float64).reshape(-1)
            ref_rhs = np.asarray(meta["native_rhs_free_impulse_Ns"], dtype=np.float64).reshape(-1)
            if ref_rhs.shape != b.shape:
                raise ValueError("reference RHS and CGLS free vector shapes differ")
            ref_ax = np.asarray(A @ x).reshape(-1)
            ref_atr = np.asarray(A.T @ r).reshape(-1)
            action_checks = {
                "available": True,
                "classification": "same-byte-compiled-world and same root/Newton rich capture; directional action parity only, not whole-matrix identity",
                "reference_run": str(ref_dir), "reference_capture_sha256": sha(ref_capture),
                "reference_source_module_sha256": sha(ref_module_path),
                "reference_prediction_inputs": ref_inputs,
                "reference_binding_source_snapshot": str(ref_snapshot),
                "compiled_pack_sha256": sha(compiled_pack),
                "compiled_pack_hash_equal": pack_equal,
                "world_fingerprint_hex": run_world_hex,
                "root": root, "newton_iteration": iteration, "environment": env,
                "free_node_count": int(meta["free_node_count"]), "operator_shape": list(A.shape), "operator_nnz": int(A.nnz),
                "native_b_vs_reference_center_rhs": compare(b, ref_rhs),
                "native_b_fp32_words_vs_reference_rhs": compare_float32_words(b, ref_rhs),
                "native_Ax_vs_reference_Ax_on_native_x": compare(native_ax, ref_ax),
                "native_s_vs_reference_ATr_on_native_r": compare(native_s, ref_atr),
                "native_recurrence_r_vs_b_minus_native_Ax": compare(r, b - native_ax),
                "reference_linear_residual_l2_from_native_Ax": float(np.linalg.norm(b - ref_ax)),
                "reference_action_residual_l2_from_native_Ax": float(np.linalg.norm(native_ax - ref_ax)),
                "threshold": None,
            }
            action_checks["native_Ax_existing_assembled_1_percent_gate"] = existing_action_gate(
                action_checks["native_Ax_vs_reference_Ax_on_native_x"])
            action_checks["native_s_existing_assembled_1_percent_gate"] = existing_action_gate(
                action_checks["native_s_vs_reference_ATr_on_native_r"])
            reference = {"capture_sha256": sha(ref_capture), "source_snapshot": str(ref_snapshot),
                         "world_fingerprint_hex": ref_world_hex, "binding": ref_bound["capture_run_fingerprints_match"]}

    baseline_result: dict[str, Any] = {"available": False,
        "reason": "no pretarget baseline run supplied"}
    baseline_records: list[dict] = []
    baseline_directions: list[dict] = []
    if args.baseline_run_dir is not None:
        baseline_dir = args.baseline_run_dir.resolve()
        baseline_pred_path = baseline_dir / "prediction.json"
        baseline_receipt_path = baseline_dir / "receipt.json"
        baseline_stderr_path = baseline_dir / "stderr.log"
        if not all(p.is_file() for p in (baseline_pred_path, baseline_receipt_path, baseline_stderr_path)):
            raise ValueError(f"baseline run is incomplete: {baseline_dir}")
        baseline_pred = json.loads(baseline_pred_path.read_text())
        baseline_snapshot = Path(baseline_pred.get("source_snapshot", "")).resolve()
        baseline_module, baseline_module_path = load_audit_module(baseline_snapshot)
        baseline_pred, baseline_rec, baseline_stderr_path, baseline_input_checks = bind_run(
            baseline_dir, baseline_snapshot, baseline_module)
        baseline_manifest_path = baseline_dir / "output/manifest.json"
        if not baseline_manifest_path.is_file():
            raise ValueError(f"baseline compiled-world manifest missing: {baseline_manifest_path}")
        baseline_manifest = json.loads(baseline_manifest_path.read_text())
        baseline_world_hex = f"{int(baseline_manifest['compiled_world_fingerprint']):016x}"
        baseline_pack_path = baseline_dir / "output/compiled.nmatterpack"
        baseline_records = parse_markers(baseline_stderr_path, "fem_residual_armijo_iteration=")
        baseline_directions = parse_markers(baseline_stderr_path, "fem_base_cgls_direction=")
        baseline_result = {
            "available": True, "run_dir": str(baseline_dir),
            "source_snapshot": str(baseline_snapshot),
            "source_module_sha256": sha(baseline_module_path),
            "prediction_sha256": sha(baseline_dir / "prediction.json"),
            "receipt_sha256": sha(baseline_dir / "receipt.json"),
            "stderr_sha256": sha(baseline_stderr_path),
            "prediction_inputs": baseline_input_checks,
            "run_exit_code": baseline_rec.get("exit_code"),
            "process_finished_at_utc": baseline_rec.get("finished_at_utc"),
            "timed_out": baseline_rec.get("timed_out"),
            "source_inputs_unchanged": baseline_rec.get("source_inputs_unchanged"),
            "compiled_world_fingerprint_hex": baseline_world_hex,
            "compiled_pack_sha256": sha(baseline_pack_path) if baseline_pack_path.is_file() else None,
            "same_compiled_world": baseline_world_hex == run_world_hex,
            "same_compiled_pack": bool(baseline_pack_path.is_file() and sha(baseline_pack_path) == sha(compiled_pack)),
        }

    selected = arrays["selected"]
    if np.any(selected > 1) or int(np.count_nonzero(selected == 1)) != 1:
        raise ValueError("snapshot must select exactly one environment")
    env = int(np.flatnonzero(selected == 1)[0])
    if env != args.operator_environment:
        raise ValueError("selected environment differs from requested operator environment")
    direction_environments = dm.get("environments", [])
    if not isinstance(direction_environments, list) or len(direction_environments) != env_count:
        raise ValueError("primary direction marker environment vector does not match snapshot E")
    direction_environment = direction_environments[env]
    eligible_bits = []
    for expected_environment, item in enumerate(direction_environments):
        if not isinstance(item, dict) or item.get("environment") != expected_environment or \
                not isinstance(item.get("primary_policy_eligible"), bool):
            raise ValueError("primary direction marker has incomplete or misindexed environment eligibility")
        eligible_bits.append(item["primary_policy_eligible"])
    snapshot_selected = arrays["selected"]
    if np.any(snapshot_selected > 1):
        raise ValueError("CGLS snapshot selected mask is outside {0,1}")
    if [bool(v) for v in snapshot_selected.tolist()] != eligible_bits:
        raise ValueError("CGLS snapshot selected-environment mask disagrees with primary eligibility marker")
    expected_env = np.concatenate((np.repeat(np.arange(env_count, dtype=np.uint32), N),
                                   np.repeat(np.arange(env_count, dtype=np.uint32), N)))
    if not np.array_equal(arrays["row_environment"], expected_env):
        raise ValueError("row-environment map is not exact E-by-N mechanical/field layout")
    for name in VECTOR_NAMES:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"CGLS vector {name} contains nonfinite data")
        if np.any(arrays[name][:, 3].view(np.uint32) != 0):
            raise ValueError(f"CGLS vector {name} has nonzero w padding")
    full_mech = env_count * N
    field_rows = np.arange(full_mech, U)
    field_checks = {n: raw_zero(arrays[n][field_rows]) for n in (*VECTOR_NAMES,)}
    if not all(field_checks.values()):
        raise ValueError(f"field backing block is not exact-zero: {field_checks}")
    quiet_checks = {}
    for qenv in range(env_count):
        if qenv == env: continue
        node_rows = np.arange(qenv * N, (qenv + 1) * N)
        rows = np.r_[node_rows, full_mech + node_rows]
        checks = {n: raw_zero(arrays[n][rows]) for n in VECTOR_NAMES}
        checks.update({"errors_zero": bool(arrays["errors"][qenv] == 0),
                       "scalars_zero": raw_zero(arrays["scalars"][qenv]),
                       "control_idle": bool(arrays["control"][qenv, 0] == 0),
                       "selected_zero": bool(arrays["selected"][qenv] == 0)})
        if not all(checks.values()):
            raise ValueError(f"quiet environment {qenv} violates zero/idle contract: {checks}")
        quiet_checks[str(qenv)] = checks

    status, completed, cap, selected_word = (int(v) for v in arrays["control"][env])
    error_bits = int(arrays["errors"][env])
    status_valid = status in (2, 3) and error_bits == 0 and selected_word == 1 and completed <= cap
    mask = arrays["mask"][:full_mech, :3]
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("mechanical free mask is outside {0,1}")
    if np.any(arrays["mask"][full_mech:, :3] != 0.0):
        raise ValueError("field backing mask is nonzero")
    reference_mask_check = {"available": False,
        "reason": "no same-world/root/Newton rich operator capture supplied"}
    if reference_free_nodes is not None:
        expected_mask = np.zeros((N, 3), dtype=np.float32)
        expected_mask[reference_free_nodes, :] = 1.0
        selected_node_mask = arrays["mask"][env * N:(env + 1) * N, :3]
        reference_mask_check = {
            "available": True,
            "classification": "exact selected-environment free-node mask comparison against the bound rich operator capture",
            "free_node_count": int(reference_free_nodes.size),
            "native_free_node_count": int(np.count_nonzero(selected_node_mask[:, 0])),
            "exact_match": bool(np.array_equal(selected_node_mask, expected_mask)),
            "mismatch_node_count": int(np.count_nonzero(np.any(selected_node_mask != expected_mask, axis=1))),
        }

    armijo_records = parse_markers(stderr_path, "fem_residual_armijo_iteration=")
    at_start = [r for r in armijo_records if int(r.get("root", -1)) == root and
                int(r.get("newton_iteration", -1)) == iteration]
    if len(at_start) != 1 or at_start[0].get("direction_kind") != "base_cgls_primary_policy":
        raise ValueError("selected Newton lacks exactly one primary-policy Armijo record")
    selected_policy_trial = summarize_primary_armijo_row(dm, at_start[0], env)
    later_policy_trials = summarize_later_primary_trials(
        armijo_records, cgls_directions, root, iteration, env,
        snapshot_header["iteration_cap"])
    pretarget = summarize_pretarget_trace(
        armijo_records, baseline_records, cgls_directions,
        baseline_directions, root, iteration, env) if baseline_result["available"] else {
        "pretarget_trace_parity_with_inert_fallbacks_discarded": False,
        "classification": "not assessed; baseline run absent"}
    if baseline_result["available"]:
        pretarget["same_compiled_world"] = baseline_result["same_compiled_world"]
        pretarget["same_compiled_pack"] = baseline_result["same_compiled_pack"]
        pretarget["pretarget_trace_parity_with_compiled_identity"] = bool(
            pretarget["pretarget_trace_parity_with_inert_fallbacks_discarded"] and
            baseline_result["same_compiled_world"] and baseline_result["same_compiled_pack"])
    coupled_capture_scope = summarize_coupled_capture_scope(stderr_path, root, iteration)
    report = {
        "schema": "numi.cardboard.native453_primary_cgls_policy_audit.v1",
        "classification": "read-only CGLS snapshot and primary-policy history audit with optional same-world directional operator replay; not a physical solve or solver certificate",
        "run_binding": {"run_dir": str(run_dir), "source_snapshot": str(source_snapshot),
            "prediction_sha256": sha(run_dir / "prediction.json"), "receipt_sha256": sha(run_dir / "receipt.json"),
            "stderr_sha256": sha(stderr_path), "source_module_sha256": sha(source_module_path),
            "prediction_inputs": input_checks, "run_exit_code": rec.get("exit_code"), "timed_out": rec.get("timed_out"),
            "execution_started": rec.get("execution_started"),
            "source_inputs_unchanged": rec.get("source_inputs_unchanged"),
            "changed_inputs": rec.get("changed_inputs"),
            "finished_at_utc": rec.get("finished_at_utc"),
            "wall_seconds": rec.get("wall_seconds"),
            "primary_policy_root": root, "primary_policy_start_newton_inclusive": primary_start,
            "primary_selector": selector,
            "CGLS_iteration_cap": snapshot_header["iteration_cap"],
            "binary_sha256": snapshot_header["sha256"], "binary_bytes": snapshot_header["file_bytes"]},
        "snapshot_header": snapshot_header,
        "native_state": {"selected_environment": env, "status": status,
            "status_name": {2:"exhausted",3:"zero_normal_residual"}.get(status,"invalid_or_breakdown"),
            "completed_iterations": completed, "cap": cap, "error_bits": error_bits,
            "selected_word": selected_word, "status_valid": bool(status_valid),
            "direction_marker_selected_environment": direction_environment,
            "primary_eligible_environment_bits": eligible_bits,
            "scalars_gamma_alpha_beta_bnorm2": arrays["scalars"][env].astype(float).tolist(),
            "x_l2": float(np.linalg.norm(arrays["x"][env*N:(env+1)*N,:3].astype(np.float64))),
            "r_l2": float(np.linalg.norm(arrays["r"][env*N:(env+1)*N,:3].astype(np.float64))),
            "s_l2": float(np.linalg.norm(arrays["s"][env*N:(env+1)*N,:3].astype(np.float64))),
            "p_l2": float(np.linalg.norm(arrays["p"][env*N:(env+1)*N,:3].astype(np.float64))),
            "Ax_l2": float(np.linalg.norm(arrays["Ax"][env*N:(env+1)*N,:3].astype(np.float64))),
            "b_l2": float(np.linalg.norm(arrays["b"][env*N:(env+1)*N,:3].astype(np.float64)))},
        "scope_checks": {"field_backing_exact_zero": field_checks,
            "quiet_environment_exact_zero_idle": quiet_checks,
            "selected_mask_free_row_count": int(np.count_nonzero(
                arrays["mask"][env*N:(env+1)*N, 0])),
            "selected_mask_vs_reference": reference_mask_check,
            "original_newton_certificate_applies_to_CGLS_direction": False},
        "selected_start_armijo": selected_policy_trial,
        "later_primary_policy_trials": later_policy_trials,
        "trace_inventory": {
            "residual_armijo_records_for_selected_root": sum(
                1 for r in armijo_records if int(r.get("root", -1)) == root),
            "CGLS_direction_records_for_selected_root": sum(
                1 for r in cgls_directions if int(r.get("root", -1)) == root),
            "policy_marker_count": len(policy_enabled),
            "all_newton_trace_cap": all_newton_enabled[0].get("maximum_trace_records"),
            "over_cap": sum(1 for r in armijo_records if int(r.get("root", -1)) == root) > 512,
            "classification": "record counts are observed trace coverage; receipt completion does not imply a successful or complete physical coupon"},
        "pretarget_history_comparison": {"baseline": baseline_result, **pretarget},
        "target_rich_operator_capture_scope": coupled_capture_scope,
        "reference_operator": reference,
        "action_comparison": action_checks,
        "limitations": [
            "Any coupled-capture skip is reported from this run's own root/Newton skip marker; an absent marker is not treated as a completed capture.",
            "The 419 CSR replay is checked only on native x and r, with same compiled world, pack, root, Newton, E, and N; this is directional action parity, not a proof of full matrix equality.",
            "All action differences are report-only and have no qualified noise-floor or acceptance threshold.",
            "Primary-policy eligibility identifies environments whose CGLS direction was selected; false eligibility retains the original Newton direction under the policy-scoped trace label.",
            "An accepted residual-Armijo trial is not a true-residual or nonlinear certificate for the selected CGLS direction; no original Newton certificate is transferred.",
            "A complete run receipt only establishes that the process ended without timeout and source-input mutation; a nonzero exit code or failed solver history is not physical success."
        ]}
    report["analyzer_binding"] = {
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha(Path(__file__).resolve()),
        "argv": list(sys.argv),
        "python_version": sys.version,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--source-snapshot", type=Path, required=True)
    ap.add_argument("--operator-environment", type=int, required=True)
    ap.add_argument("--reference-run-dir", type=Path, required=True,
                    help="source-bound 419 run containing the same-world operator-capture JSONL")
    ap.add_argument("--reference-environment", type=int)
    ap.add_argument("--baseline-run-dir", type=Path, required=True,
                    help="source-bound pretarget baseline450 used for residual-Armijo parity before Newton48")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite existing output {out}")
    try:
        report = run(args)
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=False)
        (out / "failure.json").write_text(json.dumps({"schema":"numi.cardboard.native_cgls_snapshot_audit_failure.v2","error":str(exc)},indent=2)+"\n")
        raise
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    manifest = {"files": {p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in out.iterdir()}}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"result":"AUDIT_WRITTEN","output":str(out),
        "status_valid":report["native_state"]["status_valid"],
        "selected_environment":report["native_state"]["selected_environment"],
        "root":report["run_binding"]["primary_policy_root"],
        "newton_iteration":report["run_binding"]["primary_policy_start_newton_inclusive"],
        "reference_operator_available":report["action_comparison"]["available"]},indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
