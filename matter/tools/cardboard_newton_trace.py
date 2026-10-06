#!/usr/bin/env python3
"""Decode the opt-in, bounded Matter FEM Newton trace from stderr."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ITERATE_PREFIX = "fem_newton_iterate="
TERMINAL_PREFIX = "fem_newton_terminal_certificate="
SKIP_PREFIXES = ("fem_newton_trace_skipped=", "fem_newton_certificate_skipped=")
ERROR_PREFIXES = ("fem_newton_trace_error=", "fem_newton_certificate_error=")
STAGES = (
    "before_contact_limits",
    "after_deformable_limit",
    "after_rigid_limit",
    "final_line_search",
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def parse_trace(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return parsed records and explicit errors; malformed trace lines fail closed."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    prefixes = (ITERATE_PREFIX, TERMINAL_PREFIX, *SKIP_PREFIXES, *ERROR_PREFIXES)
    for line_number, line in enumerate(text.splitlines(), 1):
        prefix = next((item for item in prefixes if item in line), None)
        if prefix is None:
            continue
        raw = line.split(prefix, 1)[1].strip()
        try:
            value = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")))
            if not isinstance(value, dict):
                raise ValueError("record must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append({"line": line_number, "prefix": prefix, "reason": str(exc)})
            continue
        kind = "iterate" if prefix == ITERATE_PREFIX else (
            "terminal" if prefix == TERMINAL_PREFIX else
            "skipped" if prefix in SKIP_PREFIXES else "error"
        )
        records.append({"kind": kind, "line": line_number, "payload": value})
    return records, errors


def _env_entry(payload: dict[str, Any], key: str, environment: int) -> dict[str, Any] | list[Any] | None:
    values = payload.get(key)
    if not isinstance(values, list) or environment >= len(values):
        return None
    entry = values[environment]
    return entry if isinstance(entry, (dict, list)) else None


def _object_entry(payload: dict[str, Any], key: str, environment: int,
                  object_index: int = 0) -> dict[str, Any] | list[Any] | None:
    values = payload.get(key)
    object_count = payload.get("object_count")
    if not isinstance(values, list) or not isinstance(object_count, int) or object_count <= 0:
        return None
    index = environment * object_count + object_index
    if index >= len(values):
        return None
    entry = values[index]
    return entry if isinstance(entry, (dict, list)) else None


def _vector(entry: Any, size: int) -> list[float | None] | None:
    if not isinstance(entry, list) or len(entry) < size:
        return None
    return [_finite_number(value) for value in entry[:size]]


def _close(a: float | None, b: float | None) -> bool:
    # Four stage copies read the same immutable FGMRES state. Identical finite
    # floats print identically with the runtime's 9-digit precision.
    return a is not None and b is not None and a == b


def analyze_trace(
    text: str,
    *,
    environment: int = 0,
    expected_root: int | None = None,
    near_full_alpha: float = 0.9,
    growth_limit: float = 1.1,
) -> dict[str, Any]:
    """Analyze capture-local residual trends; never assert final step acceptance."""
    if environment < 0:
        raise ValueError("environment must be nonnegative")
    if not math.isfinite(near_full_alpha) or not 0.0 < near_full_alpha <= 1.0:
        raise ValueError("near_full_alpha must be finite and in (0, 1]")
    if not math.isfinite(growth_limit) or growth_limit <= 1.0:
        raise ValueError("growth_limit must be finite and greater than 1")

    records, parse_errors = parse_trace(text)
    skips = [record["payload"] for record in records if record["kind"] == "skipped"]
    errors = [record["payload"] for record in records if record["kind"] == "error"]
    iterate_records = [record["payload"] for record in records if record["kind"] == "iterate"]
    terminal_records = [record["payload"] for record in records if record["kind"] == "terminal"]
    selection_mismatches: list[dict[str, Any]] = []
    if expected_root is not None:
        selection_mismatches = [
            {"reason": "record_root_does_not_match_requested_root", "record_root": payload.get("root"), "requested_root": expected_root}
            for payload in (*iterate_records, *terminal_records)
            if payload.get("root") != expected_root
        ]
        iterate_records = [r for r in iterate_records if r.get("root") == expected_root]
        terminal_records = [r for r in terminal_records if r.get("root") == expected_root]
        skips = [r for r in skips if r.get("root") in (None, expected_root)]
        errors = [r for r in errors if r.get("root") in (None, expected_root)]

    result: dict[str, Any] = {
        "schema": "cardboard.fem-newton-trace-analysis.v1",
        "environment": environment,
        "contract": {
            "alpha_boundary": "environment line-search scalar at each captured stage",
            "residual_alignment": "iteration i post-step residual is iteration i+1 nonlinear.y; terminal certificate is used only for the last captured iteration",
            "arnoldi": "diagnostics.x is a final-restart least-squares estimate normalized by diagnostics.y, which retains the Newton-start norm; it is not an independently re-evaluated b-Ax residual",
            "cycle_status": "diagnostics.z is cycle_accepted, not proof that the full nonlinear residual converged",
            "work_count": "status.fgmres_iterations_microstep_max is the max totalUsed observed in the microstep, not per-Newton work",
            "terminal_status": "raw status.code at the certificate capture is authoritative for failure; certificate.validity.w cannot override it; code zero here does not prove later commit acceptance",
            "capture_order": "terminal certificate is captured after nm_mixed_certify and before candidate masks or commit",
        },
        "parse_errors": parse_errors,
        "trace_errors": [*errors, *selection_mismatches],
        "skips": skips,
        "iterations": [],
        "screen": {"near_full_alpha": near_full_alpha, "growth_limit": growth_limit, "observations": []},
        "terminal": None,
        "integrity": "invalid" if parse_errors or errors or selection_mismatches else "ok",
    }
    if skips and not parse_errors and not errors and not iterate_records and not terminal_records:
        result["integrity"] = "skipped"
        return result

    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_stages: list[dict[str, Any]] = []
    root_values: set[int] = set()
    for payload in iterate_records:
        if payload.get("schema") != 1:
            result["trace_errors"].append({"reason": "unsupported_iterate_schema"})
            result["integrity"] = "invalid"
            continue
        root = payload.get("root")
        iteration = payload.get("iteration")
        microtick = payload.get("microtick")
        stage = payload.get("stage")
        if not isinstance(root, int) or not isinstance(iteration, int) or not isinstance(microtick, int) or stage not in STAGES:
            result["trace_errors"].append({"reason": "invalid_iterate_identity"})
            result["integrity"] = "invalid"
            continue
        root_values.add(root)
        env_count = payload.get("environment_count")
        if not isinstance(env_count, int) or environment >= env_count:
            result["trace_errors"].append({"reason": "requested_environment_unavailable", "iteration": iteration})
            result["integrity"] = "invalid"
            continue
        key = (root, iteration)
        if stage in groups[key]:
            duplicate_stages.append({"root": root, "iteration": iteration, "stage": stage})
        groups[key][stage] = payload
    if duplicate_stages:
        result["trace_errors"].append({"reason": "duplicate_stage", "items": duplicate_stages})
        result["integrity"] = "invalid"

    parsed_iterations: list[dict[str, Any]] = []
    for (root, iteration), stages in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        stage_details: dict[str, Any] = {}
        y_values: list[float | None] = []
        for stage in STAGES:
            payload = stages.get(stage)
            if payload is None:
                continue
            fgmres = _env_entry(payload, "fgmres", environment)
            status = _env_entry(payload, "status", environment)
            raw_alpha = payload.get("alpha")
            alpha_vector = _vector(raw_alpha, len(raw_alpha) if isinstance(raw_alpha, list) else 0)
            diagnostics = _vector(fgmres.get("diagnostics") if isinstance(fgmres, dict) else None, 4)
            nonlinear = _vector(fgmres.get("nonlinear") if isinstance(fgmres, dict) else None, 4)
            status_code = status.get("code") if isinstance(status, dict) else None
            object_count = payload.get("object_count")
            object_alpha = None
            if isinstance(object_count, int) and object_count > 0:
                flat = payload.get("object_line_search")
                index = environment * object_count
                if isinstance(flat, list) and index < len(flat):
                    object_alpha_vector = _vector(flat[index], 4)
                    object_alpha = object_alpha_vector[0] if object_alpha_vector else None
            if (alpha_vector is None or len(alpha_vector) <= environment or
                diagnostics is None or nonlinear is None or
                not isinstance(status_code, int) or
                not isinstance(object_count, int) or object_count <= 0 or
                object_alpha is None):
                result["trace_errors"].append({
                    "reason": "missing_or_malformed_stage_measurement",
                    "root": root, "iteration": iteration, "stage": stage,
                })
                result["integrity"] = "invalid"
            stage_details[stage] = {
                "environment_alpha": alpha_vector[environment] if alpha_vector and environment < len(alpha_vector) else None,
                "first_object_alpha": object_alpha,
                "pre_step_residual_norm": nonlinear[1] if nonlinear else None,
                "first_newton_residual_norm": nonlinear[0] if nonlinear else None,
                "last_correction_ratio": nonlinear[2] if nonlinear else None,
                "nonlinear_converged": bool(nonlinear and nonlinear[3] is not None and nonlinear[3] > 0.5),
                "arnoldi_estimate": diagnostics[0] if diagnostics else None,
                "arnoldi_normalizer": diagnostics[1] if diagnostics else None,
                "arnoldi_estimate_ratio": (diagnostics[0] / diagnostics[1]) if diagnostics and diagnostics[0] is not None and diagnostics[1] not in (None, 0.0) else None,
                "fgmres_cycle_accepted": bool(diagnostics and diagnostics[2] is not None and diagnostics[2] > 0.5),
                "fgmres_final_restart_columns": diagnostics[3] if diagnostics else None,
                "status_code_at_stage": status_code,
                "fgmres_iterations_microstep_max": status.get("fgmres_iterations_microstep_max") if isinstance(status, dict) else None,
            }
            y_values.append(nonlinear[1] if nonlinear else None)
        present = [stage for stage in STAGES if stage in stages]
        pre_norm = stage_details.get("before_contact_limits", {}).get("pre_step_residual_norm")
        if pre_norm is None:
            pre_norm = next((stage_details[s].get("pre_step_residual_norm") for s in STAGES if stage_details.get(s, {}).get("pre_step_residual_norm") is not None), None)
        for candidate in y_values:
            if candidate is not None and pre_norm is not None and not _close(pre_norm, candidate):
                result["trace_errors"].append({"reason": "residual_norm_changed_between_line_search_stages", "root": root, "iteration": iteration})
                result["integrity"] = "invalid"
                break
        final = stage_details.get("final_line_search", {})
        parsed_iterations.append({
            "root": root,
            "microtick": next(iter(stages.values())).get("microtick"),
            "iteration": iteration,
            "captured_stages": present,
            "missing_stages": [stage for stage in STAGES if stage not in stages],
            "stage_details": stage_details,
            "pre_step_residual_norm": pre_norm,
            "final_alpha": final.get("environment_alpha"),
            "first_object_alpha": final.get("first_object_alpha"),
            "final_cycle_accepted": final.get("fgmres_cycle_accepted"),
            "final_arnoldi_estimate_ratio": final.get("arnoldi_estimate_ratio"),
            "final_restart_columns_diagnostic_only": final.get("fgmres_final_restart_columns"),
            "final_status_code_at_line_search": final.get("status_code_at_stage"),
        })

    # Correctly align each chosen alpha with the next Newton assembly, never its own start norm.
    for index, item in enumerate(parsed_iterations):
        next_item = parsed_iterations[index + 1] if index + 1 < len(parsed_iterations) else None
        next_norm = None
        next_source = None
        if next_item is not None and next_item["root"] == item["root"] and next_item["microtick"] == item["microtick"] and next_item["iteration"] == item["iteration"] + 1:
            next_norm = next_item["pre_step_residual_norm"]
            next_source = "next_iteration_reassembled_nonlinear_y"
        elif terminal_records and index == len(parsed_iterations) - 1:
            terminals = [r for r in terminal_records if r.get("root") == item["root"] and r.get("microtick") == item["microtick"]]
            if terminals:
                terminal = terminals[-1]
                certificate = _object_entry(terminal, "objects", environment)
                fgmres = _env_entry(terminal, "fgmres", environment)
                cert_vector = _vector(certificate.get("nonlinear") if isinstance(certificate, dict) else None, 4)
                diag = _vector(fgmres.get("diagnostics") if isinstance(fgmres, dict) else None, 4)
                if cert_vector and cert_vector[0] is not None and diag and diag[1] is not None:
                    next_norm = cert_vector[0] * max(diag[1], 1.0)
                    next_source = "terminal_certificate_relative_residual_rescaled_by_fgmres_normalizer"
        next_status = None
        if next_item is not None and next_item["root"] == item["root"] and next_item["iteration"] == item["iteration"] + 1:
            next_status = next_item.get("stage_details", {}).get("before_contact_limits", {}).get("status_code_at_stage")
            if next_status is None:
                next_status = next_item.get("final_status_code_at_line_search")
        elif terminal_records and index == len(parsed_iterations) - 1:
            terminals = [r for r in terminal_records if r.get("root") == item["root"] and r.get("microtick") == item["microtick"]]
            if terminals:
                terminal_status = _env_entry(terminals[-1], "status", environment)
                next_status = terminal_status.get("code") if isinstance(terminal_status, dict) else None
        item["next_residual_status_code"] = next_status
        item["post_step_residual_norm"] = next_norm
        item["post_step_residual_source"] = next_source
        pre = item["pre_step_residual_norm"]
        item["residual_growth_ratio"] = next_norm / pre if next_norm is not None and pre not in (None, 0.0) else None
        alpha = item["final_alpha"]
        status_clear = item["final_status_code_at_line_search"] == 0
        item["near_full_alpha"] = alpha is not None and alpha >= near_full_alpha
        item["residual_grew"] = item["residual_growth_ratio"] is not None and item["residual_growth_ratio"] > growth_limit
        item["descriptive_screen_candidate"] = bool(
            item["near_full_alpha"] and item["residual_grew"] and
            item["final_cycle_accepted"] is True and status_clear and
            item["next_residual_status_code"] == 0 and
            pre is not None and pre > 0.0
        )
        if item["descriptive_screen_candidate"]:
            result["screen"]["observations"].append({
                "iteration": item["iteration"],
                "alpha": alpha,
                "residual_before": pre,
                "residual_after": next_norm,
                "growth_ratio": item["residual_growth_ratio"],
                "cycle_accepted": True,
                "next_status_code": item["next_residual_status_code"],
                "arnoldi_estimate_ratio_not_true_residual": item["final_arnoldi_estimate_ratio"],
                "interpretation": "descriptive candidate only; Arnoldi estimate is not a true b-Ax check or a causal explanation",
            })
    result["iterations"] = parsed_iterations
    if len(root_values) > 1:
        result["trace_errors"].append({"reason": "multiple_root_selectors_in_trace", "roots": sorted(root_values)})
        result["integrity"] = "invalid"
    for root in sorted(root_values):
        seen = sorted(item["iteration"] for item in parsed_iterations if item["root"] == root)
        if seen and seen != list(range(seen[-1] + 1)):
            result["trace_errors"].append({"reason": "noncontiguous_newton_iterations", "root": root, "seen": seen})

    terminals = [r for r in terminal_records if expected_root is None or r.get("root") == expected_root]
    if len(terminals) > 1:
        result["trace_errors"].append({"reason": "duplicate_terminal_certificates", "count": len(terminals)})
    if terminals:
        terminal = terminals[-1]
        status = _env_entry(terminal, "status", environment)
        certificate = _object_entry(terminal, "objects", environment)
        fgmres = _env_entry(terminal, "fgmres", environment)
        code = status.get("code") if isinstance(status, dict) else None
        cert_vector = _vector(certificate.get("nonlinear") if isinstance(certificate, dict) else None, 4)
        validity = _vector(certificate.get("validity") if isinstance(certificate, dict) else None, 4)
        diag = _vector(fgmres.get("diagnostics") if isinstance(fgmres, dict) else None, 4)
        nonlinear = _vector(fgmres.get("nonlinear") if isinstance(fgmres, dict) else None, 4)
        if code is None:
            disposition = "invalid_terminal_status"
        elif code != 0:
            disposition = "rejected_at_certificate_status"
        else:
            disposition = "status_clear_at_certificate_capture_not_final_commit"
        final_abs = cert_vector[0] * max(diag[1], 1.0) if cert_vector and cert_vector[0] is not None and diag and diag[1] is not None else None
        pre = nonlinear[1] if nonlinear else None
        result["terminal"] = {
            "disposition": disposition,
            "status_code_at_certificate": code,
            "completed_microsteps_at_certificate": status.get("completed_microsteps_at_certificate") if isinstance(status, dict) else None,
            "certificate_relative_residual": cert_vector[0] if cert_vector else None,
            "certificate_residual_absolute_equivalent": final_abs,
            "pre_step_residual_norm_from_last_fgmres": pre,
            "last_step_residual_growth_ratio": final_abs / pre if final_abs is not None and pre not in (None, 0.0) else None,
            "certificate_validity_w_informational_only": validity[3] if validity else None,
            "capture_order": terminal.get("capture_order"),
            "authority_note": "A nonzero raw status code rejects this capture even if validity.w is 1. A zero code here is before candidate masks and commit, so it is not final accepted-state evidence.",
        }
        if (terminal.get("schema") != 1 or not isinstance(code, int) or
            cert_vector is None or validity is None or diag is None or nonlinear is None):
            result["trace_errors"].append({"reason": "missing_or_malformed_terminal_measurement"})
            result["integrity"] = "invalid"
        matching_iterations = [item for item in parsed_iterations
                              if item["root"] == terminal.get("root") and
                              item["microtick"] == terminal.get("microtick")]
        last_seen = max((item["iteration"] for item in matching_iterations), default=None)
        if (terminal.get("microtick") != 0 or
            terminal.get("root") not in root_values or
            terminal.get("last_iteration") != last_seen):
            result["trace_errors"].append({
                "reason": "terminal_identity_mismatch",
                "root": terminal.get("root"),
                "microtick": terminal.get("microtick"),
                "last_iteration": terminal.get("last_iteration"),
                "last_iterate_seen": last_seen,
            })
            result["integrity"] = "invalid"
        if not matching_iterations:
            result["trace_errors"].append({"reason": "terminal_without_aligned_newton_trace"})
            result["integrity"] = "incomplete"
    elif parsed_iterations:
        result["terminal"] = {"disposition": "missing_terminal_certificate", "status_code_at_certificate": None}
    if result["trace_errors"] or parse_errors or errors:
        result["integrity"] = "invalid"
    elif not parsed_iterations and result["terminal"] is None and not skips:
        result["integrity"] = "empty"
    elif parsed_iterations and (result["terminal"] is None or
                                result["terminal"].get("disposition") == "missing_terminal_certificate" or
                                any(item["missing_stages"] for item in parsed_iterations)):
        result["integrity"] = "incomplete"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="runtime stderr log")
    parser.add_argument("--output", type=Path, help="write JSON analysis (exclusive create)")
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--root", type=int)
    parser.add_argument("--near-full-alpha", type=float, default=0.9)
    parser.add_argument("--growth-limit", type=float, default=1.1)
    args = parser.parse_args(argv)
    try:
        input_bytes = args.input.read_bytes()
        text = input_bytes.decode("utf-8")
        result = analyze_trace(text, environment=args.environment, expected_root=args.root,
                               near_full_alpha=args.near_full_alpha, growth_limit=args.growth_limit)
        result["evidence_binding"] = {
            "input_log_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "decoder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as stream:
                stream.write(serialized)
        else:
            sys.stdout.write(serialized)
        return 0 if result["integrity"] == "ok" else 2
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
