#!/usr/bin/env python3
"""Strict adjudication for Matter Armijo traces and selected captures.

The solver status written by nm_fem_armijo_decide is authoritative for the
Newton/control sequence. The private trial status is candidate-local: a
rejected candidate can have a nonzero status while the Newton remains healthy,
and trace records can retain an unrelated private status after acceptance.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import struct
from typing import Any


MARKER = b"fem_residual_armijo_iteration="
STATUS_BYTES = 48


class TraceError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json(data: str | bytes, where: str) -> Any:
    def reject_constant(value: str) -> None:
        raise TraceError(f"{where}: non-JSON numeric token {value}")

    try:
        return json.loads(data, parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TraceError(f"{where}: invalid JSON: {exc}") from exc


def parse_armijo_log(raw: bytes) -> list[dict[str, Any]]:
    """Parse every Armijo record in a raw stderr/stdout byte stream."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        position = line.find(MARKER)
        if position < 0:
            continue
        if line.find(MARKER, position + len(MARKER)) >= 0:
            raise TraceError(f"line {line_number}: duplicate Armijo marker")
        payload = line[position + len(MARKER):].strip()
        row = _strict_json(payload, f"line {line_number}")
        if not isinstance(row, dict):
            raise TraceError(f"line {line_number}: Armijo row is not an object")
        row["_source_line"] = line_number
        records.append(row)
    if not records:
        raise TraceError("no Armijo records found")
    return records


def adjudicate_sequence(
    records: list[dict[str, Any]], *, expected_root: int | None = None,
    environment_index: int = 0,
) -> dict[str, Any]:
    """Validate ordering and find the first fatal post-decision solver status.

    Trial status is retained in each row but never establishes a fatal Newton
    status. A row's base residual was assembled before that row's Armijo
    decision; therefore the first-fatal row can have a healthy base while its
    selected/private candidate is rejected.
    """
    if not records:
        raise TraceError("empty Armijo sequence")
    roots = {r.get("root") for r in records}
    if len(roots) != 1 or None in roots:
        raise TraceError("Armijo rows have missing or mixed roots")
    root = next(iter(roots))
    if isinstance(root, bool) or not isinstance(root, int):
        raise TraceError("Armijo root must be an integer")
    if expected_root is not None and root != expected_root:
        raise TraceError(f"root mismatch: trace={root}, expected={expected_root}")

    iterations: list[int] = []
    for row in records:
        iteration = row.get("newton_iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise TraceError("Newton iteration must be an integer")
        iterations.append(iteration)
        if row.get("schema") != 1:
            raise TraceError(f"unsupported Armijo row schema at iteration {iteration}")
    if iterations != list(range(iterations[0], iterations[0] + len(iterations))):
        raise TraceError("Newton records are missing, duplicated, or reordered")
    if iterations[0] != 0:
        raise TraceError("Armijo sequence does not begin at Newton iteration 0")

    rows: list[dict[str, Any]] = []
    first_fatal: int | None = None
    previous_solver_code = 0
    for row in records:
        iteration = row["newton_iteration"]
        envs = row.get("environments")
        if not isinstance(envs, list) or environment_index >= len(envs):
            raise TraceError(f"missing environment {environment_index} at iteration {iteration}")
        env = envs[environment_index]
        if not isinstance(env, dict):
            raise TraceError(f"environment entry malformed at iteration {iteration}")
        solver_code = env.get("solver_status_code")
        solver = env.get("solver_status")
        if isinstance(solver_code, bool) or not isinstance(solver_code, int):
            raise TraceError(f"missing solver_status_code at iteration {iteration}")
        if not isinstance(solver, dict) or solver.get("code") != solver_code:
            raise TraceError(f"solver status/code disagreement at iteration {iteration}")
        trial = env.get("trial_status")
        if (not isinstance(trial, dict) or isinstance(trial.get("code"), bool)
                or not isinstance(trial.get("code"), int)):
            raise TraceError(f"missing trial status at iteration {iteration}")
        base = env.get("base_l2")
        finite_base = isinstance(base, (int, float)) and math.isfinite(base)
        prior_fatal = first_fatal is not None
        base_valid = not prior_fatal and previous_solver_code == 0 and finite_base
        if first_fatal is None and solver_code != 0:
            first_fatal = iteration
        rows.append({
            "iteration": iteration,
            "base_l2": base,
            "base_l2_is_finite": finite_base,
            "base_is_valid_center": base_valid,
            "decision_solver_status_code": solver_code,
            "trial_status_code_non_authoritative": trial["code"],
            "trial_status_failing_index": trial.get("failing_index"),
            "trial_status_diagnostics": trial.get("diagnostics"),
            "accepted": env.get("accepted"),
            "selected_alpha": env.get("selected_alpha"),
            "environment_line_search_alpha": env.get("environment_line_search_alpha"),
            "classification": (
                "post_fatal_incomplete_metric" if prior_fatal else
                "first_fatal_decision_base_precedes_failure" if solver_code != 0 else
                "healthy_newton_record"
            ),
        })
        previous_solver_code = solver_code

    return {
        "root": root,
        "environment_index": environment_index,
        "record_count": len(records),
        "iteration_sequence": iterations,
        "first_fatal_iteration": first_fatal,
        "rows": rows,
    }


def read_capture_status(manifest_path: pathlib.Path, phase: str) -> dict[str, Any]:
    """Read NMMatterStatusGPU from a verified adjacent private-trial payload."""
    manifest_path = pathlib.Path(manifest_path)
    meta = _strict_json(manifest_path.read_bytes(), str(manifest_path))
    if meta.get("schema") != "numi.matter.coupled_armijo_trial_capture.v1":
        raise TraceError("unsupported private-trial manifest schema")
    payload_meta = meta.get("payload")
    if not isinstance(payload_meta, dict):
        raise TraceError("private-trial manifest has no payload descriptor")
    payload_name = pathlib.Path(str(payload_meta.get("file", ""))).name
    if not payload_name:
        raise TraceError("private-trial payload basename is missing")
    payload_path = manifest_path.parent / payload_name
    raw = payload_path.read_bytes()
    if len(raw) != payload_meta.get("byte_count"):
        raise TraceError("private-trial payload byte count mismatch")
    if sha256_bytes(raw) != payload_meta.get("sha256"):
        raise TraceError("private-trial payload hash mismatch")
    matches = [
        a for a in meta.get("arrays", [])
        if a.get("name") == "fem_armijo_statuses" and a.get("phase") == phase
    ]
    if len(matches) != 1:
        raise TraceError(f"expected one fem_armijo_statuses array for phase {phase}")
    arr = matches[0]
    if arr.get("dtype") != "record:NMMatterStatusGPU" or arr.get("element_bytes") != STATUS_BYTES:
        raise TraceError("unexpected NMMatterStatusGPU descriptor")
    offset = arr.get("offset")
    count = arr.get("count")
    byte_count = arr.get("byte_count")
    if (isinstance(offset, bool) or not isinstance(offset, int)
            or isinstance(count, bool) or not isinstance(count, int)
            or count < 1 or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count != count * STATUS_BYTES
            or offset < 0
            or offset + byte_count > len(raw)):
        raise TraceError("NMMatterStatusGPU extent outside payload")
    words = struct.unpack_from("<8I", raw, offset)
    diagnostics = struct.unpack_from("<4f", raw, offset + 32)
    return {
        "phase": phase,
        "code": words[0],
        "environment": words[1],
        "object_index": words[2],
        "failing_index": words[3],
        "completed_microsteps": words[4],
        "fgmres_iterations": words[5],
        "contact_count": words[6],
        "event_count": words[7],
        "diagnostics": list(diagnostics),
        "payload_sha256": payload_meta["sha256"],
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "payload_file": payload_name,
    }


def adjudicate_capture(
    sequence: dict[str, Any], capture_meta: dict[str, Any], *,
    capture_trial_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether a center/private-trial residual comparison is admissible."""
    root = capture_meta.get("root")
    iteration = capture_meta.get("iteration")
    if root != sequence["root"]:
        raise TraceError("capture root does not match Armijo trace")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TraceError("capture iteration is missing or malformed")
    if iteration not in sequence["iteration_sequence"]:
        raise TraceError("capture Newton iteration is absent from Armijo trace")
    fatal = sequence["first_fatal_iteration"]
    row = sequence["rows"][iteration]
    if fatal is None:
        post_fatal = False
        at_fatal = False
    else:
        post_fatal = iteration > fatal
        at_fatal = iteration == fatal

    comparison_allowed = True
    reasons: list[str] = []
    if post_fatal:
        comparison_allowed = False
        reasons.append("selected_newton_is_after_first_fatal_status")
    if at_fatal:
        comparison_allowed = False
        reasons.append("selected_newton_is_first_fatal_iteration; its trial is not an accepted residual pair")
    if not row["base_is_valid_center"]:
        comparison_allowed = False
        if "selected_newton_is_after_first_fatal_status" not in reasons:
            reasons.append("center_base_residual_is_not_valid")
    if capture_trial_status is not None and capture_trial_status.get("code") != 0:
        comparison_allowed = False
        reasons.append("selected_private_trial_failed_candidate_feasibility")
    return {
        "root": root,
        "selected_newton_iteration": iteration,
        "first_fatal_iteration": fatal,
        "center_base_valid": row["base_is_valid_center"],
        "capture_trial_status_code": (
            None if capture_trial_status is None else capture_trial_status.get("code")
        ),
        "center_trial_comparison_allowed": comparison_allowed,
        "reasons": reasons,
        "decision": "admissible_for_comparison" if comparison_allowed else "reject_comparison",
    }
