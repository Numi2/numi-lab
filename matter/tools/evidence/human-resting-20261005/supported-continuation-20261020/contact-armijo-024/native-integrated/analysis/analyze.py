#!/usr/bin/env python3
"""Compact analyzer for the predeclared contact Armijo 024 native trace.

The parser reads the existing runtime JSONL diagnostics from stderr, the
existing invocation/provenance/execution records, and the coupled CSV. It
does not modify the run artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PREFIX = "fem_deformable_limit_trace="
REASON_CODES = {
    1: "ccd_uncertified",
    2: "ccd_shortened_fraction",
    3: "nonfinite_toi",
    4: "hit_not_initial_touch_nonclosing",
    5: "invalid_barrier_energy",
    6: "barrier_armijo_rejected",
    7: "accepted",
}
REPO = Path(__file__).resolve().parents[6]
DEFAULT_RUN = REPO / "build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-contact-armijo-024"
DEFAULT_BASELINE = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/elastic-attachment-022/native-integrated/resting-coupled.csv"
DEFAULT_BASELINE_FAILURE = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/elastic-attachment-022/native-integrated/failure.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_input_files(invocation: dict[str, Any]) -> list[str]:
    out = []
    argv = invocation.get("argv", [])
    for item in argv[1:]:
        if isinstance(item, str) and item.startswith("/") and Path(item).is_file():
            out.append(item)
    return out


def parse_trace(stderr_path: Path, root_id: int, microtick: int, contact_slot: int) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, raw in enumerate(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.startswith(PREFIX):
            continue
        payload = raw[len(PREFIX):]
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        if event.get("schema") != 1 or event.get("root") != root_id or event.get("microtick") != microtick:
            continue
        for contact in event.get("contacts", []):
            if contact.get("contact_slot") != contact_slot:
                continue
            env = contact.get("environment")
            active_counts = event.get("active_counts", [])
            truncated = event.get("truncated", [])
            candidate = {
                "stderr_line": line_number,
                "schema": event.get("schema"),
                "root": event.get("root"),
                "microtick": event.get("microtick"),
                "iteration": event.get("iteration"),
                "stage": event.get("stage"),
                "environment": env,
                "outer_line_search_alpha": (
                    event.get("line_search_alpha", [])[env]
                    if isinstance(env, int) and env >= 0 and env < len(event.get("line_search_alpha", []))
                    else None
                ),
                "active_count": (
                    active_counts[env] if isinstance(env, int) and 0 <= env < len(active_counts) else None
                ),
                "outer_contact_trace_truncated": (
                    truncated[env] if isinstance(env, int) and 0 <= env < len(truncated) else None
                ),
                "contact": contact,
                "per_alpha_trace": contact.get("per_alpha_trace"),
                "trial_trace_capture_summary": event.get("trial_trace_capture_summary"),
            }
            candidates.append(candidate)
    return candidates, errors


def trial_tsv_row(entry: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    base = trial.get("base")
    ccd = trial.get("ccd")
    energy = trial.get("energy")
    contact = entry["contact"]
    return {
        "trace_line": entry["stderr_line"],
        "stage": entry["stage"],
        "iteration": entry["iteration"],
        "environment": entry["environment"],
        "contact_slot": contact.get("contact_slot"),
        "active_slot": contact.get("active_slot"),
        "failure": bool((entry.get("per_alpha_trace") or {}).get("is_failure")),
        "trial_index": trial.get("index"),
        "alpha": ccd[0] if isinstance(ccd, list) and len(ccd) > 0 else None,
        "reason_code": trial.get("reason_code"),
        "reason": REASON_CODES.get(trial.get("reason_code")),
        "ccd_hit": trial.get("ccd_hit"),
        "ccd_certified": trial.get("ccd_certified"),
        "tested_full_fraction": trial.get("tested_full_fraction"),
        "initial_touch_gate_evaluated": trial.get("initial_touch_gate_evaluated"),
        "initial_touch_nonclosing_or_not_required": trial.get("initial_touch_nonclosing_or_not_required"),
        "barrier_evaluated": trial.get("barrier_evaluated"),
        "finite_barrier": trial.get("finite_barrier"),
        "accepted_by_feature_gate": trial.get("accepted_by_feature_gate"),
        "base_energy": base[0] if isinstance(base, list) and len(base) > 0 else None,
        "base_directional_derivative": base[1] if isinstance(base, list) and len(base) > 1 else None,
        "base_start_alpha": base[2] if isinstance(base, list) and len(base) > 2 else None,
        "base_separation": base[3] if isinstance(base, list) and len(base) > 3 else None,
        "trial_toi": ccd[1] if isinstance(ccd, list) and len(ccd) > 1 else None,
        "trial_separation": ccd[2] if isinstance(ccd, list) and len(ccd) > 2 else None,
        "tested_fraction": ccd[3] if isinstance(ccd, list) and len(ccd) > 3 else None,
        "trial_energy": energy[0] if isinstance(energy, list) and len(energy) > 0 else None,
        "trial_energy_separation": energy[1] if isinstance(energy, list) and len(energy) > 1 else None,
        "armijo_rhs": energy[2] if isinstance(energy, list) and len(energy) > 2 else None,
    }


def row_bytes(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines(keepends=True)


def csv_step_values(path: Path, limit: int) -> list[int | None]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            try:
                rows.append(int(row["step"]))
            except (KeyError, TypeError, ValueError):
                rows.append(None)
            if len(rows) == limit:
                break
    return rows


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-failure", type=Path, default=DEFAULT_BASELINE_FAILURE)
    parser.add_argument("--root", type=int, default=20)
    parser.add_argument("--microtick", type=int, default=2)
    parser.add_argument("--contact-slot", type=int, default=1975)
    parser.add_argument("--accepted-rows", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = run_dir / "stderr.log"
    stdout_path = run_dir / "stdout.log"
    invocation_path = run_dir / "invocation.json"
    provenance_path = run_dir / "provenance.json"
    execution_path = run_dir / "execution.json"
    csv_path = run_dir / "output/resting-coupled.csv"

    missing = [str(p) for p in (stderr_path, stdout_path, invocation_path, provenance_path, csv_path) if not p.is_file()]
    if missing:
        print("analysis inputs missing: " + ", ".join(missing), file=sys.stderr)
        return 2
    if not execution_path.is_file():
        print(f"run is not complete; execution.json is absent: {execution_path}", file=sys.stderr)
        return 2
    if not args.baseline_csv.is_file():
        print(f"baseline CSV is missing: {args.baseline_csv}", file=sys.stderr)
        return 2
    if not args.baseline_failure.is_file():
        print(f"baseline failure receipt is missing: {args.baseline_failure}", file=sys.stderr)
        return 2

    invocation = read_json(invocation_path)
    provenance = read_json(provenance_path)
    execution = read_json(execution_path)
    candidates, parse_errors = parse_trace(stderr_path, args.root, args.microtick, args.contact_slot)

    qualifying_failures = []
    for entry in candidates:
        trace = entry.get("per_alpha_trace")
        contact = entry["contact"]
        if (
            entry.get("stage") == "after_deformable_limit"
            and isinstance(trace, dict)
            and trace.get("is_failure") is True
            and contact.get("diagnostic_reason_code") == 6
            and contact.get("diagnostic_reason") == "contact_barrier_armijo_failed"
        ):
            qualifying_failures.append(entry)
    selected_failure = max(qualifying_failures, key=lambda e: (e.get("iteration") or -1, e["stderr_line"])) if qualifying_failures else None

    baseline_lines = row_bytes(args.baseline_csv)
    candidate_lines = row_bytes(csv_path)
    line_count = args.accepted_rows + 1
    baseline_enough = len(baseline_lines) >= line_count
    candidate_enough = len(candidate_lines) >= line_count
    parity = baseline_enough and candidate_enough and baseline_lines[:line_count] == candidate_lines[:line_count]
    mismatch_line = None
    if baseline_enough and candidate_enough and not parity:
        for idx, (expected, actual) in enumerate(zip(baseline_lines[:line_count], candidate_lines[:line_count])):
            if expected != actual:
                mismatch_line = idx  # 0 is header; data rows are 1-based after header.
                break
    parity_rows = {
        "baseline_csv": str(args.baseline_csv.resolve()),
        "candidate_csv": str(csv_path.resolve()),
        "accepted_data_rows_required": args.accepted_rows,
        "baseline_physical_lines": len(baseline_lines),
        "candidate_physical_lines": len(candidate_lines),
        "header_and_first_n_rows_byte_identical": parity,
        "compared_physical_lines": line_count if baseline_enough and candidate_enough else min(len(baseline_lines), len(candidate_lines)),
        "first_differing_physical_line_zero_based": mismatch_line,
        "baseline_first_n_step_values": csv_step_values(args.baseline_csv, args.accepted_rows),
        "candidate_first_n_step_values": csv_step_values(csv_path, args.accepted_rows),
    }

    trial_rows: list[dict[str, Any]] = []
    selected_record = None
    capture_consistent = False
    if selected_failure is not None:
        per_alpha = selected_failure["per_alpha_trace"]
        trials = per_alpha.get("trials", [])
        indices = [t.get("index") for t in trials]
        unique_indices = len(indices) == len(set(indices))
        indices_sorted = indices == sorted(indices)
        trial_summary = selected_failure.get("trial_trace_capture_summary") or {}
        capture_consistent = (
            trial_summary.get("contact_slot") == args.contact_slot
            and trial_summary.get("trial_records_attached") == len(trials)
            and unique_indices
            and indices_sorted
        )
        for trial in trials:
            trial_rows.append(trial_tsv_row(selected_failure, trial))
        selected_record = selected_failure

    summary = {
        "schema": "numi.human.contact_armijo_024_analysis.v1",
        "analysis_status": "complete" if selected_failure and not parse_errors and capture_consistent else "incomplete_or_invalid",
        "scope": "diagnostic capture and exact accepted-CSV prefix comparison; no physical repair or qualification",
        "target": {"root": args.root, "microtick": args.microtick, "contact_slot": args.contact_slot, "required_failure_stage": "after_deformable_limit", "required_diagnostic_reason_code": 6},
        "run": {
            "run_dir": str(run_dir),
            "execution": execution,
            "invocation_sha256": sha256(invocation_path),
            "provenance_sha256": sha256(provenance_path),
            "source_base_revision": provenance.get("source_base_revision"),
            "executable_sha256": provenance.get("executable_sha256"),
            "binary_sha256_from_invocation": invocation.get("binary_sha256"),
            "stderr_path": str(stderr_path),
            "stderr_sha256": sha256(stderr_path),
            "stdout_path": str(stdout_path),
            "stdout_sha256": sha256(stdout_path),
            "file_input_count": len(source_input_files(invocation)),
            "file_inputs": source_input_files(invocation),
        },
        "trace": {
            "matching_contact_records": len(candidates),
            "qualifying_failure_records": len(qualifying_failures),
            "selected_failure_iteration": selected_failure.get("iteration") if selected_failure else None,
            "selected_failure_stage": selected_failure.get("stage") if selected_failure else None,
            "selected_failure_diagnostic": (
                {"code": selected_failure["contact"].get("diagnostic_reason_code"),
                 "reason": selected_failure["contact"].get("diagnostic_reason")}
                if selected_failure else None
            ),
            "trial_rows_attached": len(trial_rows),
            "trial_capture_consistent": capture_consistent,
            "parse_errors": parse_errors,
        },
        "csv_prefix_parity": parity_rows,
        "failure_receipt_reference": None,
        "limitations": [
            "The captured alpha sequence describes one selected dense contact slot at one root/microtick; it does not establish a general contact correction.",
            "A byte-identical accepted CSV prefix is a replay/control check, not proof of collision validity or sustained support.",
        ],
    }

    # If the invocation points to an existing common failure receipt, bind its hash and JSON value.
    env = invocation.get("environment", {})
    failure_path = env.get("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT") if isinstance(env, dict) else None
    candidate_failure = None
    if isinstance(failure_path, str) and Path(failure_path).is_file():
        fp = Path(failure_path)
        try:
            candidate_failure = read_json(fp)
        except (OSError, ValueError, json.JSONDecodeError):
            candidate_failure = None
        summary["failure_receipt_reference"] = {
            "path": str(fp),
            "sha256": sha256(fp),
            "parsed": candidate_failure,
        }

    baseline_failure = read_json(args.baseline_failure)
    failure_fields = ["segment_start", "matter_code", "matter_object_index", "matter_failing_index", "matter_microsteps"]
    expected_failure = {key: baseline_failure.get(key) for key in failure_fields}
    actual_failure = {key: candidate_failure.get(key) for key in failure_fields} if candidate_failure else None
    failure_fields_match = actual_failure == expected_failure
    trace_matches_failure_slot = bool(
        selected_failure
        and candidate_failure
        and selected_failure["contact"].get("contact_slot") == candidate_failure.get("matter_failing_index")
        and selected_failure["contact"].get("diagnostic_reason_code") == candidate_failure.get("matter_code")
        and selected_failure.get("microtick") == candidate_failure.get("matter_microsteps")
    )
    failure_reproduction = {
        "baseline_receipt": str(args.baseline_failure.resolve()),
        "baseline_sha256": sha256(args.baseline_failure),
        "compared_fields": failure_fields,
        "expected": expected_failure,
        "actual": actual_failure,
        "fields_match": failure_fields_match,
        "selected_trace_matches_slot_code_microtick": trace_matches_failure_slot,
    }
    summary["failure_reproduction_022"] = failure_reproduction
    summary["analysis_status"] = (
        "complete"
        if selected_failure and not parse_errors and capture_consistent and failure_fields_match and trace_matches_failure_slot
        else "incomplete_or_invalid"
    )

    atomic_json(output_dir / "summary.json", summary)
    selected_payload = {
        "schema": "numi.human.contact_armijo_024_selected_trace.v1",
        "stderr_path": str(stderr_path),
        "stderr_sha256": summary["run"]["stderr_sha256"],
        "target": summary["target"],
        "matching_contact_records": candidates,
        "selected_final_failure": selected_record,
        "trial_reason_codes": {str(k): v for k, v in REASON_CODES.items()},
    }
    atomic_json(output_dir / "selected-traces.json", selected_payload)

    fields = [
        "trace_line", "stage", "iteration", "environment", "contact_slot", "active_slot",
        "failure", "trial_index", "alpha", "reason_code", "reason", "ccd_hit",
        "ccd_certified", "tested_full_fraction", "initial_touch_gate_evaluated",
        "initial_touch_nonclosing_or_not_required", "barrier_evaluated", "finite_barrier",
        "accepted_by_feature_gate", "base_energy", "base_directional_derivative",
        "base_start_alpha", "base_separation", "trial_toi", "trial_separation",
        "tested_fraction", "trial_energy", "trial_energy_separation", "armijo_rhs",
    ]
    tsv_path = output_dir / "selected-traces.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(trial_rows)

    print(json.dumps({
        "analysis_status": summary["analysis_status"],
        "trace_matches": len(candidates),
        "qualifying_failures": len(qualifying_failures),
        "selected_trial_rows": len(trial_rows),
        "csv_first_n_parity": parity,
        "summary": str(output_dir / "summary.json"),
        "selected_traces": str(output_dir / "selected-traces.json"),
        "tsv": str(tsv_path),
    }, indent=2))
    if parse_errors or selected_failure is None or not capture_consistent or not failure_fields_match or not trace_matches_failure_slot:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

