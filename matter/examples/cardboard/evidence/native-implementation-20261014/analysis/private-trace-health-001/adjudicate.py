#!/usr/bin/env python3
"""CLI for strict Armijo trace / selected-capture health adjudication."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from trace_health import (
    TraceError, adjudicate_capture, adjudicate_sequence,
    parse_armijo_log, read_capture_status,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=pathlib.Path)
    parser.add_argument("--root", type=int)
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--capture", type=pathlib.Path)
    parser.add_argument("--capture-trial-status-phase",
                        default="armijo_measurement_before_decision")
    parser.add_argument("--require-comparable", action="store_true")
    args = parser.parse_args()
    try:
        records = parse_armijo_log(args.log.read_bytes())
        sequence = adjudicate_sequence(
            records, expected_root=args.root, environment_index=args.environment)
        result = {"schema": "numi.matter.armijo_trace_health.v1",
                  "sequence": sequence}
        if args.capture is not None:
            capture_meta = json.loads(args.capture.read_text())
            trial_status = None
            if capture_meta.get("schema") == \
                    "numi.matter.coupled_armijo_trial_capture.v1":
                trial_status = read_capture_status(
                    args.capture, args.capture_trial_status_phase)
            result["capture_adjudication"] = adjudicate_capture(
                sequence, capture_meta, capture_trial_status=trial_status)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"),
                         allow_nan=False))
        if args.require_comparable and args.capture is not None and \
                not result["capture_adjudication"]["center_trial_comparison_allowed"]:
            return 4
        return 0
    except (OSError, TraceError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": "numi.matter.armijo_trace_health.v1",
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
