#!/usr/bin/env python3
"""Bind retained native Armijo traces and selected captures to a health report."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path("/Users/home/cardboard-implementation-20261014")
NATIVE = BASE / "native"
sys.path.insert(0, str(HERE))
from trace_health import (  # noqa: E402
    adjudicate_capture, adjudicate_sequence, parse_armijo_log, read_capture_status,
    sha256_bytes,
)

CASES = [
    {"id": "346", "native": "ccd17-actual-private-trial20-346", "capture": "private-trial.coupled.json",
     "phase": "after_deformable_contact_before_free_body_terms", "kind": "private_trial"},
    {"id": "347", "native": "ccd17-actual-private-capture-off-347", "capture": None,
     "phase": None, "kind": "trace_only"},
    {"id": "348", "native": "ccd17-actual-center-capture-348", "capture": "center-capture.coupled.json",
     "phase": None, "kind": "center"},
    {"id": "333", "native": "center-action-feasibility-bracket-333", "capture": "private-trial.coupled.json",
     "phase": "private_contact_rebuild_before_free_body_terms", "kind": "private_trial"},
    {"id": "334", "native": "center-action-feasibility-bracket-334", "capture": "private-trial.coupled.json",
     "phase": "private_contact_rebuild_before_free_body_terms", "kind": "private_trial"},
]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_case(spec: dict[str, Any], raw_dir: pathlib.Path) -> dict[str, Any]:
    source = NATIVE / spec["native"]
    raw_log = source / "stderr.log"
    prediction_path = source / "prediction.json"
    receipt_path = source / "receipt.json"
    prediction = load_json(prediction_path)
    receipt = load_json(receipt_path)
    log_bytes = raw_log.read_bytes()
    rows = parse_armijo_log(log_bytes)
    sequence = adjudicate_sequence(rows, expected_root=1201, environment_index=0)

    # Retain the complete raw source stream and execution binding in this package.
    (raw_dir / f"{spec['id']}.stderr.log").write_bytes(log_bytes)
    (raw_dir / f"{spec['id']}.prediction.json").write_bytes(prediction_path.read_bytes())
    (raw_dir / f"{spec['id']}.receipt.json").write_bytes(receipt_path.read_bytes())
    capture_info: dict[str, Any] | None = None
    capture_decision: dict[str, Any] | None = None
    if spec["capture"]:
        capture_path = source / spec["capture"]
        capture = load_json(capture_path)
        (raw_dir / f"{spec['id']}.capture.json").write_bytes(capture_path.read_bytes())
        capture_status = None
        if spec["phase"]:
            capture_status = read_capture_status(capture_path, spec["phase"])
        capture_decision = adjudicate_capture(
            sequence, capture, capture_trial_status=capture_status)
        payload = capture.get("payload", {})
        capture_info = {
            "path": str(capture_path),
            "manifest_sha256": sha(capture_path),
            "schema": capture.get("schema"),
            "root": capture.get("root"),
            "iteration": capture.get("iteration"),
            "trial": capture.get("trial"),
            "selected_phase": capture.get("selected_phase"),
            "capture_point": capture.get("capture_point"),
            "array_count": len(capture.get("arrays", [])),
            "payload_path": str(capture_path.parent / pathlib.Path(payload.get("file", "")).name),
            "payload_byte_count": payload.get("byte_count"),
            "payload_sha256": payload.get("sha256"),
            "selected_private_status": capture_status,
        }

    source_snapshot = pathlib.Path(prediction["source_snapshot"])
    frozen_manifest = source_snapshot / "frozen-source.json"
    frozen_source = load_json(frozen_manifest)
    snapshot_key = source_snapshot.name
    relevant_source_paths = [
        "matter/src/metal/fem.metalinc",
        "matter/src/metal/fgmres.metalinc",
        "matter/src/runtime.mm",
        "matter/src/metal/contact.metalinc",
    ]
    relevant_source_hashes = {
        path: frozen_source.get("source_sha256", {}).get(path)
        for path in relevant_source_paths
    }
    if any(value is None for value in relevant_source_hashes.values()):
        raise RuntimeError(f"source manifest lacks audit files: {snapshot_key}")
    (raw_dir / f"{snapshot_key}.frozen-source.json").write_bytes(frozen_manifest.read_bytes())
    build_manifest = pathlib.Path(prediction["build_provenance"])
    if receipt.get("exit_code") != 1 or receipt.get("timed_out") is not False:
        raise RuntimeError(f"unexpected native outcome for {spec['id']}: {receipt}")
    if receipt.get("source_inputs_unchanged") is not True:
        raise RuntimeError(f"native inputs changed during run {spec['id']}")
    relevant_inputs = {
        k: v for k, v in prediction.get("inputs_sha256", {}).items()
        if k in {
            "runtime/matter/numi-matter-cardboard-full-contact-armijo-check",
            "runtime/matter/shaders/NumiMatter.metallib",
            "external/source-build.json",
        }
    }
    selected_rows = [r for r in sequence["rows"] if r["iteration"] in
                     sorted(set([0, max(0, (sequence["first_fatal_iteration"] or 0) - 1),
                                 sequence["first_fatal_iteration"] or 0,
                                 (sequence["first_fatal_iteration"] or 0) + 1,
                                 sequence["record_count"] - 1]))]
    return {
        "case_id": spec["id"],
        "native_directory": str(source),
        "kind": spec["kind"],
        "execution": {
            "exit_code": receipt.get("exit_code"),
            "timed_out": receipt.get("timed_out"),
            "source_inputs_unchanged": receipt.get("source_inputs_unchanged"),
            "wall_seconds": receipt.get("wall_seconds"),
            "prediction_sha256": sha(prediction_path),
            "receipt_sha256": sha(receipt_path),
            "stderr_sha256": sha(raw_log),
            "source_snapshot": str(source_snapshot),
            "frozen_source_manifest_sha256": sha(frozen_manifest),
            "frozen_source_files_sha256": relevant_source_hashes,
            "source_build_manifest": str(build_manifest),
            "source_build_manifest_sha256": sha(build_manifest),
            "executable_and_shader_input_hashes": relevant_inputs,
            "argv": prediction.get("argv"),
            "environment": prediction.get("environment"),
        },
        "trace": {
            **sequence,
            "rows": rows,
            "selected_summary_rows": selected_rows,
        },
        "capture": capture_info,
        "capture_decision": capture_decision,
        "physical_verdict": "failed native transaction; this trace does not qualify a converged physical step",
    }


def main() -> None:
    out = HERE
    raw_dir = out / "raw"
    raw_dir.mkdir(exist_ok=True)
    cases = [build_case(spec, raw_dir) for spec in CASES]
    report = {
        "schema": "numi.cardboard.private_trace_health_report.v1",
        "generated_from": str(NATIVE),
        "adjudicator": "trace_health.py",
        "rules": {
            "sequence_fatal_authority": "solver_status_code written by the Armijo decision; trial_status is non-authoritative for sequence health",
            "base_timing": "row base residual is measured before that row's Armijo decision; first-fatal row base remains a predecision metric",
            "capture_admission": "center/trial comparisons require a selected Newton center before first fatal status and a successful selected private-candidate phase",
            "postfatal": "post-fatal residual rows are incomplete failure-latched metrics and cannot be used as center residuals",
            "native_physical_qualification": "none; all listed runs are retained failures",
        },
        "cases": cases,
    }
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (out / "report.json").write_bytes(encoded)
    (out / "report.sha256").write_text(sha_bytes(encoded) + "  report.json\n")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
