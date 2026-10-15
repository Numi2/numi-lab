#!/usr/bin/env python3
"""Compare the feature-off 453 run with the source144 state-observer run 455."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

BASE = Path("/Users/home/cardboard-implementation-20261014")
RUNS = {
    "453": BASE / "solver/cgls-primary-after48-control1-453",
    "455": BASE / "solver/cgls-state-sidecar-observer-455",
}
TRACE_PREFIXES = (
    "fem_true_residual_cycle=",
    "fem_residual_armijo_iteration=",
    "fem_base_cgls_direction=",
)
TIMING_PREFIX = "cardboard_step_timing="
STATE_MARKER_PREFIX = "fem_base_cgls_state_capture="
STATUS_MARKER_PREFIX = "fem_status5_capture_complete="


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trace_rows(path: Path) -> tuple[dict[str, list[Any]], dict[str, int]]:
    rows = {prefix[:-1]: [] for prefix in TRACE_PREFIXES}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        for prefix in TRACE_PREFIXES:
            if line.startswith(prefix):
                name = prefix[:-1]
                rows[name].append(json.loads(line[len(prefix):]))
                counts[name] = counts.get(name, 0) + 1
                break
    return rows, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    a, b = RUNS["453"], RUNS["455"]
    output_files = sorted(str(p.relative_to(a / "output")) for p in (a / "output").rglob("*") if p.is_file())
    output_files_455 = sorted(str(p.relative_to(b / "output")) for p in (b / "output").rglob("*") if p.is_file())
    if output_files != output_files_455:
        raise ValueError("output file inventories differ")
    equal_outputs = []
    different_outputs = []
    for rel in output_files:
        pa, pb = a / "output" / rel, b / "output" / rel
        if pa.read_bytes() == pb.read_bytes():
            equal_outputs.append(rel)
        else:
            different_outputs.append(rel)
    if different_outputs != ["manifest.json"]:
        raise ValueError(f"unexpected output payload differences: {different_outputs}")
    manifest_a = json.loads((a / "output/manifest.json").read_text())
    manifest_b = json.loads((b / "output/manifest.json").read_text())
    changed_manifest_leafs = {}
    for material in ("liner", "medium", "glue"):
        key = f"{material}_path"
        old, new = manifest_a["materials"][key], manifest_b["materials"][key]
        old_name, new_name = Path(old).name, Path(new).name
        if old_name != new_name:
            raise ValueError(f"material identity changed: {material}")
        changed_manifest_leafs[key] = {"453": old, "455": new, "basename_same": True}
    normalized_a = json.loads(json.dumps(manifest_a))
    normalized_b = json.loads(json.dumps(manifest_b))
    for key in changed_manifest_leafs:
        normalized_a["materials"][key] = Path(normalized_a["materials"][key]).name
        normalized_b["materials"][key] = Path(normalized_b["materials"][key]).name
    if normalized_a != normalized_b:
        raise ValueError("output manifest differs beyond three source-snapshot material paths")

    binary_names = ["base-cgls.bin", "first-status5.bin"]
    binary_parity = {}
    for name in binary_names:
        x, y = a / name, b / name
        binary_parity[name] = {
            "453_sha256": sha(x), "455_sha256": sha(y),
            "bytes_equal": x.read_bytes() == y.read_bytes(),
            "byte_count": x.stat().st_size,
        }
        if not binary_parity[name]["bytes_equal"]:
            raise ValueError(f"physical binary differs: {name}")

    rows_a, counts_a = trace_rows(a / "stderr.log")
    rows_b, counts_b = trace_rows(b / "stderr.log")
    if rows_a != rows_b:
        raise ValueError("production residual, Armijo, or base-CGLS trace records differ")

    stderr_a = (a / "stderr.log").read_text(encoding="utf-8").splitlines()
    stderr_b = (b / "stderr.log").read_text(encoding="utf-8").splitlines()
    sidecar_markers = [line for line in stderr_b if line.startswith(STATE_MARKER_PREFIX)]
    if len(sidecar_markers) != 1:
        raise ValueError("455 must emit exactly one complete state sidecar marker")
    sidecar_meta = json.loads(sidecar_markers[0][len(STATE_MARKER_PREFIX):])
    if sidecar_meta.get("complete") is not True or sidecar_meta.get("array_count") != 24:
        raise ValueError("455 state sidecar completion marker is not complete/24 arrays")

    def normalized_non_timing_lines(lines: list[str], run_dir: Path, remove_state: bool) -> list[str]:
        out = []
        for line in lines:
            if line.startswith(TIMING_PREFIX):
                continue
            if remove_state and line.startswith(STATE_MARKER_PREFIX):
                continue
            line = line.replace(str(run_dir), "<RUN_DIR>")
            line = re.sub(r"/Users/home/numi-cardboard-20261006/build-cardboard-provenance-(?:143|144)", "<BUILD_DIR>", line)
            out.append(line)
        return out

    stderr_norm_a = normalized_non_timing_lines(stderr_a, a, False)
    stderr_norm_b = normalized_non_timing_lines(stderr_b, b, True)
    if stderr_norm_a != stderr_norm_b:
        differing = next((i for i, (x, y) in enumerate(zip(stderr_norm_a, stderr_norm_b)) if x != y), None)
        raise ValueError(f"stderr differs beyond run/build paths, timing, and new sidecar marker; first row {differing}")

    status_a = json.loads((a / "first-status5.json").read_text())
    status_b = json.loads((b / "first-status5.json").read_text())
    status_a["payload"]["path"] = "<STATUS5_PAYLOAD>"
    status_b["payload"]["path"] = "<STATUS5_PAYLOAD>"
    if status_a != status_b:
        raise ValueError("status5 manifest differs beyond its payload path")
    if status_a["payload"]["sha256"] != binary_parity["first-status5.bin"]["453_sha256"]:
        raise ValueError("status5 payload manifest SHA does not bind its binary")

    receipts = {label: json.loads((path / "receipt.json").read_text()) for label, path in RUNS.items()}
    if any(r["exit_code"] != 2 or r["timed_out"] or not r["source_inputs_unchanged"] for r in receipts.values()):
        raise ValueError("run receipts are not both expected failed attempts with unchanged inputs")
    sidecar_validation = json.loads((BASE / "analysis/base-cgls-state-reader-abi-repair-148-001/capture-455-vs-419.json").read_text())

    report = {
        "schema": "numi.matter.cgls_state_observer_parity_453_455.v1",
        "verdict": "PASS_PHYSICAL_HISTORY_PARITY_WITH_DIAGNOSTIC_SOURCE_DELTA",
        "runs": {label: {"path": str(path), "receipt": receipts[label], "stderr_sha256": sha(path / "stderr.log"), "stdout_sha256": sha(path / "stdout.log")} for label, path in RUNS.items()},
        "production_trace_record_counts": {"453": counts_a, "455": counts_b},
        "production_trace_records_exact": True,
        "non_timing_stderr_equal_after_run_build_path_and_new_state_marker_normalization": True,
        "timing_rows": {
            "453_count": sum(line.startswith(TIMING_PREFIX) for line in stderr_a),
            "455_count": sum(line.startswith(TIMING_PREFIX) for line in stderr_b),
            "compared_as_physical_history": False,
        },
        "payloads": binary_parity,
        "output_tree": {
            "file_inventory_equal": True,
            "byte_identical_files": equal_outputs,
            "manifest_only_difference": changed_manifest_leafs,
            "manifest_equal_after_path_basename_normalization": True,
        },
        "status5_manifest_equal_after_payload_path_normalization": True,
        "state_sidecar": {
            "present_only_in_455": True,
            "complete_marker": sidecar_meta,
            "strict_reader_validation": sidecar_validation["reader_validation"],
        },
        "source_build_changes": {
            "source143_runtime_sha256": "01162665b98a7f88d4d6341e2bc6ea50afb9f5d831b04fe446ce27d34e7ad5ec",
            "source144_runtime_sha256": "1a9fcbdda74c921a36b6529033851ae47a460ce51c8373e3794cc8967c5eca7f",
            "base_cgls_byte_identical": True,
            "full_application_binary_identical": False,
            "native_exit_status_is_success": False,
        },
        "limits": [
            "This proves attempted production-history parity for the failed source143/source144 pair, not completion of the coupon.",
            "Source144 adds the state observer; runtime/library/probe binaries are expected to differ and are not called byte-identical.",
            "The sidecar capture is diagnostic and records projected current-Newton history as unavailable.",
            "Runtime timing rows are preserved but excluded from physical-state parity because they measure wall/GPU time.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
