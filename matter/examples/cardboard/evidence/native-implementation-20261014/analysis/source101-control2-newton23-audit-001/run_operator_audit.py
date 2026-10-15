#!/usr/bin/env python3
"""Source-bound offline audit for the native control-2/Newton-23 capture.

This wrapper never edits the native run. It refuses to run until the sealed
prediction, terminal receipt, stderr, and rich operator capture are all present
and agree on source101 plus selector 2:23. It then delegates matrix assembly to
the source-bound auditor shipped in that exact snapshot for each environment.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

EXPECTED_SOURCE = Path(
    "/Users/home/cardboard-implementation-20261014/snapshots/"
    "source-boundary-reset-order-101"
)
EXPECTED_SELECTOR = "2:23"
EXPECTED_CAPTURE_SCHEMA = "numi.cardboard.fem-newton-capture.v1"
AUDITOR_REL = Path("matter/tools/cardboard_cached_operator_audit.py")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--source", type=Path, default=EXPECTED_SOURCE)
    ap.add_argument("--baseline-run", type=Path,
                    default=Path("/Users/home/cardboard-implementation-20261014/solver/reset-order-augmentation-complete-coupon-352"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    source = args.source.resolve()
    require(source == EXPECTED_SOURCE.resolve(), "unexpected source snapshot; this audit is bound to source101")
    prediction = run / "prediction.json"
    receipt = run / "receipt.json"
    stderr = run / "stderr.log"
    capture = run / "operator-capture"
    for path in (prediction, receipt, stderr, capture):
        require(path.is_file() and path.stat().st_size > 0, f"required completed-run artifact is absent/empty: {path}")

    pred = json.loads(prediction.read_text())
    rec = json.loads(receipt.read_text())
    baseline_run = args.baseline_run.resolve()
    baseline_prediction = baseline_run / "prediction.json"
    baseline_manifest = baseline_run / "output/manifest.json"
    baseline_files = [baseline_run / rel for rel in (
        "output/compiled.nmatterpack", "output/initial.obj", "output/mesh.json", "output/manifest.json")]
    require(all(path.is_file() for path in [baseline_prediction, baseline_manifest, *baseline_files]),
            "baseline 352 artifacts are incomplete")
    baseline_pred = json.loads(baseline_prediction.read_text())
    env = pred.get("environment", {})
    require(pred.get("schema") == "numi.cardboard.native-prediction.v1", "unexpected prediction schema")
    require(Path(pred.get("source_snapshot", "")).resolve() == source, "prediction source snapshot differs")
    require(Path(baseline_pred.get("source_snapshot", "")).resolve() == source,
            "baseline run does not use source101")
    require(env.get("NM_FEM_OPERATOR_FD_SELECTOR") == EXPECTED_SELECTOR,
            "prediction does not bind operator capture selector 2:23")
    require(env.get("NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR") == "23" and
            env.get("NM_FEM_TRUE_RESIDUAL_ROOT") == "2",
            "prediction does not bind selected true-residual trace at control 2/Newton 23")
    require(env.get("NM_FEM_RESIDUAL_ARMIJO_ROOT") == "all",
            "physical run did not request residual Armijo on all control steps")
    def argv_without_output(prediction_object: dict) -> list[str]:
        argv = prediction_object.get("argv")
        require(isinstance(argv, list), "prediction has no argument vector")
        normalized = list(argv)
        try:
            index = normalized.index("--output")
        except ValueError as error:
            raise ValueError("prediction argument vector lacks --output") from error
        require(index + 1 < len(normalized), "--output has no path")
        normalized[index + 1] = "<run-output>"
        return normalized
    require(argv_without_output(pred) == argv_without_output(baseline_pred),
            "360 physical arguments differ from baseline 352 beyond the output directory")
    require(pred.get("inputs_sha256") == baseline_pred.get("inputs_sha256"),
            "360 sealed input hashes differ from baseline 352")
    baseline_hashes = {str(path.relative_to(baseline_run)): sha256(path) for path in baseline_files}
    current_hashes = {str(path.relative_to(run)): sha256(run / path.relative_to(baseline_run))
                      for path in baseline_files}
    require(current_hashes == baseline_hashes,
            "360 compiled/mesh/manifest artifacts differ from baseline 352")
    require(rec.get("timed_out") is False and "exit_code" in rec,
            "native receipt is missing terminal completion or timed out")
    if rec.get("prediction_sha256") is not None:
        require(rec["prediction_sha256"] == sha256(prediction), "receipt prediction hash mismatch")
    if rec.get("capture_sha256") is not None:
        require(rec["capture_sha256"] == sha256(capture), "receipt capture hash mismatch")

    # Read only the first JSONL record here; the source auditor validates all
    # records, binary32 round trips, cache validity, and production fingerprints.
    with capture.open() as f:
        header = json.loads(f.readline())
    require(header.get("schema") == EXPECTED_CAPTURE_SCHEMA, "unexpected operator-capture schema")
    require((header.get("root"), header.get("newton_iteration")) == (2, 23),
            "capture is not the requested control-2/Newton-23 center")
    require(header.get("production_inputs_bitwise_unchanged") is True,
            "operator capture did not certify unchanged production inputs")
    require(header.get("cached_algorithmic_tangent_capture", {}).get("snapshot_bitwise_guard_included") is True,
            "capture lacks the tangent-cache bitwise guard")
    require(header.get("environment_count") == 2, "capture environment count differs from source101 coupon")

    stderr_text = stderr.read_text(errors="replace")
    def marker_rows(name: str) -> list[dict]:
        prefix = name + "="
        rows = []
        for line in stderr_text.splitlines():
            if line.startswith(prefix):
                try:
                    rows.append(json.loads(line[len(prefix):]))
                except json.JSONDecodeError as error:
                    raise ValueError(f"malformed {name} marker") from error
        return rows
    fd_enabled = [row for row in marker_rows("fem_operator_fd_enabled")
                  if row.get("root") == 2 and row.get("iteration") == 23]
    fd_complete = [row for row in marker_rows("fem_operator_fd_iteration")
                   if row.get("root") == 2 and row.get("newton_iteration") == 23]
    require(len(fd_enabled) == 1 and len(fd_complete) == 1,
            "stderr lacks exactly one matching completed native operator-FD record")
    true_residual = [row for row in marker_rows("fem_true_residual_cycle")
                     if row.get("root") == 2 and row.get("newton_iteration") == 23]

    spec = importlib.util.spec_from_file_location("cached_operator_audit", auditor := source / AUDITOR_REL)
    require(spec is not None and spec.loader is not None, "cannot load source-bound capture parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    header2, nodes, tets, integrity = module.capture_records(capture)
    require(header2 == header, "capture header changed during validation")
    tet_health = []
    environment_health = []
    for environment in range(2):
        rows = [tet for tet in tets if int(tet.get("environment", -1)) == environment]
        bad_projection = []
        bad_status = []
        missing_tangent = []
        for tet in rows:
            tet_id = int(tet.get("tet_id", -1))
            center = tet.get("center", {})
            projection = center.get("projection")
            terminal = center.get("flags_terminal_status_parameter_count")
            tangent = tet.get("cached_algorithmic_tangent_local")
            if not isinstance(projection, list) or len(projection) != 4 or projection[3] != 1.0:
                bad_projection.append(tet_id)
            if not isinstance(terminal, list) or len(terminal) != 4 or terminal[1] != 0.0:
                bad_status.append(tet_id)
            if not isinstance(tangent, list) or len(tangent) != 81:
                missing_tangent.append(tet_id)
        environment_health.append({
            "environment": environment,
            "tetrahedron_records": len(rows),
            "projection_invalid_tet_ids": bad_projection,
            "nonzero_or_missing_terminal_status_tet_ids": bad_status,
            "missing_tangent_tet_ids": missing_tangent,
            "all_center_records_auditable": not (bad_projection or bad_status or missing_tangent),
        })
    require(auditor.is_file(), "source-bound cached operator auditor is missing")
    args.output.mkdir(parents=True, exist_ok=False)
    health_path = args.output / "captured_material_health.json"
    health_path.write_text(json.dumps({
        "schema": "numi.cardboard.captured-material-health.v1",
        "selection": {"control_step": 2, "newton_iteration": 23},
        "integrity": integrity,
        "environments": environment_health,
        "limits": ["This checks capture fields and native status markers; it is not an independent material replay."],
    }, indent=2, sort_keys=True) + "\n")

    delegated = []
    for environment in (0, 1):
        report_path = args.output / f"environment-{environment}-operator-audit.json"
        command = [sys.executable, str(auditor), "--capture", str(capture),
                   "--prediction", str(prediction), "--receipt", str(receipt),
                   "--stderr", str(stderr), "--source-snapshot", str(source),
                   "--environment", str(environment), "--output", str(report_path)]
        completed = subprocess.run(command, capture_output=True, text=True)
        (args.output / f"environment-{environment}-operator-audit.log").write_text(
            completed.stdout + completed.stderr)
        if completed.returncode != 0:
            delegated.append({
                "environment": environment,
                "status": "blocked_or_failed",
                "exit_code": completed.returncode,
                "diagnostic": (completed.stderr or completed.stdout).strip()[-4000:],
                "captured_health": environment_health[environment],
            })
            continue
        report = json.loads(report_path.read_text())
        selection = report.get("capture_selection", {})
        require(selection == {"root": 2, "newton_iteration": 23, "environment": environment},
                f"auditor returned unexpected selection for environment {environment}")
        delegated.append({
            "environment": environment,
            "report": report_path.name,
            "report_sha256": sha256(report_path),
            "capture_fingerprints_match": report.get("binding", {}).get("capture_run_fingerprints_match"),
            "operator": report.get("operator"),
            "spectral_diagnostics": report.get("spectral_diagnostics"),
        })

    summary = {
        "schema": "numi.cardboard.source101-control2-newton23-audit.v1",
        "verdict": "offline source-bound operator reconstruction only; rejected native step remains rejected",
        "source_snapshot": str(source),
        "source_auditor_sha256": sha256(auditor),
        "prediction_sha256": sha256(prediction),
        "receipt_sha256": sha256(receipt),
        "stderr_sha256": sha256(stderr),
        "capture_sha256": sha256(capture),
        "baseline_352": {
            "run_directory": str(baseline_run),
            "prediction_sha256": sha256(baseline_prediction),
            "artifact_sha256": baseline_hashes,
            "physical_argv_equal_after_output_path_normalization": True,
            "sealed_inputs_equal": True,
        },
        "native_exit_code": rec["exit_code"],
        "selector": EXPECTED_SELECTOR,
        "captured_center": {"control_step": 2, "newton_iteration": 23},
        "environments": delegated,
        "selected_true_residual_cycle_count": len(true_residual),
        "limits": [
            "This rebuilds A from the captured production tangent cache; it does not rerun the native operator kernel.",
            "The comparison is limited to the captured Newton center and the audited FEM mechanical operator.",
            "It does not establish convergence, a corrected nonlinear step, a full contact operator, or physical qualification.",
            "Material/projection diagnosis requires a separate comparison using the exact captured F, Fdot, accepted state, low-F words, and cached tangent for this same capture.",
        ],
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"audit refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
