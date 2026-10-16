#!/usr/bin/env python3
"""Strict CPU-only comparison of CGLS diagnostic-on/off resumed segments.

This consumes retained native runs and the separately sealed 497 cached-operator
replay. It does not execute or alter the solver.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

B = Path("/Users/home/cardboard-implementation-20261014")
CASES = {
    "482": B / "solver/macmini-warm-cgls-control20-capture-off-482",
    "497": B / "solver/macmini-warm-cgls-control20-diagnostic-replay-497",
    "498": B / "solver/macmini-warm-cgls-control20-diagnostic-replay-off-498",
}
PRIOR = B / "analysis/native497-cached-cgls-audit-001"
OUT = B / "analysis/native497-cached-cgls-audit-002"
PHYSICAL_METADATA = {
    "manifest.json",
    "result.json",
    "coupon-resume-segment.json",
}
TIMING_PREFIX = "cardboard_step_timing="
DIAGNOSTIC_PREFIXES = (
    "fem_base_cgls_capture=",
    "fem_base_cgls_state_capture=",
    "fem_operator_fd_enabled=",
    "fem_operator_fd_iteration=",
    "fem_regional_rap_capture_skipped=",
    "fem_residual_armijo_trace_enabled=",
    "fem_residual_armijo_iteration=",
)
TRACE_PREFIXES = (
    "fem_true_residual_cycle=",
    "fem_base_cgls_direction=",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def output_map(case: Path) -> dict[str, Path]:
    root = case / "output"
    paths = sorted(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise AssertionError(f"symlink in native output tree: {case}")
    result = {}
    for p in paths:
        if p.is_file():
            result[p.relative_to(root).as_posix()] = p
    return result


def file_hash_map(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha(path) for name, path in paths.items()}


def normalized_log(case: Path) -> tuple[list[bytes], dict[str, int], dict[str, list[bytes]]]:
    kept: list[bytes] = []
    counts: dict[str, int] = {p[:-1]: 0 for p in TRACE_PREFIXES}
    traces: dict[str, list[bytes]] = {p[:-1]: [] for p in TRACE_PREFIXES}
    raw = (case / "stderr.log").read_bytes().splitlines()
    for line in raw:
        if line.startswith(TIMING_PREFIX.encode("ascii")):
            continue
        if any(line.startswith(p.encode("ascii")) for p in DIAGNOSTIC_PREFIXES):
            continue
        for prefix in TRACE_PREFIXES:
            bprefix = prefix.encode("ascii")
            if line.startswith(bprefix):
                key = prefix[:-1]
                counts[key] += 1
                traces[key].append(line)
        kept.append(line)
    return kept, counts, traces


def json_diff(a: Any, b: Any, path: str = "$") -> list[str]:
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}")
            else:
                out.extend(json_diff(a[k], b[k], f"{path}.{k}"))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}.length"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(json_diff(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [path]


def command_without_case_paths(prediction: dict[str, Any]) -> list[str]:
    argv = list(prediction["argv"])
    output_index = argv.index("--output") + 1
    argv[output_index] = "<CASE_OUTPUT>"
    # The source/build executable and metallib paths are provenance, not physical
    # arguments. Keep the full original argv in the report and compare remaining
    # solver arguments separately.
    return argv


def normalized_command(prediction: dict[str, Any]) -> list[str]:
    argv = list(prediction["argv"])
    if argv:
        argv[0] = "<PROBE>"
    path_flags = {
        "--output", "--metallib", "--liner-material", "--medium-material",
        "--glue-material", "--coupon-resume-checkpoint", "--coupon-resume-metadata",
    }
    i = 0
    while i < len(argv):
        if argv[i] in path_flags and i + 1 < len(argv):
            argv[i + 1] = f"<{argv[i][2:].upper()}_PATH>"
            i += 2
        else:
            i += 1
    return argv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "report-v4.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite sealed report: {report_path}")
    run_data: dict[str, dict[str, Any]] = {}
    output_maps: dict[str, dict[str, Path]] = {}
    hashes: dict[str, dict[str, str]] = {}
    normalized_logs: dict[str, list[bytes]] = {}
    traces: dict[str, dict[str, list[bytes]]] = {}
    line_counts: dict[str, dict[str, int]] = {}

    for label, case in CASES.items():
        if not case.is_dir():
            raise AssertionError(f"missing run directory {case}")
        receipt = read_json(case / "receipt.json")
        prediction = read_json(case / "prediction.json")
        native_env = read_json(case / "native-environment.json")
        captures = read_json(case / "diagnostic-captures.json")
        result = read_json(case / "output/result.json")
        segment = read_json(case / "output/coupon-resume-segment.json")
        csv_rows = list(csv.DictReader((case / "output/observations.csv").open(newline="")))
        paths = output_map(case)
        output_maps[label] = paths
        hashes[label] = file_hash_map(paths)
        normalized, counts, parsed_traces = normalized_log(case)
        normalized_logs[label] = normalized
        line_counts[label] = counts
        traces[label] = parsed_traces
        run_data[label] = {
            "case_directory": str(case),
            "receipt": receipt,
            "prediction": prediction,
            "native_environment": native_env,
            "diagnostic_captures": captures,
            "result": result,
            "segment": segment,
            "csv_rows": csv_rows,
            "output_sha256": hashes[label],
            "stderr_line_count": len((case / "stderr.log").read_bytes().splitlines()),
            "normalized_stderr_lines": len(normalized),
            "normalized_stderr_sha256": hashlib.sha256(b"\n".join(normalized)).hexdigest(),
            "trace_counts": counts,
            "stdout_sha256": sha(case / "stdout.log"),
            "stderr_sha256": sha(case / "stderr.log"),
        }
        if not (receipt.get("execution_started") and receipt.get("source_inputs_unchanged")
                and receipt.get("exit_code") == 2 and not receipt.get("timed_out")
                and not receipt.get("changed_inputs") and not receipt.get("termination_errors")):
            raise AssertionError(f"run receipt not clean expected failure: {label}")
        if result.get("status") != "resumed_diagnostic_segment_failed" or result.get("accepted_steps") != 0:
            raise AssertionError(f"unexpected segment result: {label}")
        if segment.get("segment_start_control_index") != 20 or segment.get("segment_accepted_controls") != 0:
            raise AssertionError(f"unexpected resume segment cursor: {label}")
        if len(csv_rows) != 2:
            raise AssertionError(f"expected two environment observations, got {len(csv_rows)} in {label}")

    expected_paths = set(output_maps["482"])
    if any(set(m) != expected_paths for m in output_maps.values()):
        raise AssertionError("output file path inventories differ")
    differing = {}
    for label in ("497", "498"):
        differing[label] = [name for name in sorted(expected_paths)
                            if hashes[label][name] != hashes["482"][name]]
    for label in ("482", "497", "498"):
        physical_names = expected_paths - PHYSICAL_METADATA
        if len(physical_names) != 35:
            raise AssertionError(f"expected 35 production/physical outputs, got {len(physical_names)}")
        if any(hashes[label][name] != hashes["482"][name] for name in physical_names):
            raise AssertionError(f"production/physical outputs differ for {label}")
    if differing != {
        "497": ["coupon-resume-segment.json"],
        "498": ["coupon-resume-segment.json"],
    }:
        raise AssertionError(f"unexpected bytewise output differences: {differing}")

    for pair in (("497", "498"), ("482", "498")):
        a, b = pair
        if normalized_logs[a] != normalized_logs[b]:
            raise AssertionError(f"normalized production log differs: {pair}")
        if run_data[a]["stdout_sha256"] != run_data[b]["stdout_sha256"]:
            raise AssertionError(f"stdout differs: {pair}")
        if traces[a] != traces[b]:
            raise AssertionError(f"retained true-residual/CGLS traces differ: {pair}")

    trace_summary = {}
    for label in CASES:
        trace_summary[label] = {}
        for kind, rows in traces[label].items():
            trace_summary[label][kind] = {
                "count": len(rows),
                "sha256": hashlib.sha256(b"\n".join(rows)).hexdigest(),
            }
    if line_counts["497"] != {"fem_true_residual_cycle": 4096, "fem_base_cgls_direction": 64}:
        raise AssertionError(f"unexpected retained native trace counts: {line_counts['497']}")

    # Source/protocol comparison is deliberately separate from byte-parity.
    source_versions: dict[str, Any] = {}
    source_maps: dict[str, dict[str, str]] = {}
    for label, case in CASES.items():
        pred = run_data[label]["prediction"]
        snapshot_name = Path(pred["source_snapshot"]).name
        frozen = B / "snapshots" / snapshot_name / "frozen-source.json"
        if not frozen.is_file():
            raise AssertionError(f"local frozen-source binding unavailable for {label}: {frozen}")
        frozen_doc = read_json(frozen)
        source_maps[label] = frozen_doc["source_sha256"]
        source_versions[label] = {
            "remote_snapshot": pred["source_snapshot"],
            "local_snapshot": str(frozen.parent),
            "frozen_source_json_sha256": sha(frozen),
            "source_sha256_map_sha256": hashlib.sha256(
                json.dumps(frozen_doc["source_sha256"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_file_count": len(frozen_doc["source_sha256"]),
        }
    if source_versions["497"]["frozen_source_json_sha256"] != source_versions["498"]["frozen_source_json_sha256"]:
        raise AssertionError("497/498 source snapshot mismatch")
    source_file_maps_differ = source_versions["482"]["source_sha256_map_sha256"] != source_versions["497"]["source_sha256_map_sha256"]
    source_changed_paths = [p for p in sorted(set(source_maps["482"]) | set(source_maps["497"]))
                            if source_maps["482"].get(p) != source_maps["497"].get(p)]

    # Read-only extension of sealed report-v2: derive the scales without altering
    # or rerunning its captured FP32-cache reconstruction.
    prior_report_path = PRIOR / "report-v2.json"
    prior_report = read_json(prior_report_path)
    env0 = prior_report["per_environment_operator_replay"][0]
    mn = env0["mass_stiffness_separation"]
    v = env0["vector_norms"]
    nv = env0["native_vector_comparisons"]
    x0 = mn["x0"]
    total = mn["total_X"]
    bnorm = v["original_b_l2"]
    r0 = v["r0_fp64_l2"]
    rfinal = v["r_final_fp64_l2"]
    native_r_difference = nv["captured_R_vs_FP64_b_minus_A_total_X"]["difference_l2"]
    native_r = nv["captured_R_vs_FP64_b_minus_A_total_X"]["actual_l2"]
    native_fresh_scalar = env0["CGLS_state"]["native_direction_trace"]["linear_residual_l2"]
    metrics = {
        "env0_units": "native-operator action and residual are impulse-like units as described by prior audit; offline norms inherit exact capture convention",
        "initial_guess_x0": {
            "b_l2": bnorm,
            "M_x0_l2": x0["M_x0_l2"],
            "dt2K_x0_l2": x0["dt2_K_x0_l2"],
            "A_x0_l2": x0["A_x0_l2"],
            "b_minus_Ax0_l2_fp64": r0,
            "term_norm_sum_over_A_action": (x0["M_x0_l2"] + x0["dt2_K_x0_l2"]) / x0["A_x0_l2"],
            "residual_over_b": r0 / bnorm,
            "residual_over_b_plus_Ax0": r0 / (bnorm + x0["A_x0_l2"]),
            "residual_over_term_norm_sum": r0 / (x0["M_x0_l2"] + x0["dt2_K_x0_l2"]),
        },
        "final_total_candidate_X": {
            "M_X_l2": total["M_X_l2"],
            "dt2K_X_l2": total["dt2_K_X_l2"],
            "A_X_l2": total["A_X_l2"],
            "b_minus_AX_l2_fp64": rfinal,
            "term_norm_sum_over_A_action": (total["M_X_l2"] + total["dt2_K_X_l2"]) / total["A_X_l2"],
            "residual_over_b": rfinal / bnorm,
            "residual_over_b_plus_AX": rfinal / (bnorm + total["A_X_l2"]),
            "residual_over_term_norm_sum": rfinal / (total["M_X_l2"] + total["dt2_K_X_l2"]),
        },
        "CGLS_total_delta": {
            "M_delta_l2": mn["delta"]["M_delta_l2"],
            "dt2K_delta_l2": mn["delta"]["dt2_K_delta_l2"],
            "A_delta_l2": mn["delta"]["A_delta_l2"],
            "M_over_dt2K_norm": mn["delta"]["M_delta_l2"] / mn["delta"]["dt2_K_delta_l2"],
        },
        "native_residual_comparison": {
            "native_captured_recurrence_R_l2": native_r,
            "FP64_reconstructed_b_minus_AX_l2": rfinal,
            "vector_difference_l2": native_r_difference,
            "vector_difference_over_base_r0": native_r_difference / r0,
            "vector_difference_over_rhs_b": native_r_difference / bnorm,
            "vector_difference_over_final_FP64_residual": native_r_difference / rfinal,
            "native_R_over_FP64_residual": native_r / rfinal,
            "fresh_native_linear_residual_scalar": native_fresh_scalar,
            "fresh_native_scalar_minus_FP64_residual": native_fresh_scalar - rfinal,
            "fresh_native_scalar_relative_error_vs_FP64": abs(native_fresh_scalar - rfinal) / rfinal,
            "captured_Q_vs_FP64_AX_relative_error": nv["captured_Q_vs_FP64_A_total_X"]["relative_to_reconstructed"],
            "captured_B_vs_FP64_b_minus_Ax0_relative_error": nv["captured_B_vs_FP64_b_minus_Ax0"]["relative_to_reconstructed"],
        },
        "conditioning_context_from_prior_audit": {
            "condition_estimate_status": "no condition estimate or direct solve for this exact 497 matrix is reported in the sealed replay; do not infer conditioning from residual cancellation alone",
            "source_report_keys": sorted(prior_report.keys()),
        },
    }

    # Verify the exact intended environment delta for 497/498.  Paths vary with
    # case directory, while observer enable/selector values are the intended
    # read-only instrumentation toggle.
    env497 = run_data["497"]["native_environment"]["contract"]["effective"]
    env498 = run_data["498"]["native_environment"]["contract"]["effective"]
    env_diff = {k: [env497.get(k), env498.get(k)] for k in sorted(set(env497) | set(env498)) if env497.get(k) != env498.get(k)}
    expected_env_keys = {
        "NM_FEM_BASE_CGLS_CAPTURE_NEWTON",
        "NM_FEM_BASE_CGLS_CAPTURE_PATH",
        "NM_FEM_BASE_CGLS_STATE_CAPTURE",
        "NM_FEM_OPERATOR_FD_CAPTURE_PATH",
        "NM_FEM_OPERATOR_FD_SELECTOR",
        "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT",
    }
    if set(env_diff) != expected_env_keys:
        raise AssertionError(f"unexpected 497/498 effective environment delta: {sorted(env_diff)}")
    argv497 = command_without_case_paths(run_data["497"]["prediction"])
    argv498 = command_without_case_paths(run_data["498"]["prediction"])
    if argv497 != argv498:
        raise AssertionError("497/498 native physical argv differ beyond output path")

    observer_env_keys = expected_env_keys
    contact_sort_contract = {}
    normalized_effective = {
        label: {} for label in CASES
    }
    for label in CASES:
        effective = run_data[label]["native_environment"]["contract"]["effective"]
        contact_sort_contract[label] = effective.get("NM_CONTACT_NODE_INCIDENCE_SORT")
        if contact_sort_contract[label] not in (None, {"set": False, "value": None}):
            raise AssertionError(f"contact incidence sort is unexpectedly enabled: {label}")
        normalized_effective[label] = {
            k: v for k, v in effective.items()
            if k not in observer_env_keys and k != "NM_CONTACT_NODE_INCIDENCE_SORT"
        }
    if normalized_effective["482"] != normalized_effective["497"] or normalized_effective["482"] != normalized_effective["498"]:
        raise AssertionError("solver-affecting effective native environment controls differ after observer-only keys are removed")

    cmd482 = normalized_command(run_data["482"]["prediction"])
    cmd497 = normalized_command(run_data["497"]["prediction"])
    added: list[str] = []
    removed: list[str] = []
    if cmd482 != cmd497:
        def set_delta(a: list[str], b: list[str]) -> list[str]:
            return [x for x in b if x not in a]
        added = set_delta(cmd482, cmd497)
        removed = set_delta(cmd497, cmd482)
        expected_add = ["--coupon-resume-diagnostic-replay", "--glue-material", "<GLUE-MATERIAL_PATH>"]
        if sorted(added) != sorted(expected_add) or removed:
            raise AssertionError(f"unexpected normalized command delta 482→497: added={added}, removed={removed}")

    # Classify physical segments exactly; do not call a failed resumed segment a
    # converged full-cycle simulation.
    rows = run_data["497"]["csv_rows"]
    result_classification = {
        "segment_status": run_data["497"]["result"]["status"],
        "accepted_steps": run_data["497"]["result"]["accepted_steps"],
        "observations": [
            {k: row.get(k) for k in ("step", "time_s", "environment", "arm", "target_angle_deg", "status_code", "object_index", "failing_index", "completed_microsteps", "fgmres_iterations")}
            for row in rows
        ],
        "trajectory_complete": False,
        "physical_qualification": False,
    }

    report = {
        "schema": "numi.cardboard.native497_498_482_comparison.v1",
        "verdict": "PASS_DIAGNOSTIC_ON_OFF_PHYSICAL_HISTORY_PARITY; 482_OBSERVATIONAL_PARITY_ONLY_ACROSS_SOURCE_REVISION",
        "execution": "CPU-only retained-artifact audit; no native run, GPU dispatch, source edit, or tolerance change",
        "receipt_summary": {k: {"exit_code": v["receipt"]["exit_code"], "timed_out": v["receipt"]["timed_out"], "source_inputs_unchanged": v["receipt"]["source_inputs_unchanged"], "wall_seconds": v["receipt"]["wall_seconds"]} for k, v in run_data.items()},
        "output_comparison": {
            "output_path_count_each": len(expected_paths),
            "exactly_equal_paths_count_vs_482": {k: sum(hashes[k][p] == hashes["482"][p] for p in expected_paths) for k in ("482", "497", "498")},
            "differing_paths_vs_482": differing,
            "production_physical_file_count": 35,
            "production_physical_files_byte_identical_all_three": True,
            "csv_hashes": {k: sha(CASES[k] / "output/observations.csv") for k in CASES},
            "stdout_hashes": {k: run_data[k]["stdout_sha256"] for k in CASES},
            "output_files": {k: hashes[k] for k in CASES},
            "coupon_segment_metadata_differences_497_498": json_diff(run_data["497"]["segment"], run_data["498"]["segment"]),
        },
        "log_trace_parity": {
            "normalized_stream_definition": {
                "drop_exact_prefixes": [TIMING_PREFIX, *DIAGNOSTIC_PREFIXES],
                "reason": "timing is nondeterministic; observer-only capture lines are excluded, while ordinary diagnostics and all production records remain",
            },
            "normalized_line_count": {k: len(normalized_logs[k]) for k in CASES},
            "normalized_stream_sha256": {k: hashlib.sha256(b"\n".join(normalized_logs[k])).hexdigest() for k in CASES},
            "normalized_streams_byte_identical": True,
            "trace_counts": line_counts,
            "trace_hashes": trace_summary,
            "retained_true_residual_and_CGLS_records_exact": True,
            "observer_trace_counts_497": {
                "fem_residual_armijo_iteration": 112,
                "fem_operator_fd_iteration": 1,
                "additional_capture_marker_lines": 6,
            },
        },
        "protocol_and_scope": {
            "source_versions": source_versions,
            "482_source_map_differs_from_497_498": source_file_maps_differ,
            "482_to_497_source_changed_paths": source_changed_paths,
            "497_498_same_frozen_source": True,
            "497_498_effective_environment_delta": env_diff,
            "497_498_physical_argv_same_after_output_path": True,
            "482_source_revision_caveat": "482 uses source173; 497/498 use source182. Byte-identical outputs/logs are direct observed parity, not a controlled attribution of any source change.",
            "normalized_solver_environment_equal_after_observer_controls_removed": True,
            "newly_recorded_unset_incidence_sort_contract": contact_sort_contract,
            "normalized_command_delta_482_to_497": {
                "added_source182_resume_contract_tokens": added,
                "removed_tokens": removed,
                "interpretation": "Explicit glue-material binding and diagnostic-resume flag are recorded protocol differences; no outcome is attributed to them because exact output/history parity is observational.",
            },
            "registered_read_only_capture_paths": run_data["497"]["native_environment"]["contract"].get("capture_prefixes", {}),
            "497_receipt_bound_diagnostic_files": run_data["497"]["receipt"]["diagnostic_capture_validation"].get("files", []),
        },
        "segment_outcome": result_classification,
        "cached_operator_residual_analysis": metrics,
        "cached_operator_replay_binding": {
            "prior_report": str(prior_report_path),
            "prior_report_sha256": sha(prior_report_path),
            "source_binding": prior_report["source_binding"],
            "capture_identity": prior_report["capture_identity"],
            "scope": "selected environment 0; captured FP32 tangent cache, reconstructed with offline FP64 sparse arithmetic",
        },
        "conditioning_arithmetic_interpretation": {
            "finding": "The captured x0→X action is dominated by stiffness; the residual is formed by subtracting two actions/RHS of much larger norm. The native recurrence residual vector differs from offline FP64 replay by about 2.34% of the initial residual norm and 18.44% of the final FP64 residual, although the scalar freshly recomputed residual is within about 0.194% of the FP64 norm.",
            "distinguishers": [
                "Observed cancellation is established by the M and dt^2 K action norms and b−Ax residual ratios; it does not, alone, establish ill-conditioning.",
                "This package contains no same-capture condition estimate or direct solve, so it does not isolate ill-conditioning from finite-precision or recurrence error.",
                "A small residual vector discrepancy relative to b but larger relative to the target residual can reflect arithmetic cancellation, conditioning, recurrence drift, or these combined; this comparison does not isolate a unique cause.",
            ],
            "smallest_next_diagnostic": "On the retained exact CGLS capture, independently evaluate b−Ax0 and b−AX in three arithmetic paths using the same captured FP32 cache: (1) native stored FP32 operator action/vector residual already captured, (2) FP64 sparse replay already available, and (3) compensated or higher-precision CPU replay of the identical FP32 coefficients. Compare each residual vector and backward error against ||b||, ||A||·||x||+||b||, and the target residual; pair with a documented condition estimate and forward error to an independent FP64 direct solve. This separates action/subtraction rounding from sensitivity without changing physics or tolerances. A future native capture should include the exact FP32 b, x0, total X, Q=A·X, fresh b−A·X and CGLS recurrence R from one command-buffer generation; 497 currently has enough for the offline replay but not a multi-precision native comparison.",
        },
        "limits": [
            "No full-cycle success: all three resumed segments terminate with failure at control step 20 and accept zero controls in the segment.",
            "The held-reference observation is not evidence of a completed bent coupon trajectory.",
            "497's regional CGLS observer data is a selected FEM-only block; it excludes rigid/contact/cohesive rows and does not qualify the full coupled solve.",
            "Offline replay uses the captured FP32 tangent cache and source-defined operator reconstruction; it does not reevaluate material response or execute native FP64 physics.",
            "No claim of general solver accuracy, conditioning proof, performance, or physical cardboard qualification is made.",
        ],
    }
    report["run_artifact_bindings"] = {
        label: {
            name: {"path": str(CASES[label] / name), "bytes": (CASES[label] / name).stat().st_size, "sha256": sha(CASES[label] / name)}
            for name in ("prediction.json", "native-environment.json", "receipt.json", "diagnostic-captures.json", "stdout.log", "stderr.log")
        }
        for label in CASES
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "verdict": report["verdict"], "output_paths": len(expected_paths), "trace_counts": line_counts, "env_diff": sorted(env_diff)}, indent=2))


if __name__ == "__main__":
    main()
