#!/usr/bin/env python3
"""Fail-closed verification and summary of the 2026-10-23 runtime evidence.

This reads retained evidence only. A normal solver exit is runtime evidence,
not material calibration or physical qualification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
RELEASE = "latest-20261011"
SOURCE_COMMIT = "cb760c206e346eff12161a2268b134ef9d48a4ef"
EXPECTED_RUNS = {
    "cohesive-explicit-01": {
        "case_id": "law117-mode-i-open-failure",
        "termination": "normal",
        "accepted_cycles": 1128,
    },
    "cohesive-positive-height-01": {
        "case_id": "implicit-cohesive-positive-reference-height-diagnostic",
        "termination": "nonlinear_timestep_limit",
        "accepted_cycles": 0,
    },
    "implicit-cohesive-01": {
        "case_id": "implicit-cohesive-nonlinear-force-control",
        "termination": "allocator_abort",
        "accepted_cycles": 0,
    },
    "implicit-elastic-01": {
        "case_id": "implicit-elastic-nonlinear-force-control",
        "termination": "normal",
        "accepted_cycles": 11,
    },
    "implicit-elastic-instrumented-01": {
        "case_id": "implicit-elastic-nonlinear-force-control-instrumented",
        "termination": "normal",
        "accepted_cycles": 11,
    },
}
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
TOTAL_CYCLES_RE = re.compile(r"TOTAL NUMBER OF CYCLES\s*:\s*(\d+)")
NC_RE = re.compile(r"\bNC\s*=\s*(\d+)\b")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(ValueError):
    """Raised when the copied evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path.relative_to(ROOT)}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON receipt is not an object: {path.relative_to(ROOT)}")
    return value


def _file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise EvidenceError(f"missing {label}: {path}") from exc
    _require(stat.S_ISREG(mode), f"{label} is not a regular nonsymlink file: {path}")


def _manifest_path(directory: Path, name: str, record: Any, label: str) -> Path:
    _require(isinstance(name, str) and name and Path(name).name == name,
             f"unsafe {label} manifest path: {name!r}")
    _require(isinstance(record, dict), f"invalid {label} manifest record for {name}")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    _require(isinstance(expected_bytes, int) and not isinstance(expected_bytes, bool)
             and expected_bytes >= 0, f"invalid {label} byte count for {name}")
    _require(isinstance(expected_sha, str) and SHA_RE.fullmatch(expected_sha) is not None,
             f"invalid {label} SHA-256 for {name}")
    path = directory / name
    _file(path, f"{label} {name}")
    _require(path.stat().st_size == expected_bytes,
             f"{label} byte count mismatch: {directory.name}/{name}")
    _require(_sha(path) == expected_sha,
             f"{label} SHA-256 mismatch: {directory.name}/{name}")
    return path


def _safe_recorded_name(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value and Path(value).name == value,
             f"invalid run-local {label}: {value!r}")
    return value


def _validate_runtime() -> dict[str, Any]:
    setup = ROOT / "setup"
    runtime_path = setup / "runtime-identity.json"
    packaged_path = setup / "packaged-runtime-identity.json"
    receipt_path = setup / "build-receipt.json"
    for path in (runtime_path, packaged_path, receipt_path):
        _file(path, "runtime/build identity")

    runtime = _load_json(runtime_path)
    packaged = _load_json(packaged_path)
    receipt = _load_json(receipt_path)
    runtime_sha, packaged_sha, receipt_sha = (_sha(runtime_path), _sha(packaged_path),
                                              _sha(receipt_path))
    _require(runtime.get("release") == RELEASE and runtime.get("source_commit") == SOURCE_COMMIT,
             "runtime identity is not the pinned OpenCourant release/source")
    _require(packaged.get("release") == RELEASE and packaged.get("source_commit") == SOURCE_COMMIT,
             "packaged runtime identity is not the pinned OpenCourant release/source")
    _require(runtime.get("package_sha256") == packaged.get("package_sha256"),
             "built and packaged runtime identities cite different release packages")
    _require(runtime.get("profile") == "arm64-mumps-build",
             f"unexpected build profile: {runtime.get('profile')!r}")
    _require(runtime.get("build", {}).get("source_commit") == SOURCE_COMMIT,
             "runtime build identity source does not match the pinned commit")
    _require(runtime.get("build", {}).get("tracked_source_diff") == "",
             "runtime build identity reports tracked source modifications")
    _require(runtime.get("build", {}).get("receipt_sha256") == receipt_sha,
             "runtime identity build-receipt hash does not match build-receipt.json")
    _require(receipt.get("schema") == "cardboard.opencourant.build.v1",
             "unsupported build receipt schema")
    _require(receipt.get("source_commit") == SOURCE_COMMIT
             and receipt.get("tracked_source_diff") == "",
             "build receipt source identity is inconsistent or modified")
    _require(isinstance(receipt.get("engine_sha256"), str)
             and SHA_RE.fullmatch(receipt["engine_sha256"]) is not None,
             "build receipt does not contain a valid engine SHA-256")
    for path, key in (
        (setup / "dependency-aliases.json", "dependency_aliases_sha256"),
        (setup / "dpkg-packages.txt", "dpkg_packages_sha256"),
        (ROOT.parents[3] / "tools" / "build_cardboard_opencourant_mumps.sh", "build_script_sha256"),
    ):
        _file(path, "build input")
        _require(_sha(path) == receipt.get(key), f"build input changed: {path.name}")

    runtime_files = runtime.get("files")
    packaged_files = packaged.get("files")
    _require(isinstance(runtime_files, dict) and isinstance(packaged_files, dict),
             "runtime identities lack file hash maps")
    starter_sha = runtime_files.get("exec/starter_linuxa64_gf")
    engine_sha = runtime_files.get("exec/engine_linuxa64_gf_ompi")
    converter_sha = runtime_files.get("exec/th_to_csv_linuxa64")
    packaged_engine_sha = packaged_files.get("exec/engine_linuxa64_gf_ompi")
    _require(all(isinstance(value, str) and SHA_RE.fullmatch(value) is not None
                 for value in (starter_sha, engine_sha, converter_sha, packaged_engine_sha)),
             "runtime identity lacks starter, engine, converter, or package engine hashes")
    _require(starter_sha == packaged_files.get("exec/starter_linuxa64_gf"),
             "build runtime starter differs from the pinned package starter")
    _require(converter_sha == packaged_files.get("exec/th_to_csv_linuxa64"),
             "build runtime converter differs from the pinned package converter")
    _require(engine_sha == receipt.get("engine_sha256"),
             "built engine hash differs between identity and build receipt")

    limitations = receipt.get("limitations")
    _require(isinstance(limitations, str) and limitations,
             "build receipt omits its dependency/version caveat")
    return {
        "release": RELEASE,
        "source_commit": SOURCE_COMMIT,
        "runtime_profile": runtime["profile"],
        "runtime_identity_sha256": runtime_sha,
        "packaged_runtime_identity_sha256": packaged_sha,
        "build_receipt_sha256": receipt_sha,
        "release_package_sha256": runtime["package_sha256"],
        "starter_sha256": starter_sha,
        "built_engine_sha256": engine_sha,
        "packaged_engine_sha256": packaged_engine_sha,
        "th_to_csv_sha256": converter_sha,
        "compiler": receipt.get("compiler", ""),
        "build_completed_utc": receipt.get("completed_utc"),
        "tracked_source_diff": receipt.get("tracked_source_diff"),
        "build_limitations": limitations,
    }


def _rss_bytes(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = RSS_RE.findall(text)
    _require(len(matches) == 1, f"expected one GNU time RSS record in {path.name}")
    return int(matches[0]) * 1024


def _check_diagnostic(run_receipts: dict[str, str], runtime: dict[str, Any]) -> dict[str, Any]:
    directory = ROOT / "diagnostics" / "cohesive-implicit-backtrace"
    diagnostic_path = directory / "diagnostic.json"
    trace_path = directory / "gdb.stdout"
    stderr_path = ROOT / "runs" / "implicit-cohesive-01" / "01-engine.stderr"
    for path in (diagnostic_path, trace_path, stderr_path):
        _file(path, "allocator-abort diagnostic evidence")
    diagnostic = _load_json(diagnostic_path)
    trace = trace_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    _require(diagnostic.get("parent_receipt_sha256") == run_receipts["implicit-cohesive-01"],
             "GDB diagnostic is not bound to the retained zero-height implicit cohesive run")
    _require(diagnostic.get("engine_sha256") == runtime["built_engine_sha256"],
             "GDB diagnostic engine differs from the retained runtime engine")
    _require("free(): invalid pointer" in stderr,
             "zero-height implicit cohesive stderr lacks the recorded allocator failure")
    _require("SIGABRT" in trace and "in free ()" in trace and "in resol_" in trace,
             "GDB trace does not corroborate abort in free() from resol_")
    return {
        "diagnostic_sha256": _sha(diagnostic_path),
        "gdb_trace_sha256": _sha(trace_path),
        "parent_run_receipt_sha256": diagnostic["parent_receipt_sha256"],
        "engine_sha256": diagnostic["engine_sha256"],
        "failure_evidence": "stderr reports free(): invalid pointer; GDB recorded SIGABRT in free() under resol_",
    }


def _check_postprocess(runtime: dict[str, Any], instrumented: dict[str, Any]) -> dict[str, Any]:
    post = ROOT / "post"
    long_dir = post / "implicit-elastic-instrumented-01"
    short_dir = post / "implicit-elastic-shortname-02"
    failure_path = long_dir / "failure.json"
    conversion_path = short_dir / "conversion.json"
    _file(failure_path, "converter failure receipt")
    _file(conversion_path, "converter success receipt")
    failure = _load_json(failure_path)
    conversion = _load_json(conversion_path)
    instrumented_dir = ROOT / "runs" / "implicit-elastic-instrumented-01"
    instrumented_run = _load_json(instrumented_dir / "run.json")
    instrumented_case = _load_json(instrumented_dir / "case.json")
    outputs = instrumented_run["outputs"]
    t01_name = instrumented_case["native_outputs"]["time_history"]["path"]
    t01_sha = outputs.get(t01_name, {}).get("sha256")
    titles_sha = outputs.get(t01_name + "_TITLES", {}).get("sha256")
    _require(t01_sha and titles_sha, "instrumented elastic native history is not in its output manifest")

    _require(failure.get("exit_code") == -11 and failure.get("input_sha256") == t01_sha,
             "long-path converter failure is not bound to the instrumented elastic T01")
    _require(failure.get("converter_sha256") == runtime["th_to_csv_sha256"],
             "long-path converter failure used a converter outside the runtime identity")
    _require(conversion.get("schema") == "cardboard.opencourant.postprocess.v1"
             and conversion.get("exit_code") == 0,
             "short-name conversion lacks a successful postprocess receipt")
    _require(conversion.get("input_sha256") == t01_sha
             and conversion.get("titles_sha256") == titles_sha,
             "short-name conversion receipt does not bind to retained native T01 and titles")
    _require(conversion.get("converter_sha256") == runtime["th_to_csv_sha256"],
             "short-name conversion used a converter outside the runtime identity")
    _require(conversion.get("input_relative_path")
             == f"runs/implicit-elastic-instrumented-01/{t01_name}",
             "short-name conversion points to an unexpected T01 input")
    alias = _safe_recorded_name(conversion.get("input_alias"), "converter input alias")
    alias_t01 = short_dir / alias
    alias_titles = short_dir / (alias + "_TITLES")
    csv_name = _safe_recorded_name(conversion.get("output"), "converted CSV name")
    csv_path = short_dir / csv_name
    for path, expected_sha, label in (
        (alias_t01, t01_sha, "short-name T01 alias"),
        (alias_titles, titles_sha, "short-name titles alias"),
        (csv_path, conversion.get("output_sha256"), "converted CSV"),
    ):
        _file(path, label)
        _require(_sha(path) == expected_sha, f"{label} hash does not match its receipt")
    return {
        "long_absolute_path_attempt": {
            "exit_code": -11,
            "signal": "SIGSEGV",
            "input_sha256": t01_sha,
            "converter_sha256": runtime["th_to_csv_sha256"],
        },
        "short_alias_attempt": {
            "exit_code": 0,
            "converter_sha256": runtime["th_to_csv_sha256"],
            "input_alias": alias,
            "input_sha256": t01_sha,
            "titles_sha256": titles_sha,
            "csv_sha256": conversion["output_sha256"],
            "csv_bytes": csv_path.stat().st_size,
            "interpretation": "conversion evidence only; no mechanics verdict without a measurement JSON",
        },
    }


def _optional_elastic_measurement(instrumented: dict[str, Any],
                                  conversion: dict[str, Any]) -> dict[str, Any] | None:
    candidates = sorted((ROOT / "post").rglob("*measurement*.json"))
    _require(len(candidates) <= 1,
             "expected at most one supplied elastic measurement JSON under post/")
    if not candidates:
        return None
    path = candidates[0]
    _file(path, "elastic measurement JSON")
    measurement = _load_json(path)
    instrumented_dir = ROOT / "runs" / "implicit-elastic-instrumented-01"
    instrumented_run = _load_json(instrumented_dir / "run.json")
    instrumented_case = _load_json(instrumented_dir / "case.json")
    _require(measurement.get("schema") == "cardboard.opencourant.measurement.v1",
             "supplied elastic measurement has an unsupported schema")
    _require(measurement.get("case_id") == instrumented_case.get("id"),
             "supplied elastic measurement names a different case")
    _require(measurement.get("physical_qualification") is False,
             "elastic measurement must retain its nonqualification flag")
    _require(measurement.get("overall_status") in {"pass", "fail", "inconclusive"},
             "elastic measurement has an unknown mechanics-instrument status")
    provenance = measurement.get("provenance")
    _require(isinstance(provenance, dict), "elastic measurement lacks provenance")
    outputs = instrumented_run["outputs"]
    native_name = instrumented_case["native_outputs"]["time_history"]["path"]
    _require(provenance.get("t01_sha256") == outputs[native_name]["sha256"],
             "elastic measurement T01 hash does not match the instrumented run")
    _require(provenance.get("t01_titles_sha256") == outputs[native_name + "_TITLES"]["sha256"],
             "elastic measurement TITLES hash does not match the instrumented run")
    _require(provenance.get("starter_sha256")
             == instrumented_run["inputs"][instrumented_case["starter"]]["sha256"],
             "elastic measurement starter hash does not match the instrumented run")
    _require(provenance.get("converted_csv_sha256") == conversion["short_alias_attempt"]["csv_sha256"],
             "elastic measurement CSV hash does not match the successful converter receipt")
    _require(provenance.get("converter_sha256") == conversion["short_alias_attempt"]["converter_sha256"],
             "elastic measurement converter hash does not match the successful converter receipt")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha(path),
        "overall_status": measurement["overall_status"],
        "interpretation": measurement.get("interpretation"),
        "physical_qualification": False,
        "timebase": measurement.get("timebase"),
        "displacement": measurement.get("displacement"),
        "reaction": measurement.get("reaction"),
        "energy": measurement.get("energy"),
        "external_work_crosscheck": measurement.get("external_work_crosscheck"),
        "provenance": provenance,
    }


def _analyze_run(name: str, runtime: dict[str, Any], runtime_json_sha: str) -> dict[str, Any]:
    directory = ROOT / "runs" / name
    run_path = directory / "run.json"
    _file(run_path, "run receipt")
    run = _load_json(run_path)
    inputs, outputs = run.get("inputs"), run.get("outputs")
    _require(isinstance(inputs, dict) and isinstance(outputs, dict),
             f"{name}: missing input or output manifest")
    expected_files = set(inputs) | set(outputs) | {"run.json"}
    actual_entries = list(directory.iterdir())
    _require(all(p.is_file() and not p.is_symlink() for p in actual_entries),
             f"{name}: unexpected directory, symlink, or non-file in run folder")
    actual_files = {p.name for p in actual_entries}
    _require(actual_files == expected_files,
             f"{name}: run folder file inventory differs from input/output receipts; "
             f"missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}")

    for input_name, record in inputs.items():
        _manifest_path(directory, input_name, record, "input")
    for output_name, record in outputs.items():
        _manifest_path(directory, output_name, record, "output")
    receipt_sha = _sha(run_path)

    case_name = _safe_recorded_name("case.json", "case metadata filename")
    case = _load_json(directory / case_name)
    expected = EXPECTED_RUNS[name]
    _require(run.get("schema") == "cardboard.opencourant.run.v1", f"{name}: unsupported run schema")
    _require(run.get("case_id") == expected["case_id"] and case.get("id") == expected["case_id"],
             f"{name}: run/case identity mismatch")
    _require(run.get("case_sha256") == inputs.get("case.json", {}).get("sha256"),
             f"{name}: case receipt hash mismatch")
    _require(run.get("release") == RELEASE and run.get("source_commit") == SOURCE_COMMIT,
             f"{name}: run is not bound to the pinned source/release")
    _require(run.get("runtime_profile") == runtime["runtime_profile"],
             f"{name}: runtime profile mismatch")
    _require(run.get("runtime_identity_sha256") == runtime_json_sha,
             f"{name}: runtime identity file hash mismatch")
    _require(run.get("physical_qualification") is False
             and case.get("physical_qualification") in (None, False),
             f"{name}: physical qualification must remain false or undeclared")
    _require(run.get("adapter_sha256") == outputs.get("adapter.snapshot.py", {}).get("sha256"),
             f"{name}: adapter snapshot hash differs from recorded executed adapter")
    binary = run.get("binary_sha256")
    _require(isinstance(binary, dict)
             and binary.get("starter") == runtime["starter_sha256"]
             and binary.get("engine") == runtime["built_engine_sha256"],
             f"{name}: starter/engine hashes differ from runtime identity/build receipt")
    _require(case.get("starter") in inputs and isinstance(case.get("engines"), list)
             and len(case["engines"]) == 1 and case["engines"][0] in inputs,
             f"{name}: starter/engine decks are absent from the input manifest")
    _require(math.isfinite(float(run.get("elapsed_s"))) and float(run["elapsed_s"]) >= 0,
             f"{name}: invalid total elapsed time")

    commands = run.get("commands")
    _require(isinstance(commands, list) and len(commands) == 2,
             f"{name}: expected exactly one Starter and one Engine command")
    stage_results: list[dict[str, Any]] = []
    for index, (stage, expected_deck) in enumerate((("starter", case["starter"]),
                                                     ("engine", case["engines"][0]))):
        command = commands[index]
        _require(isinstance(command, dict), f"{name}: invalid {stage} command record")
        prefix = f"{index:02d}-{stage}"
        stdout_path = _manifest_path(directory, prefix + ".stdout", outputs.get(prefix + ".stdout"), "output")
        stderr_path = _manifest_path(directory, prefix + ".stderr", outputs.get(prefix + ".stderr"), "output")
        process_path = _manifest_path(directory, prefix + ".process.json",
                                      outputs.get(prefix + ".process.json"), "output")
        process = _load_json(process_path)
        process_argv = process.get("argv")
        argv = command.get("argv")
        _require(isinstance(process_argv, list) and process_argv
                 and isinstance(argv, list) and len(argv) >= len(process_argv)
                 and argv[-len(process_argv):] == process_argv,
                 f"{name}: {stage} process argv is inconsistent with command receipt")
        _require("-i" in process_argv and process_argv[process_argv.index("-i") + 1] == expected_deck,
                 f"{name}: {stage} command did not run its recorded deck")
        _require(isinstance(command.get("elapsed_s"), (int, float))
                 and math.isfinite(command["elapsed_s"]) and command["elapsed_s"] >= 0,
                 f"{name}: invalid {stage} elapsed time")
        stage_results.append({
            "stage": stage,
            "exit_code": command.get("exit_code"),
            "normal_termination": command.get("normal_termination"),
            "error_termination": command.get("error_termination"),
            "timed_out": command.get("timed_out"),
            "elapsed_s": command["elapsed_s"],
            "max_rss_bytes": _rss_bytes(stderr_path),
            "stdout_sha256": _sha(stdout_path),
            "stderr_sha256": _sha(stderr_path),
        })

    engine_name = Path(case["engines"][0]).with_suffix(".out").name
    engine_out_path = _manifest_path(directory, engine_name, outputs.get(engine_name), "output")
    engine_out = engine_out_path.read_text(encoding="utf-8", errors="replace")
    engine_stdout_path = directory / "01-engine.stdout"
    engine_stderr_path = directory / "01-engine.stderr"
    engine_stdout = engine_stdout_path.read_text(encoding="utf-8", errors="replace")
    engine_stderr = engine_stderr_path.read_text(encoding="utf-8", errors="replace")
    total_cycles = [int(value) for value in TOTAL_CYCLES_RE.findall(engine_out)]
    observed_nc = [int(value) for value in NC_RE.findall(engine_stdout)]
    engine_command = commands[1]
    normal = (engine_command.get("exit_code") == 0
              and engine_command.get("normal_termination") is True
              and engine_command.get("error_termination") is False
              and engine_command.get("timed_out") is False
              and "NORMAL TERMINATION" in engine_out)

    if normal:
        _require(len(total_cycles) == 1, f"{name}: normal engine output lacks one total cycle count")
        termination = "normal"
        accepted_cycles = total_cycles[0]
        native_failure = None
    elif name == "implicit-cohesive-01":
        _require(engine_command.get("exit_code") == 3 and "free(): invalid pointer" in engine_stderr,
                 f"{name}: allocator abort evidence changed")
        _require(observed_nc and max(observed_nc) == 0,
                 f"{name}: allocator abort did not remain at cycle zero")
        termination = "allocator_abort"
        accepted_cycles = 0
        native_failure = {
            "type": "allocator_abort_invalid_pointer",
            "cycle": 0,
            "stderr_marker": "free(): invalid pointer",
            "diagnostic": "SIGABRT in free() under resol_",
        }
    elif name == "cohesive-positive-height-01":
        _require(engine_command.get("exit_code") == 2
                 and "SOLVER IMPLICIT STOPPED DUE TO TIMESTEP LIMIT" in engine_out
                 and "ITERATION DIVERGE with MAX_ITER REACHED" in engine_out,
                 f"{name}: implicit timestep-limit evidence changed")
        _require(observed_nc and max(observed_nc) == 0,
                 f"{name}: positive-height diagnostic advanced beyond cycle zero")
        termination = "nonlinear_timestep_limit"
        accepted_cycles = 0
        native_failure = {
            "type": "implicit_nonlinear_timestep_limit_after_iteration_divergence",
            "cycle": 0,
            "native_error": "SOLVER IMPLICIT STOPPED DUE TO TIMESTEP LIMIT",
        }
    else:
        raise EvidenceError(f"{name}: unclassified engine termination")

    _require(termination == expected["termination"]
             and accepted_cycles == expected["accepted_cycles"],
             f"{name}: native result differs from the preregistered retained outcome")
    expected_complete = termination == "normal"
    _require(run.get("execution_complete") is expected_complete
             and run.get("case_sequence_complete") is expected_complete,
             f"{name}: completion flags disagree with native termination")
    if expected_complete:
        _require(run.get("completed_engine") == case["engines"][0],
                 f"{name}: completion receipt names the wrong engine")
    else:
        _require(run.get("completed_engine") is None,
                 f"{name}: failed run claims a completed engine")
    last_reported_cycle = (max(observed_nc) if observed_nc else accepted_cycles)
    engine_rss = stage_results[1]["max_rss_bytes"]
    total_rss = max(stage["max_rss_bytes"] for stage in stage_results)
    return {
        "run": name,
        "case_id": run["case_id"],
        "termination": termination,
        "accepted_cycles": accepted_cycles,
        "last_reported_cycle": last_reported_cycle,
        "native_failure": native_failure,
        "execution_complete": expected_complete,
        "total_elapsed_s": run["elapsed_s"],
        "stages": stage_results,
        "engine_max_rss_bytes": engine_rss,
        "all_stage_max_rss_bytes": total_rss,
        "memory_limit_bytes": run.get("memory_limit_bytes"),
        "input_files_verified": len(inputs),
        "output_files_verified": len(outputs),
        "unique_run_files_verified": len(expected_files) - 1,
        "run_receipt_sha256": receipt_sha,
        "case_sha256": run["case_sha256"],
        "adapter_snapshot_sha256": run["adapter_sha256"],
        "runtime_identity_sha256": run["runtime_identity_sha256"],
        "engine_sha256": binary["engine"],
        "starter_sha256": binary["starter"],
        "loaded_library_count": len(run.get("loaded_library_sha256", {})),
    }


def analyze() -> dict[str, Any]:
    runs_directory = ROOT / "runs"
    _require(runs_directory.is_dir(), "missing runs directory")
    _require(all(p.is_dir() and not p.is_symlink() for p in runs_directory.iterdir()),
             "runs directory contains an unexpected file, symlink, or special entry")
    actual_runs = {p.name for p in runs_directory.iterdir()}
    expected_runs = set(EXPECTED_RUNS)
    _require(actual_runs == expected_runs,
             f"runset mismatch; missing={sorted(expected_runs - actual_runs)}, "
             f"extra={sorted(actual_runs - expected_runs)}")

    runtime = _validate_runtime()
    run_rows = {name: _analyze_run(name, runtime, runtime["runtime_identity_sha256"])
                for name in sorted(EXPECTED_RUNS)}
    receipt_hashes = {name: row["run_receipt_sha256"] for name, row in run_rows.items()}
    diagnostic = _check_diagnostic(receipt_hashes, runtime)
    postprocess = _check_postprocess(runtime, run_rows["implicit-elastic-instrumented-01"])
    measurement = _optional_elastic_measurement(
        run_rows["implicit-elastic-instrumented-01"], postprocess)
    if measurement is None:
        mechanics_verdict = {
            "status": "NO_MECHANICS_VERDICT",
            "reason": "No elastic measurement JSON was provided; titled CSV conversion alone is not interpreted as mechanics evidence.",
        }
    else:
        mechanics_verdict = {
            "status": measurement["overall_status"],
            "scope": "synthetic implicit elastic control only; not physical-cardboard qualification",
            "measurement_json": measurement["path"],
            "measurement_sha256": measurement["sha256"],
        }

    return {
        "schema": "cardboard.opencourant.admission-result.v2",
        "admitted": False,
        "admission_status": "NOT_ADMITTED",
        "reason": (
            "The pinned-source ARM64 GNU/MPI build completed the implicit elastic controls and "
            "the explicit cohesive runtime control, but the zero-height implicit cohesive case "
            "aborted in the native allocator at cycle 0 and the positive-reference-height "
            "diagnostic still reached the nonlinear timestep limit at cycle 0."
        ),
        "runtime": runtime,
        "verification": {
            "exact_runset": sorted(EXPECTED_RUNS),
            "exact_runset_verified": True,
            "all_run_input_output_hashes_verified": True,
            "adapter_snapshots_verified": True,
            "runtime_identity_and_build_receipt_verified": True,
            "native_abort_diagnostic_verified": True,
            "postprocess_receipts_verified": True,
        },
        "runs": [run_rows[name] for name in sorted(run_rows)],
        "cohesive_abort_diagnostic": diagnostic,
        "postprocess": postprocess,
        "elastic_control_measurement": measurement,
        "mechanics_verdict": mechanics_verdict,
        "qualification": {
            "backend_correctness": "NOT_ADMITTED",
            "implicit_cohesive_convergence": "FAILED_AT_CYCLE_ZERO",
            "implicit_elastic_runtime": "COMPLETED_RUNTIME_CONTROLS_ONLY",
            "explicit_cohesive_runtime": "COMPLETED_RUNTIME_CONTROL_ONLY",
            "component_calibration": "NOT_COMPLETE",
            "analogue_validation": "NOT_RUN",
            "physical_assembly": "NOT_RUN",
            "crush_impact_prediction": "NOT_RUN",
        },
        "forecast": {
            "full_campaign_elapsed_time_s": None,
            "credible_shipping_resource_estimate": False,
            "reason": "The required implicit cohesive control did not converge; no representative physical assembly or shipping-scale run was admitted.",
        },
        "scope_limits": [
            "Normal termination records runtime completion only; it does not establish model mechanics or calibration.",
            "The explicit cohesive control is a small synthetic runtime fixture, not a validated crush or folding simulation.",
            "The positive-reference-height case is a diagnostic geometry change only and remains failed at cycle zero.",
            "The titled elastic CSV is retained as conversion evidence; no mechanics verdict is issued unless a bound elastic measurement JSON is present.",
            "No full cardboard fold, bond-release, assembly, crush, or shipping campaign is forecast from these runs.",
            runtime["build_limitations"],
        ],
    }


def main() -> int:
    try:
        result = analyze()
        serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output_path = ROOT / "admission-result.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=ROOT,
                                         prefix=".admission-result.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except (EvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"admission evidence verification failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "run_count": len(result["runs"]),
        "input_output_hashes_verified": result["verification"]["all_run_input_output_hashes_verified"],
        "admitted": result["admitted"],
        "mechanics_verdict": result["mechanics_verdict"]["status"],
        "result": str(output_path),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
