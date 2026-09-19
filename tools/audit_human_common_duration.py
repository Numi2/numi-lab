#!/usr/bin/env python3
"""Run the production Human common-duration trace matrix on a physical Mac.

The four canonical cases advance the same 6.4 ms interval at 100, 50, 25,
and 12.5 microseconds.  This runner deliberately uses the clean production
cap-8 path: it does not stage the segment diagnostic, set its environment
override, enable Metal validation, or infer convergence thresholds.

Each case retains the complete v5 trajectory, exact source/build/input hashes,
bitwise replay status, process memory samples, machine state, and raw output.
The companion numilab-human comparator owns the cross-grid diagnostic.  A
successful run is physical execution evidence, not force convergence or a
standing qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASE_SCHEMA = "numi.human.current-refinement-case.v3"
MANIFEST_SCHEMA = "numi.human.physical-common-duration-run.v1"
TRACE_SCHEMA = "numi.human.persistent-stand-trace.v5"
COMMON_DURATION_NS = 6_400_000
CASE_SPEC = (
    ("100us", "0.0001", 100_000, 64),
    ("50us", "0.00005", 50_000, 128),
    ("25us", "0.000025", 25_000, 256),
    ("12p5us", "0.0000125", 12_500, 512),
)
RUNTIME_ENVIRONMENT_KEYS = (
    "DEVELOPER_DIR",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "SDKROOT",
    "SHELL",
    "TMPDIR",
    "USER",
)
VELOCITY_STAGE_KEYS = (
    "persistent_max_free_acceleration",
    "persistent_max_free_acceleration_dof",
    "persistent_max_constraint_delta_v",
    "persistent_max_constraint_delta_v_dof",
    "persistent_max_pre_projection_delta_v",
    "persistent_max_pre_projection_delta_v_dof",
    "persistent_max_published_delta_v",
    "persistent_max_published_delta_v_dof",
)
IMPULSE_WORK_TOTALS = {
    "contact_normal_impulse_work_j": "total_contact_normal_impulse_work_j",
    "contact_tangential_impulse_work_j":
        "total_contact_tangential_impulse_work_j",
    "equality_impulse_work_j": "total_equality_impulse_work_j",
    "source_limit_impulse_work_j": "total_source_limit_impulse_work_j",
    "contact_normal_absolute_impulse_work_j":
        "total_contact_normal_absolute_impulse_work_j",
    "contact_tangential_absolute_impulse_work_j":
        "total_contact_tangential_absolute_impulse_work_j",
    "equality_absolute_impulse_work_j":
        "total_equality_absolute_impulse_work_j",
    "source_limit_absolute_impulse_work_j":
        "total_source_limit_absolute_impulse_work_j",
}


class AuditError(RuntimeError):
    """The production execution or evidence contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def command(
    arguments: list[str], *, cwd: Path | None = None, timeout: float = 15.0
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AuditError("provenance command failed: " + " ".join(arguments)) from error


def required_output(
    arguments: list[str], *, cwd: Path | None = None, timeout: float = 15.0
) -> str:
    completed = command(arguments, cwd=cwd, timeout=timeout)
    require(
        completed.returncode == 0,
        "provenance command failed: " + " ".join(arguments),
    )
    return completed.stdout.strip()


def git_output(repository: Path, arguments: list[str]) -> str:
    return required_output(["git", "-C", str(repository), *arguments])


def git_clean(repository: Path) -> bool:
    return not required_output(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"]
    )


def strict_json_object(text: str, context: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{context} has duplicate JSON key {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise AuditError(f"{context} has non-finite JSON constant {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise AuditError(f"{context} is not strict JSON") from error
    require(isinstance(value, dict), f"{context} is not a JSON object")
    return value


def prefixed_object(text: str, prefix: str, context: str) -> dict[str, Any]:
    values = [
        line[len(prefix):]
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    require(len(values) == 1, f"{context} has missing or ambiguous {prefix[:-1]}")
    return strict_json_object(values[0], context + " " + prefix[:-1])


def parse_metrics(text: str, context: str) -> dict[str, str]:
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("myosim_articulated_mechanics=")
    ]
    require(len(rows) == 1, f"{context} has missing or ambiguous mechanics summary")
    metrics: dict[str, str] = {}
    for token in shlex.split(rows[0]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        require(key not in metrics, f"{context} has duplicate metric {key}")
        metrics[key] = value
    return metrics


def finite(value: Any, context: str) -> float:
    require(type(value) in (int, float), f"{context} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{context} is not finite")
    return result


def validate_trace(
    trace: dict[str, Any], terminal: dict[str, Any], timestep_ns: int, steps: int
) -> None:
    require(trace.get("schema") == TRACE_SCHEMA, "persistent trace schema changed")
    require(
        trace.get("endpoint_equivalent") == "bitwise"
        and trace.get("endpoint_max_q_delta") == 0
        and trace.get("endpoint_max_v_delta") == 0,
        "persistent trace replay endpoint differs",
    )
    work_scope = trace.get("work_scope")
    require(
        isinstance(work_scope, str)
        and "production_constraint_impulse_work_by_family" in work_scope
        and "exact_coordinate_projection_is_an_unowned_overwrite_not_impulse_work"
        in work_scope,
        "persistent trace work scope is incomplete",
    )
    samples = trace.get("samples")
    require(
        isinstance(samples, list) and len(samples) == steps + 1,
        "persistent trace sample count changed",
    )
    signed_absolute_pairs = (
        ("contact_normal_impulse_work_j", "contact_normal_absolute_impulse_work_j"),
        (
            "contact_tangential_impulse_work_j",
            "contact_tangential_absolute_impulse_work_j",
        ),
        ("equality_impulse_work_j", "equality_absolute_impulse_work_j"),
        ("source_limit_impulse_work_j", "source_limit_absolute_impulse_work_j"),
    )
    for index, sample in enumerate(samples):
        require(isinstance(sample, dict), f"trace sample {index} is not an object")
        require(sample.get("step") == index, f"trace sample {index} is out of order")
        expected_time = index * timestep_ns * 1.0e-9
        require(
            math.isclose(
                finite(sample.get("time_seconds"), f"trace sample {index} time"),
                expected_time,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"trace sample {index} has the wrong physical time",
        )
        q = sample.get("q")
        v = sample.get("v")
        require(isinstance(q, list) and len(q) == 129, "trace q dimension changed")
        require(isinstance(v, list) and len(v) == 128, "trace v dimension changed")
        for component_index, value in enumerate(q):
            finite(value, f"trace sample {index} q[{component_index}]")
        for component_index, value in enumerate(v):
            finite(value, f"trace sample {index} v[{component_index}]")
        finite(sample.get("normal_impulse"), f"trace sample {index} normal impulse")
        for signed, absolute in signed_absolute_pairs:
            signed_value = finite(sample.get(signed), f"trace sample {index} {signed}")
            absolute_value = finite(
                sample.get(absolute), f"trace sample {index} {absolute}"
            )
            require(absolute_value >= 0.0, f"trace sample {index} {absolute} is negative")
            require(
                absolute_value + 1.0e-12 >= abs(signed_value),
                f"trace sample {index} {absolute} hides signed work",
            )
    for sample_field, total_field in IMPULSE_WORK_TOTALS.items():
        expected = sum(
            finite(sample.get(sample_field), f"trace {sample_field}")
            for sample in samples[1:]
        )
        reported = finite(trace.get(total_field), f"trace {total_field}")
        require(
            math.isclose(reported, expected, rel_tol=1.0e-12, abs_tol=1.0e-12),
            f"trace {total_field} disagrees with samples",
        )
    require(terminal.get("step_count") == steps, "terminal step count changed")
    require(
        terminal.get("timestep_seconds") == timestep_ns * 1.0e-9,
        "terminal timestep changed",
    )
    require(terminal.get("root_assistance") is False, "terminal used root assistance")


def command_probe(arguments: list[str], timeout: float = 5.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            arguments,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": arguments,
            "exit_code": completed.returncode,
            "output": completed.stdout.strip(),
            "wall_seconds": time.monotonic() - started,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": arguments,
            "exit_code": None,
            "error": type(error).__name__,
            "wall_seconds": time.monotonic() - started,
        }


def machine_receipt() -> dict[str, Any]:
    profile = strict_json_object(
        required_output(["system_profiler", "SPHardwareDataType", "-json"]),
        "hardware profile",
    )
    items = profile.get("SPHardwareDataType")
    require(isinstance(items, list) and len(items) == 1, "hardware profile changed")
    hardware = items[0]
    require(isinstance(hardware, dict), "hardware profile row changed")
    identity_text = required_output(
        ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]
    )
    identity = ""
    for line in identity_text.splitlines():
        if '"IOPlatformUUID"' in line and "=" in line:
            identity = line.split("=", 1)[1].strip().strip('"')
            break
    require(identity, "physical machine identity is unavailable")
    receipt = {
        "architecture": required_output(["uname", "-m"]),
        "chip": hardware.get("chip_type", ""),
        "machine_identity_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "machine_model": hardware.get("machine_model", ""),
        "machine_name": hardware.get("machine_name", hardware.get("_name", "")),
        "memory": hardware.get("physical_memory", ""),
        "os_build": required_output(["sw_vers", "-buildVersion"]),
        "os_version": required_output(["sw_vers", "-productVersion"]),
    }
    require(all(receipt.values()), "physical machine receipt is incomplete")
    receipt["sha256"] = canonical_digest(receipt)
    return receipt


def system_snapshot() -> dict[str, Any]:
    load = os.getloadavg()
    processes = command_probe(
        ["ps", "-Ao", "pid=,pcpu=,pmem=,comm="], timeout=5.0
    )
    rows: list[dict[str, Any]] = []
    if processes.get("exit_code") == 0:
        for line in str(processes.get("output", "")).splitlines():
            fields = line.strip().split(None, 3)
            if len(fields) != 4:
                continue
            try:
                rows.append({
                    "pid": int(fields[0]),
                    "cpu_percent": float(fields[1]),
                    "memory_percent": float(fields[2]),
                    "command": fields[3],
                })
            except ValueError:
                continue
    rows.sort(key=lambda value: (-value["cpu_percent"], value["pid"]))
    return {
        "monotonic_seconds": time.monotonic(),
        "load_average": list(load),
        "memory_pressure": command_probe(["memory_pressure", "-Q"]),
        "swap_usage": command_probe(["sysctl", "vm.swapusage"]),
        "thermal_state": command_probe(["pmset", "-g", "therm"]),
        "top_processes_by_cpu": rows[:12],
    }


def process_sample(pid: int, elapsed_seconds: float) -> dict[str, Any] | None:
    completed = command_probe(
        ["ps", "-o", "rss=,vsz=,pcpu=", "-p", str(pid)], timeout=2.0
    )
    if completed.get("exit_code") != 0:
        return None
    fields = str(completed.get("output", "")).strip().split()
    if len(fields) != 3:
        return None
    try:
        return {
            "elapsed_seconds": elapsed_seconds,
            "resident_kib": int(fields[0]),
            "virtual_kib": int(fields[1]),
            "cpu_percent": float(fields[2]),
        }
    except ValueError:
        return None


def load_payloads(inputs: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    package = inputs / "Docs/media/native-runtime-source-package-20260915"
    receipt_path = package / "source-package-receipt-v1.json"
    require(receipt_path.is_file(), "published source package receipt is missing")
    receipt = strict_json_object(receipt_path.read_text(), "source package receipt")
    rows = receipt.get("inputs")
    require(isinstance(rows, dict) and rows, "source package input table is missing")
    payloads: dict[str, Path] = {}
    for name, row in rows.items():
        require(isinstance(name, str) and isinstance(row, dict), "payload row changed")
        path = inputs / str(row.get("path", ""))
        require(path.is_file(), f"source payload is missing: {name}")
        require(digest(path) == row.get("sha256"), f"source payload changed: {name}")
        payloads[name] = path
    for required in ("rigid", "muscle", "tendon", "support_contact", "joint_equalities"):
        require(required in payloads, f"source package omits {required}")
    return payloads, receipt


def validate_build(build: Path) -> dict[str, Any]:
    binary = build / "bin/metalrobo_numilab_human_myosim_visual_probe"
    metalrobo = build / "shaders/MetalRobo.metallib"
    matter = build / "matter/shaders/NumiMatter.metallib"
    cache = build / "CMakeCache.txt"
    for path in (binary, metalrobo, matter, cache):
        require(path.is_file(), f"production build artifact is missing: {path}")
    cache_text = cache.read_text(errors="replace")
    home_rows = [
        line.split("=", 1)[1]
        for line in cache_text.splitlines()
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")
    ]
    require(
        home_rows == [str(REPOSITORY_ROOT.resolve())],
        "build does not belong to the audited native source",
    )
    return {
        "binary": binary,
        "binary_sha256": digest(binary),
        "metalrobo_metallib_sha256": digest(metalrobo),
        "matter_metallib_sha256": digest(matter),
        "cmake_cache_sha256": digest(cache),
    }


def run_case(
    *,
    name: str,
    timestep: str,
    timestep_ns: int,
    steps: int,
    inputs: Path,
    build: Path,
    output: Path,
    payloads: dict[str, Path],
    source_receipt: dict[str, Any],
    build_receipt: dict[str, Any],
    machine: dict[str, Any],
    native_commit: str,
    input_commit: str,
    source_tree: str,
    launcher: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    directory = output / name
    directory.mkdir(parents=False, exist_ok=False)
    command_line = [
        str(launcher),
        "stand",
        str(payloads["rigid"].parent),
        str(directory / "unused-in-mechanics-mode.nhbones"),
        str(payloads["tendon"]),
        str(payloads["support_contact"]),
        str(directory / "frames"),
        "--mechanics-only",
        "--steps",
        str(steps),
        "--timestep",
        timestep,
    ]
    environment = {
        key: os.environ[key]
        for key in RUNTIME_ENVIRONMENT_KEYS
        if key in os.environ
    }
    environment.update({
        "NUMI_LAB_ROOT": str(REPOSITORY_ROOT.resolve()),
        "NUMI_BUILD_DIR": str(build.resolve()),
    })
    controlled_environment = {
        key: environment[key]
        for key in RUNTIME_ENVIRONMENT_KEYS
        if key in environment
    }
    controlled_environment.update({
        "NUMI_LAB_ROOT": environment["NUMI_LAB_ROOT"],
        "NUMI_BUILD_DIR": environment["NUMI_BUILD_DIR"],
    })
    stdout_path = directory / "stdout.txt"
    stderr_path = directory / "stderr.txt"
    before = system_snapshot()
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    timed_out = False
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        process = subprocess.Popen(
            command_line,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        deadline = started + timeout_seconds
        while process.poll() is None and time.monotonic() < deadline:
            sample = process_sample(process.pid, time.monotonic() - started)
            if sample is not None:
                samples.append(sample)
            time.sleep(1.0)
        if process.poll() is None:
            timed_out = True
            timeout_profile = directory / "timeout-profile.txt"
            with timeout_profile.open("x") as profile:
                try:
                    sampled = subprocess.run(
                        ["/usr/bin/sample", str(process.pid), "3", "1"],
                        stdout=profile,
                        stderr=subprocess.STDOUT,
                        timeout=12,
                        check=False,
                    )
                    profile.write(
                        "\nsample_exit_code=" + str(sampled.returncode) + "\n"
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    profile.write("\nsample_error=" + type(error).__name__ + "\n")
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        exit_code = process.wait()
    wall_seconds = time.monotonic() - started
    after = system_snapshot()
    stdout_text = stdout_path.read_text(errors="strict")
    stderr_text = stderr_path.read_text(errors="strict")
    metrics: dict[str, str] = {}
    trace: dict[str, Any] = {}
    terminal: dict[str, Any] = {}
    validation_error = ""
    try:
        require(not timed_out, f"{name} timed out")
        require(exit_code == 0, f"{name} exited {exit_code}")
        metrics = parse_metrics(stdout_text, name)
        trace = prefixed_object(stdout_text, "persistent_stand_trace=", name)
        terminal = prefixed_object(stdout_text, "stand_terminal_state=", name)
        validate_trace(trace, terminal, timestep_ns, steps)
    except (AuditError, KeyError, TypeError, ValueError) as error:
        validation_error = str(error)

    required_metrics = {
        "myosim_articulated_mechanics": "ok",
        "rendering_performed": "false",
        "visual_coverage_qualified": "false",
        "core_bodies": "157",
        "persistent_metal_horizon": "true",
        "persistent_completed_steps": str(steps),
        "persistent_root_assistance": "none",
        "stand_deterministic_replay": "bitwise",
        "persistent_passive_joint_law": "current_state_linear_backward_euler",
        "persistent_max_acceleration_semantics":
            "pre_projection_total_delta_v_divided_by_timestep",
    }
    mismatches = {
        key: {"observed": metrics.get(key), "required": value}
        for key, value in required_metrics.items()
        if metrics.get(key) != value
    }
    stderr_nonbanner_lines = [
        line
        for line in stderr_text.splitlines()
        if line.strip() and "Metal API Validation Enabled" not in line
    ]
    result: dict[str, Any] = {
        "schema": CASE_SCHEMA,
        "name": name,
        "native_commit": native_commit,
        "native_tree": source_tree,
        "native_worktree_clean": True,
        "input_commit": input_commit,
        "input_worktree_clean": True,
        "binary_sha256": build_receipt["binary_sha256"],
        "metalrobo_metallib_sha256":
            build_receipt["metalrobo_metallib_sha256"],
        "matter_metallib_sha256": build_receipt["matter_metallib_sha256"],
        "cmake_cache_sha256": build_receipt["cmake_cache_sha256"],
        "launcher_sha256": digest(launcher),
        "runner_sha256": digest(Path(__file__)),
        "source_package_receipt_sha256": canonical_digest(source_receipt),
        "payload_sha256": {
            key: digest(value) for key, value in sorted(payloads.items())
        },
        "physical_machine_receipt": machine,
        "physical_machine_receipt_sha256": machine["sha256"],
        "production_runtime": {
            "maximum_authoritative_submission_steps": 8,
            "diagnostic_segment_override": False,
            "metal_debug_layer": False,
            "persistent_stand_trace": True,
            "deterministic_replay": True,
            "runtime_environment": controlled_environment,
        },
        "command": command_line,
        "timestep_nanoseconds": timestep_ns,
        "step_count": steps,
        "duration_nanoseconds": timestep_ns * steps,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "wall_seconds": wall_seconds,
        "stdout_sha256": digest(stdout_path),
        "stderr_sha256": digest(stderr_path),
        "stderr_nonbanner_lines": stderr_nonbanner_lines,
        "validation_error": validation_error,
        "required_metric_mismatches": mismatches,
        "velocity_stage_diagnostics_complete":
            all(key in metrics for key in VELOCITY_STAGE_KEYS),
        "constraint_impulse_work_complete": bool(trace) and not validation_error,
        "compiled_static_balance": metrics.get("compiled_stand_balanced") == "true",
        "metrics": metrics,
        "runtime_telemetry": {
            "sample_interval_seconds": 1.0,
            "process_sample_count": len(samples),
            "peak_resident_kib": max(
                (sample["resident_kib"] for sample in samples), default=0
            ),
            "peak_virtual_kib": max(
                (sample["virtual_kib"] for sample in samples), default=0
            ),
            "maximum_process_cpu_percent": max(
                (sample["cpu_percent"] for sample in samples), default=0.0
            ),
            "before": before,
            "after": after,
        },
        "qualification": {
            "physical_execution": not validation_error,
            "bitwise_replay": trace.get("endpoint_equivalent") == "bitwise",
            "common_duration_member": timestep_ns * steps == COMMON_DURATION_NS,
            "performance": False,
            "force_convergence": False,
            "sustained_standing": False,
            "whole_human": False,
        },
        "boundary": (
            "One clean-production physical-Mac trace member with bitwise replay. "
            "Cross-grid comparison, declared acceptance thresholds, complete "
            "per-constraint reaction vectors, energy closure, sustained standing, "
            "recovery, and whole-Human behavior remain separate gates."
        ),
    }
    summary_path = directory / "case-summary.json"
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "name": name,
        "exit_code": exit_code,
        "wall_seconds": wall_seconds,
        "validation_error": validation_error,
        "peak_resident_kib":
            result["runtime_telemetry"]["peak_resident_kib"],
    }, sort_keys=True), flush=True)
    return result


def execute(
    inputs: Path, build: Path, output: Path, timeout_seconds: int
) -> int:
    inputs = inputs.resolve()
    build = build.resolve()
    output = output.resolve()
    require(sys.platform == "darwin", "common-duration execution requires macOS")
    require(timeout_seconds > 0, "timeout must be positive")
    require(not output.exists(), "output directory already exists")
    require(git_clean(REPOSITORY_ROOT), "native source worktree is not clean")
    require(git_clean(inputs), "Human input worktree is not clean")
    native_commit = git_output(REPOSITORY_ROOT, ["rev-parse", "HEAD"])
    source_tree = git_output(REPOSITORY_ROOT, ["rev-parse", "HEAD^{tree}"])
    input_commit = git_output(inputs, ["rev-parse", "HEAD"])
    payloads, source_receipt = load_payloads(inputs)
    launcher = inputs / ".numi/commands/human"
    require(launcher.is_file() and os.access(launcher, os.X_OK), "Human launcher is missing")
    build_receipt = validate_build(build)
    machine = machine_receipt()
    require(
        machine["architecture"] == "arm64" and machine["chip"] == "Apple M4 Pro",
        "physical common-duration qualification requires the selected M4 Pro",
    )
    output.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for name, timestep, timestep_ns, steps in CASE_SPEC:
        try:
            result = run_case(
                name=name,
                timestep=timestep,
                timestep_ns=timestep_ns,
                steps=steps,
                inputs=inputs,
                build=build,
                output=output,
                payloads=payloads,
                source_receipt=source_receipt,
                build_receipt=build_receipt,
                machine=machine,
                native_commit=native_commit,
                input_commit=input_commit,
                source_tree=source_tree,
                launcher=launcher,
                timeout_seconds=timeout_seconds,
            )
            results.append(result)
            if (
                result["exit_code"] != 0
                or result["timed_out"]
                or result["validation_error"]
                or result["stderr_nonbanner_lines"]
                or result["required_metric_mismatches"]
                or not result["velocity_stage_diagnostics_complete"]
                or not result["constraint_impulse_work_complete"]
                or not result["compiled_static_balance"]
            ):
                failures.append({"case": name, "error": "case contract failed"})
        except (AuditError, OSError, KeyError, TypeError, ValueError) as error:
            failures.append({"case": name, "error": str(error)})
            print(json.dumps(failures[-1], sort_keys=True), flush=True)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "passed" if not failures and len(results) == 4 else "failed",
        "native_commit": native_commit,
        "native_tree": source_tree,
        "input_commit": input_commit,
        "binary_sha256": build_receipt["binary_sha256"],
        "metalrobo_metallib_sha256":
            build_receipt["metalrobo_metallib_sha256"],
        "physical_machine_receipt": machine,
        "physical_machine_receipt_sha256": machine["sha256"],
        "common_duration_nanoseconds": COMMON_DURATION_NS,
        "case_summaries": {
            result["name"]: {
                "path": f"{result['name']}/case-summary.json",
                "sha256": digest(output / result["name"] / "case-summary.json"),
            }
            for result in results
        },
        "failures": failures,
        "qualification": {
            "physical_m4_execution": not failures and len(results) == 4,
            "all_four_common_duration_traces": not failures and len(results) == 4,
            "force_convergence": False,
            "sustained_standing": False,
            "performance": False,
        },
        "boundary": (
            "The manifest qualifies only complete source-identical physical M4 "
            "execution and retained bitwise-replayed traces. The cross-grid "
            "diagnostic and every physical acceptance claim are separate."
        ),
    }
    (output / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "output": str(output),
        "case_count": len(results),
        "failures": failures,
    }, sort_keys=True), flush=True)
    require(manifest["status"] == "passed", "common-duration matrix failed closed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    arguments = parser.parse_args()
    try:
        return execute(
            arguments.inputs,
            arguments.build,
            arguments.output,
            arguments.timeout_seconds,
        )
    except (AuditError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"physical Human common-duration audit rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
