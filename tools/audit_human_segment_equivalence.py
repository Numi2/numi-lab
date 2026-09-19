#!/usr/bin/env python3
"""Audit Human command-buffer segmentation without widening physics claims.

The diagnostic submission bound is applied only in the audit worktree. Short
cases compare segmented execution with the production monolithic route. A
separate 6.4 ms case exercises sustained cap-8 execution on a physical M4.
Neither report is a force-convergence or sustained-standing qualification.
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
import struct
import subprocess
import time

SOURCE = Path("apps/numilab_human_myosim_visual_probe.mm")
ENVIRONMENT_KEY = "NUMI_HUMAN_DIAGNOSTIC_MAXIMUM_AUTHORITATIVE_SUBMISSION_STEPS"
CASE_SCHEMA = "numi.human.segment-equivalence-case.v4"
REPORT_SCHEMA = "numi.human.segment-equivalence-audit.v4"
HOSTED_REPORT_SCHEMA = "numi.human.hosted-segment-probe.v1"
SUSTAINED_REPORT_SCHEMA = "numi.human.segment-sustained-audit.v1"
FLOAT32_EPSILON = 2.0 ** -23

WORK_PAIRS = {
    "persistent_contact_normal_impulse_work_j":
        "persistent_contact_normal_absolute_impulse_work_j",
    "persistent_contact_tangential_impulse_work_j":
        "persistent_contact_tangential_absolute_impulse_work_j",
    "persistent_equality_impulse_work_j":
        "persistent_equality_absolute_impulse_work_j",
    "persistent_source_limit_impulse_work_j":
        "persistent_source_limit_absolute_impulse_work_j",
}

CUMULATIVE_METRICS = {
    "stand_total_equality_impulse",
    "stand_total_equality_position_projection_m_or_rad",
    "stand_total_equality_velocity_projection_m_s_or_rad_s",
    "persistent_source_limit_absolute_impulse_ns_or_nms",
    *WORK_PAIRS.keys(),
    *WORK_PAIRS.values(),
}

# Wall-clock values are evidence, but are never mechanics-equivalence fields.
TIMING_METRICS = {
    "renderer_compile_ms_first_camera",
    "pose_stage_elapsed_ms",
    "selected_control_baseline_elapsed_ms",
    "stand_replay_elapsed_ms",
    "muscle_force_metal_elapsed_ms",
    "passive_fem_gpu_elapsed_ms",
    "anterior_thorax_coupled_transaction_elapsed_ms",
    "pectoralis_fascia_coupled_transaction_elapsed_ms",
    "source_support_metal_elapsed_ms",
}

DEVICE_METRICS = {
    "metal_pose_device",
    "renderer_device",
    "muscle_force_metal_device",
    "passive_fem_device",
    "anterior_thorax_device",
    "pectoralis_fascia_device",
    "source_support_metal_device",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def float32_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def ordered_float32(value: float) -> int:
    bits = float32_bits(value)
    return 0x80000000 - (bits & 0x7FFFFFFF) if bits & 0x80000000 else 0x80000000 + bits


def float32_ulp_distance(first: float, second: float) -> int:
    require(math.isfinite(first) and math.isfinite(second),
            "ULP comparison requires finite values")
    return abs(ordered_float32(first) - ordered_float32(second))


def prepare() -> int:
    text = SOURCE.read_text()
    require(text.count("#include <cstdlib>") == 1,
            "diagnostic environment support changed")
    old = """    constexpr std::uint32_t kMaximumAuthoritativeSubmissionSteps = 32u;
    const bool useSegmentedAuthoritativeHorizon =
"""
    new = """    std::uint32_t kMaximumAuthoritativeSubmissionSteps = 32u;
    if (const char* diagnosticMaximum = std::getenv(
            \"NUMI_HUMAN_DIAGNOSTIC_MAXIMUM_AUTHORITATIVE_SUBMISSION_STEPS\")) {
        std::size_t consumed = 0u;
        const unsigned long parsed = std::stoul(
            std::string(diagnosticMaximum), &consumed, 10
        );
        require(consumed == std::strlen(diagnosticMaximum) &&
                    parsed >= 1u &&
                    parsed <= MR_NUMI_HUMAN_STAND_MAX_STEPS,
                \"invalid diagnostic maximum authoritative submission steps\");
        kMaximumAuthoritativeSubmissionSteps =
            static_cast<std::uint32_t>(parsed);
    }
    const bool useSegmentedAuthoritativeHorizon =
"""
    require(text.count(old) == 1,
            "authoritative segment-bound source site changed")
    SOURCE.write_text(text.replace(old, new))
    subprocess.run(["git", "add", "--", str(SOURCE)], check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)
    changed = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], text=True
    ).splitlines()
    require(changed == [str(SOURCE)],
            f"unexpected diagnostic files: {changed}")
    return 0


def parse_output(text: str) -> tuple[dict, dict[str, str]]:
    terminals = [
        line.split("=", 1)[1]
        for line in text.splitlines()
        if line.startswith("stand_terminal_state=")
    ]
    require(len(terminals) == 1,
            "missing or ambiguous terminal state")
    terminal = json.loads(terminals[0])
    require(terminal.get("schema") == "numi.human.legacy-stand-terminal.v1",
            "terminal-state schema changed")
    summaries = [
        line for line in text.splitlines()
        if line.startswith("myosim_articulated_mechanics=")
    ]
    require(len(summaries) == 1,
            "missing or ambiguous mechanics summary")
    metrics: dict[str, str] = {}
    for token in shlex.split(summaries[0]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        require(key not in metrics, f"duplicate native metric: {key}")
        metrics[key] = value
    return terminal, metrics


def parse_stages(text: str) -> tuple[list[dict], list[dict]]:
    stages: list[dict] = []
    pending_segment: tuple[int, float] | None = None
    segments: list[dict] = []
    for line in text.splitlines():
        if not line.startswith("human_execution_stage="):
            continue
        values: dict[str, str] = {}
        for token in shlex.split(line):
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
        require(
            {"human_execution_stage", "wall_elapsed_ms", "stage_step"} <=
                values.keys(),
            "malformed Human execution stage",
        )
        stage = {
            "name": values["human_execution_stage"],
            "wall_elapsed_ms": float(values["wall_elapsed_ms"]),
            "step": int(values["stage_step"]),
        }
        stages.append(stage)
        if stage["name"] == "authoritative_segment_begin":
            require(pending_segment is None,
                    "nested authoritative segment stages")
            pending_segment = (stage["step"], stage["wall_elapsed_ms"])
        elif stage["name"] == "authoritative_segment_end":
            require(pending_segment is not None,
                    "authoritative segment ended without beginning")
            begin_step, begin_milliseconds = pending_segment
            require(stage["step"] > begin_step,
                    "authoritative segment did not advance")
            duration = stage["wall_elapsed_ms"] - begin_milliseconds
            require(duration >= 0.0 and math.isfinite(duration),
                    "authoritative segment duration is invalid")
            segments.append({
                "begin_step": begin_step,
                "end_step": stage["step"],
                "step_count": stage["step"] - begin_step,
                "wall_milliseconds": duration,
            })
            pending_segment = None
    require(pending_segment is None,
            "authoritative segment stage is incomplete")
    return stages, segments


def run_case(name: str, maximum_steps: int, metal_debug_layer: bool,
             timestep_seconds: float, step_count: int,
             deterministic_replay: bool, timeout_seconds: int,
             binary: Path, payloads: dict[str, Path], output: Path) -> dict:
    directory = output
    directory.mkdir(parents=True, exist_ok=True)
    magic = payloads["support_contact"].read_bytes()[:6]
    contacts = (
        [2, 3, 4, 5, 6, 7] if magic == b"NHCNT1" else
        [5, 6, 8, 10, 12, 14] if magic == b"NHCNT2" else None
    )
    require(contacts is not None,
            "unsupported support payload ABI")
    command = [
        str(binary),
        str(payloads["rigid"]),
        str(payloads["muscle"]),
        str(directory / "frames"),
        "--tendon-payload", str(payloads["tendon"]),
        "--support-contact-payload", str(payloads["support_contact"]),
        "--joint-equality-payload", str(payloads["joint_equalities"]),
    ]
    for dof, value in [
        (2, "0.02"), (108, "0.1"), (109, "0.1"), (110, "0.1"),
        (122, "0.1"), (123, "0.1"), (124, "0.1"),
    ]:
        command += ["--support-stance-dof", str(dof), value]
    for contact in contacts:
        command += ["--support-stance-contact", str(contact)]
    command += [
        "--muscle-step-seconds", format(timestep_seconds, ".17g"),
        "--muscle-step-count", str(step_count),
        "--persistent-metal-stand",
        "--persistent-source-passive-joint-tissue",
        "--stand-contact-iterations", "64",
        "--dimension", "640",
        "--mechanics-only",
    ]
    if deterministic_replay:
        command.append("--stand-deterministic-replay")
    environment = {
        **os.environ,
        "NUMI_HUMAN_EXECUTION_STAGES": "1",
        ENVIRONMENT_KEY: str(maximum_steps),
    }
    if metal_debug_layer:
        environment["MTL_DEBUG_LAYER"] = "1"
    else:
        environment.pop("MTL_DEBUG_LAYER", None)
    started = time.monotonic()
    stdout_path = directory / "stdout.txt"
    stderr_path = directory / "stderr.txt"
    timed_out = False
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command, env=environment, text=True, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            returncode = None
    wall_seconds = time.monotonic() - started
    stdout_text = stdout_path.read_text()
    stderr_text = stderr_path.read_text()
    terminal: dict = {}
    metrics: dict[str, str] = {}
    stages: list[dict] = []
    segments: list[dict] = []
    parse_error = ""
    try:
        stages, segments = parse_stages(stdout_text)
        if returncode == 0:
            terminal, metrics = parse_output(stdout_text)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        parse_error = str(error)
    contract_matched = (
        returncode == 0 and not parse_error and
        terminal.get("step_count") == step_count and
        terminal.get("timestep_seconds") == timestep_seconds and
        terminal.get("root_assistance") is False and
        metrics.get("persistent_completed_steps") == str(step_count) and
        metrics.get("persistent_root_assistance") == "none" and
        metrics.get("persistent_passive_joint_law") ==
            "current_state_linear_backward_euler" and
        metrics.get("stand_deterministic_replay") ==
            ("bitwise" if deterministic_replay else "not_requested")
    )
    expected_segment_count = (
        math.ceil(step_count / maximum_steps) *
        (2 if deterministic_replay else 1)
        if maximum_steps < step_count else 0
    )
    segment_contract_matched = (
        len(segments) == expected_segment_count and
        all(
            0 < segment["step_count"] <= maximum_steps and
            0 <= segment["begin_step"] < segment["end_step"] <= step_count
            for segment in segments
        )
    )
    contract_matched = contract_matched and segment_contract_matched
    result = {
        "schema": CASE_SCHEMA,
        "name": name,
        "maximum_submission_steps": maximum_steps,
        "metal_debug_layer": metal_debug_layer,
        "timestep_seconds": timestep_seconds,
        "step_count": step_count,
        "duration_seconds": timestep_seconds * step_count,
        "deterministic_replay": deterministic_replay,
        "timeout_seconds": timeout_seconds,
        "command": command,
        "exit_code": returncode,
        "timed_out": timed_out,
        "parse_error": parse_error,
        "contract_matched": contract_matched,
        "wall_seconds": wall_seconds,
        "expected_segment_count": expected_segment_count,
        "segment_contract_matched": segment_contract_matched,
        "maximum_segment_wall_milliseconds": max(
            (segment["wall_milliseconds"] for segment in segments),
            default=0.0,
        ),
        "segments": segments,
        "execution_stages": stages,
        "terminal": terminal,
        "metrics": metrics,
        "stdout_sha256": digest(stdout_path),
        "stderr_sha256": digest(stderr_path),
        "stderr_tail": stderr_text.splitlines()[-20:],
    }
    (directory / "case.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "name": name,
        "contract_matched": contract_matched,
        "exit_code": returncode,
        "maximum_segment_wall_milliseconds":
            result["maximum_segment_wall_milliseconds"],
    }, sort_keys=True))
    return result


def maximum_delta(first: list[float], second: list[float]) -> tuple[float, int]:
    require(len(first) == len(second),
            "vector dimensions differ")
    deltas = [
        abs(float(a) - float(b))
        for a, b in zip(first, second, strict=True)
    ]
    value = max(deltas, default=0.0)
    return value, deltas.index(value) if deltas else -1


def load_payloads(inputs: Path) -> dict[str, Path]:
    package = inputs / "Docs/media/native-runtime-source-package-20260915"
    receipt = json.loads(
        (package / "source-package-receipt-v1.json").read_text()
    )
    payloads = {
        name: inputs / value["path"]
        for name, value in receipt["inputs"].items()
    }
    for name, path in payloads.items():
        require(digest(path) == receipt["inputs"][name]["sha256"],
                f"source payload changed: {name}")
    return payloads


def execute_case(inputs: Path, build: Path, output: Path, name: str,
                 maximum_steps: int, metal_debug_layer: bool,
                 timestep_seconds: float, step_count: int,
                 deterministic_replay: bool, timeout_seconds: int) -> int:
    payloads = load_payloads(inputs)
    output.mkdir(parents=True, exist_ok=True)
    binary = build / "bin/metalrobo_numilab_human_myosim_visual_probe"
    require(binary.is_file(), "Human audit binary is missing")
    metallibs = {
        "metalrobo": build / "shaders/MetalRobo.metallib",
        "matter": build / "matter/shaders/NumiMatter.metallib",
    }
    for library_name, path in metallibs.items():
        require(path.is_file(),
                f"Human audit metallib is missing: {library_name}")
    result = run_case(
        name, maximum_steps, metal_debug_layer, timestep_seconds, step_count,
        deterministic_replay, timeout_seconds, binary, payloads, output
    )
    result.update({
        "native_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "input_commit": subprocess.check_output(
            ["git", "-C", str(inputs), "rev-parse", "HEAD"], text=True
        ).strip(),
        "binary_sha256": digest(binary),
        "metallib_sha256": {
            key: digest(path) for key, path in sorted(metallibs.items())
        },
        "cmake_cache_sha256": digest(build / "CMakeCache.txt"),
        "diagnostic_source_sha256": digest(SOURCE),
        "audit_script_sha256": digest(Path(__file__)),
        "diagnostic_patch_sha256": digest_bytes(subprocess.check_output(
            ["git", "diff", "--cached", "--binary", "--", str(SOURCE)]
        )),
        "payload_sha256": {
            key: digest(path) for key, path in sorted(payloads.items())
        },
    })
    (output / "case.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return 0


def arithmetic_comparison(key: str, reference: dict, candidate: dict,
                          scale: float, reduction_terms: int) -> dict:
    require(key in reference["metrics"] and key in candidate["metrics"],
            f"missing cumulative comparison metric: {key}")
    first = float(reference["metrics"][key])
    second = float(candidate["metrics"][key])
    tolerance = reduction_terms * FLOAT32_EPSILON * max(1.0e-6, scale)
    return {
        "reference": first,
        "candidate": second,
        "absolute_delta": abs(first - second),
        "absolute_tolerance": tolerance,
        "within_bound": abs(first - second) <= tolerance,
        "float32_ulp_distance": float32_ulp_distance(first, second),
    }


def load_cases(cases_root: Path, expected: dict[str, dict]) -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for path in sorted(cases_root.rglob("case.json")):
        case = json.loads(path.read_text())
        require(case.get("schema") == CASE_SCHEMA,
                f"case schema changed: {path}")
        name = case.get("name")
        require(name in expected and name not in cases,
                f"unexpected or duplicate case: {name}")
        for field, value in expected[name].items():
            require(case.get(field) == value,
                    f"case identity changed: {name}: {field}")
        cases[name] = case
    require(cases.keys() == expected.keys(),
            f"case set changed: {sorted(cases)}")
    for name, case in cases.items():
        require(case.get("contract_matched") is True,
                f"{name} did not complete its native contract")
        require(case.get("segment_contract_matched") is True,
                f"{name} segment-stage contract changed")
    return cases


def require_exact_stack(cases: dict[str, dict], reference_name: str) -> None:
    reference = cases[reference_name]
    identity_fields = [
        "native_commit", "input_commit", "binary_sha256",
        "metallib_sha256", "cmake_cache_sha256",
        "diagnostic_source_sha256", "audit_script_sha256",
        "diagnostic_patch_sha256", "payload_sha256",
    ]
    for name, case in cases.items():
        for field in identity_fields:
            require(
                field in reference and field in case and
                reference[field] == case[field],
                f"{name} exact-stack identity changed: {field}",
            )


def terminal_comparison(reference: dict, candidate: dict) -> dict:
    deltas: dict[str, dict[str, float | int]] = {}
    for field in ["initial_q", "initial_v", "q", "v"]:
        reference_values = reference["terminal"][field]
        candidate_values = candidate["terminal"][field]
        value, owner = maximum_delta(
            reference_values, candidate_values
        )
        mismatches = [
            index
            for index, (first, second) in enumerate(zip(
                reference_values, candidate_values, strict=True
            ))
            if float32_bits(first) != float32_bits(second)
        ]
        first_mismatch = mismatches[0] if mismatches else None
        deltas[field] = {
            "maximum_numeric_delta": value,
            "maximum_numeric_delta_owner": owner,
            "bitwise_mismatch_count": len(mismatches),
            "first_bitwise_mismatch": first_mismatch,
            "reference_first_mismatch_bits": (
                f"{float32_bits(reference_values[first_mismatch]):08x}"
                if first_mismatch is not None else None
            ),
            "candidate_first_mismatch_bits": (
                f"{float32_bits(candidate_values[first_mismatch]):08x}"
                if first_mismatch is not None else None
            ),
        }
    return {
        "deltas": deltas,
        "bitwise_equivalent": all(
            value["bitwise_mismatch_count"] == 0 for value in deltas.values()
        ),
    }


def mechanics_comparison(reference: dict, candidate: dict,
                         cumulative_exact: bool) -> dict:
    reference_keys = set(reference["metrics"])
    candidate_keys = set(candidate["metrics"])
    require(reference_keys == candidate_keys,
            "native mechanics summary field set changed")
    require("persistent_stand_status_non_cumulative_hex" in reference_keys,
            "normalized stand-status receipt is missing")
    for case_name, case in [
        ("reference", reference), ("candidate", candidate)
    ]:
        receipt = case["metrics"][
            "persistent_stand_status_non_cumulative_hex"
        ]
        require(len(receipt) == 544 and
                    all(character in "0123456789abcdef"
                        for character in receipt),
                f"{case_name} normalized stand-status receipt is malformed")
    timing_keys = {
        key for key in reference_keys
        if key.endswith("_elapsed_ms") or key.endswith("_ms")
    }
    require(timing_keys <= TIMING_METRICS,
            f"unclassified timing metrics: {sorted(timing_keys - TIMING_METRICS)}")
    device_keys = {
        key for key in reference_keys
        if key == "renderer_device" or key.endswith("_device")
    }
    require(device_keys <= DEVICE_METRICS,
            f"unclassified device metrics: {sorted(device_keys - DEVICE_METRICS)}")
    for key in {
        "metal_pose_device", "muscle_force_metal_device",
        "source_support_metal_device",
    }:
        require(key in device_keys, f"missing authoritative Metal device: {key}")
        require(reference["metrics"][key] not in {"", "none"} and
                    candidate["metrics"][key] not in {"", "none"},
                f"Metal device is unavailable: {key}")
    exact_keys = sorted(
        reference_keys - TIMING_METRICS - DEVICE_METRICS - CUMULATIVE_METRICS
    )
    exact_metrics = {
        key: reference["metrics"][key] == candidate["metrics"][key]
        for key in exact_keys
    }
    cumulative: dict[str, dict] = {}
    for key in sorted(CUMULATIVE_METRICS):
        require(key in reference_keys, f"missing cumulative metric: {key}")
        if key in WORK_PAIRS:
            scale_key = WORK_PAIRS[key]
            scale = max(
                abs(float(reference["metrics"][scale_key])),
                abs(float(candidate["metrics"][scale_key])),
            )
        else:
            scale = max(
                abs(float(reference["metrics"][key])),
                abs(float(candidate["metrics"][key])),
            )
        item = arithmetic_comparison(
            key, reference, candidate, scale,
            int(reference["step_count"]),
        )
        item["exact"] = (
            reference["metrics"][key] == candidate["metrics"][key]
        )
        item["accepted"] = (
            item["exact"] if cumulative_exact else item["within_bound"]
        )
        cumulative[key] = item
    terminal = terminal_comparison(reference, candidate)
    return {
        "terminal": terminal,
        "exact_metrics": exact_metrics,
        "cumulative_reductions": cumulative,
        "timing_observations": {
            key: {
                "reference": reference["metrics"].get(key),
                "candidate": candidate["metrics"].get(key),
            }
            for key in sorted(TIMING_METRICS & reference_keys)
        },
        "device_observations": {
            key: {
                "reference": reference["metrics"].get(key),
                "candidate": candidate["metrics"].get(key),
            }
            for key in sorted(DEVICE_METRICS & reference_keys)
        },
        "passed": (
            terminal["bitwise_equivalent"] and
            all(exact_metrics.values()) and
            all(value["accepted"] for value in cumulative.values())
        ),
    }


def case_summary(case: dict) -> dict:
    return {
        "maximum_submission_steps": case["maximum_submission_steps"],
        "metal_debug_layer": case["metal_debug_layer"],
        "timestep_seconds": case["timestep_seconds"],
        "step_count": case["step_count"],
        "deterministic_replay": case["deterministic_replay"],
        "wall_seconds": case["wall_seconds"],
        "maximum_segment_wall_milliseconds":
            case["maximum_segment_wall_milliseconds"],
        "segment_count": len(case["segments"]),
        "stdout_sha256": case["stdout_sha256"],
        "stderr_sha256": case["stderr_sha256"],
        "native_commit": case["native_commit"],
        "input_commit": case["input_commit"],
        "binary_sha256": case["binary_sha256"],
        "metallib_sha256": case["metallib_sha256"],
        "cmake_cache_sha256": case["cmake_cache_sha256"],
        "diagnostic_source_sha256": case["diagnostic_source_sha256"],
        "audit_script_sha256": case["audit_script_sha256"],
        "diagnostic_patch_sha256": case["diagnostic_patch_sha256"],
        "payload_sha256": case["payload_sha256"],
    }


def compare_cases(cases_root: Path, output: Path) -> int:
    shared = {
        "timestep_seconds": 0.0000125,
        "step_count": 64,
        "deterministic_replay": True,
    }
    expected = {
        "monolithic64-debug-off": {
            **shared, "maximum_submission_steps": 64,
            "metal_debug_layer": False,
        },
        "cap8-debug-off": {
            **shared, "maximum_submission_steps": 8,
            "metal_debug_layer": False,
        },
        "cap16-debug-off": {
            **shared, "maximum_submission_steps": 16,
            "metal_debug_layer": False,
        },
        "cap32-debug-off": {
            **shared, "maximum_submission_steps": 32,
            "metal_debug_layer": False,
        },
        "cap8-debug-on": {
            **shared, "maximum_submission_steps": 8,
            "metal_debug_layer": True,
        },
    }
    cases = load_cases(cases_root, expected)
    reference_name = "monolithic64-debug-off"
    require_exact_stack(cases, reference_name)
    reference = cases[reference_name]
    comparisons = {
        name: mechanics_comparison(reference, cases[name], False)
        for name in ["cap8-debug-off", "cap16-debug-off", "cap32-debug-off"]
    }
    debug_comparison = mechanics_comparison(
        cases["cap8-debug-off"], cases["cap8-debug-on"], True
    )
    segmentation_equivalent = all(
        value["passed"] for value in comparisons.values()
    )
    equivalent = segmentation_equivalent and debug_comparison["passed"]
    cap8_debug_segment_wall = cases[
        "cap8-debug-on"
    ]["maximum_segment_wall_milliseconds"]

    result = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if equivalent else "failed",
        "step_count": 64,
        "timestep_seconds": 0.0000125,
        "duration_seconds": 0.0008,
        "reference_case": reference_name,
        "comparisons": comparisons,
        "debug_layer_comparison_at_cap8": debug_comparison,
        "cap8_debug_maximum_segment_wall_milliseconds":
            cap8_debug_segment_wall,
        "case_summaries": {
            name: case_summary(case)
            for name, case in cases.items()
        },
        "qualification": {
            "segment_caps_8_16_32_equivalent_to_monolithic_64":
                segmentation_equivalent,
            "terminal_q_v_bitwise_equivalent": all(
                value["terminal"]["bitwise_equivalent"]
                for value in comparisons.values()
            ),
            "all_non_timing_non_cumulative_summary_fields_exact": all(
                all(value["exact_metrics"].values())
                for value in comparisons.values()
            ),
            "full_non_cumulative_stand_status_receipt_exact": all(
                value["exact_metrics"][
                    "persistent_stand_status_non_cumulative_hex"
                ] for value in comparisons.values()
            ),
            "cumulative_float32_reductions_within_arithmetic_bound":
                all(
                    all(item["within_bound"] for item in
                        comparison["cumulative_reductions"].values())
                    for comparison in comparisons.values()
                ),
            "debug_layer_mechanics_exact_at_cap8":
                debug_comparison["passed"],
            "hosted_watchdog_containment": False,
            "force_convergence": False,
            "sustained_standing": False,
            "physical_m4_validation": False,
        },
        "boundary": (
            "Source-identical comparison of production monolithic 64-step "
            "execution against 8, 16, and 32-step host segmentation. Initial "
            "and terminal q/v and every non-timing, non-cumulative mechanics "
            "summary field must be exact. Declared cumulative float32 "
            "reductions use an arithmetic bound; debug-on and debug-off cap-8 "
            "mechanics must be exact. Per-segment wall time is observational "
            "only and does not qualify hosted watchdog containment, physical "
            "M4 execution, performance, force convergence, sustained standing, "
            "or whole-Human behavior."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "segment-equivalence.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "equivalent": equivalent,
        "debug_layer_mechanics_exact_at_cap8": debug_comparison["passed"],
        "cap8_debug_maximum_segment_wall_milliseconds":
            cap8_debug_segment_wall,
    }, sort_keys=True))
    require(equivalent,
            "host segmentation changed the authoritative Human result or "
            "exceeded the declared cumulative rounding bound")
    return 0


def compare_hosted(cases_root: Path, output: Path) -> int:
    shared = {
        "timestep_seconds": 0.0000125,
        "step_count": 64,
        "deterministic_replay": True,
    }
    expected = {
        "cap8-debug-off": {
            **shared, "maximum_submission_steps": 8,
            "metal_debug_layer": False,
        },
        "cap16-debug-off": {
            **shared, "maximum_submission_steps": 16,
            "metal_debug_layer": False,
        },
        "cap8-debug-on": {
            **shared, "maximum_submission_steps": 8,
            "metal_debug_layer": True,
        },
    }
    cases = load_cases(cases_root, expected)
    reference_name = "cap16-debug-off"
    require_exact_stack(cases, reference_name)
    cap_comparison = mechanics_comparison(
        cases[reference_name], cases["cap8-debug-off"], False
    )
    debug_comparison = mechanics_comparison(
        cases["cap8-debug-off"], cases["cap8-debug-on"], True
    )
    passed = cap_comparison["passed"] and debug_comparison["passed"]
    result = {
        "schema": HOSTED_REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "step_count": 64,
        "timestep_seconds": 0.0000125,
        "duration_seconds": 0.0008,
        "reference_case": reference_name,
        "cap8_vs_cap16": cap_comparison,
        "debug_layer_comparison_at_cap8": debug_comparison,
        "case_summaries": {
            name: case_summary(case) for name, case in cases.items()
        },
        "qualification": {
            "bounded_cap8_and_cap16_completed_with_bitwise_replay": True,
            "cap8_equivalent_to_cap16": cap_comparison["passed"],
            "debug_layer_mechanics_exact_at_cap8":
                debug_comparison["passed"],
            "monolithic_equivalence": False,
            "cap32_hosted_watchdog_containment": False,
            "full_6_4ms_hosted_watchdog_containment": False,
            "physical_m4_validation": False,
            "force_convergence": False,
            "sustained_standing": False,
        },
        "boundary": (
            "Fresh Apple-Paravirtual runners completed isolated 64-step "
            "cap-8 and cap-16 horizons with deterministic replay. Cap-8 and "
            "cap-16 mechanics are compared, but neither is a monolithic "
            "reference. This bounded probe does not qualify cap 32, the full "
            "6.4 ms target, physical M4 execution, force convergence, or "
            "sustained standing."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "hosted-segment-probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "passed": passed,
        "cap8_equivalent_to_cap16": cap_comparison["passed"],
        "debug_layer_mechanics_exact_at_cap8":
            debug_comparison["passed"],
    }, sort_keys=True))
    require(passed,
            "hosted cap-8/cap-16 mechanics or debug-layer identity changed")
    return 0


def compare_sustained(cases_root: Path, equivalence_report: Path,
                      machine_facts: Path, output: Path) -> int:
    shared = {
        "maximum_submission_steps": 8,
        "timestep_seconds": 0.0000125,
        "step_count": 512,
        "deterministic_replay": True,
    }
    expected = {
        "sustained-cap8-debug-off": {
            **shared, "metal_debug_layer": False,
        },
        "sustained-cap8-debug-on": {
            **shared, "metal_debug_layer": True,
        },
    }
    cases = load_cases(cases_root, expected)
    reference_name = "sustained-cap8-debug-off"
    require_exact_stack(cases, reference_name)
    equivalence = json.loads(equivalence_report.read_text())
    require(equivalence.get("schema") == REPORT_SCHEMA and
                equivalence.get("status") == "passed",
            "short-horizon segment equivalence is not qualified")
    short_identity = equivalence["case_summaries"][
        equivalence["reference_case"]
    ]
    for field in [
        "native_commit", "input_commit", "binary_sha256",
        "metallib_sha256", "cmake_cache_sha256",
        "diagnostic_source_sha256", "audit_script_sha256",
        "diagnostic_patch_sha256", "payload_sha256",
    ]:
        require(short_identity[field] == cases[reference_name][field],
                f"sustained audit stack differs from equivalence: {field}")
    debug_comparison = mechanics_comparison(
        cases[reference_name], cases["sustained-cap8-debug-on"], True
    )
    facts = machine_facts.read_text()
    physical_m4_mac_mini = (
        "Model Name: Mac mini" in facts and "Chip: Apple M4" in facts
    )
    native_completed = all(
        case["contract_matched"] and case["segment_contract_matched"]
        for case in cases.values()
    )
    replay_bitwise = all(
        case["metrics"].get("stand_deterministic_replay") == "bitwise"
        for case in cases.values()
    )
    completed = (
        native_completed and replay_bitwise and
        debug_comparison["passed"] and physical_m4_mac_mini
    )
    result = {
        "schema": SUSTAINED_REPORT_SCHEMA,
        "status": "passed" if completed else "failed",
        "duration_seconds": 0.0064,
        "step_count": 512,
        "timestep_seconds": 0.0000125,
        "maximum_submission_steps": 8,
        "deterministic_replay": True,
        "reference_case": reference_name,
        "debug_layer_comparison": debug_comparison,
        "machine_facts": facts,
        "case_summaries": {
            name: case_summary(case) for name, case in cases.items()
        },
        "qualification": {
            "short_horizon_segment_equivalence": True,
            "physical_m4_mac_mini": physical_m4_mac_mini,
            "cap8_6_4ms_horizon_completed": native_completed,
            "cap8_6_4ms_replay_bitwise": replay_bitwise,
            "debug_layer_mechanics_exact": debug_comparison["passed"],
            "hosted_watchdog_containment": False,
            "performance_closure": False,
            "force_convergence": False,
            "sustained_standing": False,
            "whole_human": False,
        },
        "boundary": (
            "A physical Apple M4 Mac mini completed source-identical 512-step "
            "cap-8 horizons at 12.5 microseconds with deterministic replay and "
            "debug-on/off mechanics identity. This closes only bounded 6.4 ms "
            "execution and replay for the audited scenario. It does not prove "
            "hosted-runner watchdog containment, performance closure, force "
            "convergence, sustained standing, or whole-Human validity."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "segment-sustained.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "completed": completed,
        "physical_m4_mac_mini": physical_m4_mac_mini,
        "debug_layer_mechanics_exact": debug_comparison["passed"],
    }, sort_keys=True))
    require(completed,
            "physical M4 sustained cap-8 execution did not qualify")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    execute_parser = subparsers.add_parser("execute-case")
    execute_parser.add_argument("--inputs", type=Path, required=True)
    execute_parser.add_argument("--build", type=Path, required=True)
    execute_parser.add_argument("--output", type=Path, required=True)
    execute_parser.add_argument("--name", required=True)
    execute_parser.add_argument("--maximum-steps", type=int, required=True)
    execute_parser.add_argument(
        "--timestep-seconds", type=float, default=0.0000125
    )
    execute_parser.add_argument("--step-count", type=int, default=64)
    execute_parser.add_argument(
        "--deterministic-replay", choices=["on", "off"], default="on"
    )
    execute_parser.add_argument("--timeout-seconds", type=int, default=1200)
    execute_parser.add_argument(
        "--metal-debug-layer", choices=["on", "off"], required=True
    )
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--cases", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    hosted_parser = subparsers.add_parser("compare-hosted")
    hosted_parser.add_argument("--cases", type=Path, required=True)
    hosted_parser.add_argument("--output", type=Path, required=True)
    sustained_parser = subparsers.add_parser("compare-sustained")
    sustained_parser.add_argument("--cases", type=Path, required=True)
    sustained_parser.add_argument(
        "--equivalence-report", type=Path, required=True
    )
    sustained_parser.add_argument("--machine-facts", type=Path, required=True)
    sustained_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        return prepare()
    if arguments.command == "execute-case":
        require(1 <= arguments.maximum_steps <= arguments.step_count,
                "unsupported audit segment cap")
        require(1 <= arguments.step_count <= 512,
                "unsupported audit step count")
        require(math.isfinite(arguments.timestep_seconds) and
                    arguments.timestep_seconds > 0.0,
                "unsupported audit timestep")
        require(arguments.timeout_seconds > 0,
                "unsupported audit timeout")
        return execute_case(
            arguments.inputs, arguments.build, arguments.output,
            arguments.name, arguments.maximum_steps,
            arguments.metal_debug_layer == "on", arguments.timestep_seconds,
            arguments.step_count, arguments.deterministic_replay == "on",
            arguments.timeout_seconds,
        )
    if arguments.command == "compare-sustained":
        return compare_sustained(
            arguments.cases, arguments.equivalence_report,
            arguments.machine_facts, arguments.output,
        )
    if arguments.command == "compare-hosted":
        return compare_hosted(arguments.cases, arguments.output)
    return compare_cases(arguments.cases, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
