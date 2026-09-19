#!/usr/bin/env python3
"""Audit Human command-buffer segmentation without widening physics claims.

The diagnostic submission bound is applied only in the audit worktree. Short
cases compare segmented execution with the production monolithic route. A
separate 6.4 ms case exercises sustained cap-8 execution on a physical M4.
An optimization comparison admits only one shader source delta and keeps its
single-run timing observational. None of these reports qualifies force
convergence or sustained standing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import struct
import subprocess
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = Path("apps/numilab_human_myosim_visual_probe.mm")
SOURCE = REPOSITORY_ROOT / SOURCE_RELATIVE
OPTIMIZATION_SHADER_RELATIVE = Path("src/metal/NumiHumanStand.metal")
OPTIMIZATION_SHADER = REPOSITORY_ROOT / OPTIMIZATION_SHADER_RELATIVE
ENVIRONMENT_KEY = "NUMI_HUMAN_DIAGNOSTIC_MAXIMUM_AUTHORITATIVE_SUBMISSION_STEPS"
CASE_SCHEMA = "numi.human.segment-equivalence-case.v5"
REPORT_SCHEMA = "numi.human.segment-equivalence-audit.v4"
HOSTED_REPORT_SCHEMA = "numi.human.hosted-segment-probe.v1"
SUSTAINED_REPORT_SCHEMA = "numi.human.segment-sustained-audit.v1"
OPTIMIZATION_REPORT_SCHEMA = "numi.human.optimization-equivalence.v1"
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
WORK_METRICS = set(WORK_PAIRS) | set(WORK_PAIRS.values())

NORMALIZED_CMAKE_KEYS = {
    "BUILD_TESTING",
    "CMAKE_BUILD_TYPE",
    "CMAKE_COMMAND",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_COMPILER_LAUNCHER",
    "CMAKE_CXX_FLAGS",
    "CMAKE_CXX_FLAGS_DEBUG",
    "CMAKE_CXX_FLAGS_MINSIZEREL",
    "CMAKE_CXX_FLAGS_RELEASE",
    "CMAKE_CXX_FLAGS_RELWITHDEBINFO",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_MAKE_PROGRAM",
    "CMAKE_OSX_ARCHITECTURES",
    "CMAKE_OSX_DEPLOYMENT_TARGET",
    "CMAKE_OSX_SYSROOT",
}

RUNTIME_ENVIRONMENT_KEYS = {
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


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return digest_bytes(canonical_json_bytes(value))


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and
        all(character in "0123456789abcdef" for character in value)
    )


def command_output(command: list[str], required: bool = True) -> str:
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if required:
        require(
            completed.returncode == 0,
            "provenance command failed: " + " ".join(command),
        )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_output(arguments: list[str]) -> str:
    return command_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments]
    )


def git_paths(arguments: list[str]) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments, "-z"]
    )
    return sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in output.split(b"\0") if item
    )


def relative_to_repository(path: Path) -> Path | None:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return None


def path_is_within(path: str, roots: list[Path]) -> bool:
    candidate = Path(path)
    return any(candidate == root or root in candidate.parents for root in roots)


def repository_tree_manifest_sha256() -> str:
    entries = subprocess.check_output([
        "git", "-C", str(REPOSITORY_ROOT), "ls-tree", "-r", "-z", "HEAD",
    ]).split(b"\0")
    excluded = {
        str(SOURCE_RELATIVE).encode("utf-8"),
        str(OPTIMIZATION_SHADER_RELATIVE).encode("utf-8"),
    }
    retained: list[bytes] = []
    for entry in entries:
        if not entry:
            continue
        _, path = entry.split(b"\t", 1)
        if path not in excluded:
            retained.append(entry)
    return digest_bytes(b"\0".join(retained) + b"\0")


def source_provenance(inputs: Path, build: Path, output: Path) -> dict:
    staged_paths = git_paths(["diff", "--cached", "--name-only", "HEAD"])
    unstaged_paths = git_paths(["diff", "--name-only"])
    untracked_paths = git_paths(["ls-files", "--others", "--exclude-standard"])
    excluded_roots = [
        relative
        for relative in [
            relative_to_repository(inputs),
            relative_to_repository(build),
            relative_to_repository(output),
        ]
        if relative is not None
    ]
    source_untracked_paths = [
        path for path in untracked_paths
        if not path_is_within(path, excluded_roots)
    ]
    excluded_execution_paths = [
        path for path in untracked_paths
        if path_is_within(path, excluded_roots)
    ]
    allowed_unstaged = {str(OPTIMIZATION_SHADER_RELATIVE)}
    staged_driver_only = staged_paths == [str(SOURCE_RELATIVE)]
    unstaged_shader_only = set(unstaged_paths) <= allowed_unstaged
    source_clean = (
        staged_driver_only and unstaged_shader_only and
        not source_untracked_paths
    )
    shader_patch = subprocess.check_output([
        "git", "-C", str(REPOSITORY_ROOT), "diff", "--binary", "HEAD",
        "--", str(OPTIMIZATION_SHADER_RELATIVE),
    ])
    head_shader = subprocess.check_output([
        "git", "-C", str(REPOSITORY_ROOT), "show",
        "HEAD:" + str(OPTIMIZATION_SHADER_RELATIVE),
    ])
    receipt = {
        "native_commit": git_output(["rev-parse", "HEAD"]),
        "native_tree": git_output(["rev-parse", "HEAD^{tree}"]),
        "target_shader_path": str(OPTIMIZATION_SHADER_RELATIVE),
        "target_shader_sha256": digest(OPTIMIZATION_SHADER),
        "target_shader_head_sha256": digest_bytes(head_shader),
        "target_shader_head_relative_patch_sha256":
            digest_bytes(shader_patch),
        "target_shader_head_relative_patch_empty": not shader_patch,
        "head_non_driver_non_target_manifest_sha256":
            repository_tree_manifest_sha256(),
        "staged_changed_paths": staged_paths,
        "unstaged_changed_paths": unstaged_paths,
        "source_untracked_paths": source_untracked_paths,
        "excluded_execution_untracked_paths": excluded_execution_paths,
        "diagnostic_driver_is_only_staged_change": staged_driver_only,
        "only_allowlisted_shader_is_unstaged": unstaged_shader_only,
        "source_clean_except_diagnostic_driver_and_target_shader":
            source_clean,
    }
    receipt["sha256"] = canonical_digest(receipt)
    return receipt


def parse_cmake_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        if "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        typed_key, value = line.split("=", 1)
        key, _ = typed_key.split(":", 1)
        if (key in NORMALIZED_CMAKE_KEYS or
                key.startswith("METALROBO_") or
                key.startswith("NUMI_")):
            values[key] = value
    return values


def build_configuration_receipt(build: Path) -> dict:
    build_root = str(build.resolve())
    repository_root = str(REPOSITORY_ROOT.resolve())
    cache = {
        key: value.replace(build_root, "<BUILD_ROOT>").replace(
            repository_root, "<REPOSITORY_ROOT>"
        )
        for key, value in sorted(
            parse_cmake_cache(build / "CMakeCache.txt").items()
        )
    }
    receipt = {
        "cmake_cache": cache,
        "configuration": cache.get("CMAKE_BUILD_TYPE", ""),
        "generator": cache.get("CMAKE_GENERATOR", ""),
    }
    receipt["sha256"] = canonical_digest(receipt)
    return receipt


def resolve_executable(configured: str, fallback: str) -> str:
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())
    resolved = shutil.which(fallback)
    require(resolved is not None,
            f"required executable is unavailable: {fallback}")
    return str(Path(resolved).resolve())


def build_tool_paths(build: Path) -> dict[str, str]:
    cache = parse_cmake_cache(build / "CMakeCache.txt")
    return {
        "cmake": resolve_executable(cache.get("CMAKE_COMMAND", ""), "cmake"),
        "compiler": resolve_executable(
            cache.get("CMAKE_CXX_COMPILER", ""), "c++"
        ),
        "ninja": resolve_executable(
            cache.get("CMAKE_MAKE_PROGRAM", ""), "ninja"
        ),
    }


def toolchain_receipt(build: Path, build_configuration: dict) -> dict:
    executables = build_tool_paths(build)
    receipt = {
        "cmake": command_output([executables["cmake"], "--version"]),
        "compiler": command_output([executables["compiler"], "--version"]),
        "executables": executables,
        "metal": command_output([
            "xcrun", "-sdk", "macosx", "metal", "--version",
        ]),
        "macos_sdk_version": command_output([
            "xcrun", "-sdk", "macosx", "--show-sdk-version",
        ]),
        "ninja": command_output([executables["ninja"], "--version"]),
        "xcode": command_output(["xcodebuild", "-version"]),
    }
    receipt["sha256"] = canonical_digest(receipt)
    return receipt


def shader_build_binding_receipt(build: Path, metallib: Path) -> dict:
    build_ninja = build / "build.ninja"
    ninja = build_tool_paths(build)["ninja"]
    receipt = {
        "build_graph_available": build_ninja.is_file(),
        "metalrobo_metallib_sha256": digest(metallib),
        "target_shader_sha256": digest(OPTIMIZATION_SHADER),
    }
    if build_ninja.is_file():
        target = str(metallib.resolve().relative_to(build.resolve()))
        clean = subprocess.run(
            [ninja, "-C", str(build), "-t", "clean", target],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        require(clean.returncode == 0,
                "failed to clean the audited MetalRobo metallib target")
        rebuild = subprocess.run(
            [ninja, "-C", str(build), target],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        require(rebuild.returncode == 0,
                "failed to force-rebuild the audited MetalRobo metallib")
        inputs_text = command_output([
            ninja, "-C", str(build), "-t", "inputs", target,
        ])
        commands_text = command_output([
            ninja, "-C", str(build), "-t", "commands", target,
        ])
        normalized_commands = commands_text.replace(
            str(build.resolve()), "<BUILD_ROOT>"
        ).replace(
            str(REPOSITORY_ROOT.resolve()), "<REPOSITORY_ROOT>"
        ).splitlines()
        normalized_inputs: list[dict[str, str | None]] = []
        shader_is_input = False
        for raw_value in inputs_text.splitlines():
            value = raw_value.strip()
            if not value:
                continue
            input_path = Path(value)
            if not input_path.is_absolute():
                input_path = build / input_path
            resolved = input_path.resolve()
            shader_is_input = shader_is_input or (
                resolved == OPTIMIZATION_SHADER.resolve()
            )
            normalized_inputs.append({
                "path": str(resolved).replace(
                    str(build.resolve()), "<BUILD_ROOT>"
                ).replace(
                    str(REPOSITORY_ROOT.resolve()), "<REPOSITORY_ROOT>"
                ),
                "sha256": digest(resolved) if resolved.is_file() else None,
            })
        dry_run = subprocess.run(
            [ninja, "-C", str(build), "-n", target],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        receipt.update({
            "metallib_target": target,
            "forced_clean_completed": True,
            "forced_rebuild_completed": True,
            "forced_rebuild_output_sha256":
                digest_bytes(rebuild.stdout.encode("utf-8")),
            "normalized_commands": normalized_commands,
            "normalized_transitive_inputs": sorted(
                normalized_inputs, key=lambda value: value["path"]
            ),
            "shader_is_transitive_input": shader_is_input,
            "target_up_to_date": (
                dry_run.returncode == 0 and
                "no work to do" in dry_run.stdout.lower()
            ),
            "target_up_to_date_output": dry_run.stdout.strip(),
        })
    else:
        receipt.update({
            "metallib_target": "",
            "forced_clean_completed": False,
            "forced_rebuild_completed": False,
            "forced_rebuild_output_sha256": "",
            "normalized_commands": [],
            "normalized_transitive_inputs": [],
            "shader_is_transitive_input": False,
            "target_up_to_date": False,
            "target_up_to_date_output": "build graph unavailable",
        })
    receipt["metalrobo_metallib_sha256"] = digest(metallib)
    receipt["sha256"] = canonical_digest(receipt)
    return receipt


def physical_machine_receipt() -> dict:
    profile_text = command_output([
        "system_profiler", "SPHardwareDataType", "-json",
    ])
    profile = json.loads(profile_text)
    hardware_items = profile.get("SPHardwareDataType", [])
    require(len(hardware_items) == 1,
            "physical machine receipt is unavailable")
    hardware = hardware_items[0]
    identity_text = command_output([
        "ioreg", "-rd1", "-c", "IOPlatformExpertDevice",
    ])
    identity = ""
    for line in identity_text.splitlines():
        if '"IOPlatformUUID"' in line and "=" in line:
            identity = line.split("=", 1)[1].strip().strip('"')
            break
    require(identity, "physical machine identity is unavailable")
    receipt = {
        "architecture": command_output(["uname", "-m"]),
        "chip": hardware.get("chip_type", ""),
        "machine_identity_sha256": digest_bytes(identity.encode("utf-8")),
        "machine_model": hardware.get("machine_model", ""),
        "machine_name": hardware.get("machine_name", hardware.get("_name", "")),
        "memory": hardware.get("physical_memory", ""),
        "os_build": command_output(["sw_vers", "-buildVersion"]),
        "os_version": command_output(["sw_vers", "-productVersion"]),
    }
    require(
        all(receipt[key] for key in [
            "architecture", "chip", "machine_identity_sha256",
            "machine_model", "machine_name", "memory", "os_build",
            "os_version",
        ]),
        "physical machine receipt is incomplete",
    )
    receipt["sha256"] = canonical_digest(receipt)
    return receipt


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
    subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "add", "--",
         str(SOURCE_RELATIVE)], check=True
    )
    subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "diff", "--cached", "--check"],
        check=True
    )
    changed = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "diff", "--cached",
         "--name-only"], text=True
    ).splitlines()
    require(changed == [str(SOURCE_RELATIVE)],
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
    directory = output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    magic = payloads["support_contact"].read_bytes()[:6]
    contacts = (
        [2, 3, 4, 5, 6, 7] if magic == b"NHCNT1" else
        [5, 6, 8, 10, 12, 14] if magic == b"NHCNT2" else None
    )
    require(contacts is not None,
            "unsupported support payload ABI")
    stance_dofs = [
        (2, "0.02"), (108, "0.1"), (109, "0.1"), (110, "0.1"),
        (122, "0.1"), (123, "0.1"), (124, "0.1"),
    ]
    environment = {
        key: os.environ[key]
        for key in sorted(RUNTIME_ENVIRONMENT_KEYS)
        if key in os.environ
    }
    environment.update({
        "NUMI_HUMAN_EXECUTION_STAGES": "1",
        ENVIRONMENT_KEY: str(maximum_steps),
    })
    if metal_debug_layer:
        environment["MTL_DEBUG_LAYER"] = "1"
    scenario_configuration = {
        "schema": "numi.human.segment-audit-scenario.v1",
        "deterministic_replay": deterministic_replay,
        "dimension": 640,
        "execution_stages": True,
        "maximum_submission_steps": maximum_steps,
        "mechanics_only": True,
        "metal_debug_layer": metal_debug_layer,
        "persistent_source_passive_joint_tissue": True,
        "persistent_stand_trace": False,
        "root_assistance": False,
        "runtime_environment": environment,
        "runtime_working_directory": "<REPOSITORY_ROOT>",
        "stand_contact_iterations": 64,
        "step_count": step_count,
        "support_contact_payload_magic": magic.decode("ascii"),
        "support_stance_contacts": contacts,
        "support_stance_dofs": [
            {"dof": dof, "value": value}
            for dof, value in stance_dofs
        ],
        "timestep_seconds": timestep_seconds,
        "timeout_seconds": timeout_seconds,
    }
    command = [
        str(binary.resolve()),
        str(payloads["rigid"].resolve()),
        str(payloads["muscle"].resolve()),
        str(directory / "frames"),
        "--tendon-payload", str(payloads["tendon"].resolve()),
        "--support-contact-payload", str(payloads["support_contact"].resolve()),
        "--joint-equality-payload", str(payloads["joint_equalities"].resolve()),
    ]
    for dof, value in stance_dofs:
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
    started = time.monotonic()
    stdout_path = directory / "stdout.txt"
    stderr_path = directory / "stderr.txt"
    timed_out = False
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command, env=environment, text=True, stdout=stdout, stderr=stderr,
            cwd=REPOSITORY_ROOT, start_new_session=True,
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
        "scenario_configuration": scenario_configuration,
        "scenario_configuration_sha256":
            canonical_digest(scenario_configuration),
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
        for a, b in zip(first, second)
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
    provenance = source_provenance(inputs, build, output)
    require(
        provenance[
            "source_clean_except_diagnostic_driver_and_target_shader"
        ],
        "native source has changes beyond the diagnostic driver and "
        "allowlisted Human stand shader",
    )
    build_configuration = build_configuration_receipt(build)
    toolchain = toolchain_receipt(build, build_configuration)
    machine = physical_machine_receipt()
    shader_build_binding = shader_build_binding_receipt(
        build, metallibs["metalrobo"]
    )
    result = run_case(
        name, maximum_steps, metal_debug_layer, timestep_seconds, step_count,
        deterministic_replay, timeout_seconds, binary, payloads, output
    )
    result.update({
        "native_commit": provenance["native_commit"],
        "source_provenance": provenance,
        "source_provenance_sha256": provenance["sha256"],
        "input_commit": subprocess.check_output(
            ["git", "-C", str(inputs), "rev-parse", "HEAD"], text=True
        ).strip(),
        "binary_sha256": digest(binary),
        "metallib_sha256": {
            key: digest(path) for key, path in sorted(metallibs.items())
        },
        "cmake_cache_sha256": digest(build / "CMakeCache.txt"),
        "normalized_build_configuration": build_configuration,
        "normalized_build_configuration_sha256":
            build_configuration["sha256"],
        "toolchain_receipt": toolchain,
        "toolchain_receipt_sha256": toolchain["sha256"],
        "physical_machine_receipt": machine,
        "physical_machine_receipt_sha256": machine["sha256"],
        "shader_build_binding": shader_build_binding,
        "shader_build_binding_sha256": shader_build_binding["sha256"],
        "diagnostic_source_sha256": digest(SOURCE),
        "audit_script_sha256": digest(Path(__file__)),
        "diagnostic_patch_sha256": digest_bytes(subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "diff", "--cached",
             "--binary", "--", str(SOURCE_RELATIVE)]
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
                reference_values, candidate_values
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


def validate_embedded_receipt(case: dict, field: str,
                              digest_field: str) -> None:
    receipt = case.get(field)
    require(isinstance(receipt, dict), f"missing case receipt: {field}")
    embedded_digest = receipt.get("sha256")
    require(isinstance(embedded_digest, str),
            f"missing embedded receipt digest: {field}")
    unhashed = dict(receipt)
    del unhashed["sha256"]
    require(canonical_digest(unhashed) == embedded_digest,
            f"embedded receipt digest changed: {field}")
    require(case.get(digest_field) == embedded_digest,
            f"outer receipt digest changed: {digest_field}")


def validate_optimization_case(case: dict, label: str) -> None:
    require(case.get("schema") == CASE_SCHEMA,
            f"{label} case schema changed")
    require(case.get("contract_matched") is True,
            f"{label} native contract did not complete")
    require(case.get("segment_contract_matched") is True,
            f"{label} segment contract did not complete")
    require(case.get("maximum_submission_steps") == 8,
            f"{label} is not a cap-8 case")
    require(case.get("metal_debug_layer") is False,
            f"{label} enabled the Metal debug layer")
    require(case.get("timestep_seconds") == 0.0000125,
            f"{label} timestep changed")
    require(case.get("step_count") in {64, 512},
            f"{label} step count is not an optimization audit horizon")
    require(case.get("deterministic_replay") is True,
            f"{label} did not request deterministic replay")
    require(case.get("metrics", {}).get("stand_deterministic_replay") ==
                "bitwise",
            f"{label} deterministic replay is not bitwise")
    scenario = case.get("scenario_configuration")
    require(isinstance(scenario, dict),
            f"{label} scenario configuration is missing")
    require(canonical_digest(scenario) ==
                case.get("scenario_configuration_sha256"),
            f"{label} scenario digest changed")
    require(
        scenario.get("persistent_stand_trace") is False and
        scenario.get("maximum_submission_steps") == 8 and
        scenario.get("metal_debug_layer") is False and
        scenario.get("timestep_seconds") == 0.0000125 and
        scenario.get("step_count") == case["step_count"] and
        scenario.get("deterministic_replay") is True,
        f"{label} canonical scenario is not the optimization contract",
    )
    runtime_environment = scenario.get("runtime_environment")
    require(isinstance(runtime_environment, dict),
            f"{label} controlled runtime environment is missing")
    allowed_environment = RUNTIME_ENVIRONMENT_KEYS | {
        "NUMI_HUMAN_EXECUTION_STAGES", ENVIRONMENT_KEY,
    }
    require(set(runtime_environment) <= allowed_environment and
                runtime_environment.get("NUMI_HUMAN_EXECUTION_STAGES") == "1" and
                runtime_environment.get(ENVIRONMENT_KEY) == "8" and
                "MTL_DEBUG_LAYER" not in runtime_environment and
                scenario.get("runtime_working_directory") ==
                    "<REPOSITORY_ROOT>",
            f"{label} runtime environment is not controlled")
    source = case.get("source_provenance")
    require(isinstance(source, dict),
            f"{label} source provenance is missing")
    validate_embedded_receipt(
        case, "source_provenance", "source_provenance_sha256"
    )
    require(source.get("native_commit") == case.get("native_commit"),
            f"{label} native commit provenance changed")
    require(source.get("target_shader_path") ==
                str(OPTIMIZATION_SHADER_RELATIVE),
            f"{label} optimization shader path changed")
    for field in [
        "target_shader_sha256", "target_shader_head_sha256",
        "target_shader_head_relative_patch_sha256",
        "head_non_driver_non_target_manifest_sha256",
    ]:
        require(is_sha256(source.get(field)),
                f"{label} source digest is malformed: {field}")
    require(source.get("staged_changed_paths") == [str(SOURCE_RELATIVE)],
            f"{label} staged paths include more than the diagnostic driver")
    require(
        set(source.get("unstaged_changed_paths", [])) <=
            {str(OPTIMIZATION_SHADER_RELATIVE)},
        f"{label} has a non-allowlisted unstaged source change",
    )
    require(not source.get("source_untracked_paths"),
            f"{label} has untracked source inputs")
    require(source.get(
        "source_clean_except_diagnostic_driver_and_target_shader"
    ) is True, f"{label} source cleanliness contract failed")
    validate_embedded_receipt(
        case, "normalized_build_configuration",
        "normalized_build_configuration_sha256",
    )
    validate_embedded_receipt(
        case, "toolchain_receipt", "toolchain_receipt_sha256"
    )
    validate_embedded_receipt(
        case, "physical_machine_receipt",
        "physical_machine_receipt_sha256",
    )
    validate_embedded_receipt(
        case, "shader_build_binding", "shader_build_binding_sha256"
    )
    binding = case["shader_build_binding"]
    require(binding.get("build_graph_available") is True and
                binding.get("forced_clean_completed") is True and
                binding.get("forced_rebuild_completed") is True and
                binding.get("shader_is_transitive_input") is True and
                binding.get("target_up_to_date") is True and
                bool(binding.get("normalized_commands")) and
                bool(binding.get("normalized_transitive_inputs")) and
                all(is_sha256(value.get("sha256")) for value in
                    binding.get("normalized_transitive_inputs", [])),
            f"{label} metallib is not proven current for the target shader")
    require(binding.get("target_shader_sha256") ==
                source.get("target_shader_sha256"),
            f"{label} shader/build binding source hash changed")
    require(binding.get("metalrobo_metallib_sha256") ==
                case.get("metallib_sha256", {}).get("metalrobo"),
            f"{label} shader/build binding metallib hash changed")
    machine = case["physical_machine_receipt"]
    require(machine.get("machine_identity_sha256"),
            f"{label} physical machine identity is unavailable")
    require(math.isfinite(float(case.get("wall_seconds", math.nan))) and
                float(case["wall_seconds"]) > 0.0,
            f"{label} wall time is invalid")


def optimization_mechanics_comparison(reference: dict,
                                      candidate: dict) -> dict:
    reference_keys = set(reference["metrics"])
    candidate_keys = set(candidate["metrics"])
    require(reference_keys == candidate_keys,
            "optimization mechanics summary field set changed")
    require(WORK_METRICS <= reference_keys,
            "optimization work telemetry is incomplete")
    require("persistent_stand_status_non_cumulative_hex" in reference_keys,
            "optimization normalized stand-status receipt is missing")
    for label, case in [
        ("baseline", reference), ("optimized", candidate)
    ]:
        status_receipt = case["metrics"][
            "persistent_stand_status_non_cumulative_hex"
        ]
        require(
            len(status_receipt) == 544 and
            all(character in "0123456789abcdef"
                for character in status_receipt),
            f"{label} normalized stand-status receipt is malformed",
        )
    timing_keys = {
        key for key in reference_keys
        if key.endswith("_elapsed_ms") or key.endswith("_ms")
    }
    require(timing_keys <= TIMING_METRICS,
            "unclassified optimization timing metrics: " +
            str(sorted(timing_keys - TIMING_METRICS)))
    device_keys = {
        key for key in reference_keys
        if key == "renderer_device" or key.endswith("_device")
    }
    require(device_keys <= DEVICE_METRICS,
            "unclassified optimization device metrics: " +
            str(sorted(device_keys - DEVICE_METRICS)))
    for key in {
        "metal_pose_device", "muscle_force_metal_device",
        "source_support_metal_device",
    }:
        require(key in device_keys,
                f"missing authoritative optimization device: {key}")
        require(reference["metrics"][key] not in {"", "none"} and
                    candidate["metrics"][key] not in {"", "none"},
                f"optimization Metal device is unavailable: {key}")
    exact_keys = sorted(reference_keys - TIMING_METRICS - WORK_METRICS)
    exact_metrics = {
        key: reference["metrics"][key] == candidate["metrics"][key]
        for key in exact_keys
    }
    work_comparisons: dict[str, dict] = {}
    for key in sorted(WORK_METRICS):
        reference_value = float(reference["metrics"][key])
        candidate_value = float(candidate["metrics"][key])
        require(math.isfinite(reference_value) and
                    math.isfinite(candidate_value),
                f"non-finite optimization work metric: {key}")
        scale_key = WORK_PAIRS.get(key, key)
        reference_scale = float(reference["metrics"][scale_key])
        candidate_scale = float(candidate["metrics"][scale_key])
        require(math.isfinite(reference_scale) and
                    math.isfinite(candidate_scale),
                f"non-finite optimization work scale: {scale_key}")
        scale = max(abs(reference_scale), abs(candidate_scale))
        tolerance = int(reference["step_count"]) * FLOAT32_EPSILON * scale
        delta = abs(reference_value - candidate_value)
        work_comparisons[key] = {
            "reference": reference_value,
            "candidate": candidate_value,
            "absolute_delta": delta,
            "absolute_tolerance": tolerance,
            "scale": scale,
            "scale_metric": scale_key,
            "within_bound": delta <= tolerance,
            "float32_ulp_distance": float32_ulp_distance(
                reference_value, candidate_value
            ),
        }
    signed_absolute_invariants: dict[str, dict] = {}
    for signed_key, absolute_key in sorted(WORK_PAIRS.items()):
        per_case: dict[str, dict] = {}
        for label, case in [
            ("baseline", reference), ("optimized", candidate)
        ]:
            signed = float(case["metrics"][signed_key])
            absolute = float(case["metrics"][absolute_key])
            tolerance = (
                int(case["step_count"]) * FLOAT32_EPSILON * abs(absolute)
            )
            per_case[label] = {
                "signed_magnitude": abs(signed),
                "absolute_work": absolute,
                "absolute_work_nonnegative": absolute >= 0.0,
                "signed_within_absolute_work":
                    abs(signed) <= absolute + tolerance,
                "rounding_tolerance": tolerance,
            }
        signed_absolute_invariants[signed_key] = per_case
    terminal = terminal_comparison(reference, candidate)
    exact_passed = all(exact_metrics.values())
    work_passed = all(
        comparison["within_bound"]
        for comparison in work_comparisons.values()
    )
    invariants_passed = all(
        values["absolute_work_nonnegative"] and
        values["signed_within_absolute_work"]
        for pair in signed_absolute_invariants.values()
        for values in pair.values()
    )
    return {
        "terminal": terminal,
        "exact_non_timing_non_work_metrics": exact_metrics,
        "work_telemetry": work_comparisons,
        "signed_absolute_work_invariants": signed_absolute_invariants,
        "timing_metrics": {
            key: {
                "baseline": reference["metrics"][key],
                "optimized": candidate["metrics"][key],
            }
            for key in sorted(timing_keys)
        },
        "passed": (
            terminal["bitwise_equivalent"] and exact_passed and
            work_passed and invariants_passed
        ),
    }


def compare_optimization(baseline_path: Path, optimized_path: Path,
                         output: Path) -> int:
    baseline = json.loads(baseline_path.read_text())
    optimized = json.loads(optimized_path.read_text())
    validate_optimization_case(baseline, "baseline")
    validate_optimization_case(optimized, "optimized")
    for field in [
        "maximum_submission_steps", "metal_debug_layer",
        "timestep_seconds", "step_count", "deterministic_replay",
        "timeout_seconds", "scenario_configuration",
        "scenario_configuration_sha256", "input_commit", "payload_sha256",
        "binary_sha256", "cmake_cache_sha256",
        "diagnostic_source_sha256", "audit_script_sha256",
        "diagnostic_patch_sha256", "normalized_build_configuration_sha256",
        "toolchain_receipt_sha256", "physical_machine_receipt_sha256",
    ]:
        require(baseline.get(field) == optimized.get(field),
                f"optimization anti-confound field changed: {field}")
    baseline_source = baseline["source_provenance"]
    optimized_source = optimized["source_provenance"]
    require(baseline["native_commit"] == optimized["native_commit"],
            "uncommitted optimization cases do not share one native HEAD")
    require(
        baseline_source["target_shader_head_sha256"] ==
            optimized_source["target_shader_head_sha256"],
        "optimization cases do not share one HEAD shader",
    )
    require(
        baseline_source["target_shader_head_relative_patch_empty"] is True and
        baseline_source["target_shader_sha256"] ==
            baseline_source["target_shader_head_sha256"] and
        baseline_source["unstaged_changed_paths"] == [],
        "baseline is not the clean HEAD shader",
    )
    require(
        optimized_source["target_shader_head_relative_patch_empty"] is False and
        optimized_source["target_shader_sha256"] !=
            optimized_source["target_shader_head_sha256"] and
        optimized_source["unstaged_changed_paths"] ==
            [str(OPTIMIZATION_SHADER_RELATIVE)],
        "optimized case is not one uncommitted allowlisted shader patch",
    )
    require(
        baseline_source["head_non_driver_non_target_manifest_sha256"] ==
            optimized_source[
                "head_non_driver_non_target_manifest_sha256"
            ],
        "non-allowlisted native source changed between optimization cases",
    )
    require(
        baseline_source["target_shader_sha256"] !=
            optimized_source["target_shader_sha256"],
        "optimization did not change the allowlisted Human stand shader",
    )
    require(
        baseline["metallib_sha256"]["metalrobo"] !=
            optimized["metallib_sha256"]["metalrobo"],
        "optimization did not change the MetalRobo metallib",
    )
    require(
        baseline["metallib_sha256"]["matter"] ==
            optimized["metallib_sha256"]["matter"],
        "optimization changed the unrelated NumiMatter metallib",
    )
    baseline_segment_schedule = [
        {
            "begin_step": segment["begin_step"],
            "end_step": segment["end_step"],
            "step_count": segment["step_count"],
        }
        for segment in baseline["segments"]
    ]
    optimized_segment_schedule = [
        {
            "begin_step": segment["begin_step"],
            "end_step": segment["end_step"],
            "step_count": segment["step_count"],
        }
        for segment in optimized["segments"]
    ]
    require(baseline_segment_schedule == optimized_segment_schedule,
            "optimization changed the authoritative segment schedule")
    baseline_stage_schedule = [
        {"name": stage["name"], "step": stage["step"]}
        for stage in baseline["execution_stages"]
    ]
    optimized_stage_schedule = [
        {"name": stage["name"], "step": stage["step"]}
        for stage in optimized["execution_stages"]
    ]
    require(baseline_stage_schedule == optimized_stage_schedule,
            "optimization changed the native execution-stage schedule")
    baseline_binding = baseline["shader_build_binding"]
    optimized_binding = optimized["shader_build_binding"]
    for field in [
        "build_graph_available", "forced_clean_completed",
        "forced_rebuild_completed", "metallib_target",
        "normalized_commands", "shader_is_transitive_input", "target_up_to_date",
    ]:
        require(baseline_binding[field] == optimized_binding[field],
                f"shader build binding changed: {field}")
    baseline_inputs = {
        value["path"]: value["sha256"]
        for value in baseline_binding["normalized_transitive_inputs"]
    }
    optimized_inputs = {
        value["path"]: value["sha256"]
        for value in optimized_binding["normalized_transitive_inputs"]
    }
    require(baseline_inputs.keys() == optimized_inputs.keys(),
            "shader build transitive input paths changed")
    shader_input = "<REPOSITORY_ROOT>/" + str(
        OPTIMIZATION_SHADER_RELATIVE
    )
    require(shader_input in baseline_inputs,
            "allowlisted shader is absent from normalized build inputs")
    require(
        baseline_inputs[shader_input] ==
            baseline_source["target_shader_sha256"] and
        optimized_inputs[shader_input] ==
            optimized_source["target_shader_sha256"],
        "shader build input digest is not bound to source provenance",
    )
    for path in baseline_inputs.keys() - {shader_input}:
        if path == "<BUILD_ROOT>" or path.startswith("<BUILD_ROOT>/"):
            continue
        require(baseline_inputs[path] == optimized_inputs[path],
                f"non-target shader build input changed: {path}")
    mechanics = optimization_mechanics_comparison(baseline, optimized)
    baseline_wall = float(baseline["wall_seconds"])
    optimized_wall = float(optimized["wall_seconds"])
    baseline_segment_wall = sum(
        float(segment["wall_milliseconds"])
        for segment in baseline["segments"]
    ) / 1000.0
    optimized_segment_wall = sum(
        float(segment["wall_milliseconds"])
        for segment in optimized["segments"]
    ) / 1000.0
    result = {
        "schema": OPTIMIZATION_REPORT_SCHEMA,
        "status": "passed" if mechanics["passed"] else "failed",
        "baseline_case": str(baseline_path),
        "optimized_case": str(optimized_path),
        "step_count": baseline["step_count"],
        "timestep_seconds": baseline["timestep_seconds"],
        "mechanics_comparison": mechanics,
        "provenance": {
            "baseline_native_commit": baseline["native_commit"],
            "optimized_native_commit": optimized["native_commit"],
            "native_commit_changed":
                baseline["native_commit"] != optimized["native_commit"],
            "baseline_target_shader_sha256":
                baseline_source["target_shader_sha256"],
            "optimized_target_shader_sha256":
                optimized_source["target_shader_sha256"],
            "baseline_target_shader_patch_sha256": baseline_source[
                "target_shader_head_relative_patch_sha256"
            ],
            "optimized_target_shader_patch_sha256": optimized_source[
                "target_shader_head_relative_patch_sha256"
            ],
            "baseline_binary_sha256": baseline["binary_sha256"],
            "optimized_binary_sha256": optimized["binary_sha256"],
            "binary_changed":
                baseline["binary_sha256"] != optimized["binary_sha256"],
            "baseline_metalrobo_metallib_sha256":
                baseline["metallib_sha256"]["metalrobo"],
            "optimized_metalrobo_metallib_sha256":
                optimized["metallib_sha256"]["metalrobo"],
            "only_allowlisted_shader_differs_beyond_diagnostic_driver": True,
            "authoritative_segment_schedule_exact": True,
            "native_execution_stage_schedule_exact": True,
            "scenario_configuration_sha256":
                baseline["scenario_configuration_sha256"],
            "normalized_build_configuration_sha256":
                baseline["normalized_build_configuration_sha256"],
            "toolchain_receipt_sha256":
                baseline["toolchain_receipt_sha256"],
            "physical_machine_receipt_sha256":
                baseline["physical_machine_receipt_sha256"],
        },
        "timing_observation": {
            "sample_count_per_variant": 1,
            "baseline_wall_seconds": baseline_wall,
            "optimized_wall_seconds": optimized_wall,
            "observed_wall_speedup": baseline_wall / optimized_wall,
            "baseline_authoritative_segment_wall_seconds":
                baseline_segment_wall,
            "optimized_authoritative_segment_wall_seconds":
                optimized_segment_wall,
            "observed_authoritative_segment_speedup": (
                baseline_segment_wall / optimized_segment_wall
                if optimized_segment_wall > 0.0 else None
            ),
            "performance_qualified": False,
        },
        "qualification": {
            "optimization_mechanics_equivalent_for_audited_scenario":
                mechanics["passed"],
            "terminal_q_v_bitwise_equivalent":
                mechanics["terminal"]["bitwise_equivalent"],
            "all_non_timing_non_work_metrics_exact": all(
                mechanics[
                    "exact_non_timing_non_work_metrics"
                ].values()
            ),
            "eight_work_metrics_within_paired_absolute_work_bounds": all(
                value["within_bound"]
                for value in mechanics["work_telemetry"].values()
            ),
            "performance_qualified": False,
            "force_convergence": False,
            "sustained_standing": False,
            "whole_human": False,
        },
        "boundary": (
            "One immutable baseline and one immutable optimized cap-8, "
            "debug-off case on the same machine and toolchain establish only "
            "bitwise trajectory preservation, exact non-timing/non-work "
            "telemetry, and bounded constraint-work telemetry for this "
            "scenario. Wall-time ratios are observational single samples. "
            "They do not qualify performance, force convergence, sustained "
            "standing, or whole-Human behavior; a performance claim requires "
            "repeated interleaved runs with load, thermal, and counter evidence."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "optimization-equivalence.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "mechanics_equivalent": mechanics["passed"],
        "observed_wall_speedup": baseline_wall / optimized_wall,
        "performance_qualified": False,
    }, sort_keys=True))
    require(mechanics["passed"],
            "optimization changed authoritative mechanics or exceeded the "
            "paired absolute-work arithmetic bound")
    return 0


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
        "normalized_build_configuration_sha256":
            case["normalized_build_configuration_sha256"],
        "toolchain_receipt_sha256": case["toolchain_receipt_sha256"],
        "physical_machine_receipt_sha256":
            case["physical_machine_receipt_sha256"],
        "shader_build_binding_sha256":
            case["shader_build_binding_sha256"],
        "scenario_configuration_sha256":
            case["scenario_configuration_sha256"],
        "source_provenance_sha256": case["source_provenance_sha256"],
        "source_provenance": case["source_provenance"],
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
    optimization_parser = subparsers.add_parser("compare-optimization")
    optimization_parser.add_argument(
        "--baseline-case", type=Path, required=True
    )
    optimization_parser.add_argument(
        "--optimized-case", type=Path, required=True
    )
    optimization_parser.add_argument("--output", type=Path, required=True)
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
    if arguments.command == "compare-optimization":
        return compare_optimization(
            arguments.baseline_case, arguments.optimized_case,
            arguments.output,
        )
    return compare_cases(arguments.cases, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
