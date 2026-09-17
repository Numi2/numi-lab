#!/usr/bin/env python3
"""Compare a 64-step Human horizon with and without host segmentation.

The diagnostic override is applied only in the Actions worktree. The audit
requires source-identical inputs and exact terminal q/v equality. A mismatch is
retained as evidence and fails the job; it is not converted into a tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

SOURCE = Path("apps/numilab_human_myosim_visual_probe.mm")
ENVIRONMENT_KEY = "NUMI_HUMAN_DIAGNOSTIC_MAXIMUM_AUTHORITATIVE_SUBMISSION_STEPS"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare() -> int:
    text = SOURCE.read_text()
    require("#include <cstdlib>" not in text,
            "diagnostic include already exists")
    text = text.replace("#include <cstdint>\n", "#include <cstdint>\n#include <cstdlib>\n", 1)
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
    require(changed == [str(SOURCE)], f"unexpected diagnostic files: {changed}")
    return 0


def parse_output(text: str) -> tuple[dict, dict[str, str]]:
    terminals = [line.split("=", 1)[1] for line in text.splitlines()
                 if line.startswith("stand_terminal_state=")]
    require(len(terminals) == 1, "missing or ambiguous terminal state")
    terminal = json.loads(terminals[0])
    require(terminal.get("schema") == "numi.human.legacy-stand-terminal.v1",
            "terminal-state schema changed")
    summaries = [line for line in text.splitlines()
                 if line.startswith("myosim_articulated_mechanics=")]
    require(len(summaries) == 1, "missing or ambiguous mechanics summary")
    metrics: dict[str, str] = {}
    for token in shlex.split(summaries[0]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        require(key not in metrics, f"duplicate native metric: {key}")
        metrics[key] = value
    return terminal, metrics


def run_case(name: str, maximum_steps: int, binary: Path,
             payloads: dict[str, Path], output: Path) -> dict:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    magic = payloads["support_contact"].read_bytes()[:6]
    contacts = ([2, 3, 4, 5, 6, 7] if magic == b"NHCNT1" else
                [5, 6, 8, 10, 12, 14] if magic == b"NHCNT2" else None)
    require(contacts is not None, "unsupported support payload ABI")
    command = [str(binary), str(payloads["rigid"]), str(payloads["muscle"]),
               str(directory / "frames"),
               "--tendon-payload", str(payloads["tendon"]),
               "--support-contact-payload", str(payloads["support_contact"]),
               "--joint-equality-payload", str(payloads["joint_equalities"])]
    for dof, value in [(2, "0.02"), (108, "0.1"), (109, "0.1"),
                       (110, "0.1"), (122, "0.1"), (123, "0.1"),
                       (124, "0.1")]:
        command += ["--support-stance-dof", str(dof), value]
    for contact in contacts:
        command += ["--support-stance-contact", str(contact)]
    command += ["--muscle-step-seconds", "0.00005",
                "--muscle-step-count", "64",
                "--persistent-metal-stand",
                "--persistent-source-passive-joint-tissue",
                "--stand-contact-iterations", "64",
                "--dimension", "640", "--mechanics-only"]
    environment = {**os.environ, "MTL_DEBUG_LAYER": "1",
                   "NUMI_HUMAN_EXECUTION_STAGES": "1",
                   ENVIRONMENT_KEY: str(maximum_steps)}
    started = time.monotonic()
    completed = subprocess.run(command, env=environment, text=True,
                               capture_output=True, timeout=1200)
    wall_seconds = time.monotonic() - started
    stdout_path = directory / "stdout.txt"
    stderr_path = directory / "stderr.txt"
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    require(completed.returncode == 0,
            f"{name} failed: {completed.returncode}\n{completed.stderr}")
    terminal, metrics = parse_output(completed.stdout)
    require(terminal["step_count"] == 64 and
            terminal["timestep_seconds"] == 0.00005 and
            terminal["root_assistance"] is False,
            f"{name} terminal contract changed")
    require(metrics.get("persistent_completed_steps") == "64" and
            metrics.get("persistent_root_assistance") == "none" and
            metrics.get("persistent_passive_joint_law") ==
                "current_state_linear_backward_euler",
            f"{name} mechanics contract changed")
    return {
        "name": name,
        "maximum_submission_steps": maximum_steps,
        "wall_seconds": wall_seconds,
        "terminal": terminal,
        "metrics": metrics,
        "stdout_sha256": digest(stdout_path),
        "stderr_sha256": digest(stderr_path),
    }


def maximum_delta(first: list[float], second: list[float]) -> tuple[float, int]:
    require(len(first) == len(second), "vector dimensions differ")
    deltas = [abs(float(a) - float(b)) for a, b in zip(first, second, strict=True)]
    value = max(deltas, default=0.0)
    return value, deltas.index(value) if deltas else -1


def execute(inputs: Path, build: Path, output: Path) -> int:
    package = inputs / "Docs/media/native-runtime-source-package-20260915"
    receipt = json.loads((package / "source-package-receipt-v1.json").read_text())
    payloads = {name: inputs / value["path"]
                for name, value in receipt["inputs"].items()}
    for name, path in payloads.items():
        require(digest(path) == receipt["inputs"][name]["sha256"],
                f"source payload changed: {name}")
    output.mkdir(parents=True, exist_ok=True)
    binary = build / "bin/metalrobo_numilab_human_myosim_visual_probe"
    binary_hash = digest(binary)
    segmented = run_case("segmented-32", 32, binary, payloads, output)
    monolithic = run_case("monolithic-64", 64, binary, payloads, output)
    q_delta, q_owner = maximum_delta(segmented["terminal"]["q"],
                                     monolithic["terminal"]["q"])
    v_delta, v_owner = maximum_delta(segmented["terminal"]["v"],
                                     monolithic["terminal"]["v"])
    initial_q_delta, _ = maximum_delta(segmented["terminal"]["initial_q"],
                                        monolithic["terminal"]["initial_q"])
    initial_v_delta, _ = maximum_delta(segmented["terminal"]["initial_v"],
                                        monolithic["terminal"]["initial_v"])
    status_fields = [
        "persistent_max_acceleration",
        "persistent_max_free_acceleration",
        "persistent_max_constraint_delta_v",
        "persistent_max_pre_projection_delta_v",
        "persistent_max_published_delta_v",
        "persistent_max_penetration_m",
        "persistent_normal_impulse",
        "stand_max_equality_velocity_error",
        "stand_total_equality_impulse",
        "muscle_step_max_velocity_delta",
        "muscle_step_max_configuration_delta",
    ]
    metric_deltas = {}
    for key in status_fields:
        require(key in segmented["metrics"] and key in monolithic["metrics"],
                f"missing comparison metric: {key}")
        metric_deltas[key] = abs(float(segmented["metrics"][key]) -
                                 float(monolithic["metrics"][key]))
    equivalent = (initial_q_delta == 0.0 and initial_v_delta == 0.0 and
                  q_delta == 0.0 and v_delta == 0.0 and
                  all(value == 0.0 for value in metric_deltas.values()))
    result = {
        "schema": "numi.human.segment-equivalence-audit.v1",
        "status": "passed" if equivalent else "failed",
        "binary_sha256": binary_hash,
        "step_count": 64,
        "timestep_seconds": 0.00005,
        "duration_seconds": 0.0032,
        "initial_q_max_delta": initial_q_delta,
        "initial_v_max_delta": initial_v_delta,
        "terminal_q_max_delta": q_delta,
        "terminal_q_max_delta_index": q_owner,
        "terminal_v_max_delta": v_delta,
        "terminal_v_max_delta_index": v_owner,
        "metric_deltas": metric_deltas,
        "segmented": segmented,
        "monolithic": monolithic,
        "bitwise_equivalent": equivalent,
        "qualification": {
            "host_segmentation_equivalent_to_monolithic": equivalent,
            "force_convergence": False,
            "sustained_standing": False,
            "physical_m4_validation": False,
        },
        "boundary": (
            "Source-identical hosted Apple comparison of one 64-step command "
            "against two 32-step commands. Exact equality is required; no "
            "acceptance tolerance is inferred."
        ),
    }
    (output / "segment-equivalence.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "bitwise_equivalent": equivalent,
        "terminal_q_max_delta": q_delta,
        "terminal_v_max_delta": v_delta,
        "q_owner": q_owner,
        "v_owner": v_owner,
    }, sort_keys=True))
    require(equivalent,
            "host segmentation changed the authoritative Human trajectory")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--inputs", type=Path, required=True)
    execute_parser.add_argument("--build", type=Path, required=True)
    execute_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        return prepare()
    return execute(arguments.inputs, arguments.build, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
