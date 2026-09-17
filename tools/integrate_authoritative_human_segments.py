#!/usr/bin/env python3
"""Prepare and qualify watchdog-safe authoritative Numi Human horizons.

This tool is deliberately separate from GitHub workflow YAML. ``prepare``
applies one hash-bound source patch and the two reviewed follow-up corrections.
``execute`` runs the exact 157-body source package for 128 unassisted steps and
writes a fail-closed evidence receipt. It does not claim force convergence,
sustained standing, walking, or physical-M4 qualification.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zlib

PATCH_BYTES = 15451
PATCH_SHA256 = "1c439dbcd19f17ec3f429f77b750fe028b7f573a99ff62673536d906404389bf"
SOURCE_PATH = Path("apps/numilab_human_myosim_visual_probe.mm")


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
    carrier = Path(".github/workflows/integrate-segmented-human-horizon.yml")
    match = re.search(r"segmented_encoded = '([^']+)'", carrier.read_text())
    require(match is not None, "segmented source patch carrier is missing")
    patch = zlib.decompress(base64.b64decode(match.group(1), validate=True))
    require(len(patch) == PATCH_BYTES, "segmented source patch length changed")
    require(hashlib.sha256(patch).hexdigest() == PATCH_SHA256,
            "segmented source patch hash changed")
    subprocess.run(["git", "apply", "--check", "-"], input=patch, check=True)
    subprocess.run(["git", "apply", "--index", "-"], input=patch, check=True)

    text = SOURCE_PATH.read_text()
    old_capture = """    const auto runAuthoritativeHorizon = [&context, &model,
                                           &mergeStandStatus,
                                           useSegmentedAuthoritativeHorizon](
"""
    new_capture = """    const auto runAuthoritativeHorizon = [&context, &model,
                                           &mergeStandStatus,
                                           useSegmentedAuthoritativeHorizon,
                                           kMaximumAuthoritativeSubmissionSteps](
"""
    require(text.count(old_capture) == 1,
            "authoritative horizon capture site changed")
    text = text.replace(old_capture, new_capture)

    old_head = """    bool tendonBorrowedConsumerVerified = false;
    if (!tendonProgram.bindings.empty()) {
        tendonBorrowedConsumerVerified =
"""
    new_head = """    bool tendonBorrowedConsumerVerified = false;
    if (!tendonProgram.bindings.empty()) {
        const auto* borrowedTendonStatus =
            static_cast<const MRNumiHumanStandStatusGPU*>(
                acceptedTendonConsumer.statusSnapshot.contents
            );
        const std::uint32_t borrowedTendonStatusSteps =
            useSegmentedAuthoritativeHorizon
                ? ((stepCount - 1u) %
                    kMaximumAuthoritativeSubmissionSteps) + 1u
                : stepCount;
        const bool borrowedTendonStatusMatchesPublication =
            acceptedTendonConsumer.statusSnapshot != nil &&
            (!useSegmentedAuthoritativeHorizon
                ? std::memcmp(
                    acceptedTendonConsumer.statusSnapshot.contents,
                    metalResult.standStatuses.data(),
                    sizeof(MRNumiHumanStandStatusGPU)
                  ) == 0
                : borrowedTendonStatus != nullptr &&
                  borrowedTendonStatus->code == status.code &&
                  borrowedTendonStatus->environment == status.environment &&
                  borrowedTendonStatus->completedSteps ==
                    borrowedTendonStatusSteps &&
                  borrowedTendonStatus->failingIndex == status.failingIndex &&
                  borrowedTendonStatus->flags == status.flags &&
                  borrowedTendonStatus->activeContactCount ==
                    status.activeContactCount &&
                  borrowedTendonStatus->contactAndAcceleration.z ==
                    status.contactAndAcceleration.z &&
                  borrowedTendonStatus->tendonTransferCount ==
                    tendonProgram.bindings.size() *
                    borrowedTendonStatusSteps &&
                  borrowedTendonStatus->tendonEnvelopeTransferCount ==
                    tendonEnvelopeBindingCount *
                    borrowedTendonStatusSteps &&
                  borrowedTendonStatus->tendonPointTransferCount ==
                    (tendonProgram.bindings.size() -
                     tendonEnvelopeBindingCount) *
                    borrowedTendonStatusSteps &&
                  borrowedTendonStatus->tendonFailureCount == 0u);
        tendonBorrowedConsumerVerified =
"""
    require(text.count(old_head) == 1,
            "borrowed tendon verification site changed")
    text = text.replace(old_head, new_head)

    old_status = """            std::memcmp(
                acceptedTendonConsumer.statusSnapshot.contents,
                metalResult.standStatuses.data(),
                sizeof(MRNumiHumanStandStatusGPU)
            ) == 0;
"""
    require(text.count(old_status) == 1,
            "borrowed tendon status comparison site changed")
    text = text.replace(old_status,
                        "            borrowedTendonStatusMatchesPublication;\n")
    SOURCE_PATH.write_text(text)
    subprocess.run(["git", "add", "--", str(SOURCE_PATH)], check=True)
    changed = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], text=True
    ).splitlines()
    require(changed == [str(SOURCE_PATH)],
            f"unexpected staged Human files: {changed}")
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)
    print(json.dumps({
        "schema": "numi.human.authoritative-segment-source-preparation.v1",
        "source_path": str(SOURCE_PATH),
        "base_patch_bytes": PATCH_BYTES,
        "base_patch_sha256": PATCH_SHA256,
        "maximum_submission_steps": 32,
        "segmented_tendon_status_accounting": True,
    }, sort_keys=True))
    return 0


def load_metrics(stdout: str) -> dict[str, str]:
    summaries = [line for line in stdout.splitlines()
                 if line.startswith("myosim_articulated_mechanics=")]
    require(len(summaries) == 1, "missing or ambiguous mechanics summary")
    metrics: dict[str, str] = {}
    for token in shlex.split(summaries[0]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        require(key not in metrics, f"duplicate native metric: {key}")
        metrics[key] = value
    return metrics


def execute(inputs: Path, build: Path, output: Path) -> int:
    package = inputs / "Docs/media/native-runtime-source-package-20260915"
    receipt = json.loads((package / "source-package-receipt-v1.json").read_text())
    payloads = {name: inputs / value["path"]
                for name, value in receipt["inputs"].items()}
    for name, path in payloads.items():
        require(digest(path) == receipt["inputs"][name]["sha256"],
                f"source payload changed: {name}")

    magic = payloads["support_contact"].read_bytes()[:6]
    contacts = ([2, 3, 4, 5, 6, 7] if magic == b"NHCNT1" else
                [5, 6, 8, 10, 12, 14] if magic == b"NHCNT2" else None)
    require(contacts is not None, "unsupported support payload ABI")

    output.mkdir(parents=True, exist_ok=True)
    binary = build / "bin/metalrobo_numilab_human_myosim_visual_probe"
    command = [str(binary), str(payloads["rigid"]), str(payloads["muscle"]),
               str(output / "frames"),
               "--tendon-payload", str(payloads["tendon"]),
               "--support-contact-payload", str(payloads["support_contact"]),
               "--joint-equality-payload", str(payloads["joint_equalities"])]
    for dof, value in [(2, "0.02"), (108, "0.1"), (109, "0.1"),
                       (110, "0.1"), (122, "0.1"), (123, "0.1"),
                       (124, "0.1")]:
        command += ["--support-stance-dof", str(dof), value]
    for contact in contacts:
        command += ["--support-stance-contact", str(contact)]
    command += [
        "--muscle-step-seconds", "0.00005",
        "--muscle-step-count", "128",
        "--persistent-metal-stand",
        "--persistent-source-passive-joint-tissue",
        "--stand-contact-iterations", "64",
        "--stand-deterministic-replay",
        "--dimension", "640",
        "--mechanics-only",
    ]

    started = time.monotonic()
    completed = subprocess.run(
        command,
        env={**os.environ, "MTL_DEBUG_LAYER": "1",
             "NUMI_HUMAN_EXECUTION_STAGES": "1"},
        text=True, capture_output=True, timeout=1800,
    )
    wall_seconds = time.monotonic() - started
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    require(completed.returncode == 0,
            f"native segmented horizon failed: {completed.returncode}\n"
            f"{completed.stderr}")

    metrics = load_metrics(completed.stdout)
    expected = {
        "myosim_articulated_mechanics": "ok",
        "core_bodies": "157",
        "compiled_stand_recruited_muscles": "416",
        "persistent_metal_horizon": "true",
        "persistent_completed_steps": "128",
        "persistent_root_assistance": "none",
        "persistent_source_passive_joint_tissue": "true",
        "persistent_passive_joint_law":
            "current_state_linear_backward_euler",
        "stand_deterministic_replay": "bitwise",
        "persistent_stand_trace": "not_requested",
        "rendering_performed": "false",
        "visual_coverage_qualified": "false",
    }
    changed = {key: (metrics.get(key), value)
               for key, value in expected.items()
               if metrics.get(key) != value}
    require(not changed, f"execution contract changed: {changed}")

    stages = [line for line in completed.stdout.splitlines()
              if line.startswith("human_execution_stage=")]
    for stage in ["native_horizon_end", "deterministic_replay_end"]:
        require(any(f"human_execution_stage={stage}" in line and
                    "stage_step=128" in line for line in stages),
                f"{stage} did not report 128 steps")

    result = {
        "schema": "numi.human.authoritative-segmented-horizon.v2",
        "status": "partial",
        "step_count": 128,
        "timestep_seconds": 0.00005,
        "duration_seconds": 0.0064,
        "maximum_submission_steps": 32,
        "submission_count_per_horizon": 4,
        "deterministic_replay": True,
        "root_assistance": "none",
        "trace_requested": False,
        "wall_seconds": wall_seconds,
        "metrics": metrics,
        "binary_sha256": digest(binary),
        "stdout_sha256": digest(stdout_path),
        "stderr_sha256": digest(stderr_path),
        "input_revision": subprocess.check_output(
            ["git", "-C", str(inputs), "rev-parse", "HEAD"],
            text=True).strip(),
        "qualification": {
            "authoritative_horizon_completed": True,
            "watchdog_safe_submission_bound": True,
            "bitwise_replay": True,
            "borrowed_tendon_segment_accounting": True,
            "force_convergence": False,
            "sustained_standing": False,
            "physical_m4_validation": False,
        },
        "boundary": (
            "Hosted Apple 6.4 ms mechanics-only horizon. Segmentation, "
            "transaction accounting and deterministic replay are established; "
            "force convergence, sustained standing and physical-M4 "
            "qualification are not."
        ),
    }
    (output / "receipt.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "completed_steps": 128,
        "wall_seconds": wall_seconds,
        "device": metrics.get("source_support_metal_device"),
        "maximum_free_force_acceleration":
            metrics.get("persistent_max_free_force_acceleration"),
        "maximum_constraint_velocity_delta":
            metrics.get("persistent_max_constraint_velocity_delta"),
        "maximum_published_velocity_delta":
            metrics.get("persistent_max_published_velocity_delta"),
    }, sort_keys=True))
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
