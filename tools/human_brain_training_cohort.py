#!/usr/bin/env python3
"""Run independent native Brain/Human episodes as one bounded training job.

This process never steps physics or creates learning records. Each worker is the
same native Human/Brain executable used for a single physical rollout.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import statistics
import subprocess
import sys
import time


GIB = 1024**3
PROGRESS = re.compile(r"^human_standing_progress=accepted step=(\d+)\b")
PROFILE = re.compile(r"^human_training_step_profile=accepted step=(\d+)\b")
WITNESS = re.compile(r"^human_brain_joint_commit=accepted step=(\d+)\b")
STAGE = re.compile(
    r"^human_execution_stage=(native_horizon_begin|native_horizon_end) "
    r"wall_elapsed_ms=([0-9.eE+-]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(log: Path, steps: int, exit_code: int) -> dict:
    progress, profiles, witnesses = [], [], []
    physical_trace = hashlib.sha256()
    joint_fingerprints = []
    witness_integrity = True
    physical_gpu_ms, brain_completion_ms = [], []
    stages = {}
    static_cache_status = None
    max_penetration = 0.0
    minimum_contacts = None
    terminal_root_xyz = None
    unassisted = True
    failed = False
    with log.open(errors="replace") as source:
        for line in source:
            if (match := PROGRESS.match(line)):
                progress.append(int(match.group(1)))
                physical_trace.update(line.encode())
                for name in ("root_assistance_force_n", "root_assistance_torque_nm"):
                    value = re.search(rf"\b{name}=([0-9.eE+-]+)", line)
                    unassisted &= value is not None and float(value.group(1)) == 0.0
                penetration = re.search(r"\bpenetration_m=([0-9.eE+-]+)", line)
                contacts = re.search(r"\bcontact_count=(\d+)", line)
                position = re.search(
                    r"\broot_xyz_m=\[([0-9.eE+-]+),([0-9.eE+-]+),([0-9.eE+-]+)\]",
                    line)
                if position:
                    terminal_root_xyz = [float(position.group(i)) for i in (1, 2, 3)]
                if penetration:
                    max_penetration = max(max_penetration, float(penetration.group(1)))
                if contacts:
                    count = int(contacts.group(1))
                    minimum_contacts = count if minimum_contacts is None else min(minimum_contacts, count)
            if (match := PROFILE.match(line)):
                profiles.append(int(match.group(1)))
                for field, values in (("physical_gpu_ms", physical_gpu_ms),
                                      ("brain_completion_wall_ms", brain_completion_ms)):
                    metric = re.search(rf"\b{field}=([0-9.eE+-]+)", line)
                    if metric:
                        values.append(float(metric.group(1)))
            if (match := WITNESS.match(line)):
                step = int(match.group(1))
                witnesses.append(step)
                generation = re.search(r"\bbrain_generation=(\d+)", line)
                fingerprint = re.search(r"\bjoint_commit_fingerprint=(\d+)", line)
                witness_integrity &= (
                    generation is not None and int(generation.group(1)) == step
                    and fingerprint is not None
                    and all(f"{flag}=true" in line for flag in (
                        "physical_motor_same_command",
                        "accepted_consequence_followup_command",
                        "same_native_owner_queue"))
                )
                if fingerprint:
                    joint_fingerprints.append(int(fingerprint.group(1)))
            if (match := STAGE.match(line)):
                stages[match.group(1)] = float(match.group(2))
            if line.startswith("human_static_equilibrium_cache="):
                static_cache_status = line.split("=", 1)[1].split(" ", 1)[0]
            failed |= "human_brain_completion=failed" in line
    expected = list(range(1, steps + 1))
    expected_witnesses = sorted({1, steps, *range(1000, steps + 1, 1000)})
    accepted = (
        exit_code == 0 and not failed and unassisted
        and progress == expected and profiles == expected
        and len(physical_gpu_ms) == steps and len(brain_completion_ms) == steps
        and min(physical_gpu_ms) > 0 and min(brain_completion_ms) > 0
        and witnesses == expected_witnesses
        and witness_integrity
        and set(stages) == {"native_horizon_begin", "native_horizon_end"}
        and stages.get("native_horizon_end", 0) > stages.get("native_horizon_begin", 0)
    )
    return {
        "accepted": accepted,
        "exitCode": exit_code,
        "acceptedPhysicalSteps": len(progress),
        "profiledSteps": len(profiles),
        "jointCommitWitnessSteps": witnesses,
        "jointCommitFingerprints": joint_fingerprints,
        "expectedJointCommitWitnessSteps": expected_witnesses,
        "jointWitnessIntegrity": witness_integrity,
        "physicalProgressSHA256": physical_trace.hexdigest(),
        "medianPhysicalGPUMilliseconds": (
            statistics.median(physical_gpu_ms) if physical_gpu_ms else None
        ),
        "medianBrainCompletionMilliseconds": (
            statistics.median(brain_completion_ms) if brain_completion_ms else None
        ),
        "unassisted": unassisted,
        "minimumContacts": minimum_contacts,
        "terminalRootXYZMeters": terminal_root_xyz,
        "maximumPenetrationM": max_penetration,
        "nativeHorizonSeconds": (
            (stages["native_horizon_end"] - stages["native_horizon_begin"]) / 1000
            if set(stages) == {"native_horizon_begin", "native_horizon_end"} else None
        ),
        "staticEquilibriumCache": static_cache_status,
        "logSHA256": sha256(log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--brain-dylib", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--bones", type=Path, required=True)
    parser.add_argument("--muscle-surfaces", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1314213193)
    parser.add_argument("--same-seed", action="store_true",
                        help="Use one seed for a matched throughput benchmark")
    parser.add_argument("--sensor-audit", action="store_true")
    parser.add_argument("--push-start-step", type=int)
    parser.add_argument("--push-duration-steps", type=int)
    parser.add_argument("--push-x-force-n", type=float, action="append", default=[],
                        help="One x-axis force per worker; repeat once per worker")
    parser.add_argument("--source-revision", default="unavailable")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or not 1 <= args.workers <= 16:
        parser.error("steps must be 1..10000 and workers must be 1..16")
    if args.seed < 0 or args.seed + args.workers > 2**32:
        parser.error("seed range exceeds UInt32")
    push_requested = bool(args.push_x_force_n)
    if push_requested != (args.push_start_step is not None):
        parser.error("push forces and push start step must be supplied together")
    if push_requested != (args.push_duration_steps is not None):
        parser.error("push forces and push duration must be supplied together")
    if push_requested and (
        len(args.push_x_force_n) != args.workers
        or args.push_start_step < 1
        or args.push_duration_steps < 1
        or args.push_start_step + args.push_duration_steps - 1 > args.steps
        or not all(math.isfinite(force) for force in args.push_x_force_n)
    ):
        parser.error("push schedule needs one finite force per worker within the horizon")
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    required = {
        "launcher": args.launcher,
        "binary": args.build_dir / "bin/metalrobo_numilab_human_myosim_visual_probe",
        "labLibrary": args.build_dir / "lib/libmetalrobo.dylib",
        "humanMetallib": args.build_dir / "shaders/MetalRobo.metallib",
        "brainDylib": args.brain_dylib,
        "program": args.program,
        "rigid": args.source_dir / "myosim-fullbody-core-reference.nhrigid",
        "muscle": args.source_dir / "myosim-fullbody-muscle-reference.nhmyo",
        "contacts": args.source_dir / "myosim-fullbody-support-contact.nhcnt",
        "equalities": args.source_dir / "myosim-fullbody-joint-equalities.nheq",
        "tendons": args.source_dir / "numi-human-tendon-attachments.nhtendon",
        "bones": args.bones,
        "muscleSurfaces": args.muscle_surfaces,
    }
    absent = [name for name, path in required.items() if not path.is_file()]
    if absent:
        parser.error("missing inputs: " + ", ".join(absent))
    bundle = args.build_dir / "bin/NumiBrain_NumiBrainMetal.bundle"
    resources = sorted(path for path in bundle.rglob("*") if path.is_file())
    if not resources:
        parser.error("Brain Metal resource bundle is missing")
    output_parent = args.output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    with (output_parent / ".human-brain-training.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another Human Brain training job owns this output root")
        if shutil.disk_usage(output_parent).free < 5.5 * GIB:
            parser.error("Data volume is below the 5.5 GiB training start floor")
        args.output.mkdir()
        manifest = {
            "purpose": "native Brain/Human training throughput cohort",
            "evidenceBoundary": "bounded physical rollout, not learned behavior",
            "stepsPerWorker": args.steps,
            "workers": args.workers,
            "sameSeedBenchmark": args.same_seed,
            "sourceRevision": args.source_revision,
            "seeds": [args.seed if args.same_seed else args.seed + i
                      for i in range(args.workers)],
            "push": ({"startStep": args.push_start_step,
                      "durationSteps": args.push_duration_steps,
                      "xForceNewtons": args.push_x_force_n}
                     if push_requested else None),
            "artifactSHA256": {name: sha256(path) for name, path in required.items()},
            "brainBundleSHA256": {
                str(path.relative_to(bundle)): sha256(path) for path in resources
            },
        }
        static_cache_key = hashlib.sha256(json.dumps({
            "format": "numi-human-static-activation-cache-v1",
            "artifacts": manifest["artifactSHA256"],
            "brainBundle": manifest["brainBundleSHA256"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        static_cache_path = output_parent / ".human-brain-static-cache" / (
            static_cache_key + ".nhstatic")
        manifest["staticEquilibriumCacheKey"] = static_cache_key
        manifest["staticEquilibriumCachePath"] = str(static_cache_path)
        static_cache_path.parent.mkdir(exist_ok=True)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        env_base = os.environ.copy()
        env_base.update({
            "NUMI_HUMAN_BRAIN_BUILD_DIR": str(args.build_dir),
            "NUMI_HUMAN_BRAIN_BINARY": str(required["binary"]),
            "NUMI_HUMAN_BRAIN_DYLIB": str(args.brain_dylib),
            "NUMI_HUMAN_BRAIN_SOURCE_DIR": str(args.source_dir),
            "NUMI_HUMAN_BRAIN_BONES": str(args.bones),
            "NUMI_HUMAN_BRAIN_MUSCLE_SURFACES": str(args.muscle_surfaces),
            "NUMI_HUMAN_BRAIN_PROGRAM": str(args.program),
            "NUMI_HUMAN_BRAIN_STEPS": str(args.steps),
            "NUMI_HUMAN_BRAIN_SENSOR_AUDIT": "1" if args.sensor_audit else "0",
            "NUMI_HUMAN_BRAIN_MECHANICS_ONLY": "1",
            "NUMI_HUMAN_BRAIN_JOINT_PATH_CALIBRATION": "1",
            "NUMI_HUMAN_TRAINING_PROFILE": "1",
            "NUMI_HUMAN_EXECUTION_STAGES": "1",
            "NUMI_HUMAN_BRAIN_SOURCE_REVISION": args.source_revision,
            "NUMI_HUMAN_STATIC_EQUILIBRIUM_CACHE_PATH": str(static_cache_path),
            "NUMI_HUMAN_STATIC_EQUILIBRIUM_CACHE_KEY": static_cache_key,
            "DYLD_LIBRARY_PATH": str(args.build_dir / "lib"),
        })
        running = []
        started = time.monotonic()
        try:
            for index, seed in enumerate(manifest["seeds"]):
                target = args.output / f"worker-{index:02d}"
                env = env_base | {"NUMI_HUMAN_BRAIN_SEED": str(seed)}
                if push_requested:
                    env.update({
                        "NUMI_HUMAN_BRAIN_PUSH_START_STEP": str(args.push_start_step),
                        "NUMI_HUMAN_BRAIN_PUSH_DURATION_STEPS": str(args.push_duration_steps),
                        "NUMI_HUMAN_BRAIN_PUSH_FORCE_X_N": str(args.push_x_force_n[index]),
                        "NUMI_HUMAN_BRAIN_PUSH_FORCE_Y_N": "0",
                        "NUMI_HUMAN_BRAIN_PUSH_FORCE_Z_N": "0",
                    })
                stream = (args.output / f"worker-{index:02d}.launcher.log").open("wb")
                process = subprocess.Popen(
                    [str(args.launcher), str(target)], env=env,
                    stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                running.append((index, process, stream, target))
            stop_reason = None
            peak_aggregate_rss_kib = 0
            while any(process.poll() is None for _, process, _, _ in running):
                if shutil.disk_usage(output_parent).free < 5 * GIB:
                    stop_reason = "storage_floor"
                    for _, process, _, _ in running:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                    break
                process_groups = {process.pid for _, process, _, _ in running
                                  if process.poll() is None}
                listing = subprocess.run(
                    ["ps", "-A", "-o", "pgid=,rss="],
                    capture_output=True, text=True, check=False)
                aggregate_rss_kib = 0
                for line in listing.stdout.splitlines():
                    try:
                        group, rss = map(int, line.split())
                    except ValueError:
                        continue
                    if group in process_groups:
                        aggregate_rss_kib += rss
                peak_aggregate_rss_kib = max(
                    peak_aggregate_rss_kib, aggregate_rss_kib)
                if aggregate_rss_kib > 16 * GIB // 1024:
                    stop_reason = "aggregate_rss_limit"
                    for _, process, _, _ in running:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                    break
                time.sleep(0.5)
            results = []
            for index, process, stream, target in running:
                try:
                    exit_code = process.wait(timeout=10 if stop_reason else None)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    exit_code = process.wait()
                stream.close()
                log = target / "launch.log"
                item = receipt(log, args.steps, exit_code) if log.is_file() else {
                    "accepted": False, "exitCode": exit_code, "error": "native log missing"
                }
                item.update({"worker": index, "seed": manifest["seeds"][index]})
                results.append(item)
            wall = time.monotonic() - started
            all_accepted = stop_reason is None and all(item["accepted"] for item in results)
            unique_physical_traces = len({
                item.get("physicalProgressSHA256") for item in results
                if item.get("physicalProgressSHA256")
            })
            summary = {
                "status": "ACCEPTED_NATIVE_COHORT" if all_accepted else "FAILED",
                "resourceStopReason": stop_reason,
                "results": results,
                "aggregateAcceptedSteps": args.steps * args.workers if all_accepted else 0,
                "uniquePhysicalTraceCount": unique_physical_traces,
                "wallSeconds": wall,
                "acceptedStepsPerWallHour": (
                    args.steps * args.workers * 3600 / wall if all_accepted else 0
                ),
                "peakAggregateRSSKiB": peak_aggregate_rss_kib,
                "freeBytesAfter": shutil.disk_usage(output_parent).free,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(json.dumps({key: summary[key] for key in (
                "status", "aggregateAcceptedSteps", "wallSeconds",
                "acceptedStepsPerWallHour", "freeBytesAfter")}, sort_keys=True))
            return 0 if all_accepted else 1
        finally:
            for _, process, stream, _ in running:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                if not stream.closed:
                    stream.close()


if __name__ == "__main__":
    sys.exit(main())
