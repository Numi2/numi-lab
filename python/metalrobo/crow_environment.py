"""Crow experiment orchestration; all transitions remain in native Metal.

Every invocation creates a new evidence directory and writes its plan before
execution. This module must never implement a second task/physics transition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


VARIANTS = {
    "v9-visual-neural": "crow_journey_window/crow-journey.sensor-fast.visual-observation.json",
    "v10-world-model": "crow_navigation_course/crow-navigation.sensor-fast.visual-observation.json",
}
CONTRACT = ("world_fingerprint", "task_fingerprint", "observation_fingerprint", "action_fingerprint")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    # Exclusive creation makes partial and failed trials durable, never resumable
    # by silently overwriting a result or its prospective plan.
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def capture(arguments: list[str], root: Path) -> str:
    return subprocess.check_output(arguments, cwd=root, text=True).strip()


def runtime(root: Path) -> dict:
    build = Path(os.environ.get("NUMI_CROW_BUILD_DIR", os.environ.get("NUMI_BUILD_DIR", root / "build"))).resolve()
    return {
        "rollout": build / "bin/metalrobo_task_rollout",
        "train": build / "bin/metalrobo_task_train",
        "library": build / "lib/libmetalrobo.dylib",
        "metallib": build / "shaders/MetalRobo.metallib",
        "head_check": build / "bin/metalrobo_crow_embodiment_check",
    }


def preflight(root: Path, profile: dict) -> dict:
    import mlx.core as mx
    import numpy as np
    from metalrobo import mlx_policy_worker  # actual learner dependencies

    del mlx_policy_worker
    paths = runtime(root)
    paths.update({name: root / "assets" / relative for name, relative in VARIANTS.items()})
    paths["profile"] = root / "profiles/american-crow-rl-v1.json"
    missing = [str(path) for path in paths.values() if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError(f"Missing runtime inputs: {missing}")
    if not mx.metal.is_available():
        raise RuntimeError("The native crow environment requires Apple Metal")
    free = shutil.disk_usage(root).free
    if free < profile["minimum_free_disk_gib"] * 1024**3:
        raise RuntimeError("Free disk is below the project's minimum")
    return {
        "schema": "numi.crow-preflight.v1", "host": platform.node(),
        "machine": platform.machine(), "macos": platform.mac_ver()[0],
        "python": sys.executable, "numpy": np.__version__,
        "mlx": importlib.metadata.version("mlx"),
        "metal_device": mx.device_info(), "free_disk_bytes": free,
        "inputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        "source_commit": capture(["git", "rev-parse", "HEAD"], root),
        "status": capture(["git", "status", "--porcelain=v1"], root),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)).hexdigest(),
        "untracked_inputs": {
            name: sha(root / name) for name in
            capture(["git", "ls-files", "--others", "--exclude-standard"], root).splitlines()
            if (root / name).is_file()
        },
        "readiness": "native engineering trials only; ecological qualification pending",
    }


def execute(output: Path, name: str, command: list[str], root: Path, timeout: int) -> dict:
    started = time.monotonic()
    exit_code = None
    timed_out = False
    with (output / f"{name}.json").open("x") as stdout, (output / f"{name}.stderr").open("x") as stderr:
        try:
            result = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, timeout=timeout)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    receipt = {
        "command": command, "exit_code": exit_code, "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": sha(output / f"{name}.json"),
        "stderr_sha256": sha(output / f"{name}.stderr"),
    }
    write(output / f"{name}.receipt.json", receipt)
    if timed_out or exit_code != 0:
        raise RuntimeError(f"{name} failed; retained receipt and stderr in {output}")
    with (output / f"{name}.json").open() as handle:
        return json.load(handle)


def rollout(root: Path, variant: str, envs: int, steps: int, chunk: int, seed: int,
            band: int, policy: Path | None = None, course: str = "training") -> list[str]:
    paths = runtime(root)
    command = [str(paths["rollout"]), "--metallib", str(paths["metallib"]),
               "--birdflow-american-crow-journey", "--birdflow-journey-variant", variant,
               "--envs", str(envs), "--steps", str(steps), "--repeats", "1",
               "--chunk", str(chunk), "--seed", str(seed), "--no-scheduled-resets",
               "--minimum-difficulty-band", str(band), "--maximum-difficulty-band", str(band)]
    if variant in VARIANTS:
        command += ["--visual-observation-config", str(root / "assets" / VARIANTS[variant])]
    command += ["--policy-pack", str(policy)] if policy else ["--native-policy"]
    if variant == "v10-world-model":
        command += ["--birdflow-navigation-course", course]
    return command


def validate_native(value: dict, *, envs: int, steps: int, seed: int, policy: bool,
                    variant: str | None = None, action_source: str | None = None) -> None:
    expected = {"environments": envs, "steps_per_repeat": steps, "repeats": 1,
                "benchmark_seed": seed, "scheduled_resets": False,
                "birdflow_journey_teacher": False, "visual_observation": variant != "v11-embodied",
                "action_source": action_source or ("policy_pack" if policy else "compiled_policy"),
                "failed_environment_steps": 0}
    if variant is not None:
        expected["birdflow_journey_variant"] = variant
    failures = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if failures:
        raise RuntimeError(f"Native evidence violated trial identity/validity: {failures}")
    if value.get("birdflow_journey_variant") == "v10-world-model":
        evidence = value.get("first_episode_navigation")
        if not isinstance(evidence, dict) or evidence.get("environment_count") != envs:
            raise RuntimeError("First-episode navigation evidence missing or incomplete")
        validate_first_episode(evidence, envs, steps)


def validate_first_episode(evidence: dict, envs: int, horizon: int) -> None:
    fields = ("steps", "maximum_waypoints", "completion_seen", "safe_completion",
              "physical_failure", "closed", "termination_reason", "first_completion_step")
    arrays = {field: evidence.get(field + "_by_environment") for field in fields}
    if any(not isinstance(array, list) or len(array) != envs for array in arrays.values()):
        raise RuntimeError("Incomplete first-episode lane arrays")
    for lane in range(envs):
        row = {key: array[lane] for key, array in arrays.items()}
        if type(row["steps"]) is not int or not 1 <= row["steps"] <= horizon:
            raise RuntimeError("Invalid first-episode duration")
        waypoints = row["maximum_waypoints"]
        if not isinstance(waypoints, (float, int)) or not math.isfinite(waypoints) or not 0 <= waypoints <= 5 or int(waypoints) != waypoints:
            raise RuntimeError("Invalid accepted waypoint count")
        for field in ("completion_seen", "safe_completion", "physical_failure", "closed"):
            if type(row[field]) is not bool:
                raise RuntimeError("First-episode lane flags must be Boolean")
        reason = row["termination_reason"]
        if type(reason) is not int or not 0 <= reason <= 7 or row["closed"] != (reason > 0):
            raise RuntimeError("Invalid first-episode termination identity")
        if row["physical_failure"] != (row["closed"] and reason not in (4, 7)):
            raise RuntimeError("Physical failure and termination disagree")
        if row["safe_completion"] != (row["completion_seen"] and not row["physical_failure"]):
            raise RuntimeError("Inconsistent safe-completion evidence")
        first = row["first_completion_step"]
        if row["completion_seen"]:
            if waypoints != 5 or type(first) is not int or not 1 <= first <= row["steps"]:
                raise RuntimeError("Completion lacks accepted full-route evidence")
        elif first is not None:
            raise RuntimeError("Unexpected completion timestamp")
        if not row["closed"] and row["steps"] != horizon:
            raise RuntimeError("Truncated first episode cannot satisfy a full horizon")
    counts = {
        "safe_completion_count": sum(arrays["safe_completion"]),
        "completion_seen_count": sum(arrays["completion_seen"]),
        "physical_failure_count": sum(arrays["physical_failure"]),
        "closed_episode_count": sum(arrays["closed"]),
        "horizon_censored_count": envs - sum(arrays["closed"]),
    }
    if any(type(evidence.get(key)) is not int or evidence[key] != count for key, count in counts.items()):
        raise RuntimeError("First-episode aggregate counts disagree with lanes")


def probe(root: Path, output: Path, profile: dict) -> dict:
    settings = profile["probe"]
    seed, envs, steps = settings["seed"], settings["environments"], settings["steps"]
    records = []
    baseline = None
    for name, chunk in [("replay-a", 1), ("replay-b", 1), ("replay-chunk8", 8)]:
        value = execute(output, name, rollout(root, "v9-visual-neural", envs, steps, chunk, seed, 4), root, settings["timeout_seconds"])
        validate_native(value, envs=envs, steps=steps, seed=seed, policy=False)
        keys = (*CONTRACT, "sensor_fingerprint", "policy_rollout_fingerprint")
        fingerprints = {key: value[key] for key in keys}
        if baseline is None:
            baseline = fingerprints
        elif fingerprints != baseline:
            raise RuntimeError(f"Deterministic replay failed at {name}")
        records.append({"name": name, "chunk": chunk, "fingerprints": fingerprints})
    for envs in settings["scale_environments"]:
        name = f"scale-{envs}"
        value = execute(output, name, rollout(root, "v9-visual-neural", envs, steps, 8, seed, 4), root, settings["timeout_seconds"])
        validate_native(value, envs=envs, steps=steps, seed=seed, policy=False)
        records.append({"name": name, "environments": envs,
                        "native_environment_steps_per_second": value["environment_control_steps_per_second"],
                        "retained_buffer_bytes": value["retained_buffer_bytes"],
                        "peak_aliased_bytes": value["peak_aliased_bytes"]})
    return {"status": "passed", "claim": "native replay and bounded rollout scaling only",
            "policy": "untrained compiled engineering policy", "records": records,
            "ecological_environment_qualified": False}


def bootstrap(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    source = args.source_policy.resolve(strict=True)
    layout = execute(output, "native-contract", rollout(root, args.variant, 1, 1, 1,
                     profile["probe"]["seed"], 0), root, 180)
    validate_native(layout, envs=1, steps=1, seed=profile["probe"]["seed"], policy=False)
    command = [sys.executable, "-m", "metalrobo.mlx_policy_worker", "initialize",
               "--actor-policy-pack", str(source), "--actor-fresh-critic", "--deterministic",
               "--actor-observations", str(layout["actor_observation_count"]),
               "--critic-observations", str(layout["critic_observation_count"]),
               "--actions", str(layout["action_count"]),
               "--policy-id", f"crow-explicit-actor-transfer-{args.variant}",
               "--seed", str(profile["development_seeds"][0]),
               "--native-library", str(runtime(root)["library"]),
               "--output", str(output / "deployment.policypack")]
    for field in CONTRACT:
        command += ["--" + field.replace("_", "-"), str(layout[field])]
    if args.variant == "v10-world-model":
        command += ["--actor-observation-extension-offset", "84"]
    result = execute(output, "actor-transfer", command, root, 180)
    policy = output / "deployment.policypack"
    check = execute(output, "loaded-policy-check", rollout(root, args.variant, 8, 32, 8,
                    profile["probe"]["seed"], 0, policy), root, 180)
    validate_native(check, envs=8, steps=32, seed=profile["probe"]["seed"], policy=True)
    return {"status": "created_unqualified_actor_transfer", "source_sha256": sha(source),
            "policy_sha256": sha(policy), "critic": "fresh initialization",
            "training_updates": 0, "native_contract": {key: layout[key] for key in CONTRACT},
            "initialization": result, "ecological_environment_qualified": False}


def evaluate(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    policy = args.policy.resolve(strict=True)
    # Development uses its own seed namespace; held-out seeds are never spent
    # automatically by engineering probes or a development evaluation.
    seeds = profile["development_seeds"] if args.mode == "development" else profile["qualification_seeds"]
    gate = profile["navigation_gate"]
    envs, steps = gate["environments"], gate["steps"]
    courses = ["training"] if args.mode == "development" else ["training", "held-out-a", "held-out-b"]
    results = []
    for course in courses:
        for seed in seeds:
            name = f"{course}-{seed}"
            value = execute(output, name, rollout(root, args.variant, envs, steps, 8, seed, 10, policy, course), root, 300)
            validate_native(value, envs=envs, steps=steps, seed=seed, policy=True)
            if value.get("birdflow_navigation_course") != course or value.get("birdflow_navigation_curriculum") is not False:
                raise RuntimeError("Course identity or curriculum isolation failed")
            first = value["first_episode_navigation"]
            results.append({"course": course, "seed": seed, "first_episode": first,
                            "elapsed_seconds": value["elapsed_seconds"],
                            "device_failure_steps": value["failed_environment_steps"]})
    failures = []
    for course in courses:
        group = [row for row in results if row["course"] == course]
        for row in group:
            first = row["first_episode"]
            if first["safe_completion_count"] < gate["minimum_successful_lanes_per_seed"]:
                failures.append(f"{course}/{row['seed']}: completion minimum")
            if first["physical_failure_count"]:
                failures.append(f"{course}/{row['seed']}: first-episode physical failure")
        if sum(row["first_episode"]["safe_completion_count"] for row in group) / (envs * len(group)) < gate["minimum_aggregate_success_fraction"]:
            failures.append(f"{course}: aggregate completion minimum")
    return {"status": "passed" if not failures else "rejected", "mode": args.mode,
            "policy_sha256": sha(policy), "failures": failures, "results": results,
            "boundary": "waypoint navigation only; supported landing and ecological tasks remain separate",
            "ecological_environment_qualified": False}


def learn_probe(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    source = args.source_policy.resolve(strict=True) if args.source_policy else None
    variant = args.variant
    if variant == "v9-visual-neural" and source is None:
        raise ValueError("The v9 learning probe requires an explicitly selected source actor")
    paths = runtime(root)
    records = []
    for stage, updates in (("initial", 2), ("resume", 1)):
        command = [str(paths["train"]), "--metallib", str(paths["metallib"]),
                   "--native-library", str(paths["library"]), "--mlx-python", sys.executable,
                   "--python-root", str(root / "python"), "--birdflow-american-crow-journey",
                   "--birdflow-journey-variant", variant,
                   "--envs", "512", "--steps", "128", "--updates", str(updates), "--chunk", "8",
                   "--minimum-difficulty-band", "4", "--maximum-difficulty-band", "4",
                   "--seed", str(profile["probe"]["seed"]),
                   "--learner-seed", str(profile["development_seeds"][0]),
                   "--update-epochs", "1", "--minibatch-size", "8192", "--learning-rate", "0.0001",
                   "--policy-pack", str(output / ("initial.policypack" if stage == "initial" else "initial-updated.policypack")),
                   "--updated-policy-pack", str(output / f"{stage}-updated.policypack"),
                   "--deployment-policy-pack", str(output / f"{stage}-deployment.policypack"),
                   "--incumbent-policy-pack", str(output / f"{stage}-incumbent.policypack"),
                   "--rollout-pack", str(output / f"{stage}.rolloutpack"),
                   "--output-learner-state", str(output / f"{stage}.learner.safetensors")]
        if variant in VARIANTS:
            command += ["--visual-observation-config", str(root / "assets" / VARIANTS[variant])]
        if stage == "initial":
            command += ["--initialize-policy", "crow-engineering-learning-probe"]
            if source:
                command += ["--initialize-actor-policy-pack", str(source), "--initialize-actor-fresh-critic"]
            else:
                command += ["--zero-actor-output"]
        else:
            command += ["--learner-state", str(output / "initial.learner.safetensors")]
        value = execute(output, f"learner-{stage}", command, root, 300)
        if value.get("failed_environment_steps") != 0 or value.get("training_samples") != 512 * 128 * updates:
            raise RuntimeError("Learner probe invalid sample or device status")
        if value.get("final_policy_revision") != value.get("initial_policy_revision", -1) + updates:
            raise RuntimeError("Learner did not publish every expected policy revision")
        if stage == "resume" and value.get("learner_state_restored") is not True:
            raise RuntimeError("Learner checkpoint was not restored")
        if not (output / f"{stage}.learner.safetensors").is_file():
            raise RuntimeError("Learner checkpoint missing")
        receipt = json.loads((output / f"learner-{stage}.receipt.json").read_text())
        records.append({"stage": stage, "updates": updates,
                        "samples": value["training_samples"],
                        "training_loop_steps_per_second": value["end_to_end_environment_steps_per_second"],
                        "whole_process_steps_per_second": value["training_samples"] / receipt["wall_seconds"],
                        "initial_revision": value["initial_policy_revision"],
                        "final_revision": value["final_policy_revision"],
                        "learner_state_restored": value["learner_state_restored"],
                        "checkpoint_sha256": sha(output / f"{stage}.learner.safetensors")})
    return {"status": "passed", "claim": "native/MLX updates, publication, and learner-only restore",
            "variant": variant, "actor_initialization": "explicit source" if source else "new zero-output actor",
            "records": records, "candidate_promoted": False, "persistent_world_restore_tested": False,
            "ecological_environment_qualified": False}


def motor_probe(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    policy = args.policy.resolve(strict=True)
    seed = profile["development_seeds"][0]
    rows = []
    for band in (0, 1, 2, 4, 7, 9, 10):
        value = execute(output, f"motor-band{band}", rollout(root, "v9-visual-neural",
                        32, 1600, 8, seed, band, policy), root, 300)
        validate_native(value, envs=32, steps=1600, seed=seed, policy=True)
        if value.get("minimum_sampled_difficulty_band") != band or value.get("maximum_sampled_difficulty_band") != band:
            raise RuntimeError("Motor probe difficulty identity failed")
        rows.append({"band": band, "mean_tracking_score": value["mean_tracking_score"],
                     "physical_terminations": value["termination_count"] - value["timeout_count"],
                     "maximum_root_height_m": value["maximum_root_height"],
                     "maximum_tilt_rad": value["maximum_tilt"],
                     "outcomes": value["outcomes"]})
    return {"status": "measured", "claim": "single-development-seed motor diagnosis only",
            "seed": seed, "policy_sha256": sha(policy), "records": rows,
            "policy_qualified": False, "ecological_environment_qualified": False}


def partition_probe(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    policy = args.policy.resolve(strict=True)
    base = args.base_policy.resolve(strict=True)
    rows = []
    for mode in ("ordinary", "residual"):
        fingerprints = []
        for chunk in (1, 8):
            command = rollout(root, args.variant, 32, 128, chunk,
                              profile["probe"]["seed"], 4, policy)
            if mode == "residual":
                command += ["--base-policy-pack", str(base)]
            value = execute(output, f"{mode}-chunk{chunk}", command, root, 180)
            validate_native(value, envs=32, steps=128, seed=profile["probe"]["seed"], policy=True,
                            variant=args.variant)
            fingerprints.append(value["policy_rollout_fingerprint"])
        if fingerprints[0] != fingerprints[1]:
            raise RuntimeError(f"{mode} policy sampling depends on submission partition")
        rows.append({"mode": mode, "fingerprint": fingerprints[0], "chunks": [1, 8]})
    return {"status": "passed", "claim": "ordinary and residual native policy partition invariance",
            "records": rows, "ecological_environment_qualified": False}


def body_admission(root: Path, output: Path, profile: dict) -> dict:
    import numpy as np
    seed = profile["probe"]["seed"]
    rows = []
    final = {}
    for control in ("neutral", "head-beak-command"):
        actions = np.zeros((128, 8, 18), dtype="<f4")
        if control != "neutral":
            actions[16:, :, 15:] = [0.4, -0.3, 0.5]
        stream = output / f"{control}.actions.f32"
        with stream.open("xb") as handle:
            handle.write(actions.tobytes())
        trace = output / f"{control}.state.tsv"
        command = rollout(root, "v11-embodied", 8, 128, 1, seed, 0)
        command.remove("--native-policy")
        command += ["--action-stream", str(stream), "--state-trace", str(trace),
                    "--state-trace-environment", "0"]
        value = execute(output, control, command, root, 180)
        validate_native(value, envs=8, steps=128, seed=seed, policy=False,
                        variant="v11-embodied", action_source="foundation_action_stream")
        if value.get("task") != "american_crow_embodied_development_v11" or value.get("action_count") != 18 or value.get("actor_observation_count") != 93:
            raise RuntimeError("Embodied task/action/proprioception contract mismatch")
        if value.get("termination_count") != value.get("timeout_count"):
            raise RuntimeError("Embodied neutral/command control physically terminated")
        samples = np.loadtxt(trace, comments="#", ndmin=2)
        if samples.shape[0] != 128 or samples[-1,0] != 128 or not np.isfinite(samples).all():
            raise RuntimeError("Incomplete accepted-state trace")
        final[control] = samples[-1,21:24]
        rows.append({"control": control, "action_stream_sha256": sha(stream),
                     "accepted_state_trace_sha256": sha(trace),
                     "final_head_beak_angles_rad": final[control].tolist(),
                     "device_failure_steps": value["failed_environment_steps"],
                     "physical_terminations": value["termination_count"]-value["timeout_count"]})
    delta = final["head-beak-command"] - final["neutral"]
    if not (delta[0] > 0.05 and delta[1] < -0.03 and delta[2] > 0.1):
        raise RuntimeError(f"Accepted head/beak response did not satisfy prospective directional thresholds: {delta}")
    return {"status": "passed", "claim": "accepted native contact-world head/beak actuation control",
            "records": rows, "command_minus_control_angles_rad": delta.tolist(),
            "boundary": "one traced lane, eight environment validity control, 2.56 seconds; no sensing or behavioral qualification",
            "ecological_environment_qualified": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("doctor")
    commands.add_parser("probe").add_argument("--output", type=Path, required=True)
    commands.add_parser("head-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("body-admission").add_argument("--output", type=Path, required=True)
    init = commands.add_parser("bootstrap")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--source-policy", type=Path, required=True)
    init.add_argument("--variant", choices=VARIANTS, required=True)
    learning = commands.add_parser("learn-probe")
    learning.add_argument("--output", type=Path, required=True)
    learning.add_argument("--source-policy", type=Path)
    learning.add_argument("--variant", choices=("v9-visual-neural", "v11-embodied"), default="v9-visual-neural")
    motor = commands.add_parser("motor-probe")
    motor.add_argument("--output", type=Path, required=True)
    motor.add_argument("--policy", type=Path, required=True)
    partition = commands.add_parser("partition-probe")
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--policy", type=Path, required=True)
    partition.add_argument("--base-policy", type=Path, required=True)
    partition.add_argument("--variant", choices=("v9-visual-neural", "v11-embodied"), default="v9-visual-neural")
    assessment = commands.add_parser("evaluate")
    assessment.add_argument("--output", type=Path, required=True)
    assessment.add_argument("--policy", type=Path, required=True)
    assessment.add_argument("--variant", choices=("v10-world-model",), default="v10-world-model")
    assessment.add_argument("--mode", choices=("development", "qualification"), default="development")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    profile = json.loads((root / "profiles/american-crow-rl-v1.json").read_text())
    if args.command == "status":
        print(json.dumps(profile, indent=2))
        return 0
    checked = preflight(root, profile)
    if args.command == "doctor":
        print(json.dumps(checked, indent=2))
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "numi.crow-trial-plan.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": args.command, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "preflight": checked, "profile": profile,
            "prediction": "Exact same-seed replay and partition invariance; imported historical actor may fail full-route completion",
            "control": "repeat same seed with submission chunk 1 versus 8; first episode only",
            "validity": "native accepted state, exact identity, finite first-episode evidence, no device failures or hidden teacher",
            "stopping": "bounded complete runs, timeout retained as failure; no overwrite or automatic retry"}
    if args.command in ("bootstrap", "learn-probe", "evaluate", "motor-probe", "partition-probe"):
        source = args.source_policy if args.command in ("bootstrap", "learn-probe") else args.policy
        plan["source_policy_sha256"] = sha(source.resolve(strict=True)) if source else None
    if args.command == "partition-probe":
        plan["base_policy_sha256"] = sha(args.base_policy.resolve(strict=True))
    write(output / "plan.json", plan)
    try:
        if args.command == "probe":
            result = probe(root, output, profile)
        elif args.command == "head-admission":
            result = execute(output, "head-mechanics", [str(runtime(root)["head_check"])], root, 180)
            if result.get("status") != "passed" or result.get("exact_replay") is not True:
                raise RuntimeError("Native head mechanics admission failed")
        elif args.command == "body-admission":
            result = body_admission(root, output, profile)
        elif args.command == "bootstrap":
            result = bootstrap(root, output, args, profile)
        elif args.command == "learn-probe":
            result = learn_probe(root, output, args, profile)
        elif args.command == "motor-probe":
            result = motor_probe(root, output, args, profile)
        elif args.command == "partition-probe":
            result = partition_probe(root, output, args, profile)
        else:
            result = evaluate(root, output, args, profile)
        write(output / "summary.json", result)
        print(json.dumps({"output": str(output), **result}, indent=2))
        return 0 if result["status"] != "rejected" else 2
    except Exception as error:
        write(output / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
