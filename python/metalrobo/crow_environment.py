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
        "foraging_check": build / "bin/metalrobo_crow_foraging_check",
        "eye_check": build / "bin/metalrobo_crow_eye_check",
        "retinal_check": build / "bin/metalrobo_retinal_stream_check",
        "retinal_contract_check": build / "bin/metalrobo_crow_retinal_contract_check",
        "retinal_metallib": build / "shaders/RetinalStream.metallib",
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



def retain_source_delta(root: Path, output: Path, checked: dict) -> None:
    """Retain the exact executable source delta, including new native files."""
    directory = output / "source-inputs"
    directory.mkdir()
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    if hashlib.sha256(patch).hexdigest() != checked["tracked_diff_sha256"]:
        raise RuntimeError("Source changed between preflight and retention")
    (directory / "tracked.patch").write_bytes(patch)
    retained, external = {}, {}
    for name, expected in checked["untracked_inputs"].items():
        # Scientific payload archives have their own receipt/inventory. New
        # runtime code must be portable even when the trial precedes a commit.
        if Path(name).parts[0] == "artifacts":
            external[name] = expected
            continue
        source = root / name
        target = directory / "untracked" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != expected:
            raise RuntimeError(f"Source changed during retention: {name}")
        retained[name] = expected
    write(directory / "manifest.json", {"base_commit": checked["source_commit"],
        "tracked_patch_sha256": checked["tracked_diff_sha256"],
        "retained_untracked": retained, "external_artifact_inputs": external})


def execute(output: Path, name: str, command: list[str], root: Path, timeout: int) -> dict:
    started = time.monotonic()
    exit_code = None
    timed_out = False
    interrupted = False
    with (output / f"{name}.json").open("x") as stdout, (output / f"{name}.stderr").open("x") as stderr:
        try:
            result = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, timeout=timeout)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except KeyboardInterrupt:
            interrupted = True
    receipt = {
        "command": command, "exit_code": exit_code, "timed_out": timed_out, "interrupted": interrupted,
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": sha(output / f"{name}.json"),
        "stderr_sha256": sha(output / f"{name}.stderr"),
    }
    write(output / f"{name}.receipt.json", receipt)
    if interrupted:
        raise KeyboardInterrupt("Native trial interrupted; retained receipt")
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
                "birdflow_journey_teacher": False,
                "visual_observation": variant not in ("v11-embodied", "v12-feeding-station"),
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
                   "--minimum-difficulty-band", "0" if variant=="v12-feeding-station" else "4",
                   "--maximum-difficulty-band", "0" if variant=="v12-feeding-station" else "4",
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
        if args.fixed_learning_rate:
            command += ["--fixed-learning-rate"]
        if args.initial_log_standard_deviation is not None:
            command += ["--initial-log-standard-deviation", str(args.initial_log_standard_deviation)]
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
        update = value["last_learning_update"]
        records.append({"stage": stage, "updates": updates,
                        "samples": value["training_samples"],
                        "training_loop_steps_per_second": value["end_to_end_environment_steps_per_second"],
                        "whole_process_steps_per_second": value["training_samples"] / receipt["wall_seconds"],
                        "initial_revision": value["initial_policy_revision"],
                        "final_revision": value["final_policy_revision"],
                        "learner_state_restored": value["learner_state_restored"],
                        "checkpoint_sha256": sha(output / f"{stage}.learner.safetensors"),
                        "last_update_mean_reward": update["mean_reward"],
                        "last_update_maximum_reward": update["maximum_reward"],
                        "last_update_intake_utility_mean": update.get("outcomes", {}).get("accepted_intake_utility", {}).get("mean"),
                        "last_update_learning_rate": update["learning_rate"],
                        "last_update_action_standard_deviation": update["mean_action_standard_deviation"],
                        "mlx_peak_memory_bytes": update["mlx_peak_memory_bytes"]})
    if args.require_intake_reward and not any((row["last_update_intake_utility_mean"] or 0)>0 for row in records):
        write(output / "learning-no-intake-reward.json", {"records": records})
        raise RuntimeError("No accepted intake reward in reported learning updates")
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
                              profile["probe"]["seed"], 0 if args.variant=="v12-feeding-station" else 4, policy)
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


def validate_feeding_accounting(value: dict, envs: int, steps: int) -> dict:
    accounting = value.get("finite_resource_accounting", {})
    fields = ("remaining_micrograms", "crop_micrograms", "consumed_micrograms", "bite_count",
              "initialized", "accepted_control_ticks")
    arrays = {field: accounting.get(field + "_by_environment") for field in fields}
    if any(not isinstance(array, list) or len(array) != envs for array in arrays.values()):
        raise RuntimeError("Complete per-lane native finite-resource evidence is required")
    for lane in range(envs):
        row = {field: arrays[field][lane] for field in fields}
        if any(type(item) is not int for item in row.values()):
            raise RuntimeError("Resource evidence must retain integer micrograms and ticks")
        consumed = row["consumed_micrograms"]
        if (consumed not in (0, 20000, 40000, 45000) or row["remaining_micrograms"] + consumed != 45000
            or row["crop_micrograms"] != consumed or row["bite_count"] != {0: 0, 20000: 1, 40000: 2, 45000: 3}[consumed]
            or row["initialized"] != 1 or row["accepted_control_ticks"] != steps + 1):
            raise RuntimeError("Native feeding inventory, crop, bites or accepted lifetime clock are inconsistent")
    utility = sum(arrays["consumed_micrograms"]) * .0001 / envs
    reported = (value.get("mean_reward"), value.get("outcomes", {}).get("accepted_intake_utility", {}).get("mean"))
    if any(not isinstance(mean, (int, float)) or not math.isfinite(mean)
           or abs(mean * steps - utility) > 3e-6 for mean in reported):
        raise RuntimeError("Published food reward/outcome disagrees with accepted physical resource accounting")
    return {"completed_lanes": arrays["consumed_micrograms"].count(45000),
            "environment_count": envs, "complete_fraction": arrays["consumed_micrograms"].count(45000) / envs,
            "mean_consumed_micrograms": sum(arrays["consumed_micrograms"]) / envs,
            "mean_accepted_intake_utility_per_lane": utility}


def feeding_probe(root: Path, output: Path, profile: dict) -> dict:
    import numpy as np
    records = []
    fingerprints = []
    for name, chunk, miss, bites in (("feeding-chunk8", 8, False, True),
                                     ("feeding-chunk64", 64, False, True),
                                     ("head-yaw-miss", 8, True, True),
                                     ("held-closed", 8, False, False)):
        actions = np.zeros((448, 8, 18), dtype="<f4")
        if miss:
            actions[:, :, 15] = 1.0
        if bites:
            for step in range(64, 320):
                if ((step-64)//32) % 2 == 0:
                    actions[step, :, 17] = 0.9
        stream = output / f"{name}.actions.f32"
        with stream.open("xb") as handle:
            handle.write(actions.tobytes())
        command = rollout(root, "v12-feeding-station", 8, 448, chunk, profile["probe"]["seed"], 0)
        command.remove("--native-policy")
        command += ["--action-stream", str(stream)]
        value = execute(output, name, command, root, 180)
        validate_native(value, envs=8, steps=448, seed=profile["probe"]["seed"], policy=False,
                        variant="v12-feeding-station", action_source="foundation_action_stream")
        if (value.get("task") != "american_crow_feeding_station_v12" or
            value.get("actor_observation_count") != 97 or value.get("critic_observation_count") != 97 or
            value.get("action_count") != 18 or value.get("termination_count") != 0):
            raise RuntimeError("Native feeding contract or physical validity failed")
        utility = value["mean_reward"] * 448
        accepted = value["outcomes"]["accepted_intake_utility"]["mean"] * 448
        expected = 4.5 if bites and not miss else 0.0
        if abs(utility-expected)>2e-6 or abs(accepted-expected)>2e-6:
            raise RuntimeError(f"Feeding reward violates finite debit identity: {utility}, {accepted}, {expected}")
        inventory = validate_feeding_accounting(value, 8, 448)
        if inventory["mean_consumed_micrograms"] != (45000 if expected else 0):
            raise RuntimeError("Authored feeding control has the wrong accepted inventory")
        if not miss and bites:
            fingerprints.append(value["policy_rollout_fingerprint"])
        records.append({"control": name, "action_stream_sha256": sha(stream),
                        "accepted_intake_utility_per_lane": utility,
                        "native_outcome_utility_per_lane": accepted,
                        "physical_terminations": value["termination_count"],
                        "failed_physics_steps": value["failed_environment_steps"], "inventory": inventory})
    if len(set(fingerprints)) != 1:
        raise RuntimeError("Feeding observations/rewards depend on submission partition")
    return {"status": "passed", "claim": "native feeding sensor/reward and finite-debit controls",
            "records": records, "nutrition_inputs": 4, "actor_inputs": 97, "actions": 18,
            "behavioral_policy_qualified": False, "persistent_world_restore_tested": False,
            "ecological_environment_qualified": False}


def feeding_policy_probe(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    from .mlx_policy_learning import read_policy_pack
    policy = read_policy_pack(args.policy)
    stochastic = bool(getattr(args, "stochastic", False))
    if bool(policy.action_log_standard_deviation.size) != stochastic:
        raise ValueError("Feeding screen requires the explicitly selected deterministic/stochastic PolicyPack mode")
    envs = getattr(args, "environments", 32)
    rows=[]
    for seed in profile["development_seeds"]:
        value=execute(output,f"seed-{seed}",rollout(root,"v12-feeding-station",envs,1600,8,seed,0,args.policy),root,180)
        validate_native(value,envs=envs,steps=1600,seed=seed,policy=True,variant="v12-feeding-station")
        if value.get("termination_count")!=0:
            raise RuntimeError("Feeding behavior screen requires a complete physically valid first horizon")
        utility=value["mean_reward"]*1600
        if not math.isfinite(utility) or not -1e-6<=utility<=4.5+2e-6:
            raise RuntimeError("Deployed feeding reward exceeds the finite lifetime food budget")
        rows.append({"seed":seed,"mean_accepted_intake_utility_per_lane":utility,
                     "fraction_of_available_food_utility":utility/4.5,
                     "physical_terminations":0,"failed_physics_steps":0,
                     "inventory": validate_feeding_accounting(value, envs, 1600)})
    return {"status":"measured_unqualified","claim":"development-only feeding policy screen",
            "records":rows,"behavioral_policy_qualified":False, "stochastic_policy": stochastic,
            "boundary":"canonical station with per-lane finite inventory; no search, physical food handling or learned memory",
            "ecological_environment_qualified":False}


def feeding_learning_campaign(root: Path, output: Path, profile: dict) -> dict:
    """Bounded development experiment; each process is an independent lifetime.

    Only policy/optimizer state crosses lifetimes. Native resource state is
    never replenished at PPO update boundaries. This is not life-state resume.
    """
    paths = runtime(root)
    if shutil.disk_usage(root).free < 150 * 1024**3:
        raise RuntimeError("Feeding campaign requires 150 GiB free at admission")
    active = capture(["ps", "-axo", "pid,comm"], root)
    if any(line.rsplit("/", 1)[-1] in ("metalrobo_task_train", "metalrobo_task_rollout")
           for line in active.splitlines()):
        raise RuntimeError("Another native training/rollout worker is active")
    write(output / "worker-admission.json", {"processes": active,
          "free_disk_bytes": shutil.disk_usage(root).free})
    started = time.monotonic()
    records, screens = [], []
    previous = None
    for lifetime in range(1, 33):
        if time.monotonic() - started > 1500:
            raise RuntimeError("Campaign reached its prospective 1500-second admission bound")
        if shutil.disk_usage(root).free < 110 * 1024**3:
            raise RuntimeError("Campaign reached its 110 GiB free-space stopping floor")
        directory = output / f"lifetime-{lifetime:02d}"
        directory.mkdir()
        command = [str(paths["train"]), "--metallib", str(paths["metallib"]),
                   "--native-library", str(paths["library"]), "--mlx-python", sys.executable,
                   "--python-root", str(root / "python"), "--birdflow-american-crow-journey",
                   "--birdflow-journey-variant", "v12-feeding-station",
                   "--envs", "512", "--steps", "128", "--updates", "4", "--chunk", "8",
                   "--minimum-difficulty-band", "0", "--maximum-difficulty-band", "0",
                   "--seed", str(2651108100 + lifetime), "--learner-seed", str(2651108200 + lifetime),
                   "--update-epochs", "4", "--minibatch-size", "8192", "--learning-rate", "0.0001",
                   "--fixed-learning-rate", "--initial-log-standard-deviation", "0",
                   "--policy-pack", str(directory / "initial.policypack" if previous is None else previous / "updated.policypack"),
                   "--updated-policy-pack", str(directory / "updated.policypack"),
                   "--deployment-policy-pack", str(directory / "deployment.policypack"),
                   "--incumbent-policy-pack", str(directory / "incumbent.policypack"),
                   "--rollout-pack", str(directory / "final.rolloutpack"),
                   "--output-learner-state", str(directory / "learner.safetensors")]
        if previous is None:
            command += ["--initialize-policy", "crow-feeding-development-campaign", "--zero-actor-output"]
        else:
            command += ["--learner-state", str(previous / "learner.safetensors")]
        write(directory / "input-plan.json", {"command": command, "world_seed": 2651108100 + lifetime,
              "learner_seed": 2651108200 + lifetime, "independent_new_finite_lifetime": True,
              "restored_world": False, "source_learner_sha256": sha(previous / "learner.safetensors") if previous else None,
              "source_policy_sha256": sha(previous / "updated.policypack") if previous else None})
        value = execute(directory, "training", command, root, 90)
        expected_initial_revision = records[-1]["final_revision"] if records else 1
        expected = {"training_samples": 512 * 128 * 4, "failed_environment_steps": 0,
                    "host_requested_resets": 0, "learner_state_restored": previous is not None,
                    "birdflow_journey_variant": "v12-feeding-station", "visual_observation": False,
                    "birdflow_journey_teacher": False, "initial_policy_revision": expected_initial_revision,
                    "final_policy_revision": expected_initial_revision + 4}
        if any(value.get(key) != target for key, target in expected.items()):
            raise RuntimeError("Feeding campaign violated its native learner/lifetime contract")
        history = value.get("learning_update_history", [])
        if len(history) != 4:
            raise RuntimeError("Complete per-update feeding metrics are required")
        for index, update in enumerate(history):
            if (update.get("samples") != 65536 or update.get("done_count") != 0 or update.get("timeout_count") != 0
                or update.get("policy_revision_before") != expected_initial_revision + index
                or update.get("policy_revision_after") != expected_initial_revision + index + 1):
                raise RuntimeError("Feeding update omitted samples, ended a life, or changed revision unexpectedly")
            reward = update.get("mean_reward")
            intake = update.get("outcomes", {}).get("accepted_intake_utility", {}).get("mean")
            if (not isinstance(reward, (int, float)) or not math.isfinite(reward) or reward < 0
                or intake != reward or not 0 <= update.get("maximum_reward", math.inf) <= 2.000002):
                raise RuntimeError("Feeding update has invalid accepted intake reward")
        utility = sum(update["mean_reward"] * 128 for update in history)
        if not 0 <= utility <= 4.500004:
            raise RuntimeError("Training exceeded its finite 45 mg per-lane lifetime budget")
        receipt = json.loads((directory / "training.receipt.json").read_text())
        record = {"lifetime": lifetime, "mean_intake_utility_per_lane": utility,
                  "fraction_of_available_food_utility": utility / 4.5,
                  "initial_revision": expected_initial_revision, "final_revision": value["final_policy_revision"],
                  "samples": value["training_samples"], "whole_process_steps_per_second": value["training_samples"] / receipt["wall_seconds"],
                  "mlx_peak_memory_bytes": max(update["mlx_peak_memory_bytes"] for update in history),
                  "learner_sha256": sha(directory / "learner.safetensors"),
                  "deployment_sha256": sha(directory / "deployment.policypack")}
        write(directory / "analysis.json", record)
        records.append(record)
        print(json.dumps({"feeding_lifetime": lifetime, **record}), file=sys.stderr, flush=True)
        if lifetime == 1:
            baseline = output / "baseline-screen"
            baseline.mkdir()
            result = feeding_policy_probe(root, baseline,
                        argparse.Namespace(policy=directory / "incumbent.policypack"), profile)
            write(baseline / "summary.json", result)
            screens.append({"after_lifetimes": 0, **result})
        if lifetime in (8, 16, 32):
            screen = output / f"development-screen-{lifetime:02d}"
            screen.mkdir()
            result = feeding_policy_probe(root, screen,
                        argparse.Namespace(policy=directory / "deployment.policypack"), profile)
            write(screen / "summary.json", result)
            screens.append({"after_lifetimes": lifetime, **result})
            print(json.dumps({"development_screen": lifetime, "records": result["records"]}), file=sys.stderr, flush=True)
        previous = directory
    final_minimum = min(row["fraction_of_available_food_utility"] for row in screens[-1]["records"])
    return {"status": "measured_unqualified", "claim": "bounded canonical-station PPO development experiment",
            "prediction_supported": final_minimum >= .8, "predicted_minimum_final_food_fraction": .8,
            "final_minimum_development_food_fraction": final_minimum,
            "records": records, "development_screens": screens, "training_samples": 32 * 512 * 128 * 4,
            "world_restore_tested": False, "qualification_conditions_used": False,
            "behavioral_policy_qualified": False, "ecological_environment_qualified": False}


def feeding_policy_comparison(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    import numpy as np
    from .mlx_policy_learning import read_policy_pack
    sources = {"initial-deterministic": args.untrained_deployment_policy,
               "initial-stochastic": args.untrained_policy,
               "trained-deterministic": args.trained_deployment_policy,
               "trained-stochastic": args.trained_policy}
    packs = {name: read_policy_pack(path) for name, path in sources.items()}
    for prefix in ("initial", "trained"):
        deterministic, stochastic = (packs[prefix + "-" + mode] for mode in ("deterministic", "stochastic"))
        if deterministic.action_log_standard_deviation.size or stochastic.action_log_standard_deviation.size != 18:
            raise ValueError("Comparison requires explicitly deterministic and stochastic artifacts")
        for key in CONTRACT + ("observation_clip", "action_clip"):
            if getattr(deterministic, key) != getattr(stochastic, key):
                raise ValueError("Policy modes do not share the same exact actor contract")
        for key in ("observation_mean", "observation_inverse_standard_deviation", "action_bias", "action_scale"):
            if not np.array_equal(getattr(deterministic, key), getattr(stochastic, key)):
                raise ValueError("Policy modes changed actor normalization or action transforms")
        if len(deterministic.layers) != len(stochastic.layers):
            raise ValueError("Policy modes changed actor architecture")
        for left, right in zip(deterministic.layers, stochastic.layers):
            if (left.input_count != right.input_count or left.output_count != right.output_count
                or left.activation != right.activation or not np.array_equal(left.weights, right.weights)
                or not np.array_equal(left.bias, right.bias)):
                raise ValueError("Policy modes changed actor coefficients")
    initial = packs["initial-deterministic"]
    if np.any(initial.layers[-1].weights) or np.any(initial.layers[-1].bias):
        raise ValueError("The untrained control must retain the original zero-output actor")
    if any(getattr(initial, key) != getattr(packs["trained-stochastic"], key) for key in CONTRACT):
        raise ValueError("Initial and trained policies do not belong to the same task")
    write(output / "actor-mode-admission.json", {name: {
        "path": str(sources[name].resolve()), "sha256": sha(sources[name]), "revision": pack.revision,
        "stochastic": bool(pack.action_log_standard_deviation.size),
        "standard_deviation": np.exp(pack.action_log_standard_deviation).tolist(),
        "contract": {key: getattr(pack, key) for key in CONTRACT}} for name, pack in packs.items()})
    order = ("initial-deterministic", "trained-stochastic", "initial-stochastic", "trained-deterministic")
    rows = []
    for index, seed in enumerate((2651108401, 2651108402, 2651108403)):
        for name in order[index:] + order[:index]:
            directory = output / f"seed-{seed}" / name
            directory.mkdir(parents=True)
            result = feeding_policy_probe(root, directory,
                argparse.Namespace(policy=sources[name], stochastic=name.endswith("-stochastic"), environments=512),
                profile | {"development_seeds": [seed]})
            row = {"arm": name, **result["records"][0]}
            rows.append(row)
            write(directory / "analysis.json", row)
            print(json.dumps(row), file=sys.stderr, flush=True)
    candidate = [row for row in rows if row["arm"] == "trained-stochastic"]
    null = [row for row in rows if row["arm"] == "initial-stochastic"]
    prediction_supported = (all(row["inventory"]["complete_fraction"] >= .95 for row in candidate)
                            and all(row["inventory"]["complete_fraction"] <= .30 for row in null))
    return {"status": "measured_unqualified", "claim": "frozen-policy canonical feeding comparison",
            "prediction_supported": prediction_supported, "records": rows,
            "native_trajectories": 12 * 512, "accepted_test_transitions": 12 * 512 * 1600,
            "independently_trained_candidates": 1, "behavioral_policy_qualified": False,
            "boundary": "simulation feeding primitive; estimated supported body, fixed intake tracer, no search or solid food manipulation",
            "ecological_environment_qualified": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("doctor")
    commands.add_parser("probe").add_argument("--output", type=Path, required=True)
    commands.add_parser("head-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("eye-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("retina-admission").add_argument("--output", type=Path, required=True)
    retinal_contract=commands.add_parser("retinal-contract-admission")
    retinal_contract.add_argument("--output",type=Path,required=True)
    retinal_contract.add_argument("--legacy-metallib",type=Path)
    commands.add_parser("body-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("foraging-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("feeding-probe").add_argument("--output", type=Path, required=True)
    commands.add_parser("feeding-learning-campaign").add_argument("--output", type=Path, required=True)
    feeding_policy=commands.add_parser("feeding-policy-probe")
    feeding_policy.add_argument("--output",type=Path,required=True)
    feeding_policy.add_argument("--policy",type=Path,required=True)
    feeding_policy.add_argument("--stochastic", action="store_true")
    feeding_policy.add_argument("--environments", type=int, choices=(32, 512), default=32)
    comparison = commands.add_parser("feeding-policy-comparison")
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--calibration", type=Path, required=True)
    for name in ("untrained-policy", "untrained-deployment-policy", "trained-policy", "trained-deployment-policy"):
        comparison.add_argument("--" + name, type=Path, required=True)
    init = commands.add_parser("bootstrap")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--source-policy", type=Path, required=True)
    init.add_argument("--variant", choices=VARIANTS, required=True)
    learning = commands.add_parser("learn-probe")
    learning.add_argument("--output", type=Path, required=True)
    learning.add_argument("--source-policy", type=Path)
    learning.add_argument("--variant", choices=("v9-visual-neural", "v11-embodied", "v12-feeding-station"), default="v9-visual-neural")
    learning.add_argument("--initial-log-standard-deviation", type=float)
    learning.add_argument("--fixed-learning-rate", action="store_true")
    learning.add_argument("--require-intake-reward", action="store_true")
    motor = commands.add_parser("motor-probe")
    motor.add_argument("--output", type=Path, required=True)
    motor.add_argument("--policy", type=Path, required=True)
    partition = commands.add_parser("partition-probe")
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--policy", type=Path, required=True)
    partition.add_argument("--base-policy", type=Path, required=True)
    partition.add_argument("--variant", choices=("v9-visual-neural", "v11-embodied", "v12-feeding-station"), default="v9-visual-neural")
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
    if args.command=="retinal-contract-admission":
        args.legacy_metallib=(args.legacy_metallib or root/"artifacts/crow-rl-retinal-contract-20261107/inputs/legacy-MetalRobo.metallib").resolve(strict=True)
        checked["inputs"]["retinal_legacy_shader"]={"path":str(args.legacy_metallib),"sha256":sha(args.legacy_metallib)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "numi.crow-trial-plan.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": args.command, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "preflight": checked, "profile": profile,
            "prediction": "Exact same-seed replay and partition invariance; imported historical actor may fail full-route completion",
            "control": "repeat same seed with submission chunk 1 versus 8; first episode only",
            "validity": "native accepted state, exact identity, finite first-episode evidence, no device failures or hidden teacher",
            "stopping": "bounded complete runs, timeout retained as failure; no overwrite or automatic retry"}
    if args.command in ("bootstrap", "learn-probe", "evaluate", "motor-probe", "partition-probe", "feeding-policy-probe"):
        source = args.source_policy if args.command in ("bootstrap", "learn-probe") else args.policy
        plan["source_policy_sha256"] = sha(source.resolve(strict=True)) if source else None
    if args.command == "partition-probe":
        plan["base_policy_sha256"] = sha(args.base_policy.resolve(strict=True))
    if args.command == "feeding-learning-campaign":
        plan.update({"purpose": "development exploration; no qualification claim",
          "question": "Does longer PPO learning produce deterministic feeding instead of reward from random jaw closures?",
          "model_revision": "fresh zero-output MLP, unit exploration, four PPO epochs per update",
          "previous_evidence": "crow-rl-feeding-task-20261107: three updates deploy with zero food despite stochastic reward",
          "prediction": "Final deterministic policy consumes at least 80% of finite food utility on each of three development seeds",
          "refutation": "Any final development seed below 80% contradicts this bounded training prediction",
          "control": "initial zero-output policy, identical canonical station and development seeds, 32 lanes and 1600 steps",
          "unit": "one independent learner trajectory; 32 finite-world training lifetimes are correlated learning exposures, not independent trained agents",
          "budget": {"lifetimes": 32, "envs": 512, "steps_per_update": 128, "updates_per_lifetime": 4,
                     "total_transitions": 8388608, "world_seeds": [2651108101, 2651108132],
                     "learner_seeds": [2651108201, 2651108232], "food_mg_per_lane_lifetime": 45},
          "analysis": "all lifetime reward sums constrained to [0,4.5] utility/lane; report all screens, no inferential population interval",
          "validity": "zero device failures, physical terminations and resets; complete update history/revisions; native finite ledger; no teacher",
          "stopping": "exactly 32 lifetimes; no success-driven extension; stop on invalidity, per-process 90s timeout, 1500s admission bound or free space below 110GiB",
          "unused": "qualification seeds/layouts remain untouched; future independent learner replicates required"})
    if args.command == "feeding-policy-comparison":
        calibration = json.loads((args.calibration / "summary.json").read_text())
        calibration_plan = json.loads((args.calibration / "plan.json").read_text())
        if calibration.get("status") != "passed" or len(calibration.get("records", [])) != 4:
            raise ValueError("Feeding comparison requires passed native inventory controls")
        for name in ("rollout", "library", "metallib"):
            if calibration_plan["preflight"]["inputs"][name]["sha256"] != checked["inputs"][name]["sha256"]:
                raise ValueError("Feeding calibration does not bind the current native runtime")
        plan.update({"purpose": "development mechanism comparison on unused seed identities; one frozen trained candidate",
          "parent_attempt": "20261107-feeding-learning-campaign-01: interrupted invalid deterministic baseline; eight completed learning lifetimes",
          "source_policies": {name: {"path": str(getattr(args, name).resolve(strict=True)), "sha256": sha(getattr(args, name))}
                              for name in ("untrained_policy", "untrained_deployment_policy", "trained_policy", "trained_deployment_policy")},
          "calibration": {"path": str(args.calibration.resolve()), "summary_sha256": sha(args.calibration / "summary.json"),
                          "plan_sha256": sha(args.calibration / "plan.json")},
          "question": "Has learning improved stochastic feeding reliability, or is positive intake explained by untrained exploration alone?",
          "prediction": "Trained stochastic actor depletes all 45 mg in at least 95% of lanes on EACH seed; initial stochastic actor completes at most 30%",
          "refutation": "Any seed outside either interval contradicts the prediction; deterministic modes are separately reported diagnostic controls",
          "control": "initial and trained actors each tested with explicit deterministic and stochastic artifacts; each mode pair has bit-identical actor coefficients",
          "unit": "6,144 trajectories, 512 lanes per arm/seed; one trained agent, not 6,144 independent learned animals",
          "conditions": {"seeds": [2651108401,2651108402,2651108403], "steps":1600, "dt_seconds":.02,
                         "food_micrograms_per_lane":45000, "arms":4, "additional_learning_updates":0},
          "order": "per-seed cyclic rotation of initial deterministic, trained stochastic, initial stochastic, trained deterministic",
          "analysis": "per-lane integer inventory/crop/bites/clock conservation and reward agreement; all per-seed completion fractions and mass, no population generalization",
          "validity": "passed native positive/null/partition inventory calibration, exact actor-mode identities, zero device errors/physical ends/resets",
          "stopping": "exactly twelve native runs; 180s timeout per run; abort invalidity; no retry, extra learning, or promotion",
          "limitations": "policy revision changes random key; same seed identifies matched world conditions but does not create identical stochastic noise across policy revisions; learned variance is part of the tested policy",
          "unused": "original qualification seeds/layouts remain unspent"})
    if args.command == "retinal-contract-admission":
        plan.update({"purpose":"native retinal actor contract and pre-submission rejection instrument",
          "prediction":"1251 actor/97 critic/18 action channels; 12 malformed manifests and 9 same-dimension sensor changes reject; missing observer and old shader cannot execute; legacy task ABI and mechanics unchanged",
          "boundary":"compiler metadata and native rejection probes only; no rendered actor images or learner integration"})
    if args.command == "retina-admission":
        plan.update({"purpose": "native RGB timing instrument calibration",
          "prediction": "integer 3/10 sample cadence, two accepted-tick delay; zero premature/reset-stale frames; feature error <=3e-6 against FP64 oracle; chunk1/8 bit-identical",
          "control": "three native fixture lanes, one per-lane episode reset, duplicate accepted tick with altered RGB, clock-gap and nonfinite negatives",
          "boundary": "synthetic device-input instrument only; no renderer-to-actor or learner integration; existing task semantics unchanged"})
    if args.command == "eye-admission":
        plan.update({"purpose": "native geometric instrument calibration",
          "prediction": "Accepted physical head transforms determine both camera poses within 2e-6 components; stable interior depth matches FP64 physical-shape rays within 0.0002m; zero identity errors",
          "control": "visible/physically occluded target, opposite head yaw, head pitch, exact renderer repeats, native chunks 1/8",
          "boundary": "estimated parallel head cameras with collision-proxy assets; publication-time inspection; no biological eye calibration, actor image integration or sensor-cadence admission"})
    write(output / "plan.json", plan)
    try:
        retain_source_delta(root, output, checked)
        if args.command == "probe":
            result = probe(root, output, profile)
        elif args.command == "head-admission":
            result = execute(output, "head-mechanics", [str(runtime(root)["head_check"])], root, 180)
            if result.get("status") != "passed" or result.get("exact_replay") is not True:
                raise RuntimeError("Native head mechanics admission failed")
        elif args.command == "retinal-contract-admission":
            result=execute(output,"retinal-contract",[str(runtime(root)["retinal_contract_check"]),
                           str(output/"native-evidence"),str(args.legacy_metallib)],root,180)
            if result.get("status")!="passed" or result.get("stale_shader_rejected") is not True:
                raise RuntimeError("Native retinal contract admission failed")
        elif args.command == "retina-admission":
            result = execute(output, "retinal-timing", [str(runtime(root)["retinal_check"]),
                             str(output / "native-evidence")], root, 180)
            if result.get("status") != "passed" or result.get("exact_partition_invariance") is not True:
                raise RuntimeError("Native retinal timing admission failed")
        elif args.command == "eye-admission":
            result = execute(output, "head-camera-geometry", [str(runtime(root)["eye_check"]),
                             str(output / "native-evidence")], root, 180)
            if result.get("status") != "passed" or result.get("geometry_and_camera_pose_admitted") is not True:
                raise RuntimeError("Native head camera geometry admission failed")
        elif args.command == "body-admission":
            result = body_admission(root, output, profile)
        elif args.command == "feeding-probe":
            result = feeding_probe(root, output, profile)
        elif args.command == "feeding-policy-probe":
            result = feeding_policy_probe(root,output,args,profile)
        elif args.command == "feeding-learning-campaign":
            result = feeding_learning_campaign(root,output,profile)
        elif args.command == "feeding-policy-comparison":
            result = feeding_policy_comparison(root,output,args,profile)
        elif args.command == "foraging-admission":
            result = execute(output, "foraging-ledger", [str(runtime(root)["foraging_check"]),
                             str(output / "native-evidence")], root, 180)
            if result.get("status") != "passed" or result.get("exact_partition_replay") is not True:
                raise RuntimeError("Native finite-resource admission failed")
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
    except (Exception, KeyboardInterrupt) as error:
        write(output / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
