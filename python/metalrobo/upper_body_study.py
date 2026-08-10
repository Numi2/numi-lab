"""Run the PQI-II generated upper-body robustness study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np


SEEDS = (2650443581, 2650443582, 2650443583, 2650443584, 2650443585)
TASKS = {
    "raise-right-hand": {"steps": 40, "joints": (22,)},
    "raise-left-hand": {"steps": 40, "joints": (15,)},
    "raise-both-hands": {"steps": 36, "joints": (15, 22)},
}
METHODS = {
    "raw-generated": {
        "directory": "{task}",
        "pack": "ardy-g1.interactionpack",
        "clip": "ardy-g1",
    },
    "stable-base-composition": {
        "directory": "{task}-composed",
        "pack": "motion.interactionpack",
        "clip": "{task}",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, text=True, capture_output=True,
    ).stdout.strip()


def _metrics(result: dict[str, Any], trace: np.ndarray, joints: tuple[int, ...]) -> dict[str, Any]:
    shoulders = []
    for joint in joints:
        values = trace[:, 1 + 7 + joint]
        shoulders.append({
            "joint_index": joint,
            "initial_rad": float(values[0]),
            "final_rad": float(values[-1]),
            "minimum_rad": float(np.min(values)),
            "excursion_rad": float(values[0] - np.min(values)),
        })
    stable = (
        result["failed_environment_steps"] == 0
        and result["termination_count"] == 0
        and result["maximum_tilt"] < 0.50
        and min(result["minimum_root_height_by_environment"]) > 0.64
    )
    raised = all(item["minimum_rad"] <= -0.15 for item in shoulders)
    return {
        "failed_environment_steps": result["failed_environment_steps"],
        "termination_count": result["termination_count"],
        "minimum_root_height_m": min(result["minimum_root_height_by_environment"]),
        "maximum_tilt_rad": result["maximum_tilt"],
        "mean_tracking_score": result["mean_tracking_score"],
        "shoulders": shoulders,
        "stable": stable,
        "raised": raised,
        "qualified_success": stable and raised,
        "policy": "negative shoulder pitch raises the corresponding G1 hand forward",
    }


def run(
    output: Path, inputs: Path, protocol_path: Path, *, allow_dirty: bool = False,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "numi.pqi2-upper-body-protocol.v1" or protocol.get("status") != "preregistered":
        raise ValueError("upper-body protocol must be preregistered v1")
    if tuple(protocol.get("seeds", ())) != SEEDS or set(protocol.get("tasks", {})) != set(TASKS):
        raise ValueError("upper-body protocol does not match the executable study contract")
    protocol_sha256 = _sha256(protocol_path)
    if _git("branch", "--show-current") != "numisolver":
        raise RuntimeError("PQI-II must run on numisolver")
    status = _git("status", "--porcelain")
    if status and not allow_dirty:
        raise RuntimeError("PQI-II confirmatory study requires a clean worktree")
    revision = _git("rev-parse", "HEAD")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for task, task_spec in TASKS.items():
        for method, method_spec in METHODS.items():
            directory = inputs / method_spec["directory"].format(task=task)
            pack = directory / method_spec["pack"]
            clip = method_spec["clip"].format(task=task)
            if not pack.is_file():
                raise FileNotFoundError(pack)
            for seed in SEEDS:
                run_id = f"{task}--{method}--seed-{seed}"
                run_directory = output / run_id
                run_directory.mkdir(parents=True, exist_ok=True)
                trace_path = run_directory / "state.tsv"
                evidence_path = run_directory / "evidence.json"
                command = [
                    "build/bin/metalrobo_task_rollout",
                    "--task", "velocity", "--scene", "ground",
                    "--minimum-difficulty-band", "1",
                    "--maximum-difficulty-band", "1",
                    "--envs", "1", "--steps", str(task_spec["steps"]),
                    "--repeats", "1", "--chunk", "1", "--zero-actions",
                    "--no-scheduled-resets", "--seed", str(seed),
                    "--interaction-pack", str(pack),
                    "--interaction-clip", clip,
                    "--interaction-student-authority", "0",
                    "--state-trace", str(trace_path),
                ]
                completed = subprocess.run(command, text=True, capture_output=True)
                (run_directory / "stderr.log").write_text(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(f"{run_id} failed: {completed.stderr.strip()}")
                native = json.loads(completed.stdout)
                trace = np.loadtxt(trace_path, comments="#")
                record = {
                    "schema": "numi.pqi2-upper-body-run.v1",
                    "protocol_sha256": protocol_sha256,
                    "run_id": run_id,
                    "revision": revision,
                    "worktree_status": status,
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "steps": task_spec["steps"],
                    "interaction_pack": str(pack),
                    "interaction_pack_sha256": _sha256(pack),
                    "command": command,
                    **_metrics(native, trace, task_spec["joints"]),
                }
                evidence_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                records.append(record)
    qualified = [record["qualified_success"] for record in records]
    summary = {
        "schema": "numi.pqi2-upper-body-summary.v1",
        "protocol_sha256": protocol_sha256,
        "revision": revision,
        "worktree_status": status,
        "planned_runs": len(records),
        "completed_runs": len(records),
        "failed_environment_steps": sum(record["failed_environment_steps"] for record in records),
        "qualified_successes": sum(qualified),
        "claims_unlocked": len(records) == 30 and not status and all(
            record["failed_environment_steps"] == 0 for record in records
        ),
        "by_method": {
            method: {
                "runs": sum(record["method"] == method for record in records),
                "qualified_successes": sum(
                    record["qualified_success"] for record in records
                    if record["method"] == method
                ),
            }
            for method in METHODS
        },
        "records": records,
    }
    (output / "study-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output / "study-summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task", "method", "seed", "stable", "raised", "qualified_success",
                         "terminations", "max_tilt_rad", "min_root_height_m"))
        for record in records:
            writer.writerow((record["task"], record["method"], record["seed"],
                             record["stable"], record["raised"], record["qualified_success"],
                             record["termination_count"], record["maximum_tilt_rad"],
                             record["minimum_root_height_m"]))
    return summary


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(".numi/runs/pqi2-inputs"))
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("research/pqi2/upper_body_protocol.json"),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    options = parser.parse_args(arguments)
    print(json.dumps(run(
        options.output, options.inputs, options.protocol,
        allow_dirty=options.allow_dirty,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
