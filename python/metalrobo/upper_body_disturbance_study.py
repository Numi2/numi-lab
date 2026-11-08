"""Run the preregistered PQI-II invariant-base disturbance study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np


WRIST_COLUMNS = {"left": 3, "right": 12}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, text=True, capture_output=True,
    ).stdout.strip()


def run(
    output: Path, inputs: Path, protocol_path: Path,
    policies: dict[str, Path], *, allow_dirty: bool = False,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "numi.pqi2-disturbance-protocol.v1" or protocol.get("status") != "preregistered":
        raise ValueError("disturbance protocol must be preregistered v1")
    if _git("branch", "--show-current") != "numisolver":
        raise RuntimeError("disturbance study must run on numisolver")
    status = _git("status", "--porcelain")
    if status and not allow_dirty:
        raise RuntimeError("confirmatory disturbance study requires a clean worktree")
    revision = _git("rev-parse", "HEAD")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for task, specification in protocol["tasks"].items():
        policy = policies[task]
        if not policy.is_file():
            raise FileNotFoundError(policy)
        pack = inputs / f"{task}-composed" / "motion.interactionpack"
        for method in protocol["methods"]:
            for seed in protocol["confirmatory_seeds"]:
                run_id = f"{task}--{method}--seed-{seed}"
                directory = output / run_id
                directory.mkdir(parents=True, exist_ok=True)
                state = directory / "state.tsv"
                motion = directory / "wrist-motion.tsv"
                command = [
                    "build/bin/metalrobo_task_rollout",
                    "--task", "velocity", "--scene", "ground",
                    "--minimum-difficulty-band", "1",
                    "--maximum-difficulty-band", "1",
                    "--envs", "1", "--steps", str(specification["steps"]),
                    "--repeats", "1", "--chunk", "1",
                    "--no-scheduled-resets", "--seed", str(seed),
                    "--interaction-pack", str(pack),
                    "--interaction-clip", task,
                    "--interaction-reset-phase-fraction", "0",
                    "--interaction-push-maximum-velocity",
                    str(specification["push_maximum_velocity_mps"]),
                    "--interaction-push-interval-seconds",
                    str(protocol["push_interval_seconds"]),
                    "--state-trace", str(state),
                    "--motion-feature-trace", str(motion),
                ]
                if method == "fixed-base":
                    command += ["--zero-actions", "--interaction-student-authority", "0"]
                else:
                    command += ["--policy-pack", str(policy), "--interaction-student-authority", "0.1"]
                completed = subprocess.run(command, text=True, capture_output=True)
                (directory / "stderr.log").write_text(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(f"{run_id} failed: {completed.stderr.strip()}")
                (directory / "result.json").write_text(completed.stdout)
                native = json.loads(completed.stdout)
                trace = np.loadtxt(motion, comments="#")
                wrists = []
                for side in specification["requested_wrists"]:
                    values = trace[:, WRIST_COLUMNS[side]]
                    wrists.append({
                        "side": side,
                        "peak_height_m": float(np.max(values)),
                        "upward_excursion_m": float(np.max(values) - values[0]),
                    })
                stable = (
                    native["failed_environment_steps"] == 0
                    and native["termination_count"] == 0
                    and native["maximum_tilt"] < 0.50
                    and min(native["minimum_root_height_by_environment"]) > 0.64
                )
                raised = all(
                    wrist["peak_height_m"] >= 0.88
                    and wrist["upward_excursion_m"] >= 0.20
                    for wrist in wrists
                )
                records.append({
                    "run_id": run_id, "revision": revision,
                    "protocol_sha256": _sha256(protocol_path),
                    "task": task, "method": method, "seed": seed,
                    "policy_pack": str(policy) if method != "fixed-base" else None,
                    "policy_pack_sha256": _sha256(policy) if method != "fixed-base" else None,
                    "interaction_pack_sha256": _sha256(pack),
                    "command": command,
                    "state_trace_sha256": _sha256(state),
                    "motion_trace_sha256": _sha256(motion),
                    "failed_environment_steps": native["failed_environment_steps"],
                    "termination_count": native["termination_count"],
                    "minimum_root_height_m": min(native["minimum_root_height_by_environment"]),
                    "maximum_tilt_rad": native["maximum_tilt"],
                    "wrists": wrists, "stable": stable, "raised": raised,
                    "qualified_success": stable and raised,
                })
    learned = [record for record in records if record["method"] == "learned-invariant-base"]
    fixed = [record for record in records if record["method"] == "fixed-base"]
    planned_runs = (
        len(protocol["tasks"])
        * len(protocol["methods"])
        * len(protocol["confirmatory_seeds"])
    )
    claims_unlocked = (
        len(records) == planned_runs
        and all(record["failed_environment_steps"] == 0 for record in records)
        and all(record["qualified_success"] for record in learned)
        and all(any(
            item["task"] == task and not item["qualified_success"]
            for item in fixed
        ) for task in protocol["tasks"])
    )
    summary = {
        "schema": "numi.pqi2-disturbance-summary.v1",
        "revision": revision, "worktree_status": status,
        "protocol_sha256": _sha256(protocol_path),
        "planned_runs": planned_runs, "completed_runs": len(records),
        "claims_unlocked": claims_unlocked, "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(".numi/runs/pqi2-inputs"))
    parser.add_argument("--protocol", type=Path, default=Path("research/pqi2/upper_body_disturbance_protocol.json"))
    parser.add_argument("--right-policy", type=Path, required=True)
    parser.add_argument("--left-policy", type=Path, required=True)
    parser.add_argument("--both-policy", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    options = parser.parse_args(arguments)
    summary = run(options.output, options.inputs, options.protocol, {
        "raise-right-hand": options.right_policy,
        "raise-left-hand": options.left_policy,
        "raise-both-hands": options.both_policy,
    }, allow_dirty=options.allow_dirty)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
