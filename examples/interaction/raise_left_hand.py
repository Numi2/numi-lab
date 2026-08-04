#!/usr/bin/env python3
"""Author and optionally execute the minimal G1 InteractionPack example."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))

from metalrobo.ardy_interaction_convert import convert  # noqa: E402


RESET_JOINTS = np.asarray(
    (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    ),
    dtype=np.float32,
)


def author(output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_count = 61
    qpos = np.tile(
        np.concatenate(
            (
                # ARDY CSV convention: root xyz, quaternion wxyz, joints.
                np.asarray(
                    (0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
                    dtype=np.float32,
                ),
                RESET_JOINTS,
            )
        ),
        (frame_count, 1),
    )
    for frame in range(frame_count):
        progress = float(np.clip((frame - 5) / 35, 0.0, 1.0))
        smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
        # Negative G1 shoulder pitch raises the left hand forward.
        qpos[frame, 7 + 15] = 0.35 + (-1.0 - 0.35) * smooth

    qpos_path = output_directory / "raise-left-hand-qpos.csv"
    motion_path = output_directory / "raise-left-hand-motion.npz"
    pack_path = output_directory / "raise-left-hand.interactionpack"
    np.savetxt(qpos_path, qpos, delimiter=",", fmt="%.9g")
    np.savez(
        motion_path,
        foot_contacts=np.ones((frame_count, 4), dtype=np.float32),
        fps=np.asarray((50.0,), dtype=np.float32),
        text=np.asarray(("raise left hand",)),
    )
    convert(
        SimpleNamespace(
            motion_npz=motion_path,
            qpos_csv=qpos_path,
            output=pack_path,
            id="raise-left-hand",
            clip_id="raise-left-hand",
            desired_outcome="raise left hand",
            source_repository="MetalRobo/examples/interaction",
            source_revision="raise-left-hand-v1",
            license="Apache-2.0",
            left_contact_group="left_foot_contact",
            right_contact_group="right_foot_contact",
            counterpart="locomotion_ground",
            contact_confidence=0.5,
            joint_limit_margin=1.0e-4,
            loop=False,
        )
    )
    return pack_path


def execute(pack_path: Path, rollout: Path) -> None:
    trace_path = pack_path.with_suffix(".state.tsv")
    completed = subprocess.run(
        (
            str(rollout),
            "--task", "velocity",
            "--scene", "ground",
            "--envs", "1",
            "--steps", "50",
            "--repeats", "1",
            "--chunk", "1",
            "--zero-actions",
            "--no-scheduled-resets",
            "--interaction-pack", str(pack_path),
            "--interaction-clip", "raise-left-hand",
            "--state-trace", str(trace_path),
        ),
        check=True,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    trace = np.loadtxt(trace_path, comments="#")
    shoulder = trace[:, 1 + 7 + 15]
    if (
        result["failed_environment_steps"] != 0
        or result["termination_count"] != 0
        or result["standing_step_count"] != 50
        or shoulder[-1] > -0.7
    ):
        raise RuntimeError("native raise-left-hand realization did not qualify")
    print(
        json.dumps(
            {
                "pack": str(pack_path),
                "state_trace": str(trace_path),
                "device": result["device"],
                "standing_steps": result["standing_step_count"],
                "failed_environment_steps": result[
                    "failed_environment_steps"
                ],
                "terminations": result["termination_count"],
                "initial_left_shoulder_pitch": float(shoulder[0]),
                "final_left_shoulder_pitch": float(shoulder[-1]),
                "mean_tracking_score": result["mean_tracking_score"],
                "maximum_tilt": result["maximum_tilt"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--rollout",
        type=Path,
        default=REPOSITORY / "build/bin/metalrobo_task_rollout",
    )
    parser.add_argument("--generate-only", action="store_true")
    arguments = parser.parse_args()
    pack_path = author(arguments.output_directory.resolve())
    if arguments.generate_only:
        print(pack_path)
    else:
        execute(pack_path, arguments.rollout.resolve())


if __name__ == "__main__":
    main()
