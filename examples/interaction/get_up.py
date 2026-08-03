#!/usr/bin/env python3
"""Author the complete G1 supine-to-restored-stance interaction intent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))

from metalrobo.ardy_interaction_convert import (  # noqa: E402
    write_interaction_pack,
)


FPS = 50.0
FRAME_COUNT = 751
FREE = 0
APPROACH = 1
STICK = 2
RELEASE = 5

SUPINE = np.asarray(
    (
        -0.3600, 0.2481, 1.6115, -0.0647, -0.8612, -0.1226,
        -0.3878, 0.3584, 1.5328, 0.1519, -0.8651, 0.2362,
        -0.0357, 0.0685, -0.5200,
        0.4665, 0.8218, 0.4253, 1.2972, 0.0, 0.0, 0.0,
        0.1429, -1.0324, -0.4241, 1.4075, 0.0, 0.0, 0.0,
    ),
    dtype=np.float32,
)

STANDING = np.asarray(
    (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    ),
    dtype=np.float32,
)

BRACE = SUPINE.copy()
BRACE[15:22] = (-1.35, 1.50625, 1.25, -0.625, 0.0, 0.0, 0.0)
BRACE[22:29] = (-3.0, -0.35, -1.25, -0.8125, 0.0, 0.0, 0.0)

TUCK = BRACE.copy()
TUCK[0:6] = (-1.18, 0.15, 0.0, 2.22, -0.84, 0.0)
TUCK[6:12] = (-1.18, -0.15, 0.0, 2.22, -0.84, 0.0)
TUCK[12:15] = (0.0, 0.0, 0.47)

SUPPORTED_SQUAT = TUCK.copy()
SUPPORTED_SQUAT[0:6] = (-1.25, 0.10, 0.0, 2.35, -0.86, 0.0)
SUPPORTED_SQUAT[6:12] = (-1.25, -0.10, 0.0, 2.35, -0.86, 0.0)
SUPPORTED_SQUAT[12:15] = (0.0, 0.0, 0.38)

CROUCH = STANDING.copy()
CROUCH[0:6] = (-0.90, 0.08, 0.0, 1.80, -0.80, 0.0)
CROUCH[6:12] = (-0.90, -0.08, 0.0, 1.80, -0.80, 0.0)
CROUCH[14] = 0.22
CROUCH[15:19] = (-0.90, 0.35, 0.0, 0.40)
CROUCH[22:26] = (-0.90, -0.35, 0.0, 0.40)

HALF_STAND = STANDING.copy()
HALF_STAND[0:6] = (-0.48, 0.04, 0.0, 1.02, -0.48, 0.0)
HALF_STAND[6:12] = (-0.48, -0.04, 0.0, 1.02, -0.48, 0.0)
HALF_STAND[14] = 0.10
HALF_STAND[15:19] = (-0.35, 0.25, 0.0, 0.55)
HALF_STAND[22:26] = (-0.35, -0.25, 0.0, 0.55)


def _normalized(quaternion: np.ndarray) -> np.ndarray:
    return quaternion / np.linalg.norm(quaternion)


def _slerp(left: np.ndarray, right: np.ndarray, amount: float) -> np.ndarray:
    first = _normalized(left.astype(np.float64))
    second = _normalized(right.astype(np.float64))
    cosine = float(np.dot(first, second))
    if cosine < 0.0:
        second = -second
        cosine = -cosine
    if cosine > 0.9995:
        return _normalized(first + amount * (second - first)).astype(np.float32)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return (
        np.sin((1.0 - amount) * angle) / np.sin(angle) * first
        + np.sin(amount * angle) / np.sin(angle) * second
    ).astype(np.float32)


def _smooth(amount: float) -> float:
    clipped = float(np.clip(amount, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _interpolate_keyframes(
    frames: tuple[int, ...],
    values: tuple[np.ndarray, ...],
) -> np.ndarray:
    if len(frames) != len(values) or frames[0] != 0:
        raise ValueError("keyframes must start at zero and match their values")
    result = np.empty((frames[-1] + 1, values[0].size), dtype=np.float32)
    for segment in range(len(frames) - 1):
        start = frames[segment]
        end = frames[segment + 1]
        for frame in range(start, end + 1):
            blend = _smooth((frame - start) / max(end - start, 1))
            result[frame] = values[segment] + blend * (
                values[segment + 1] - values[segment]
            )
    return result


def _root_targets() -> np.ndarray:
    frames = (0, 70, 150, 260, 380, 500, 620, 750)
    positions = (
        np.asarray((0.0, 0.0, 0.01850436), dtype=np.float32),
        np.asarray((-0.04, 0.0, 0.05), dtype=np.float32),
        np.asarray((-0.12, 0.0, 0.12), dtype=np.float32),
        np.asarray((-0.10, 0.0, 0.22), dtype=np.float32),
        np.asarray((-0.04, 0.0, 0.40), dtype=np.float32),
        np.asarray((0.0, 0.0, 0.56), dtype=np.float32),
        np.asarray((0.0, 0.0, 0.72396994), dtype=np.float32),
        np.asarray((0.0, 0.0, 0.72396994), dtype=np.float32),
    )
    supine_q = np.asarray(
        (-0.0098234, 0.49986, 0.017525, -0.86587),
        dtype=np.float32,
    )
    identity = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    orientations = (
        supine_q,
        supine_q,
        _slerp(supine_q, identity, 0.20),
        _slerp(supine_q, identity, 0.45),
        _slerp(supine_q, identity, 0.70),
        _slerp(supine_q, identity, 0.88),
        identity,
        identity,
    )
    roots = np.empty((FRAME_COUNT, 7), dtype=np.float32)
    for segment in range(len(frames) - 1):
        start = frames[segment]
        end = frames[segment + 1]
        for frame in range(start, end + 1):
            blend = _smooth((frame - start) / max(end - start, 1))
            roots[frame, :3] = positions[segment] + blend * (
                positions[segment + 1] - positions[segment]
            )
            roots[frame, 3:] = _slerp(
                orientations[segment],
                orientations[segment + 1],
                blend,
            )
    return roots


def author(
    output_directory: Path,
    start_frame: int = 0,
) -> tuple[Path, str, int]:
    if start_frame < 0 or start_frame >= FRAME_COUNT - 1:
        raise ValueError("start frame must leave at least two recovery frames")
    output_directory.mkdir(parents=True, exist_ok=True)
    joints = _interpolate_keyframes(
        (0, 70, 150, 260, 380, 500, 620, 750),
        (
            SUPINE,
            SUPINE,
            BRACE,
            TUCK,
            SUPPORTED_SQUAT,
            CROUCH,
            STANDING,
            STANDING,
        ),
    )
    # The half-stand waypoint replaces the midpoint of the crouch-to-stand
    # segment without introducing a velocity discontinuity.
    joints[500:621] = _interpolate_keyframes(
        (0, 60, 120),
        (CROUCH, HALF_STAND, STANDING),
    )

    tracks = (
        ("left_foot", "left_foot_contact", "locomotion_ground"),
        ("right_foot", "right_foot_contact", "locomotion_ground"),
        ("left_hand", "left_hand_contact", "locomotion_ground"),
        ("right_hand", "right_hand_contact", "locomotion_ground"),
        ("left_knee", "left_knee_contact", "locomotion_ground"),
        ("right_knee", "right_knee_contact", "locomotion_ground"),
        ("trunk", "trunk_contact", "locomotion_ground"),
    )
    modes = np.full((FRAME_COUNT, len(tracks)), FREE, dtype=np.uint32)
    modes[80:120, 0:2] = APPROACH
    modes[120:, 0:2] = STICK
    modes[45:90, 2:4] = APPROACH
    modes[90:390, 2:4] = STICK
    modes[390:440, 2:4] = RELEASE
    modes[220:275, 4:6] = APPROACH
    modes[275:420, 4:6] = STICK
    modes[420:465, 4:6] = RELEASE
    modes[:180, 6] = STICK
    modes[180:230, 6] = RELEASE
    confidence = np.where(modes == STICK, 0.75, 0.35).astype(np.float32)
    root_targets = _root_targets()[start_frame:]
    joint_targets = joints[start_frame:]
    modes = modes[start_frame:]
    confidence = confidence[start_frame:]
    identity = (
        "get-up-restored"
        if start_frame == 0
        else f"get-up-restored-from-{start_frame:04d}"
    )
    pack_path = output_directory / f"{identity}.interactionpack"
    target, content_hash = write_interaction_pack(
        output=pack_path,
        pack_id=identity,
        clip_id=identity,
        desired_outcome=(
            "brace, tuck, squat, stand, and restore the pre-fall stance"
        ),
        source_repository="MetalRobo/examples/interaction",
        source_revision="contact-chain-restoration-v1",
        license_name="Apache-2.0",
        frames_per_second=FPS,
        root_targets=root_targets,
        joint_targets=joint_targets,
        tracks=tracks,
        contact_modes=modes,
        contact_confidence=confidence,
    )
    print(
        json.dumps(
            {
                "pack": str(target),
                "frames": int(root_targets.shape[0]),
                "source_start_frame": start_frame,
                "tracks": [track[0] for track in tracks],
                "content_hash": content_hash,
            },
            indent=2,
        )
    )
    return target, identity, int(root_targets.shape[0])


def execute(
    pack_path: Path,
    clip_id: str,
    frame_count: int,
    rollout: Path,
) -> None:
    trace = pack_path.with_suffix(".state.tsv")
    completed = subprocess.run(
        (
            str(rollout),
            "--task", "supine-get-up",
            "--scene", "ground",
            "--envs", "1",
            "--steps", str(frame_count - 2),
            "--repeats", "1",
            "--chunk", "1",
            "--zero-actions",
            "--no-scheduled-resets",
            "--interaction-pack", str(pack_path),
            "--interaction-clip", clip_id,
            "--state-trace", str(trace),
        ),
        check=False,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "recovery rollout failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result = json.loads(completed.stdout)
    if result["failed_environment_steps"] != 0:
        raise RuntimeError("generated recovery intent caused a physics failure")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--rollout",
        type=Path,
        default=REPOSITORY / "build/bin/metalrobo_task_rollout",
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    arguments = parser.parse_args()
    pack, clip_id, frame_count = author(
        arguments.output_directory.resolve(),
        arguments.start_frame,
    )
    if not arguments.generate_only:
        execute(pack, clip_id, frame_count, arguments.rollout.resolve())


if __name__ == "__main__":
    main()
