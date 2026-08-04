#!/usr/bin/env python3
"""Author the complete G1 supine-to-restored-stance interaction intent."""

from __future__ import annotations

import argparse
import hashlib
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
ROOT_CENTER_OF_MASS_LOCAL = np.asarray(
    (0.0, 0.0, -0.076030060304), dtype=np.float32
)

CONTACT_TRACKS = (
    ("left_foot", "left_foot_contact", "locomotion_ground"),
    ("right_foot", "right_foot_contact", "locomotion_ground"),
    ("left_hand", "left_hand_contact", "locomotion_ground"),
    ("right_hand", "right_hand_contact", "locomotion_ground"),
    ("left_knee", "left_knee_contact", "locomotion_ground"),
    ("right_knee", "right_knee_contact", "locomotion_ground"),
    ("trunk", "trunk_contact", "locomotion_ground"),
)

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
        # Native standing sweeps found +0.22 normalized symmetric ankle pitch
        # to be the best tested balance command: in the refined 256-step
        # sweep it reduced resets to 12/16 environments and raised restored
        # incidence to 0.918. The get-up action range maps that command to
        # this physical target (-0.2 + 0.22 * 0.7236 = -0.040808 rad). ARDY supplies
        # the imagined target; NumiSolver still decides whether it can stand.
        -0.1, 0.0, 0.0, 0.3, -0.040808, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.040808, 0.0,
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

# Squat motion is authored as a displacement around the mechanism's nominal
# stance. The standing policy then contributes its complete learned residual,
# including the calibrated ankle bias, without double-counting that bias in
# the ARDY reference.
SQUAT_STANCE = STANDING.copy()
SQUAT_STANCE[4] = -0.20
SQUAT_STANCE[10] = -0.20

# A deliberately shallow, quasi-static squat for the first cyclic skill. It
# preserves bilateral foot support and asks the existing balance policy to
# solve only a small center-of-mass excursion before depth is increased.
SHALLOW_SQUAT = SQUAT_STANCE.copy()
SHALLOW_SQUAT[0:6] = (-0.20, 0.0, 0.0, 0.48, -0.28, 0.0)
SHALLOW_SQUAT[6:12] = (-0.20, 0.0, 0.0, 0.48, -0.28, 0.0)


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


def _solver_roots_to_link(roots: np.ndarray) -> np.ndarray:
    """Convert solver COM poses to InteractionPack root-link poses."""

    result = np.asarray(roots, dtype=np.float32).copy()
    quaternion = result[:, 3:7]
    vector = np.repeat(
        ROOT_CENTER_OF_MASS_LOCAL[None, :], result.shape[0], axis=0
    )
    first_cross = np.cross(quaternion[:, :3], vector)
    rotated = vector + 2.0 * (
        quaternion[:, 3:4] * first_cross
        + np.cross(quaternion[:, :3], first_cross)
    )
    result[:, :3] -= rotated
    return result


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
    return _solver_roots_to_link(roots)


def author(
    output_directory: Path,
    start_frame: int = 0,
    hold_frames: int = 0,
    episode_frames: int | None = None,
) -> tuple[Path, str, int]:
    if start_frame < 0 or start_frame >= FRAME_COUNT - 1:
        raise ValueError("start frame must leave at least two recovery frames")
    if hold_frames < 0:
        raise ValueError("hold frames must be non-negative")
    if episode_frames is not None and episode_frames < 2:
        raise ValueError("episode frames must be at least two")
    if episode_frames is not None and hold_frames:
        raise ValueError(
            "episode frames and hold frames are mutually exclusive"
        )
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

    tracks = CONTACT_TRACKS
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
    if episode_frames is not None:
        root_targets = root_targets[:episode_frames]
        joint_targets = joint_targets[:episode_frames]
        modes = modes[:episode_frames]
        confidence = confidence[:episode_frames]
    if hold_frames:
        root_targets = np.concatenate(
            (
                root_targets,
                np.repeat(root_targets[-1:], hold_frames, axis=0),
            )
        )
        joint_targets = np.concatenate(
            (
                joint_targets,
                np.repeat(joint_targets[-1:], hold_frames, axis=0),
            )
        )
        modes = np.concatenate(
            (modes, np.repeat(modes[-1:], hold_frames, axis=0))
        )
        confidence = np.concatenate(
            (
                confidence,
                np.repeat(confidence[-1:], hold_frames, axis=0),
            )
        )
    identity = (
        "get-up-restored"
        if start_frame == 0
        else f"get-up-restored-from-{start_frame:04d}"
    )
    if hold_frames:
        identity += f"-hold-{hold_frames:04d}"
    if episode_frames is not None:
        identity += f"-frames-{root_targets.shape[0]:04d}"
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


def author_solver_state(
    output_directory: Path,
    state_trace: Path,
    state_step: int,
    episode_frames: int,
) -> tuple[Path, str, int]:
    """Author a reset curriculum from one accepted native solver state."""

    if state_step <= 0:
        raise ValueError("solver state step must be positive")
    if episode_frames < 2:
        raise ValueError("episode frames must be at least two")
    payload = state_trace.read_bytes()
    lines = payload.decode("utf-8").splitlines()
    rows = [line for line in lines if line and not line.startswith("#")]
    if state_step > len(rows):
        raise ValueError("solver state step exceeds the trace length")
    fields = rows[state_step - 1].split("\t")
    if len(fields) < 37 or int(fields[0]) != state_step:
        raise ValueError("solver state trace row is malformed or misordered")
    state = np.asarray(fields[1:37], dtype=np.float32)
    if state.shape != (36,) or not np.all(np.isfinite(state)):
        raise ValueError("solver state must contain finite root plus 29 joints")
    quaternion_norm = float(np.linalg.norm(state[3:7]))
    if abs(quaternion_norm - 1.0) > 1.0e-3:
        raise ValueError("solver state root quaternion is not normalized")

    output_directory.mkdir(parents=True, exist_ok=True)
    root_targets = _solver_roots_to_link(
        np.repeat(state[:7][None, :], episode_frames, axis=0)
    )
    joint_targets = np.repeat(
        state[7:][None, :],
        episode_frames,
        axis=0,
    )
    # A trace proves accepted coordinates, not future contact. Keep every
    # contact prediction free; NumiSolver remains the only contact authority.
    modes = np.full(
        (episode_frames, len(CONTACT_TRACKS)),
        FREE,
        dtype=np.uint32,
    )
    confidence = np.full(modes.shape, 0.35, dtype=np.float32)
    identity = (
        f"solver-state-{state_step:04d}-frames-{episode_frames:04d}"
    )
    target, content_hash = write_interaction_pack(
        output=output_directory / f"{identity}.interactionpack",
        pack_id=identity,
        clip_id=identity,
        desired_outcome="recover from an accepted native solver state",
        source_repository="MetalRobo/native-state-trace",
        source_revision="sha256:" + hashlib.sha256(payload).hexdigest(),
        license_name="Apache-2.0",
        frames_per_second=FPS,
        root_targets=root_targets,
        joint_targets=joint_targets,
        tracks=CONTACT_TRACKS,
        contact_modes=modes,
        contact_confidence=confidence,
    )
    print(
        json.dumps(
            {
                "pack": str(target),
                "frames": episode_frames,
                "solver_state_step": state_step,
                "source_trace_sha256": hashlib.sha256(payload).hexdigest(),
                "content_hash": content_hash,
            },
            indent=2,
        )
    )
    return target, identity, episode_frames


def author_solver_state_to_stand(
    output_directory: Path,
    state_trace: Path,
    state_step: int,
    episode_frames: int,
) -> tuple[Path, str, int]:
    """Imagine a squat-to-stand action sequence from accepted solver state."""

    if episode_frames < 161:
        raise ValueError("solver-state stand authoring requires 161+ frames")
    payload = state_trace.read_bytes()
    rows = [
        line for line in payload.decode("utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if state_step <= 0 or state_step > len(rows):
        raise ValueError("solver state step exceeds the trace length")
    fields = rows[state_step - 1].split("\t")
    if len(fields) < 37 or int(fields[0]) != state_step:
        raise ValueError("solver state trace row is malformed or misordered")
    state = np.asarray(fields[1:37], dtype=np.float32)
    if state.shape != (36,) or not np.all(np.isfinite(state)):
        raise ValueError("solver state must contain finite root plus 29 joints")
    if abs(float(np.linalg.norm(state[3:7])) - 1.0) > 1.0e-3:
        raise ValueError("solver state root quaternion is not normalized")

    settle = min(30, episode_frames // 8)
    half = min(140, episode_frames - 50)
    stand = min(220, episode_frames - 1)
    joint_targets = _interpolate_keyframes(
        (0, settle, half, stand, episode_frames - 1),
        (state[7:], state[7:], HALF_STAND, STANDING, STANDING),
    )
    root_targets = np.empty((episode_frames, 7), dtype=np.float32)
    identity_q = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    for frame in range(episode_frames):
        blend = _smooth((frame - settle) / max(stand - settle, 1))
        root_targets[frame, :3] = state[:3] + blend * (
            np.asarray((state[0], state[1], 0.72396994), dtype=np.float32)
            - state[:3]
        )
        root_targets[frame, 3:] = _slerp(state[3:7], identity_q, blend)
    root_targets = _solver_roots_to_link(root_targets)
    modes = np.full(
        (episode_frames, len(CONTACT_TRACKS)), FREE, dtype=np.uint32
    )
    confidence = np.full(modes.shape, 0.35, dtype=np.float32)
    identity = f"solver-squat-{state_step:04d}-to-stand-{episode_frames:04d}"
    output_directory.mkdir(parents=True, exist_ok=True)
    target, content_hash = write_interaction_pack(
        output=output_directory / f"{identity}.interactionpack",
        pack_id=identity,
        clip_id=identity,
        desired_outcome="extend from a solver-accepted squat into quiet stance",
        source_repository="MetalRobo/native-state-trace",
        source_revision="sha256:" + hashlib.sha256(payload).hexdigest(),
        license_name="Apache-2.0",
        frames_per_second=FPS,
        root_targets=root_targets,
        joint_targets=joint_targets,
        tracks=CONTACT_TRACKS,
        contact_modes=modes,
        contact_confidence=confidence,
    )
    print(json.dumps({
        "pack": str(target),
        "frames": episode_frames,
        "solver_state_step": state_step,
        "target": "stand",
        "source_trace_sha256": hashlib.sha256(payload).hexdigest(),
        "content_hash": content_hash,
    }, indent=2))
    return target, identity, episode_frames


def author_squat_cycles(
    output_directory: Path,
    cycle_count: int,
) -> tuple[Path, str, int]:
    """Author slow bilateral shallow-squat cycles from nominal stance."""

    if cycle_count <= 0:
        raise ValueError("squat cycle count must be positive")
    # Keep the first motor primitive short: 0.9 s down, 0.1 s settle,
    # 0.9 s up, 0.1 s settle. ARDY owns the smooth motion while the imported
    # standing controller spends its authority on balance rather than waiting
    # through a long static prelude.
    hold_frames = 5
    move_frames = 45
    keyframes = [0]
    poses = [SQUAT_STANCE]
    heights = [0.72396994]
    frame = hold_frames
    keyframes.append(frame)
    poses.append(SQUAT_STANCE)
    heights.append(0.72396994)
    for _ in range(cycle_count):
        frame += move_frames
        keyframes.append(frame)
        poses.append(SHALLOW_SQUAT)
        heights.append(0.68)
        frame += hold_frames
        keyframes.append(frame)
        poses.append(SHALLOW_SQUAT)
        heights.append(0.68)
        frame += move_frames
        keyframes.append(frame)
        poses.append(SQUAT_STANCE)
        heights.append(0.72396994)
        frame += hold_frames
        keyframes.append(frame)
        poses.append(SQUAT_STANCE)
        heights.append(0.72396994)

    joint_targets = _interpolate_keyframes(
        tuple(keyframes),
        tuple(poses),
    )
    root_targets = np.empty((frame + 1, 7), dtype=np.float32)
    root_targets[:, :2] = 0.0
    root_targets[:, 3:] = np.asarray(
        (0.0, 0.0, 0.0, 1.0), dtype=np.float32
    )
    root_targets[:, 2] = _interpolate_keyframes(
        tuple(keyframes),
        tuple(np.asarray((height,), dtype=np.float32) for height in heights),
    )[:, 0]
    root_targets = _solver_roots_to_link(root_targets)
    tracks = CONTACT_TRACKS[:2]
    modes = np.full(
        (frame + 1, len(tracks)), STICK, dtype=np.uint32
    )
    confidence = np.full(modes.shape, 0.35, dtype=np.float32)
    confidence[:, :] = 0.90
    identity = f"standing-shallow-squat-cycles-{cycle_count:02d}"
    output_directory.mkdir(parents=True, exist_ok=True)
    target, content_hash = write_interaction_pack(
        output=output_directory / f"{identity}.interactionpack",
        pack_id=identity,
        clip_id=identity,
        desired_outcome=(
            "lower into a bilateral shallow squat and return to quiet stance"
        ),
        source_repository="MetalRobo/examples/interaction",
        source_revision="shallow-squat-cycle-v2",
        license_name="Apache-2.0",
        frames_per_second=FPS,
        root_targets=root_targets,
        joint_targets=joint_targets,
        tracks=tracks,
        contact_modes=modes,
        contact_confidence=confidence,
    )
    print(json.dumps({
        "pack": str(target),
        "frames": frame + 1,
        "squat_cycles": cycle_count,
        "content_hash": content_hash,
    }, indent=2))
    return target, identity, frame + 1


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
    parser.add_argument("--hold-frames", type=int, default=0)
    parser.add_argument("--episode-frames", type=int)
    parser.add_argument("--squat-cycles", type=int, default=0)
    parser.add_argument("--solver-state-trace", type=Path)
    parser.add_argument("--solver-state-step", type=int)
    parser.add_argument(
        "--solver-state-target",
        choices=("hold", "stand"),
        default="hold",
    )
    arguments = parser.parse_args()
    if (arguments.solver_state_trace is None) != (
        arguments.solver_state_step is None
    ):
        parser.error(
            "--solver-state-trace and --solver-state-step are required together"
        )
    if arguments.squat_cycles:
        if (arguments.solver_state_trace is not None or
                arguments.start_frame != 0 or arguments.hold_frames or
                arguments.episode_frames is not None):
            parser.error(
                "--squat-cycles cannot combine with recovery authoring options"
            )
        pack, clip_id, frame_count = author_squat_cycles(
            arguments.output_directory.resolve(),
            arguments.squat_cycles,
        )
    elif arguments.solver_state_trace is not None:
        if arguments.episode_frames is None:
            parser.error("solver-state authoring requires --episode-frames")
        if arguments.start_frame != 0 or arguments.hold_frames:
            parser.error(
                "solver-state authoring cannot combine with start/hold frames"
            )
        author_from_state = (
            author_solver_state_to_stand
            if arguments.solver_state_target == "stand"
            else author_solver_state
        )
        pack, clip_id, frame_count = author_from_state(
            arguments.output_directory.resolve(),
            arguments.solver_state_trace.resolve(),
            arguments.solver_state_step,
            arguments.episode_frames,
        )
    else:
        pack, clip_id, frame_count = author(
            arguments.output_directory.resolve(),
            arguments.start_frame,
            arguments.hold_frames,
            arguments.episode_frames,
        )
    if not arguments.generate_only:
        execute(pack, clip_id, frame_count, arguments.rollout.resolve())


if __name__ == "__main__":
    main()
