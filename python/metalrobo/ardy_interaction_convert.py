"""Convert one NVIDIA ARDY G1 generation into InteractionPack v1.

ARDY's G1 CSV already uses MuJoCo's z-up/x-forward frame and carries root
quaternions as wxyz. This authoring bridge reorders them to MetalRobo xyzw,
validates the canonical 29-DoF G1 joint contract, and imports ARDY's four
binary foot channels only as predicted contact modes. It deliberately leaves
wrench, CoP, area, and pressure validity masks empty until NumiSolver certifies
those physical fields.
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

_HEADER = struct.Struct("<8sIIQQ")
_INTERACTION_VERSION = 1
_INTERACTION_KIND = 5
_CONTACT_FEATURE_COUNT = 13
_CONTACT_MODE_FREE = 0
_CONTACT_MODE_STICK = 2
_SAMPLE_PREDICTED = 1
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211

_G1_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_G1_JOINT_LOWER = np.asarray(
    (
        -2.5307,
        -0.5236,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.5307,
        -2.9671,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.618,
        -0.52,
        -0.52,
        -3.0892,
        -1.5882,
        -2.618,
        -1.0472,
        -1.972222054,
        -1.614429558,
        -1.614429558,
        -3.0892,
        -2.2515,
        -2.618,
        -1.0472,
        -1.972222054,
        -1.614429558,
        -1.614429558,
    ),
    dtype=np.float32,
)
_G1_JOINT_UPPER = np.asarray(
    (
        2.8798,
        2.9671,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.8798,
        0.5236,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.618,
        0.52,
        0.52,
        2.6704,
        2.2515,
        2.618,
        2.0944,
        1.972222054,
        1.614429558,
        1.614429558,
        2.6704,
        1.5882,
        2.618,
        2.0944,
        1.972222054,
        1.614429558,
        1.614429558,
    ),
    dtype=np.float32,
)
_G1_JOINT_VELOCITY = np.asarray(
    (
        32.0,
        20.0,
        32.0,
        20.0,
        30.0,
        30.0,
        32.0,
        20.0,
        32.0,
        20.0,
        30.0,
        30.0,
        32.0,
        30.0,
        30.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        22.0,
        22.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        22.0,
        22.0,
    ),
    dtype=np.float32,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _strings(values: tuple[str, ...]) -> bytes:
    return struct.pack("<Q", len(values)) + b"".join(_string(value) for value in values)


def _vector(values: np.ndarray, dtype: str) -> bytes:
    packed = np.asarray(values, dtype=dtype).reshape(-1)
    return struct.pack("<Q", packed.size) + packed.tobytes()


def _content_hash(payload: bytes) -> int:
    result = _FNV_OFFSET
    for value in payload:
        result ^= value
        result = (result * _FNV_PRIME) & ((1 << 64) - 1)
    return result or 1


def _load_inputs(
    qpos_csv: Path,
    motion_npz: Path,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    qpos = np.asarray(np.loadtxt(qpos_csv, delimiter=","), dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    with np.load(motion_npz, allow_pickle=False) as archive:
        contacts = np.asarray(archive["foot_contacts"])
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        source_text = np.asarray(archive["text"]).reshape(-1)[0]
        desired_outcome = (
            source_text.decode("utf-8")
            if isinstance(source_text, bytes)
            else str(source_text)
        )
    binary_contacts = np.logical_or(
        np.isclose(contacts, 0.0, atol=1.0e-6),
        np.isclose(contacts, 1.0, atol=1.0e-6),
    )
    if (
        qpos.ndim != 2
        or qpos.shape[0] < 2
        or qpos.shape[1] != 7 + len(_G1_JOINTS)
        or contacts.shape != (qpos.shape[0], 4)
        or not np.isfinite(qpos).all()
        or not np.isfinite(contacts).all()
        or not binary_contacts.all()
        or not np.isfinite(fps)
        or fps <= 0.0
        or not desired_outcome.strip()
    ):
        raise ValueError("ARDY qpos, contacts, fps, or desired outcome is invalid")
    return qpos, contacts > 0.5, fps, desired_outcome.strip()


def _metalrobo_targets(
    qpos: np.ndarray,
    fps: float,
    joint_limit_margin: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    joints = np.asarray(qpos[:, 7:], dtype=np.float32)
    below = joints < (_G1_JOINT_LOWER - joint_limit_margin)
    above = joints > (_G1_JOINT_UPPER + joint_limit_margin)
    if np.any(below | above):
        frame, joint = np.argwhere(below | above)[0]
        raise ValueError(
            f"ARDY joint target violates MetalRobo G1 limits: frame={frame} "
            f"joint={_G1_JOINTS[joint]} value={joints[frame, joint]:.6f} "
            f"range=[{_G1_JOINT_LOWER[joint]:.6f}, {_G1_JOINT_UPPER[joint]:.6f}]"
        )
    velocity = np.abs(np.diff(joints, axis=0)) * fps
    velocity_ratio = velocity / _G1_JOINT_VELOCITY[None, :]
    peak_velocity_ratio = float(np.max(velocity_ratio))
    if peak_velocity_ratio > 1.0 + 1.0e-5:
        frame, joint = np.unravel_index(np.argmax(velocity_ratio), velocity_ratio.shape)
        raise ValueError(
            f"ARDY joint target violates MetalRobo G1 velocity limit: "
            f"transition={frame}->{frame + 1} joint={_G1_JOINTS[joint]} "
            f"ratio={velocity_ratio[frame, joint]:.6f}"
        )

    quaternion_wxyz = np.asarray(qpos[:, 3:7], dtype=np.float32)
    norms = np.linalg.norm(quaternion_wxyz, axis=1, keepdims=True)
    if np.any(norms < 1.0e-6):
        raise ValueError("ARDY root quaternion cannot be normalized")
    if np.any(np.abs(norms - 1.0) > 5.0e-2):
        raise ValueError("ARDY root quaternion norm is outside tolerance")
    quaternion_wxyz = quaternion_wxyz / norms
    for frame in range(1, quaternion_wxyz.shape[0]):
        if float(np.dot(quaternion_wxyz[frame - 1], quaternion_wxyz[frame])) < 0.0:
            quaternion_wxyz[frame] *= -1.0
    quaternion_xyzw = quaternion_wxyz[:, (1, 2, 3, 0)]
    root = np.concatenate((qpos[:, :3], quaternion_xyzw), axis=1)
    return root.astype(np.float32), joints, peak_velocity_ratio


def convert(arguments: argparse.Namespace) -> None:
    qpos, contacts, fps, source_outcome = _load_inputs(
        arguments.qpos_csv,
        arguments.motion_npz,
    )
    root, joints, peak_velocity_ratio = _metalrobo_targets(
        qpos,
        fps,
        arguments.joint_limit_margin,
    )
    frame_count = qpos.shape[0]
    desired_outcome = (arguments.desired_outcome or source_outcome).strip()
    contact_by_track = np.stack(
        (
            np.any(contacts[:, :2], axis=1),
            np.any(contacts[:, 2:], axis=1),
        ),
        axis=1,
    )
    modes = np.where(
        contact_by_track,
        _CONTACT_MODE_STICK,
        _CONTACT_MODE_FREE,
    ).astype("<u4")
    contact_sample_count = frame_count * 2
    masks = np.zeros(contact_sample_count, dtype="<u4")
    flags = np.full(contact_sample_count, _SAMPLE_PREDICTED, dtype="<u4")
    confidence = np.full(
        contact_sample_count,
        arguments.contact_confidence,
        dtype="<f4",
    )
    contact_values = np.zeros(
        contact_sample_count * _CONTACT_FEATURE_COUNT,
        dtype="<f4",
    )

    tracks = (
        ("left_foot", arguments.left_contact_group, arguments.counterpart),
        ("right_foot", arguments.right_contact_group, arguments.counterpart),
    )
    payload = b"".join(
        (
            _string(arguments.id),
            _string(arguments.source_repository),
            _string(arguments.source_revision),
            _string(arguments.license),
            _string("metalrobo_z_up_x_forward_xyzw"),
            _strings(_G1_JOINTS),
            struct.pack("<Q", len(tracks)),
            b"".join(
                _string(track_id) + _string(group) + _string(counterpart)
                for track_id, group, counterpart in tracks
            ),
            struct.pack("<Q", 1),
            _string(arguments.clip_id),
            _string(desired_outcome),
            struct.pack("<fII", fps, frame_count, int(arguments.loop)),
            _vector(root, "<f4"),
            _vector(joints, "<f4"),
            _vector(modes, "<u4"),
            _vector(masks, "<u4"),
            _vector(flags, "<u4"),
            _vector(confidence, "<f4"),
            _vector(contact_values, "<f4"),
            _vector(contact_values, "<f4"),
        )
    )
    header = _HEADER.pack(
        b"MRLEARN\0",
        _INTERACTION_VERSION,
        _INTERACTION_KIND,
        len(payload),
        _content_hash(payload),
    )
    target = arguments.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(header)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(
        f"wrote {target} frames={frame_count} fps={fps:.6f} "
        f"left_contact_frames={int(np.count_nonzero(contact_by_track[:, 0]))} "
        f"right_contact_frames={int(np.count_nonzero(contact_by_track[:, 1]))} "
        f"peak_joint_velocity_ratio={peak_velocity_ratio:.6f} "
        f"content_hash={_content_hash(payload)} bytes={target.stat().st_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-npz", type=Path, required=True)
    parser.add_argument("--qpos-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--clip-id", default="ardy-g1")
    parser.add_argument("--desired-outcome")
    parser.add_argument(
        "--source-repository",
        default="https://github.com/nv-tlabs/ardy",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license", default="Apache-2.0")
    parser.add_argument("--left-contact-group", default="left_foot_contact")
    parser.add_argument("--right-contact-group", default="right_foot_contact")
    parser.add_argument("--counterpart", default="locomotion_ground")
    parser.add_argument("--contact-confidence", type=float, default=0.5)
    parser.add_argument("--joint-limit-margin", type=float, default=1.0e-4)
    parser.add_argument("--loop", action="store_true")
    arguments = parser.parse_args()
    if not 0.0 <= arguments.contact_confidence <= 1.0:
        parser.error("--contact-confidence must be in [0, 1]")
    if arguments.joint_limit_margin < 0.0:
        parser.error("--joint-limit-margin must be nonnegative")
    convert(arguments)


if __name__ == "__main__":
    main()
