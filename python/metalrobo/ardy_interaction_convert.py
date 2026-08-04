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


def foot_contact_contract(
    foot_contacts: np.ndarray,
    foot_contact_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve ARDY's four contact probabilities as two foot intents."""

    contacts = np.asarray(foot_contacts, dtype=np.uint8)
    scores = np.asarray(foot_contact_scores, dtype=np.float32)
    if (
        contacts.ndim != 2
        or contacts.shape[1] != 4
        or scores.shape != contacts.shape
        or not np.isfinite(scores).all()
        or not np.array_equal(contacts != 0, scores > 0.5)
    ):
        raise ValueError(
            "ARDY foot contacts and scores must be finite, aligned [frames, 4] arrays"
        )
    contact_by_foot = np.stack(
        (np.any(contacts[:, :2], axis=1), np.any(contacts[:, 2:], axis=1)),
        axis=1,
    )
    modes = np.where(
        contact_by_foot, _CONTACT_MODE_STICK, _CONTACT_MODE_FREE
    ).astype(np.uint32)
    probabilities = np.stack(
        (np.max(scores[:, :2], axis=1), np.max(scores[:, 2:], axis=1)),
        axis=1,
    )
    confidence = np.clip(
        2.0 * np.abs(probabilities - 0.5), 0.0, 1.0
    ).astype(np.float32)
    return modes, confidence
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


def write_interaction_pack(
    *,
    output: Path,
    pack_id: str,
    clip_id: str,
    desired_outcome: str,
    source_repository: str,
    source_revision: str,
    license_name: str,
    frames_per_second: float,
    root_targets: np.ndarray,
    joint_targets: np.ndarray,
    tracks: tuple[tuple[str, str, str], ...],
    contact_modes: np.ndarray,
    contact_confidence: np.ndarray,
    contact_feature_masks: np.ndarray | None = None,
    contact_sample_flags: np.ndarray | None = None,
    contact_targets: np.ndarray | None = None,
    contact_tolerances: np.ndarray | None = None,
    loop: bool = False,
    joint_names: tuple[str, ...] = _G1_JOINTS,
    joint_lower: np.ndarray = _G1_JOINT_LOWER,
    joint_upper: np.ndarray = _G1_JOINT_UPPER,
    joint_velocity: np.ndarray = _G1_JOINT_VELOCITY,
) -> tuple[Path, int]:
    """Write one generated-intent clip through the canonical pack ABI."""
    root = np.asarray(root_targets, dtype=np.float32)
    joints = np.asarray(joint_targets, dtype=np.float32)
    modes = np.asarray(contact_modes, dtype=np.uint32)
    confidence = np.asarray(contact_confidence, dtype=np.float32)
    names = tuple(str(name) for name in joint_names)
    lower = np.asarray(joint_lower, dtype=np.float32)
    upper = np.asarray(joint_upper, dtype=np.float32)
    velocity = np.asarray(joint_velocity, dtype=np.float32)
    if (
        not names
        or len(set(names)) != len(names)
        or lower.shape != (len(names),)
        or upper.shape != (len(names),)
        or velocity.shape != (len(names),)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.isfinite(velocity).all()
        or np.any(lower > upper)
        or np.any(velocity <= 0.0)
    ):
        raise ValueError("joint semantics and mechanism limits are invalid")
    if not np.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError("frames per second must be finite and positive")
    if root.ndim != 2 or root.shape[1] != 7:
        raise ValueError("root targets must have shape [frames, 7]")
    frame_count = root.shape[0]
    if frame_count < 2 or not np.isfinite(root).all():
        raise ValueError("root targets must contain at least two finite frames")
    if joints.shape != (frame_count, len(names)):
        raise ValueError("joint targets must cover every authored joint")
    if not np.isfinite(joints).all():
        raise ValueError("joint targets must be finite")
    if np.any(joints < lower) or np.any(joints > upper):
        frame, joint = np.argwhere(
            (joints < lower) | (joints > upper)
        )[0]
        raise ValueError(
            f"joint target is outside the mechanism limit: frame={frame} "
            f"joint={names[joint]} value={joints[frame, joint]:.6f}"
        )
    joint_velocity_ratio = (
        np.abs(np.diff(joints, axis=0)) * frames_per_second
        / velocity[None, :]
    )
    if np.any(joint_velocity_ratio > 1.0 + 1.0e-5):
        frame, joint = np.unravel_index(
            np.argmax(joint_velocity_ratio),
            joint_velocity_ratio.shape,
        )
        raise ValueError(
            f"joint target exceeds the mechanism velocity limit: "
            f"transition={frame}->{frame + 1} "
            f"joint={names[joint]} "
            f"ratio={joint_velocity_ratio[frame, joint]:.6f}"
        )
    quaternions = root[:, 3:]
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if np.any(np.abs(quaternion_norms - 1.0) > 1.0e-3):
        raise ValueError("root target quaternions must be normalized")
    if modes.shape != (frame_count, len(tracks)):
        raise ValueError("contact modes must have shape [frames, tracks]")
    if len({track[0] for track in tracks}) != len(tracks):
        raise ValueError("contact tracks must be uniquely named")
    if any(not all(field.strip() for field in track) for track in tracks):
        raise ValueError("contact track identities must be nonempty")
    if np.any(modes > 5):
        raise ValueError("contact modes must use the canonical 0 through 5 enum")
    if confidence.shape != modes.shape or np.any(~np.isfinite(confidence)):
        raise ValueError(
            "contact confidence must be finite and match contact modes"
        )
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("contact confidence must lie in [0, 1]")
    sample_count = frame_count * len(tracks)
    masks = (
        np.zeros(sample_count, dtype=np.uint32)
        if contact_feature_masks is None
        else np.asarray(contact_feature_masks, dtype=np.uint32).reshape(-1)
    )
    flags = (
        np.full(sample_count, _SAMPLE_PREDICTED, dtype=np.uint32)
        if contact_sample_flags is None
        else np.asarray(contact_sample_flags, dtype=np.uint32).reshape(-1)
    )
    values = (
        np.zeros(
            sample_count * _CONTACT_FEATURE_COUNT,
            dtype=np.float32,
        )
        if contact_targets is None
        else np.asarray(contact_targets, dtype=np.float32).reshape(-1)
    )
    tolerances = (
        np.zeros_like(values)
        if contact_tolerances is None
        else np.asarray(contact_tolerances, dtype=np.float32).reshape(-1)
    )
    if masks.size != sample_count or flags.size != sample_count:
        raise ValueError("contact metadata must cover every frame and track")
    unknown_feature_bits = np.uint32(
        ~((1 << _CONTACT_FEATURE_COUNT) - 1) & 0xFFFFFFFF
    )
    if np.any(masks & unknown_feature_bits):
        raise ValueError("contact feature mask contains unknown bits")
    if np.any(flags & np.uint32(0xFFFFFFFC)):
        raise ValueError("contact sample flags contain unknown bits")
    if (
        values.size != sample_count * _CONTACT_FEATURE_COUNT
        or tolerances.size != values.size
    ):
        raise ValueError("contact fields must cover all 13 values per sample")
    if not np.isfinite(values).all() or not np.isfinite(tolerances).all():
        raise ValueError("contact targets and tolerances must be finite")
    if np.any(tolerances < 0.0):
        raise ValueError("contact tolerances must be nonnegative")

    payload = b"".join(
        (
            _string(pack_id),
            _string(source_repository),
            _string(source_revision),
            _string(license_name),
            _string("metalrobo_z_up_x_forward_xyzw"),
            _strings(names),
            struct.pack("<Q", len(tracks)),
            b"".join(
                _string(track_id) + _string(group) + _string(counterpart)
                for track_id, group, counterpart in tracks
            ),
            struct.pack("<Q", 1),
            _string(clip_id),
            _string(desired_outcome),
            struct.pack(
                "<fII",
                frames_per_second,
                frame_count,
                int(loop),
            ),
            _vector(root, "<f4"),
            _vector(joints, "<f4"),
            _vector(modes, "<u4"),
            _vector(masks, "<u4"),
            _vector(flags, "<u4"),
            _vector(confidence, "<f4"),
            _vector(values, "<f4"),
            _vector(tolerances, "<f4"),
        )
    )
    content_hash = _content_hash(payload)
    header = _HEADER.pack(
        b"MRLEARN\0",
        _INTERACTION_VERSION,
        _INTERACTION_KIND,
        len(payload),
        content_hash,
    )
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(header)
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return target, content_hash


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


def write_retarget_interaction_pack(
    *,
    output: Path,
    root_targets: np.ndarray,
    joint_targets: np.ndarray,
    frames_per_second: float,
    desired_outcome: str,
    source_repository: str,
    source_revision: str,
    pack_id: str,
    foot_contacts: np.ndarray | None = None,
    foot_contact_scores: np.ndarray | None = None,
    clip_id: str = "ardy-g1",
    counterpart: str = "locomotion_ground",
) -> tuple[Path, int]:
    """Compile ARDY-retargeted G1 intent without inventing contact truth."""
    joints = np.asarray(joint_targets, dtype=np.float32)
    if joints.ndim != 2:
        raise ValueError("retargeted joint intent must be frame-major")
    frame_count = joints.shape[0]
    if (foot_contacts is None) != (foot_contact_scores is None):
        raise ValueError(
            "retargeted contact intent requires both contacts and scores"
        )
    if foot_contacts is None:
        tracks: tuple[tuple[str, str, str], ...] = ()
        modes = np.empty((frame_count, 0), dtype=np.uint32)
        confidence = np.empty((frame_count, 0), dtype=np.float32)
    else:
        tracks = (
            ("left_foot", "left_foot_contact", counterpart),
            ("right_foot", "right_foot_contact", counterpart),
        )
        modes, confidence = foot_contact_contract(
            foot_contacts, foot_contact_scores
        )
    return write_interaction_pack(
        output=output,
        pack_id=pack_id,
        clip_id=clip_id,
        desired_outcome=desired_outcome,
        source_repository=source_repository,
        source_revision=source_revision,
        license_name="Apache-2.0",
        frames_per_second=frames_per_second,
        root_targets=root_targets,
        joint_targets=joints,
        tracks=tracks,
        contact_modes=modes,
        contact_confidence=confidence,
    )


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
    confidence = np.full(
        (frame_count, 2),
        0.0 if arguments.contact_confidence is None
        else arguments.contact_confidence,
        dtype=np.float32,
    )

    tracks = (
        ("left_foot", arguments.left_contact_group, arguments.counterpart),
        ("right_foot", arguments.right_contact_group, arguments.counterpart),
    )
    target, content_hash = write_interaction_pack(
        output=arguments.output,
        pack_id=arguments.id,
        clip_id=arguments.clip_id,
        desired_outcome=desired_outcome,
        source_repository=arguments.source_repository,
        source_revision=arguments.source_revision,
        license_name=arguments.license,
        frames_per_second=fps,
        root_targets=root,
        joint_targets=joints,
        tracks=tracks,
        contact_modes=modes,
        contact_confidence=confidence,
        loop=arguments.loop,
    )
    print(
        f"wrote {target} frames={frame_count} fps={fps:.6f} "
        f"left_contact_frames={int(np.count_nonzero(contact_by_track[:, 0]))} "
        f"right_contact_frames={int(np.count_nonzero(contact_by_track[:, 1]))} "
        f"peak_joint_velocity_ratio={peak_velocity_ratio:.6f} "
        f"content_hash={content_hash} bytes={target.stat().st_size}"
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
    parser.add_argument("--contact-confidence", type=float)
    parser.add_argument("--joint-limit-margin", type=float, default=1.0e-4)
    parser.add_argument("--loop", action="store_true")
    arguments = parser.parse_args()
    if (
        arguments.contact_confidence is not None
        and not 0.0 <= arguments.contact_confidence <= 1.0
    ):
        parser.error("--contact-confidence must be in [0, 1]")
    if arguments.joint_limit_margin < 0.0:
        parser.error("--joint-limit-margin must be nonnegative")
    convert(arguments)


if __name__ == "__main__":
    main()
