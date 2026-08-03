"""Retarget a cskel27 motion proposal onto authored Unitree G1 mechanics."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .ardy_interaction_convert import (
    _G1_JOINT_LOWER,
    _G1_JOINT_UPPER,
    _G1_JOINT_VELOCITY,
    _G1_JOINTS,
)


G1_RETARGET_FORMAT = "numi.motion-retarget.v1"
G1_SOURCE_REVISION = "aa0f5c68b5aba347bad409e71b6430407da758d7"
G1_RESET_Q = np.asarray(
    (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    ),
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _vector(text: str | None, default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    result = np.fromstring(text, dtype=np.float64, sep=" ")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"invalid URDF vector: {text}")
    return result


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class G1Kinematics:
    links: tuple[str, ...]
    joints: tuple[Joint, ...]
    joint_indices: Mapping[str, int]
    source_sha256: str

    @classmethod
    def from_urdf(cls, path: Path) -> "G1Kinematics":
        root = ET.parse(path).getroot()
        links = tuple(link.attrib["name"] for link in root.findall("link"))
        joints: list[Joint] = []
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            joints.append(
                Joint(
                    name=element.attrib["name"],
                    kind=element.attrib["type"],
                    parent=element.find("parent").attrib["link"],
                    child=element.find("child").attrib["link"],
                    origin=_transform(
                        _vector(
                            None if origin is None else origin.attrib.get("xyz"),
                            (0.0, 0.0, 0.0),
                        ),
                        _vector(
                            None if origin is None else origin.attrib.get("rpy"),
                            (0.0, 0.0, 0.0),
                        ),
                    ),
                    axis=_vector(
                        None if axis is None else axis.attrib.get("xyz"),
                        (1.0, 0.0, 0.0),
                    ),
                )
            )
        indices = {name: index for index, name in enumerate(_G1_JOINTS)}
        movable = tuple(
            joint.name for joint in joints if joint.kind in {"revolute", "continuous"}
        )
        if set(movable) != set(_G1_JOINTS) or len(movable) != len(_G1_JOINTS):
            raise ValueError("URDF does not contain the canonical 29-DoF G1")
        if "pelvis" not in links:
            raise ValueError("G1 URDF has no pelvis root")
        return cls(links, tuple(joints), indices, _sha256(path))

    def forward(self, q: np.ndarray) -> dict[str, np.ndarray]:
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ValueError("G1 q must be finite with shape [29]")
        poses: dict[str, np.ndarray] = {"pelvis": np.eye(4, dtype=np.float64)}
        pending = list(self.joints)
        while pending:
            remaining: list[Joint] = []
            advanced = False
            for joint in pending:
                parent = poses.get(joint.parent)
                if parent is None:
                    remaining.append(joint)
                    continue
                pose = parent @ joint.origin
                if joint.kind in {"revolute", "continuous"}:
                    angle = q[self.joint_indices[joint.name]]
                    motion = np.eye(4, dtype=np.float64)
                    motion[:3, :3] = Rotation.from_rotvec(
                        joint.axis * angle
                    ).as_matrix()
                    pose = pose @ motion
                poses[joint.child] = pose
                advanced = True
            if not advanced:
                raise ValueError("G1 URDF joint graph is disconnected")
            pending = remaining
        if set(poses) != set(self.links):
            raise ValueError("G1 URDF forward kinematics missed links")
        return poses


def _cont6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = value[..., :3]
    second = value[..., 3:]
    x = first / np.linalg.norm(first, axis=-1, keepdims=True)
    z = np.cross(x, second)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    result = np.stack((x, y, z), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("ARDY continuous-6D rotation is degenerate")
    return result


def _source_joints(
    archive: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    root = np.asarray(archive["root_positions"], dtype=np.float64)
    local = np.asarray(archive["local_joint_positions"], dtype=np.float64)
    joints = np.concatenate((root[:, None, :], local), axis=1)
    joints[:, 1:, 0] += root[:, None, 0]
    joints[:, 1:, 2] += root[:, None, 2]
    rotations = _cont6d_to_matrix(
        np.asarray(archive["global_rotations_6d"], dtype=np.float64)
    )
    if (
        joints.shape != (40, 27, 3)
        or rotations.shape != (40, 27, 3, 3)
        or not np.isfinite(joints).all()
    ):
        raise ValueError("source must contain one finite 40-frame cskel27 motion")
    return joints, rotations


def _to_g1_direction(
    vector: np.ndarray,
    source_root_rotation: np.ndarray,
) -> np.ndarray:
    source_local = source_root_rotation.T @ vector
    mapped = np.asarray(
        (source_local[2], source_local[0], source_local[1]),
        dtype=np.float64,
    )
    norm = np.linalg.norm(mapped)
    if norm <= 1.0e-8:
        raise ValueError("ARDY limb segment collapsed during retargeting")
    return mapped / norm


def _chain_targets(
    source: np.ndarray,
    shoulder: np.ndarray,
    lengths: tuple[float, float, float],
    indices: tuple[int, int, int, int],
    source_root_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second, third, fourth = indices
    elbow = shoulder + lengths[0] * _to_g1_direction(
        source[second] - source[first], source_root_rotation
    )
    wrist = elbow + lengths[1] * _to_g1_direction(
        source[third] - source[second], source_root_rotation
    )
    hand = wrist + lengths[2] * _to_g1_direction(
        source[fourth] - source[third], source_root_rotation
    )
    return elbow, wrist, hand


def _point(poses: Mapping[str, np.ndarray], link: str) -> np.ndarray:
    if link not in poses:
        raise ValueError(f"G1 URDF is missing retarget link {link}")
    return poses[link][:3, 3]


def _segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Shortest distance between two finite 3D line segments."""
    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    aa = float(first @ first)
    bb = float(first @ second)
    cc = float(second @ second)
    dd = float(first @ offset)
    ee = float(second @ offset)
    denominator = aa * cc - bb * bb
    if aa <= 1.0e-12 or cc <= 1.0e-12:
        return float(min(
            np.linalg.norm(first_start - second_start),
            np.linalg.norm(first_end - second_end),
        ))
    first_t = np.clip((bb * ee - cc * dd) / denominator, 0.0, 1.0) \
        if denominator > 1.0e-12 else 0.0
    second_t = np.clip((aa * ee - bb * dd) / denominator, 0.0, 1.0) \
        if denominator > 1.0e-12 else np.clip(ee / cc, 0.0, 1.0)
    first_t = np.clip((bb * second_t - dd) / aa, 0.0, 1.0)
    second_t = np.clip((bb * first_t + ee) / cc, 0.0, 1.0)
    return float(np.linalg.norm(
        first_start + first_t * first - second_start - second_t * second
    ))


def _arm_clearances(
    poses: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    torso_start = np.asarray((0.0, 0.0, 0.08), dtype=np.float64)
    torso_end = np.asarray((0.0, 0.0, 0.48), dtype=np.float64)
    forearms: list[tuple[np.ndarray, np.ndarray]] = []
    torso_clearances: list[float] = []
    for side in ("left", "right"):
        elbow = _point(poses, f"{side}_elbow_link")
        wrist = _point(poses, f"{side}_wrist_roll_link")
        hand = _point(poses, f"{side}_rubber_hand")
        torso_clearances.extend((
            _segment_distance(elbow, wrist, torso_start, torso_end),
            _segment_distance(wrist, hand, torso_start, torso_end),
        ))
        forearms.append((elbow, hand))
    separation = _segment_distance(*forearms[0], *forearms[1])
    return np.asarray(torso_clearances, dtype=np.float64), separation


def _leg_clearances(
    poses: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    left_knee = _point(poses, "left_knee_link")
    left_ankle = _point(poses, "left_ankle_roll_link")
    right_knee = _point(poses, "right_knee_link")
    right_ankle = _point(poses, "right_ankle_roll_link")
    lateral_margins = np.asarray(
        (left_knee[1], left_ankle[1], -right_knee[1], -right_ankle[1]),
        dtype=np.float64,
    )
    shank_separation = _segment_distance(
        left_knee, left_ankle, right_knee, right_ankle
    )
    return lateral_margins, shank_separation


def retarget_g1(
    proposal_directory: Path,
    urdf_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    evidence_path = proposal_directory / "evidence.json"
    source_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if source_evidence.get("format") != "numi.motion-proposal.v1":
        raise ValueError("source is not a Numi motion proposal")
    with np.load(proposal_directory / "motion_proposal.npz") as archive:
        source, source_rotations = _source_joints(archive)
        root_positions = np.asarray(archive["root_positions"], dtype=np.float64)

    model = G1Kinematics.from_urdf(urdf_path)
    rest = model.forward(G1_RESET_Q)
    arms = (
        {
            "side": "right",
            "source": (8, 9, 10, 11),
            "links": (
                "right_shoulder_pitch_link",
                "right_elbow_link",
                "right_wrist_roll_link",
                "right_rubber_hand",
            ),
            "active": tuple(range(22, 29)),
        },
        {
            "side": "left",
            "source": (14, 15, 16, 17),
            "links": (
                "left_shoulder_pitch_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_rubber_hand",
            ),
            "active": tuple(range(15, 22)),
        },
    )
    legs = (
        {
            "side": "right",
            "source": (19, 20, 21),
            "links": (
                "right_hip_pitch_link",
                "right_knee_link",
                "right_ankle_pitch_link",
            ),
        },
        {
            "side": "left",
            "source": (23, 24, 25),
            "links": (
                "left_hip_pitch_link",
                "left_knee_link",
                "left_ankle_pitch_link",
            ),
        },
    )
    chain_lengths: dict[str, tuple[float, float, float]] = {}
    for arm in arms:
        links = arm["links"]
        points = tuple(_point(rest, link) for link in links)
        chain_lengths[str(arm["side"])] = tuple(
            float(np.linalg.norm(points[index + 1] - points[index]))
            for index in range(3)
        )

    leg_lengths: dict[str, tuple[float, float]] = {}
    for leg in legs:
        points = tuple(_point(rest, link) for link in leg["links"])
        leg_lengths[str(leg["side"])] = tuple(
            float(np.linalg.norm(points[index + 1] - points[index]))
            for index in range(2)
        )
    head_length = float(np.linalg.norm(_point(rest, "head_link")))

    q_frames = np.empty((source.shape[0], 29), dtype=np.float64)
    endpoint_errors = np.empty((source.shape[0], 11), dtype=np.float64)
    previous = G1_RESET_Q.copy()
    active = np.arange(29, dtype=np.int64)
    position_lower = _G1_JOINT_LOWER.astype(np.float64)[active]
    position_upper = _G1_JOINT_UPPER.astype(np.float64)[active]
    maximum_frame_delta = (
        _G1_JOINT_VELOCITY.astype(np.float64)[active]
        / float(source_evidence["fps"])
    )

    for frame_index, source_frame in enumerate(source):
        source_root_rotation = source_rotations[frame_index, 0]
        base_pose = model.forward(previous)
        targets: list[tuple[str, np.ndarray]] = []
        for arm in arms:
            links = arm["links"]
            shoulder = _point(base_pose, links[0])
            chain = _chain_targets(
                source_frame,
                shoulder,
                chain_lengths[str(arm["side"])],
                arm["source"],
                source_root_rotation,
            )
            targets.extend(zip(links[1:], chain, strict=True))
        for leg in legs:
            links = leg["links"]
            source_indices = leg["source"]
            hip = _point(base_pose, links[0])
            knee = hip + leg_lengths[str(leg["side"])][0] * _to_g1_direction(
                source_frame[source_indices[1]] - source_frame[source_indices[0]],
                source_root_rotation,
            )
            ankle = knee + leg_lengths[str(leg["side"])][1] * _to_g1_direction(
                source_frame[source_indices[2]] - source_frame[source_indices[1]],
                source_root_rotation,
            )
            targets.extend(((links[1], knee), (links[2], ankle)))
        head_target = head_length * _to_g1_direction(
            source_frame[6] - source_frame[0],
            source_root_rotation,
        )
        targets.append(("head_link", head_target))

        def residual(active_q: np.ndarray) -> np.ndarray:
            q = previous.copy()
            q[active] = active_q
            poses = model.forward(q)
            positions = np.concatenate(
                [(_point(poses, link) - target) / 0.01 for link, target in targets]
            )
            continuity = (active_q - previous[active]) / 0.28
            posture = (active_q - G1_RESET_Q[active]) / 3.5
            torso_clearances, forearm_separation = _arm_clearances(poses)
            leg_margins, shank_separation = _leg_clearances(poses)
            collision = np.concatenate((
                np.maximum(0.0, 0.17 - torso_clearances) / 0.003,
                np.asarray((
                    max(0.0, 0.09 - forearm_separation) / 0.003,
                    *(
                        np.maximum(0.0, 0.025 - leg_margins) / 0.001
                    ),
                    max(0.0, 0.075 - shank_separation) / 0.001,
                )),
            ))
            return np.concatenate((positions, continuity, posture, collision))

        solved = least_squares(
            residual,
            previous[active],
            bounds=(
                np.maximum(position_lower, previous[active] - maximum_frame_delta),
                np.minimum(position_upper, previous[active] + maximum_frame_delta),
            ),
            method="trf",
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
            max_nfev=100,
        )
        q = previous.copy()
        q[active] = solved.x
        poses = model.forward(q)
        endpoint_errors[frame_index] = np.asarray(
            [np.linalg.norm(_point(poses, link) - target) for link, target in targets]
        )
        q_frames[frame_index] = q
        previous = q

    fps = float(source_evidence["fps"])
    if np.any(q_frames < _G1_JOINT_LOWER) or np.any(q_frames > _G1_JOINT_UPPER):
        raise RuntimeError("retargeting produced an out-of-limit G1 joint")

    source_origin = root_positions[0]
    root_world = np.empty((source.shape[0], 7), dtype=np.float64)
    root_world[:, 0] = root_positions[:, 2] - source_origin[2]
    root_world[:, 1] = root_positions[:, 0] - source_origin[0]
    root_world[:, 2] = 0.793 + (root_positions[:, 1] - source_origin[1])
    ardy_to_g1 = np.asarray(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    root_rotations = (
        ardy_to_g1[None, :, :]
        @ source_rotations[:, 0]
        @ ardy_to_g1.T[None, :, :]
    )
    root_world[:, 3:] = Rotation.from_matrix(root_rotations).as_quat()

    terminal_frame_count = 30
    touchdown_frame_count = 28
    foot_orientation_frame_count = 10
    terminal_phase = np.linspace(
        1.0 / terminal_frame_count,
        1.0,
        terminal_frame_count,
        dtype=np.float64,
    )
    terminal_blend = terminal_phase * terminal_phase * (3.0 - 2.0 * terminal_phase)
    touchdown_phase = np.minimum(
        np.arange(1, terminal_frame_count + 1, dtype=np.float64)
        / touchdown_frame_count,
        1.0,
    )
    touchdown_blend = (
        touchdown_phase * touchdown_phase * (3.0 - 2.0 * touchdown_phase)
    )
    foot_orientation_phase = np.minimum(
        np.arange(1, terminal_frame_count + 1, dtype=np.float64)
        / foot_orientation_frame_count,
        1.0,
    )
    foot_orientation_blend = (
        foot_orientation_phase
        * foot_orientation_phase
        * (3.0 - 2.0 * foot_orientation_phase)
    )
    terminal_root = np.repeat(root_world[-1][None, :], terminal_frame_count, axis=0)
    terminal_root[:, 2] = (
        root_world[-1, 2] * (1.0 - touchdown_blend) + 0.793 * touchdown_blend
    )
    terminal_root[:, 3:] = Slerp(
        (0.0, 1.0),
        Rotation.from_quat((root_world[-1, 3:], (0.0, 0.0, 0.0, 1.0))),
    )(touchdown_blend).as_quat()

    foot_links = ("left_ankle_roll_link", "right_ankle_roll_link")
    source_terminal_poses = model.forward(q_frames[-1])
    source_terminal_rotation = Rotation.from_quat(root_world[-1, 3:]).as_matrix()
    touchdown_root_position = terminal_root[-1, :3]
    touchdown_targets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for foot_link in foot_links:
        source_local = source_terminal_poses[foot_link]
        source_position = (
            root_world[-1, :3]
            + source_terminal_rotation @ source_local[:3, 3]
        )
        source_orientation = source_terminal_rotation @ source_local[:3, :3]
        final_local = rest[foot_link]
        final_position = touchdown_root_position + final_local[:3, 3]
        touchdown_targets.append((
            source_position,
            final_position,
            source_orientation,
            final_local[:3, :3],
        ))

    terminal_q = np.empty((terminal_frame_count, 29), dtype=np.float64)
    terminal_foot_errors = np.empty((terminal_frame_count, 2), dtype=np.float64)
    terminal_active = np.arange(15, dtype=np.int64)
    terminal_previous = q_frames[-1].copy()
    for terminal_index in range(terminal_frame_count):
        posture_blend = terminal_blend[terminal_index]
        contact_blend = touchdown_blend[terminal_index]
        reference = (
            q_frames[-1] * (1.0 - posture_blend)
            + G1_RESET_Q * posture_blend
        )
        reference[terminal_active] = (
            q_frames[-1, terminal_active] * (1.0 - contact_blend)
            + G1_RESET_Q[terminal_active] * contact_blend
        )
        ankle_indices = np.asarray((4, 5, 10, 11), dtype=np.int64)
        orientation_blend = foot_orientation_blend[terminal_index]
        reference[ankle_indices] = (
            q_frames[-1, ankle_indices] * (1.0 - orientation_blend)
            + G1_RESET_Q[ankle_indices] * orientation_blend
        )
        root_rotation = Rotation.from_quat(
            terminal_root[terminal_index, 3:]
        ).as_matrix()
        foot_targets: list[tuple[str, np.ndarray, np.ndarray]] = []
        for foot_link, target in zip(foot_links, touchdown_targets, strict=True):
            source_position, final_position, source_orientation, final_orientation = target
            target_position = (
                source_position * (1.0 - contact_blend)
                + final_position * contact_blend
            )
            target_orientation = Slerp(
                (0.0, 1.0),
                Rotation.from_matrix((source_orientation, final_orientation)),
            )((orientation_blend,)).as_matrix()[0]
            foot_targets.append((foot_link, target_position, target_orientation))

        def terminal_residual(active_q: np.ndarray) -> np.ndarray:
            q = reference.copy()
            q[terminal_active] = active_q
            poses = model.forward(q)
            foot_terms: list[np.ndarray] = []
            for foot_link, target_position, target_orientation in foot_targets:
                local = poses[foot_link]
                world_position = (
                    terminal_root[terminal_index, :3]
                    + root_rotation @ local[:3, 3]
                )
                world_orientation = root_rotation @ local[:3, :3]
                foot_terms.extend((
                    (world_position - target_position) / 0.0005,
                    Rotation.from_matrix(
                        target_orientation.T @ world_orientation
                    ).as_rotvec() / 0.03,
                ))
            continuity = (
                active_q - terminal_previous[terminal_active]
            ) / 0.24
            posture = (
                active_q - reference[terminal_active]
            ) / 0.16
            return np.concatenate((*foot_terms, continuity, posture))

        solved = least_squares(
            terminal_residual,
            terminal_previous[terminal_active],
            bounds=(
                np.maximum(
                    _G1_JOINT_LOWER[terminal_active],
                    terminal_previous[terminal_active]
                    - maximum_frame_delta[terminal_active],
                ),
                np.minimum(
                    _G1_JOINT_UPPER[terminal_active],
                    terminal_previous[terminal_active]
                    + maximum_frame_delta[terminal_active],
                ),
            ),
            method="trf",
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=160,
        )
        q = reference.copy()
        q[terminal_active] = solved.x
        if terminal_index >= touchdown_frame_count - 1:
            q = G1_RESET_Q.copy()
        poses = model.forward(q)
        for foot_index, (foot_link, target_position, _) in enumerate(foot_targets):
            local = poses[foot_link]
            world_position = (
                terminal_root[terminal_index, :3]
                + root_rotation @ local[:3, 3]
            )
            terminal_foot_errors[terminal_index, foot_index] = np.linalg.norm(
                world_position - target_position
            )
        terminal_q[terminal_index] = q
        terminal_previous = q
    q_frames = np.concatenate((q_frames, terminal_q), axis=0)
    root_world = np.concatenate((root_world, terminal_root), axis=0)
    velocity_ratio = (
        np.abs(np.diff(q_frames, axis=0)) * fps
        / _G1_JOINT_VELOCITY.astype(np.float64)[None, :]
    )
    if np.max(velocity_ratio, initial=0.0) > 1.0 + 1.0e-6:
        raise RuntimeError("terminal completion exceeds an authored G1 velocity limit")

    link_transforms = np.empty((q_frames.shape[0], len(model.links), 7), dtype=np.float64)
    clearance_frames = np.empty((q_frames.shape[0], 10), dtype=np.float64)
    for frame_index, q in enumerate(q_frames):
        root_rotation = Rotation.from_quat(root_world[frame_index, 3:]).as_matrix()
        local_poses = model.forward(q)
        torso_clearances, forearm_separation = _arm_clearances(local_poses)
        clearance_frames[frame_index, :4] = torso_clearances
        clearance_frames[frame_index, 4] = forearm_separation
        leg_margins, shank_separation = _leg_clearances(local_poses)
        clearance_frames[frame_index, 5:9] = leg_margins
        clearance_frames[frame_index, 9] = shank_separation
        for link_index, link in enumerate(model.links):
            local = local_poses[link]
            link_transforms[frame_index, link_index, :3] = (
                root_world[frame_index, :3] + root_rotation @ local[:3, 3]
            )
            link_transforms[frame_index, link_index, 3:] = Rotation.from_matrix(
                root_rotation @ local[:3, :3]
            ).as_quat()

    arrays = {
        "root_position_quaternion_xyzw": root_world.astype(np.float32),
        "joint_positions": q_frames.astype(np.float32),
        "endpoint_errors_m": endpoint_errors.astype(np.float32),
        "link_names": np.asarray(model.links, dtype=np.str_),
        "link_position_quaternion_xyzw": link_transforms.astype(np.float32),
    }
    result_evidence = {
        "format": G1_RETARGET_FORMAT,
        "status": "kinematic-retarget",
        "robot": "unitree_g1_29dof_rev_1_0",
        "robot_source_revision": G1_SOURCE_REVISION,
        "robot_urdf": {
            "path": str(urdf_path),
            "sha256": model.source_sha256,
        },
        "source_motion": {
            "path": str(proposal_directory),
            "evidence_sha256": _sha256(evidence_path),
            "arrays_fingerprint": source_evidence["motion_proposal"][
                "arrays_fingerprint"
            ],
            "prompt": source_evidence["prompt"],
            "model_repository": source_evidence["model_repository"],
            "model_revision": source_evidence["model_revision"],
            "selected_provider": source_evidence["selected_provider"],
        },
        "fps": int(fps),
        "frame_count": int(q_frames.shape[0]),
        "joint_order": list(_G1_JOINTS),
        "retarget": {
            "method": "bounded sequential full-body endpoint IK",
            "mean_endpoint_error_m": float(np.mean(endpoint_errors)),
            "maximum_endpoint_error_m": float(np.max(endpoint_errors)),
            "maximum_joint_velocity_ratio": float(
                np.max(velocity_ratio, initial=0.0)
            ),
            "joint_limits_satisfied": True,
            "source_human_proportions_copied": False,
            "minimum_arm_torso_clearance_m": float(
                np.min(clearance_frames[:, :4])
            ),
            "minimum_forearm_separation_m": float(
                np.min(clearance_frames[:, 4])
            ),
            "self_collision_objective": (
                "torso/forearm clearance plus sided knees, ankles, and shanks"
            ),
            "minimum_leg_lateral_margin_m": float(
                np.min(clearance_frames[:, 5:9])
            ),
            "minimum_shank_separation_m": float(
                np.min(clearance_frames[:, 9])
            ),
            "source_frame_count": int(source.shape[0]),
            "terminal_completion": {
                "frame_count": terminal_frame_count,
                "method": (
                    "foot-locked touchdown IK then bounded posture completion"
                ),
                "touchdown_frame_count": touchdown_frame_count,
                "foot_orientation_frame_count": foot_orientation_frame_count,
                "maximum_post_touchdown_foot_position_error_m": float(
                    np.max(terminal_foot_errors[touchdown_frame_count - 1 :])
                ),
                "target_root_height_m": 0.793,
                "target_root_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "authority": (
            "kinematic preview only; NumiSolver contact, balance, and actuation "
            "have not been applied"
        ),
    }
    return arrays, result_evidence


def write_retarget(
    output_directory: Path,
    arrays: Mapping[str, np.ndarray],
    evidence: Mapping[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "g1_motion_retarget.npz"
    np.savez_compressed(path, **arrays)
    document = dict(evidence)
    document["retarget_artifact"] = {
        "path": path.name,
        "sha256": _sha256(path),
        "arrays_fingerprint": _array_fingerprint(arrays),
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    (output_directory / "evidence.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
