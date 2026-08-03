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

    foot_links = ("left_ankle_roll_link", "right_ankle_roll_link")
    source_frame_count = q_frames.shape[0]
    root_rise = float(np.max(root_world[:, 2]) - root_world[0, 2])
    root_rotations_world = Rotation.from_quat(root_world[:, 3:])
    root_turn = float(
        np.max((root_rotations_world[0].inv() * root_rotations_world).magnitude())
    )
    aerial_completion = root_rise > 0.20 and root_turn > np.deg2rad(90.0)
    phase_ids = np.zeros(source_frame_count, dtype=np.uint8)
    completion_evidence: dict[str, Any] = {"applied": False}
    if aerial_completion:
        peak_frame = int(np.argmax(root_world[:, 2]))
        splice_frame = source_frame_count - 1
        minimum_airborne_ankle_height = 0.15
        for frame_index in range(peak_frame + 1, source_frame_count):
            poses = model.forward(q_frames[frame_index])
            root_rotation = Rotation.from_quat(
                root_world[frame_index, 3:]
            ).as_matrix()
            ankle_heights = [
                float(
                    (
                        root_world[frame_index, :3]
                        + root_rotation @ poses[foot_link][:3, 3]
                    )[2]
                )
                for foot_link in foot_links
            ]
            if min(ankle_heights) < minimum_airborne_ankle_height:
                splice_frame = max(peak_frame, frame_index - 1)
                break

        retained_source_count = splice_frame + 1
        alignment_frame_count = 14
        descent_frame_count = 14
        support_frame_count = 16
        landing_q = G1_RESET_Q.copy()
        landing_q[[0, 6]] = -0.45
        landing_q[[1, 2, 5, 7, 8, 11, 12, 13, 14]] = 0.0
        landing_q[[3, 9]] = 0.90
        landing_q[[4, 10]] = -0.45

        final_root = np.asarray(
            (root_world[splice_frame, 0], root_world[splice_frame, 1], 0.793),
            dtype=np.float64,
        )
        target_foot_positions = np.asarray(
            [final_root + rest[foot_link][:3, 3] for foot_link in foot_links],
            dtype=np.float64,
        )
        landing_poses = model.forward(landing_q)
        landing_foot_positions = np.asarray(
            [landing_poses[foot_link][:3, 3] for foot_link in foot_links],
            dtype=np.float64,
        )
        contact_root = np.mean(
            target_foot_positions - landing_foot_positions,
            axis=0,
        )
        airborne_root = contact_root.copy()
        airborne_root[2] = max(
            float(root_world[splice_frame, 2]),
            float(contact_root[2] + 0.32),
        )

        alignment_phase = np.linspace(
            1.0 / alignment_frame_count,
            1.0,
            alignment_frame_count,
            dtype=np.float64,
        )
        alignment_blend = (
            alignment_phase * alignment_phase * (3.0 - 2.0 * alignment_phase)
        )
        alignment_q = (
            q_frames[splice_frame][None, :] * (1.0 - alignment_blend[:, None])
            + landing_q[None, :] * alignment_blend[:, None]
        )
        alignment_root = np.empty((alignment_frame_count, 7), dtype=np.float64)
        alignment_root[:, :3] = (
            root_world[splice_frame, :3][None, :]
            * (1.0 - alignment_blend[:, None])
            + airborne_root[None, :] * alignment_blend[:, None]
        )
        alignment_root[:, 3:] = Slerp(
            (0.0, 1.0),
            Rotation.from_quat((
                root_world[splice_frame, 3:],
                (0.0, 0.0, 0.0, 1.0),
            )),
        )(alignment_blend).as_quat()

        descent_phase = np.linspace(
            1.0 / descent_frame_count,
            1.0,
            descent_frame_count,
            dtype=np.float64,
        )
        descent_blend = (
            descent_phase * descent_phase * (3.0 - 2.0 * descent_phase)
        )
        descent_q = np.repeat(
            landing_q[None, :], descent_frame_count, axis=0
        )
        descent_root = np.empty((descent_frame_count, 7), dtype=np.float64)
        descent_root[:, :3] = (
            airborne_root[None, :] * (1.0 - descent_blend[:, None])
            + contact_root[None, :] * descent_blend[:, None]
        )
        descent_root[:, 3:] = (0.0, 0.0, 0.0, 1.0)

        support_phase = np.linspace(
            1.0 / support_frame_count,
            1.0,
            support_frame_count,
            dtype=np.float64,
        )
        support_blend = (
            support_phase * support_phase * (3.0 - 2.0 * support_phase)
        )
        support_q = (
            landing_q[None, :] * (1.0 - support_blend[:, None])
            + G1_RESET_Q[None, :] * support_blend[:, None]
        )
        support_root = np.empty((support_frame_count, 7), dtype=np.float64)
        support_errors = np.empty((support_frame_count, 2), dtype=np.float64)
        for frame_index, q in enumerate(support_q):
            poses = model.forward(q)
            local_feet = np.asarray(
                [poses[foot_link][:3, 3] for foot_link in foot_links],
                dtype=np.float64,
            )
            support_root[frame_index, :3] = np.mean(
                target_foot_positions - local_feet,
                axis=0,
            )
            support_root[frame_index, 3:] = (0.0, 0.0, 0.0, 1.0)
            support_errors[frame_index] = np.linalg.norm(
                support_root[frame_index, :3][None, :]
                + local_feet
                - target_foot_positions,
                axis=1,
            )
        maximum_support_error = float(np.max(support_errors))
        if maximum_support_error > 1.0e-9:
            raise RuntimeError(
                "aerial completion failed to preserve bilateral foot support"
            )

        q_frames = np.concatenate((
            q_frames[:retained_source_count],
            alignment_q,
            descent_q,
            support_q,
        ))
        root_world = np.concatenate((
            root_world[:retained_source_count],
            alignment_root,
            descent_root,
            support_root,
        ))
        endpoint_errors = endpoint_errors[:retained_source_count]
        phase_ids = np.concatenate((
            np.zeros(retained_source_count, dtype=np.uint8),
            np.ones(alignment_frame_count, dtype=np.uint8),
            np.full(descent_frame_count, 2, dtype=np.uint8),
            np.full(support_frame_count, 3, dtype=np.uint8),
        ))
        touchdown_frame = (
            retained_source_count
            + alignment_frame_count
            + descent_frame_count
            - 1
        )
        completion_root = root_world[retained_source_count - 1 :]
        completion_linear_speed = np.linalg.norm(
            np.diff(completion_root[:, :3], axis=0), axis=1
        ) * fps
        completion_delta_rotation = (
            Rotation.from_quat(completion_root[:-1, 3:]).inv()
            * Rotation.from_quat(completion_root[1:, 3:])
        )
        completion_angular_speed = completion_delta_rotation.magnitude() * fps
        completion_evidence = {
            "applied": True,
            "method": (
                "truncate corrupt source tail; airborne attitude alignment; "
                "flat-foot descent; exact bilateral support settle"
            ),
            "source_frame_count": source_frame_count,
            "retained_source_frame_count": retained_source_count,
            "rejected_source_tail_frame_count": (
                source_frame_count - retained_source_count
            ),
            "splice_frame": splice_frame,
            "splice_minimum_ankle_height_m": minimum_airborne_ankle_height,
            "alignment_frame_count": alignment_frame_count,
            "descent_frame_count": descent_frame_count,
            "support_frame_count": support_frame_count,
            "touchdown_frame": touchdown_frame,
            "landing_joint_targets_rad": {
                "hip_pitch": -0.45,
                "hip_roll": 0.0,
                "hip_yaw": 0.0,
                "knee": 0.90,
                "ankle_pitch": -0.45,
                "ankle_roll": 0.0,
            },
            "maximum_support_foot_position_error_m": float(
                maximum_support_error
            ),
            "maximum_completion_root_linear_speed_mps": float(
                np.max(completion_linear_speed, initial=0.0)
            ),
            "maximum_completion_root_angular_speed_radps": float(
                np.max(completion_angular_speed, initial=0.0)
            ),
            "contact_root_height_m": float(contact_root[2]),
            "airborne_alignment_root_height_m": float(airborne_root[2]),
            "final_root_height_m": float(support_root[-1, 2]),
        }
    velocity_ratio = (
        np.abs(np.diff(q_frames, axis=0)) * fps
        / _G1_JOINT_VELOCITY.astype(np.float64)[None, :]
    )
    if np.max(velocity_ratio, initial=0.0) > 1.0 + 1.0e-6:
        raise RuntimeError("terminal completion exceeds an authored G1 velocity limit")
    root_linear_speed = np.linalg.norm(
        np.diff(root_world[:, :3], axis=0), axis=1
    ) * fps
    root_delta_rotation = (
        Rotation.from_quat(root_world[:-1, 3:]).inv()
        * Rotation.from_quat(root_world[1:, 3:])
    )
    root_angular_speed = root_delta_rotation.magnitude() * fps

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
        "phase_ids": phase_ids,
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
            "method": (
                "bounded source endpoint IK plus robot-authored aerial completion"
            ),
            "mean_endpoint_error_m": float(np.mean(endpoint_errors)),
            "maximum_endpoint_error_m": float(np.max(endpoint_errors)),
            "maximum_joint_velocity_ratio": float(
                np.max(velocity_ratio, initial=0.0)
            ),
            "maximum_root_linear_speed_mps": float(
                np.max(root_linear_speed, initial=0.0)
            ),
            "maximum_root_angular_speed_radps": float(
                np.max(root_angular_speed, initial=0.0)
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
            "terminal_completion": completion_evidence,
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
