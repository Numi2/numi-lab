"""Convert native ARDY-G1 skeleton motion to Unitree G1 mechanism intent.

The joint-frame conversion follows NVIDIA ARDY's MujocoQposConverter at the
pinned source revision below. It changes coordinates and representation only;
it does not filter motion, repair contacts, or author dynamics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.transform import Rotation

from .ardy_interaction_convert import (
    _CONTACT_MODE_FREE,
    _CONTACT_MODE_STICK,
    _G1_JOINT_LOWER,
    _G1_JOINTS,
    _G1_JOINT_UPPER,
    _G1_JOINT_VELOCITY,
    _metalrobo_targets,
    write_interaction_pack,
)
from .g1_motion_retarget import G1Kinematics


ARDY_G1_MECHANISM_FORMAT = "numi.ardy-g1-mechanism.v1"
ARDY_G1_CONVERTER_REPOSITORY = "https://github.com/nv-tlabs/ardy"
ARDY_G1_CONVERTER_REVISION = "693f74d13b3d04a0a22ce127ee79c929dd89756b"
ARDY_G1_LICENSE = "NVIDIA Open Model License"

_G1_SKEL_JOINTS = tuple(name.replace("_joint", "_skel") for name in _G1_JOINTS)
_MUJOCO_TO_ARDY = np.asarray(
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    dtype=np.float64,
)

# MuJoCo hinge axes and authored body-frame quaternion offsets from ARDY's
# g1skel34/xml/g1.xml. Quaternions are scalar-first in MuJoCo coordinates.
_JOINT_FRAME_CONTRACT = (
    ("left_hip_pitch_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("left_hip_roll_joint", "x", (0.996179, 0.0, -0.0873386, 0.0)),
    ("left_hip_yaw_joint", "z", (1.0, 0.0, 0.0, 0.0)),
    ("left_knee_joint", "y", (0.996179, 0.0, 0.0873386, 0.0)),
    ("left_ankle_pitch_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("left_ankle_roll_joint", "x", (1.0, 0.0, 0.0, 0.0)),
    ("right_hip_pitch_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("right_hip_roll_joint", "x", (0.996179, 0.0, -0.0873386, 0.0)),
    ("right_hip_yaw_joint", "z", (1.0, 0.0, 0.0, 0.0)),
    ("right_knee_joint", "y", (0.996179, 0.0, 0.0873386, 0.0)),
    ("right_ankle_pitch_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("right_ankle_roll_joint", "x", (1.0, 0.0, 0.0, 0.0)),
    ("waist_yaw_joint", "z", (1.0, 0.0, 0.0, 0.0)),
    ("waist_roll_joint", "x", (1.0, 0.0, 0.0, 0.0)),
    ("waist_pitch_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("left_shoulder_pitch_joint", "y", (0.990264, 0.139201, 1.38722e-05, -9.86868e-05)),
    ("left_shoulder_roll_joint", "x", (0.990268, -0.139172, 0.0, 0.0)),
    ("left_shoulder_yaw_joint", "z", (1.0, 0.0, 0.0, 0.0)),
    ("left_elbow_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("left_wrist_roll_joint", "x", (1.0, 0.0, 0.0, 0.0)),
    ("left_wrist_pitch_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("left_wrist_yaw_joint", "z", (1.0, 0.0, 0.0, 0.0)),
    ("right_shoulder_pitch_joint", "y", (0.990264, -0.139201, 1.38722e-05, 9.86868e-05)),
    ("right_shoulder_roll_joint", "x", (0.990268, 0.139172, 0.0, 0.0)),
    ("right_shoulder_yaw_joint", "z", (1.0, 0.0, 0.0, 0.0)),
    ("right_elbow_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("right_wrist_roll_joint", "x", (1.0, 0.0, 0.0, 0.0)),
    ("right_wrist_pitch_joint", "y", (1.0, 0.0, 0.0, 0.0)),
    ("right_wrist_yaw_joint", "z", (1.0, 0.0, 0.0, 0.0)),
)


def _cont6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = value[..., :3]
    second = value[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1.0e-8):
        raise ValueError("ARDY-G1 rotation has a degenerate first axis")
    x = first / first_norm
    z = np.cross(x, second)
    z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(z_norm < 1.0e-8):
        raise ValueError("ARDY-G1 rotation has collinear 6D axes")
    z /= z_norm
    y = np.cross(z, x)
    result = np.stack((x, y, z), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("ARDY-G1 rotation conversion is non-finite")
    return result


def _global_to_local(global_rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.empty_like(global_rotations)
    for joint, parent in enumerate(parents):
        local[:, joint] = (
            global_rotations[:, joint]
            if parent < 0
            else np.swapaxes(global_rotations[:, parent], -1, -2)
            @ global_rotations[:, joint]
        )
    return local


def _joint_frame_offsets() -> Mapping[str, tuple[str, np.ndarray]]:
    result: dict[str, tuple[str, np.ndarray]] = {}
    for name, axis, quaternion_wxyz in _JOINT_FRAME_CONTRACT:
        mujoco_rotation = Rotation.from_quat(
            quaternion_wxyz, scalar_first=True
        ).as_matrix()
        ardy_rotation = (
            _MUJOCO_TO_ARDY @ mujoco_rotation @ _MUJOCO_TO_ARDY.T
        )
        result[name] = (axis, ardy_rotation.T)
    return result


def _qpos_from_motion(
    root_positions: np.ndarray,
    global_rotations_6d: np.ndarray,
    joint_names: tuple[str, ...],
    joint_parents: np.ndarray,
) -> np.ndarray:
    global_rotations = _cont6d_to_matrix(global_rotations_6d)
    local_rotations = _global_to_local(global_rotations, joint_parents)
    frame_contract = _joint_frame_offsets()
    joint_index = {name: index for index, name in enumerate(joint_names)}
    if tuple(name for name, _, _ in _JOINT_FRAME_CONTRACT) != _G1_JOINTS:
        raise RuntimeError("ARDY-G1 mechanism mapping order changed")
    missing = [name for name in _G1_SKEL_JOINTS if name not in joint_index]
    if missing:
        raise ValueError("ARDY-G1 skeleton is missing mechanism joints: " + ", ".join(missing))

    joint_dofs = np.empty((root_positions.shape[0], len(_G1_JOINTS)), dtype=np.float64)
    for output_index, mechanism_name in enumerate(_G1_JOINTS):
        axis, frame_offset = frame_contract[mechanism_name]
        rotation = frame_offset @ local_rotations[:, joint_index[_G1_SKEL_JOINTS[output_index]]]
        x_angle = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
        y_angle = np.arctan2(rotation[:, 0, 2], rotation[:, 0, 0])
        z_angle = np.arctan2(rotation[:, 1, 0], rotation[:, 1, 1])
        # ARDY maps MuJoCo x/y/z hinge axes to z/x/y in its y-up frame.
        joint_dofs[:, output_index] = {
            "x": z_angle,
            "y": x_angle,
            "z": y_angle,
        }[axis]

    ardy_to_mujoco = _MUJOCO_TO_ARDY.T
    root_xyz = np.einsum("ij,tj->ti", ardy_to_mujoco, root_positions)
    root_rotation = (
        ardy_to_mujoco[None]
        @ local_rotations[:, 0]
        @ _MUJOCO_TO_ARDY[None]
    )
    root_quaternion_wxyz = Rotation.from_matrix(root_rotation).as_quat(
        scalar_first=True
    )
    return np.concatenate(
        (root_xyz, root_quaternion_wxyz, joint_dofs), axis=1
    ).astype(np.float32)


def _project_mechanism_trajectory(
    joints: np.ndarray, fps: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Closest L2 trajectory satisfying authored position and velocity limits."""
    source = np.asarray(joints, dtype=np.float64)
    frame_count = source.shape[0]
    difference = np.zeros((frame_count - 1, frame_count), dtype=np.float64)
    rows = np.arange(frame_count - 1)
    difference[rows, rows] = -1.0
    difference[rows, rows + 1] = 1.0
    projected = np.empty_like(source)
    solver_iterations = []
    for joint in range(source.shape[1]):
        maximum_delta = float(_G1_JOINT_VELOCITY[joint]) / fps
        initial = np.clip(
            source[:, joint],
            float(_G1_JOINT_LOWER[joint]),
            float(_G1_JOINT_UPPER[joint]),
        )
        # Produce a feasible starting point before solving the convex projection.
        for _ in range(2):
            for frame in range(1, frame_count):
                initial[frame] = np.clip(
                    initial[frame],
                    initial[frame - 1] - maximum_delta,
                    initial[frame - 1] + maximum_delta,
                )
            for frame in range(frame_count - 2, -1, -1):
                initial[frame] = np.clip(
                    initial[frame],
                    initial[frame + 1] - maximum_delta,
                    initial[frame + 1] + maximum_delta,
                )
        result = minimize(
            lambda value: 0.5 * float(np.sum((value - source[:, joint]) ** 2)),
            initial,
            jac=lambda value: value - source[:, joint],
            method="SLSQP",
            bounds=Bounds(
                float(_G1_JOINT_LOWER[joint]),
                float(_G1_JOINT_UPPER[joint]),
            ),
            constraints=(LinearConstraint(
                difference, -maximum_delta, maximum_delta
            ),),
            options={"ftol": 1.0e-12, "maxiter": 300},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(
                f"ARDY-G1 feasibility projection failed for {_G1_JOINTS[joint]}: "
                f"{result.message}"
            )
        projected[:, joint] = result.x
        solver_iterations.append(int(result.nit))
    correction = projected - source
    per_joint_maximum = np.max(np.abs(correction), axis=0)
    return projected.astype(np.float32), {
        "method": "closest L2 trajectory under Unitree position and velocity limits",
        "source_preserved": True,
        "corrected_sample_count": int(np.count_nonzero(np.abs(correction) > 1.0e-7)),
        "mean_absolute_correction_rad": float(np.mean(np.abs(correction))),
        "maximum_absolute_correction_rad": float(np.max(np.abs(correction))),
        "maximum_correction_joint": _G1_JOINTS[int(np.argmax(per_joint_maximum))],
        "per_joint_maximum_correction_rad": {
            name: float(value)
            for name, value in zip(_G1_JOINTS, per_joint_maximum, strict=True)
        },
        "maximum_solver_iterations": max(solver_iterations),
    }


def native_g1_mechanism(
    proposal_directory: Path,
    urdf_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert native g1skel34 output to exact 29-DoF G1 coordinates."""
    evidence_path = proposal_directory / "evidence.json"
    motion_path = proposal_directory / "motion_proposal.npz"
    source = json.loads(evidence_path.read_text(encoding="utf-8"))
    skeleton = source.get("skeleton", {})
    if skeleton.get("id") != "g1skel34":
        raise ValueError("native G1 conversion requires an ARDY g1skel34 proposal")
    names = tuple(str(value) for value in skeleton.get("joint_names", ()))
    parents = np.asarray(skeleton.get("joint_parents", ()), dtype=np.int64)
    if len(names) != 34 or parents.shape != (34,) or parents[0] != -1:
        raise ValueError("ARDY-G1 skeleton hierarchy is invalid")

    with np.load(motion_path, allow_pickle=False) as archive:
        root_positions = np.asarray(archive["root_positions"], dtype=np.float64)
        rotations = np.asarray(archive["global_rotations_6d"], dtype=np.float64)
        contacts = np.asarray(archive["foot_contacts"], dtype=np.uint8)
    frame_count = int(source["frame_count"])
    fps = float(source["fps"])
    if (
        root_positions.shape != (frame_count, 3)
        or rotations.shape != (frame_count, 34, 6)
        or contacts.shape != (frame_count, 4)
        or not np.isfinite(root_positions).all()
        or not np.isfinite(rotations).all()
    ):
        raise ValueError("ARDY-G1 proposal arrays do not match the model contract")

    qpos = _qpos_from_motion(root_positions, rotations, names, parents)
    raw_joint_targets = qpos[:, 7:].copy()
    projected_joint_targets, projection_evidence = _project_mechanism_trajectory(
        raw_joint_targets, fps
    )
    qpos[:, 7:] = projected_joint_targets
    root_targets, joint_targets, peak_velocity_ratio = _metalrobo_targets(
        qpos, fps, 1.0e-6
    )
    model = G1Kinematics.from_urdf(urdf_path)
    link_transforms = np.empty(
        (frame_count, len(model.links), 7), dtype=np.float64
    )
    for frame, joint_values in enumerate(joint_targets):
        root_rotation = Rotation.from_quat(root_targets[frame, 3:]).as_matrix()
        local_poses = model.forward(joint_values.astype(np.float64))
        for link_index, link in enumerate(model.links):
            local_pose = local_poses[link]
            link_transforms[frame, link_index, :3] = (
                root_targets[frame, :3] + root_rotation @ local_pose[:3, 3]
            )
            link_transforms[frame, link_index, 3:] = Rotation.from_matrix(
                root_rotation @ local_pose[:3, :3]
            ).as_quat()

    arrays = {
        "root_position_quaternion_xyzw": root_targets,
        "ardy_joint_positions": raw_joint_targets,
        "joint_positions": joint_targets,
        "foot_contacts": contacts,
        "phase_ids": np.zeros(frame_count, dtype=np.uint8),
        "link_names": np.asarray(model.links, dtype=np.str_),
        "link_position_quaternion_xyzw": link_transforms.astype(np.float32),
    }
    evidence = {
        "format": ARDY_G1_MECHANISM_FORMAT,
        "status": "native-g1-coordinate-conversion",
        "robot": "unitree_g1_29dof_rev_1_0",
        "source_motion": {
            "path": str(proposal_directory),
            "prompt": source["prompt"],
            "model_repository": source["model_repository"],
            "model_revision": source["model_revision"],
            "arrays_fingerprint": source["motion_proposal"]["arrays_fingerprint"],
            "skeleton": "g1skel34",
        },
        "converter": {
            "repository": ARDY_G1_CONVERTER_REPOSITORY,
            "revision": ARDY_G1_CONVERTER_REVISION,
            "method": "NVIDIA ARDY G1 joint-frame projection without IK",
        },
        "fps": fps,
        "frame_count": frame_count,
        "joint_order": list(_G1_JOINTS),
        "maximum_joint_velocity_ratio": peak_velocity_ratio,
        "joint_limits_satisfied": True,
        "mechanism_projection": projection_evidence,
        "authority": (
            "native ARDY-G1 kinematic intent only; gravity, contact, balance, "
            "and actuation remain NumiSolver outcomes"
        ),
    }
    return arrays, evidence


def write_native_g1_interaction_pack(
    *,
    output: Path,
    arrays: Mapping[str, np.ndarray],
    evidence: Mapping[str, Any],
    desired_outcome: str,
    clip_id: str = "ardy-g1",
) -> tuple[Path, int]:
    contacts = np.asarray(arrays["foot_contacts"], dtype=np.uint8)
    contact_by_foot = np.stack(
        (np.any(contacts[:, :2], axis=1), np.any(contacts[:, 2:], axis=1)),
        axis=1,
    )
    modes = np.where(
        contact_by_foot, _CONTACT_MODE_STICK, _CONTACT_MODE_FREE
    ).astype(np.uint32)
    source = evidence["source_motion"]
    return write_interaction_pack(
        output=output,
        pack_id="ardy-g1-native-" + str(source["arrays_fingerprint"])[:16],
        clip_id=clip_id,
        desired_outcome=desired_outcome,
        source_repository=str(source["model_repository"]),
        source_revision=str(source["model_revision"]),
        license_name=ARDY_G1_LICENSE,
        frames_per_second=float(evidence["fps"]),
        root_targets=np.asarray(arrays["root_position_quaternion_xyzw"]),
        joint_targets=np.asarray(arrays["joint_positions"]),
        tracks=(
            ("left_foot", "left_foot_contact", "locomotion_ground"),
            ("right_foot", "right_foot_contact", "locomotion_ground"),
        ),
        contact_modes=modes,
        contact_confidence=np.full(modes.shape, 0.5, dtype=np.float32),
    )
