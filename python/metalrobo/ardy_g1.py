"""Convert native ARDY-G1 skeleton motion to Unitree G1 mechanism intent.

The joint-frame conversion follows NVIDIA ARDY's MujocoQposConverter at the
pinned source revision below. It changes coordinates and representation only;
it does not filter motion, repair contacts, or author dynamics.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.transform import Rotation

from .ardy_interaction_convert import (
    _G1_JOINT_LOWER,
    _G1_JOINTS,
    _G1_JOINT_UPPER,
    _G1_JOINT_VELOCITY,
    _metalrobo_targets,
    foot_contact_contract,
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
    (
        "right_shoulder_pitch_joint",
        "y",
        (0.990264, -0.139201, 1.38722e-05, 9.86868e-05),
    ),
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
        ardy_rotation = _MUJOCO_TO_ARDY @ mujoco_rotation @ _MUJOCO_TO_ARDY.T
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
        raise ValueError(
            "ARDY-G1 skeleton is missing mechanism joints: " + ", ".join(missing)
        )

    joint_dofs = np.empty((root_positions.shape[0], len(_G1_JOINTS)), dtype=np.float64)
    for output_index, mechanism_name in enumerate(_G1_JOINTS):
        axis, frame_offset = frame_contract[mechanism_name]
        rotation = (
            frame_offset
            @ local_rotations[:, joint_index[_G1_SKEL_JOINTS[output_index]]]
        )
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
    root_rotation = ardy_to_mujoco[None] @ local_rotations[:, 0] @ _MUJOCO_TO_ARDY[None]
    root_quaternion_wxyz = Rotation.from_matrix(root_rotation).as_quat(
        scalar_first=True
    )
    return np.concatenate((root_xyz, root_quaternion_wxyz, joint_dofs), axis=1).astype(
        np.float32
    )


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
            constraints=(LinearConstraint(difference, -maximum_delta, maximum_delta),),
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
        contact_scores = np.asarray(archive["foot_contact_scores"], dtype=np.float32)
    frame_count = int(source["frame_count"])
    fps = float(source["fps"])
    if (
        root_positions.shape != (frame_count, 3)
        or rotations.shape != (frame_count, 34, 6)
        or contacts.shape != (frame_count, 4)
        or contact_scores.shape != (frame_count, 4)
        or not np.isfinite(root_positions).all()
        or not np.isfinite(rotations).all()
        or not np.isfinite(contact_scores).all()
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
    link_transforms = np.empty((frame_count, len(model.links), 7), dtype=np.float64)
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
        "foot_contact_scores": contact_scores,
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


def prepare_single_support_recovery_handoff(
    arrays: Mapping[str, np.ndarray],
    evidence: Mapping[str, Any],
    urdf_path: Path,
    *,
    swing_side: str,
    minimum_landing_dwell_frames: int = 5,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select an authored landing boundary for physical recovery control.

    ARDY remains the motion-intent owner: this function only rejects an
    inconsistent single-support move and removes an unstable suffix after a
    real authored landing.  It never inserts, blends, or changes a pose.  A
    non-looping InteractionPack holds the selected landing target while the
    closed-loop policy and NumiSolver own stabilization.
    """

    if swing_side not in ("left", "right"):
        raise ValueError("single-support handoff side must be left or right")
    if minimum_landing_dwell_frames < 2:
        raise ValueError("landing dwell must contain at least two frames")
    joints = np.asarray(arrays["joint_positions"], dtype=np.float64)
    roots = np.asarray(arrays["root_position_quaternion_xyzw"], dtype=np.float64)
    raw_contacts = np.asarray(arrays["foot_contacts"], dtype=np.uint8)
    frame_count = joints.shape[0]
    fps = float(evidence["fps"])
    if (
        joints.ndim != 2
        or roots.shape != (frame_count, 7)
        or raw_contacts.shape != (frame_count, 4)
        or frame_count < minimum_landing_dwell_frames + 3
        or not np.isfinite(joints).all()
        or not np.isfinite(roots).all()
        or not np.isfinite(fps)
        or fps <= 0.0
    ):
        raise ValueError("single-support recovery input is invalid")

    foot_contact = np.column_stack(
        (
            np.any(raw_contacts[:, :2] != 0, axis=1),
            np.any(raw_contacts[:, 2:] != 0, axis=1),
        )
    )
    swing_index = 0 if swing_side == "left" else 1
    support_index = 1 - swing_index
    off_frames = np.flatnonzero(~foot_contact[:, swing_index])
    if off_frames.size == 0:
        raise ValueError(f"ARDY motion never releases the {swing_side} foot")
    takeoff = int(off_frames[0])
    contiguous = off_frames[off_frames == np.arange(takeoff, takeoff + off_frames.size)]
    if contiguous.size == 0:
        raise ValueError("ARDY swing contact interval is invalid")
    last_airborne = int(contiguous[-1])
    landing = last_airborne + 1
    if (
        takeoff == 0
        or landing >= frame_count
        or not foot_contact[0, swing_index]
        or not np.all(foot_contact[takeoff : landing + 1, support_index])
        or not foot_contact[landing, swing_index]
    ):
        raise ValueError(
            "ARDY motion lacks supported takeoff followed by an authored landing"
        )

    bilateral = np.all(foot_contact, axis=1)
    dwell_end = landing + minimum_landing_dwell_frames - 1
    if dwell_end >= frame_count or not np.all(bilateral[landing : dwell_end + 1]):
        raise ValueError("ARDY landing lacks bilateral support dwell")
    candidates = np.flatnonzero(bilateral & (np.arange(frame_count) >= dwell_end))
    candidates = candidates[
        np.asarray([np.all(bilateral[landing : frame + 1]) for frame in candidates])
    ]
    if candidates.size == 0:
        raise ValueError("ARDY landing has no valid recovery handoff frame")

    joint_velocity_ratio = np.zeros(frame_count, dtype=np.float64)
    joint_velocity_ratio[1:] = np.max(
        np.abs(np.diff(joints, axis=0))
        * fps
        / _G1_JOINT_VELOCITY.astype(np.float64)[None, :],
        axis=1,
    )
    root_linear_speed = np.zeros(frame_count, dtype=np.float64)
    root_linear_speed[1:] = np.linalg.norm(np.diff(roots[:, :3], axis=0), axis=1) * fps
    root_angular_speed = np.zeros(frame_count, dtype=np.float64)
    root_angular_speed[1:] = (
        Rotation.from_quat(roots[:-1, 3:]).inv() * Rotation.from_quat(roots[1:, 3:])
    ).magnitude() * fps
    quiet_score = (
        joint_velocity_ratio + 0.25 * root_linear_speed + 0.05 * root_angular_speed
    )
    handoff = int(candidates[np.argmin(quiet_score[candidates])])

    model = G1Kinematics.from_urdf(urdf_path)
    link = f"{swing_side}_ankle_pitch_link"
    foot_position = np.asarray(
        [model.forward(frame)[link][:3, 3] for frame in joints[: handoff + 1]]
    )
    displacement = foot_position - foot_position[0]
    peak_forward_frame = int(np.argmax(displacement[:, 0]))
    peak_lift_frame = int(np.argmax(displacement[:, 2]))
    peak_forward = float(displacement[peak_forward_frame, 0])
    peak_lift = float(displacement[peak_lift_frame, 2])
    peak_lateral = float(abs(displacement[peak_forward_frame, 1]))
    return_error = float(np.linalg.norm(displacement[handoff]))
    if (
        peak_forward < 0.20
        or peak_lift < 0.08
        or peak_lateral > 0.20
        or return_error > 0.12
    ):
        raise ValueError(
            "ARDY single-support motion does not contain a front strike and return"
        )

    prepared: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        table = np.asarray(value)
        prepared[name] = (
            table[: handoff + 1].copy()
            if table.ndim > 0 and table.shape[0] == frame_count
            else table.copy()
        )
    prepared_evidence = deepcopy(dict(evidence))
    prepared_evidence["frame_count"] = handoff + 1
    prepared_evidence["execution_handoff"] = {
        "method": "authored unilateral swing to quiet bilateral support",
        "poses_inserted": 0,
        "poses_modified": 0,
        "source_frame_count": frame_count,
        "selected_frame": handoff,
        "selected_frame_count": handoff + 1,
        "swing_side": swing_side,
        "takeoff_frame": takeoff,
        "last_airborne_frame": last_airborne,
        "landing_frame": landing,
        "landing_dwell_frames": handoff - landing + 1,
        "peak_forward_m": peak_forward,
        "peak_forward_frame": peak_forward_frame,
        "peak_lift_m": peak_lift,
        "peak_lift_frame": peak_lift_frame,
        "peak_lateral_at_strike_m": peak_lateral,
        "terminal_foot_return_error_m": return_error,
        "terminal_maximum_joint_velocity_ratio": float(joint_velocity_ratio[handoff]),
        "terminal_root_linear_speed_mps": float(root_linear_speed[handoff]),
        "terminal_root_angular_speed_radps": float(root_angular_speed[handoff]),
        "post_handoff_owner": "HyperPolicy residual plus NumiSolver contact",
    }
    return prepared, prepared_evidence


def write_native_g1_interaction_pack(
    *,
    output: Path,
    arrays: Mapping[str, np.ndarray],
    evidence: Mapping[str, Any],
    desired_outcome: str,
    clip_id: str = "ardy-g1",
) -> tuple[Path, int]:
    modes, contact_confidence = foot_contact_contract(
        arrays["foot_contacts"], arrays["foot_contact_scores"]
    )
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
        contact_confidence=contact_confidence,
    )
