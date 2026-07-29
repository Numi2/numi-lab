"""GPU-resident Franka action adaptation and pick-place task semantics.

The policy emits a Cartesian delta twist plus one gripper command. Exact
kinematic constants come from MetalRobo's pinned FER Franka model; all forward
kinematics, analytic Jacobians, damped least-squares adaptation, contact
evidence, phase transitions, and rewards remain MLX array programs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import mlx.core as mx

from .mlx_world import ContactEvidence, SampledWorldFamilyState


class FrankaTaskPhase(IntEnum):
    approach = 0
    pregrasp = 1
    contact = 2
    grasp = 3
    lift = 4
    transport = 5
    place = 6
    release = 7
    settle = 8


class FrankaActionMode(IntEnum):
    joint_position = 0
    cartesian_delta = 1
    operational_space_impedance = 2


class FrankaKinematics(NamedTuple):
    position: mx.array
    orientation: mx.array
    linear_jacobian: mx.array
    angular_jacobian: mx.array


class AdaptedFrankaAction(NamedTuple):
    joint_targets: mx.array
    feedforward_torque: mx.array
    kinematics: FrankaKinematics


class FrankaPickPlaceTaskState(NamedTuple):
    phase: mx.array
    phase_steps: mx.array
    grasp_steps: mx.array
    settle_steps: mx.array
    initial_object_position: mx.array
    previous_object_position: mx.array


class FrankaTaskEvidence(NamedTuple):
    bilateral_contact: mx.array
    stable_grasp: mx.array
    lifted: mx.array
    within_target: mx.array
    released: mx.array
    support_contact: mx.array
    settled: mx.array
    success: mx.array
    task_margin: mx.array
    safety_margin: mx.array
    reward: mx.array
    failure_mask: mx.array


@dataclass(frozen=True, slots=True)
class CartesianActionAdapterConfig:
    translation_scale: float = 0.035
    rotation_scale: float = 0.12
    damping: float = 0.08
    nullspace_gain: float = 0.035
    joint_limit_margin: float = 0.01

    def validate(self) -> None:
        if (
            self.translation_scale <= 0.0
            or self.rotation_scale <= 0.0
            or self.damping <= 0.0
            or self.nullspace_gain < 0.0
            or self.joint_limit_margin < 0.0
        ):
            raise ValueError("Cartesian action-adapter parameters are invalid")


@dataclass(frozen=True, slots=True)
class OperationalSpaceImpedanceConfig:
    translation_stiffness: float = 180.0
    rotation_stiffness: float = 25.0
    translation_damping: float = 26.0
    rotation_damping: float = 8.0
    nullspace_stiffness: float = 7.5
    joint_damping: float = 1.5
    maximum_torque_rate: float = 800.0

    def validate(self) -> None:
        if (
            self.translation_stiffness <= 0.0
            or self.rotation_stiffness <= 0.0
            or self.translation_damping < 0.0
            or self.rotation_damping < 0.0
            or self.nullspace_stiffness < 0.0
            or self.joint_damping < 0.0
            or self.maximum_torque_rate <= 0.0
        ):
            raise ValueError(
                "operational-space impedance parameters are invalid"
            )


@dataclass(frozen=True, slots=True)
class FrankaPickPlaceTaskConfig:
    object_scene_index: int = 0
    target_scene_index: int = 2
    object_collider: int = 32
    left_finger_first_collider: int = 24
    left_finger_collider_count: int = 4
    right_finger_first_collider: int = 28
    right_finger_collider_count: int = 4
    ground_collider: int = 33
    target_collider: int = 34
    object_half_height: float = 0.025
    pregrasp_distance: float = 0.10
    stable_grasp_steps: int = 3
    lift_height: float = 0.04
    target_radius: float = 0.05
    target_height_tolerance: float = 0.025
    release_aperture: float = 0.055
    settle_speed: float = 0.05
    settle_steps: int = 6

    def validate(self) -> None:
        if (
            min(
                self.object_scene_index,
                self.target_scene_index,
                self.object_collider,
                self.left_finger_first_collider,
                self.right_finger_first_collider,
                self.ground_collider,
                self.target_collider,
            )
            < 0
            or self.left_finger_collider_count <= 0
            or self.right_finger_collider_count <= 0
            or self.object_half_height <= 0.0
            or self.pregrasp_distance <= 0.0
            or self.stable_grasp_steps <= 0
            or self.lift_height <= 0.0
            or self.target_radius <= 0.0
            or self.target_height_tolerance <= 0.0
            or self.release_aperture <= 0.0
            or self.settle_speed <= 0.0
            or self.settle_steps <= 0
        ):
            raise ValueError("Franka task configuration is invalid")


_LINK_COM = (
    (-0.0172, 0.0004, 0.0745),
    (0.00033, -0.02204, -0.04762),
    (0.00038, -0.09211, 0.01908),
    (0.05152, 0.01696, -0.02971),
    (-0.05113, 0.05825, 0.01698),
    (-0.00005, 0.03730, -0.09280),
    (0.06572, -0.00371, 0.00153),
    (0.00089, -0.00044, 0.05491),
)
_JOINT_OFFSETS = (
    (0.0, 0.0, 0.333),
    (0.0, 0.0, 0.0),
    (0.0, -0.316, 0.0),
    (0.0825, 0.0, 0.0),
    (-0.0825, 0.384, 0.0),
    (0.0, 0.0, 0.0),
    (0.088, 0.0, 0.0),
)
_JOINT_ROLLS = (
    0.0,
    -0.5 * math.pi,
    0.5 * math.pi,
    0.5 * math.pi,
    -0.5 * math.pi,
    0.5 * math.pi,
    0.5 * math.pi,
)
_JOINT_LOWER = (
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
)
_JOINT_UPPER = (
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973,
)
_JOINT_VELOCITY = (2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610)
_JOINT_TORQUE = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
_HAND_TOOL_OFFSET = 0.1034


def _rotation_x(angle: float) -> mx.array:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return mx.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, cosine, -sine),
            (0.0, sine, cosine),
        ),
        dtype=mx.float32,
    )


def _rotation_z(angle: mx.array) -> mx.array:
    cosine = mx.cos(angle)
    sine = mx.sin(angle)
    zero = mx.zeros_like(angle)
    one = mx.ones_like(angle)
    return mx.stack(
        (
            mx.stack((cosine, -sine, zero), axis=-1),
            mx.stack((sine, cosine, zero), axis=-1),
            mx.stack((zero, zero, one), axis=-1),
        ),
        axis=-2,
    )


def _matvec(matrix: mx.array, vector: mx.array) -> mx.array:
    return mx.einsum("nij,nj->ni", matrix, vector)


def _cross(left: mx.array, right: mx.array) -> mx.array:
    return mx.stack(
        (
            left[:, 1] * right[:, 2] - left[:, 2] * right[:, 1],
            left[:, 2] * right[:, 0] - left[:, 0] * right[:, 2],
            left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0],
        ),
        axis=-1,
    )


def _batched_spd_solve(
    matrix: mx.array,
    right_hand_side: mx.array,
    *,
    iterations: int = 8,
) -> mx.array:
    """Fixed-iteration batched CG for small GPU-resident SPD systems."""

    solution = mx.zeros_like(right_hand_side)
    residual = right_hand_side
    direction = residual
    residual_norm = mx.sum(residual * residual, axis=-1)
    for _ in range(iterations):
        product = mx.matmul(matrix, direction[:, :, None])[:, :, 0]
        denominator = mx.maximum(
            mx.sum(direction * product, axis=-1),
            1.0e-12,
        )
        step = residual_norm / denominator
        solution = solution + step[:, None] * direction
        next_residual = residual - step[:, None] * product
        next_norm = mx.sum(next_residual * next_residual, axis=-1)
        direction = next_residual + (
            next_norm / mx.maximum(residual_norm, 1.0e-12)
        )[:, None] * direction
        residual = next_residual
        residual_norm = next_norm
    return solution


def franka_kinematics(q: mx.array) -> FrankaKinematics:
    """Return the canonical FER hand tool pose and analytic 6x7 Jacobian."""

    if q.ndim != 2 or int(q.shape[1]) < 7:
        raise ValueError("Franka q must have shape [environment, >=7]")
    environment_count = int(q.shape[0])
    rotation = mx.broadcast_to(
        mx.eye(3, dtype=mx.float32)[None, :, :],
        (environment_count, 3, 3),
    )
    origin = mx.zeros((environment_count, 3), dtype=mx.float32)
    joint_axes: list[mx.array] = []
    joint_positions: list[mx.array] = []
    for joint in range(7):
        parent_rotation = rotation
        parent_origin = origin
        parent_to_joint = mx.matmul(
            parent_rotation,
            _rotation_x(_JOINT_ROLLS[joint]),
        )
        joint_axes.append(parent_to_joint[:, :, 2])
        parent_anchor = mx.array(
            tuple(
                _JOINT_OFFSETS[joint][axis]
                - _LINK_COM[joint][axis]
                for axis in range(3)
            ),
            dtype=mx.float32,
        )
        child_anchor = mx.array(
            tuple(-value for value in _LINK_COM[joint + 1]),
            dtype=mx.float32,
        )
        joint_position = parent_origin + _matvec(
            parent_rotation,
            mx.broadcast_to(
                parent_anchor[None, :],
                (environment_count, 3),
            ),
        )
        joint_positions.append(joint_position)
        rotation = mx.matmul(
            parent_to_joint,
            _rotation_z(q[:, joint]),
        )
        origin = joint_position - _matvec(
            rotation,
            mx.broadcast_to(
                child_anchor[None, :],
                (environment_count, 3),
            ),
        )

    link7_com = _LINK_COM[7]
    flange_anchor = mx.array(
        (
            -link7_com[0],
            -link7_com[1],
            0.107 - link7_com[2],
        ),
        dtype=mx.float32,
    )
    flange = origin + _matvec(
        rotation,
        mx.broadcast_to(
            flange_anchor[None, :],
            (environment_count, 3),
        ),
    )
    hand_rotation = mx.matmul(
        rotation,
        _rotation_z(
            mx.full(
                (environment_count,),
                -0.25 * math.pi,
                dtype=mx.float32,
            )
        ),
    )
    tool = flange + _matvec(
        hand_rotation,
        mx.broadcast_to(
            mx.array(
                (0.0, 0.0, _HAND_TOOL_OFFSET),
                dtype=mx.float32,
            )[None, :],
            (environment_count, 3),
        ),
    )
    linear_columns = [
        _cross(axis, tool - position)
        for axis, position in zip(
            joint_axes,
            joint_positions,
            strict=True,
        )
    ]
    return FrankaKinematics(
        position=tool,
        orientation=hand_rotation,
        linear_jacobian=mx.stack(linear_columns, axis=-1),
        angular_jacobian=mx.stack(joint_axes, axis=-1),
    )


def adapt_cartesian_action(
    q: mx.array,
    normalized_action: mx.array,
    *,
    control_period_seconds: float,
    config: CartesianActionAdapterConfig = CartesianActionAdapterConfig(),
) -> AdaptedFrankaAction:
    """Map a Cartesian delta twist and gripper scalar to joint targets."""

    config.validate()
    if (
        q.ndim != 2
        or int(q.shape[1]) != 9
        or normalized_action.shape != (int(q.shape[0]), 7)
        or control_period_seconds <= 0.0
    ):
        raise ValueError(
            "Cartesian Franka action requires q [N,9] and action [N,7]"
        )
    kinematics = franka_kinematics(q)
    jacobian = mx.concatenate(
        (
            kinematics.linear_jacobian,
            kinematics.angular_jacobian,
        ),
        axis=1,
    )
    twist = mx.concatenate(
        (
            normalized_action[:, :3] * config.translation_scale,
            normalized_action[:, 3:6] * config.rotation_scale,
        ),
        axis=-1,
    )
    jacobian_t = mx.swapaxes(jacobian, -1, -2)
    identity6 = mx.broadcast_to(
        mx.eye(6, dtype=mx.float32)[None, :, :],
        (int(q.shape[0]), 6, 6),
    )
    gram = mx.matmul(jacobian, jacobian_t) + (
        config.damping * config.damping
    ) * identity6
    task_delta = mx.matmul(
        jacobian_t,
        _batched_spd_solve(gram, twist)[:, :, None],
    )[:, :, 0]
    lower = mx.array(_JOINT_LOWER, dtype=mx.float32)
    upper = mx.array(_JOINT_UPPER, dtype=mx.float32)
    midpoint = 0.5 * (lower + upper)
    span = upper - lower
    null_direction = (midpoint[None, :] - q[:, :7]) / (
        span[None, :] * span[None, :]
    )
    projected = null_direction - mx.matmul(
        jacobian_t,
        _batched_spd_solve(
            gram,
            mx.matmul(jacobian, null_direction[:, :, None])[:, :, 0],
        )[:, :, None],
    )[:, :, 0]
    delta = task_delta + config.nullspace_gain * projected
    maximum_delta = (
        mx.array(_JOINT_VELOCITY, dtype=mx.float32)
        * control_period_seconds
    )
    delta = mx.clip(
        delta,
        -maximum_delta[None, :],
        maximum_delta[None, :],
    )
    arm_target = mx.clip(
        q[:, :7] + delta,
        lower[None, :] + config.joint_limit_margin,
        upper[None, :] - config.joint_limit_margin,
    )
    finger_target = mx.clip(
        0.02 * (normalized_action[:, 6:7] + 1.0),
        0.0,
        0.04,
    )
    return AdaptedFrankaAction(
        joint_targets=mx.concatenate(
            (arm_target, finger_target, finger_target),
            axis=-1,
        ),
        feedforward_torque=mx.zeros_like(q),
        kinematics=kinematics,
    )


def adapt_joint_position_action(
    q: mx.array,
    normalized_action: mx.array,
    *,
    config: CartesianActionAdapterConfig = CartesianActionAdapterConfig(),
) -> AdaptedFrankaAction:
    """Compatibility mapping from normalized joint actions to FER targets."""

    config.validate()
    if q.ndim != 2 or int(q.shape[1]) != 9 or q.shape != normalized_action.shape:
        raise ValueError("joint-position action requires q/action [N,9]")
    lower = mx.array(_JOINT_LOWER, dtype=mx.float32)
    upper = mx.array(_JOINT_UPPER, dtype=mx.float32)
    arm = 0.5 * (
        lower[None, :]
        + upper[None, :]
        + mx.clip(normalized_action[:, :7], -1.0, 1.0)
        * (upper - lower)[None, :]
    )
    fingers = 0.02 * (
        mx.clip(normalized_action[:, 7:9], -1.0, 1.0) + 1.0
    )
    return AdaptedFrankaAction(
        joint_targets=mx.concatenate(
            (
                mx.clip(
                    arm,
                    lower[None, :] + config.joint_limit_margin,
                    upper[None, :] - config.joint_limit_margin,
                ),
                fingers,
            ),
            axis=-1,
        ),
        feedforward_torque=mx.zeros_like(q),
        kinematics=franka_kinematics(q),
    )


def adapt_operational_space_action(
    q: mx.array,
    v: mx.array,
    normalized_action: mx.array,
    *,
    control_period_seconds: float,
    previous_torque: mx.array | None = None,
    payload_mass_kg: float | mx.array = 0.0,
    action_config: CartesianActionAdapterConfig = (
        CartesianActionAdapterConfig()
    ),
    impedance: OperationalSpaceImpedanceConfig = (
        OperationalSpaceImpedanceConfig()
    ),
) -> AdaptedFrankaAction:
    """Produce DLS joint targets plus bounded operational-space torque."""

    impedance.validate()
    if v.shape != q.shape:
        raise ValueError("operational-space impedance requires v shaped like q")
    adapted = adapt_cartesian_action(
        q,
        normalized_action,
        control_period_seconds=control_period_seconds,
        config=action_config,
    )
    kinematics = adapted.kinematics
    jacobian = mx.concatenate(
        (
            kinematics.linear_jacobian,
            kinematics.angular_jacobian,
        ),
        axis=1,
    )
    desired_delta = mx.concatenate(
        (
            normalized_action[:, :3]
            * action_config.translation_scale,
            normalized_action[:, 3:6]
            * action_config.rotation_scale,
        ),
        axis=-1,
    )
    task_velocity = mx.matmul(
        jacobian,
        v[:, :7, None],
    )[:, :, 0]
    stiffness = mx.array(
        (
            impedance.translation_stiffness,
            impedance.translation_stiffness,
            impedance.translation_stiffness,
            impedance.rotation_stiffness,
            impedance.rotation_stiffness,
            impedance.rotation_stiffness,
        ),
        dtype=mx.float32,
    )
    damping = mx.array(
        (
            impedance.translation_damping,
            impedance.translation_damping,
            impedance.translation_damping,
            impedance.rotation_damping,
            impedance.rotation_damping,
            impedance.rotation_damping,
        ),
        dtype=mx.float32,
    )
    wrench = (
        stiffness[None, :] * desired_delta
        - damping[None, :] * task_velocity
    )
    torque = mx.matmul(
        mx.swapaxes(jacobian, -1, -2),
        wrench[:, :, None],
    )[:, :, 0]
    midpoint = 0.5 * (
        mx.array(_JOINT_LOWER, dtype=mx.float32)
        + mx.array(_JOINT_UPPER, dtype=mx.float32)
    )
    torque += (
        impedance.nullspace_stiffness
        * (midpoint[None, :] - q[:, :7])
        - impedance.joint_damping * v[:, :7]
    )
    payload = (
        mx.full(
            (int(q.shape[0]),),
            float(payload_mass_kg),
            dtype=mx.float32,
        )
        if isinstance(payload_mass_kg, (float, int))
        else payload_mass_kg.astype(mx.float32)
    )
    if payload.shape != (int(q.shape[0]),):
        raise ValueError("payload mass must be scalar or per environment")
    payload_force = mx.stack(
        (
            mx.zeros_like(payload),
            mx.zeros_like(payload),
            payload * 9.80665,
        ),
        axis=-1,
    )
    torque += mx.matmul(
        mx.swapaxes(kinematics.linear_jacobian, -1, -2),
        payload_force[:, :, None],
    )[:, :, 0]
    previous = (
        mx.zeros_like(torque)
        if previous_torque is None
        else previous_torque[:, :7]
    )
    if previous.shape != torque.shape:
        raise ValueError("previous torque has the wrong shape")
    maximum_delta = (
        impedance.maximum_torque_rate * control_period_seconds
    )
    torque = previous + mx.clip(
        torque - previous,
        -maximum_delta,
        maximum_delta,
    )
    limits = mx.array(_JOINT_TORQUE, dtype=mx.float32)
    torque = mx.clip(torque, -limits[None, :], limits[None, :])
    return AdaptedFrankaAction(
        joint_targets=adapted.joint_targets,
        feedforward_torque=mx.concatenate(
            (
                torque,
                mx.zeros(
                    (int(q.shape[0]), 2),
                    dtype=mx.float32,
                ),
            ),
            axis=-1,
        ),
        kinematics=kinematics,
    )


class FrankaActionAdapter:
    """Policy-independent static action port selected at graph compile time."""

    def __init__(
        self,
        mode: FrankaActionMode = FrankaActionMode.cartesian_delta,
    ) -> None:
        self.mode = FrankaActionMode(mode)

    def adapt(
        self,
        q: mx.array,
        action: mx.array,
        *,
        control_period_seconds: float,
        v: mx.array | None = None,
        previous_torque: mx.array | None = None,
        payload_mass_kg: float | mx.array = 0.0,
    ) -> AdaptedFrankaAction:
        if self.mode == FrankaActionMode.joint_position:
            return adapt_joint_position_action(q, action)
        if self.mode == FrankaActionMode.cartesian_delta:
            return adapt_cartesian_action(
                q,
                action,
                control_period_seconds=control_period_seconds,
            )
        if v is None:
            raise ValueError(
                "operational-space impedance requires joint velocity"
            )
        return adapt_operational_space_action(
            q,
            v,
            action,
            control_period_seconds=control_period_seconds,
            previous_torque=previous_torque,
            payload_mass_kg=payload_mass_kg,
        )


def initial_franka_pick_place_task(
    sampled: SampledWorldFamilyState,
    *,
    config: FrankaPickPlaceTaskConfig = FrankaPickPlaceTaskConfig(),
) -> FrankaPickPlaceTaskState:
    config.validate()
    object_position = sampled.world.scene_bodies.position[
        :, config.object_scene_index, :3
    ]
    environment_count = int(object_position.shape[0])
    zeros = mx.zeros((environment_count,), dtype=mx.uint32)
    return FrankaPickPlaceTaskState(
        phase=zeros,
        phase_steps=zeros,
        grasp_steps=zeros,
        settle_steps=zeros,
        initial_object_position=object_position,
        previous_object_position=object_position,
    )


def select_franka_pick_place_task(
    current: FrankaPickPlaceTaskState,
    replacement: FrankaPickPlaceTaskState,
    reset_mask: mx.array,
) -> FrankaPickPlaceTaskState:
    if reset_mask.shape != current.phase.shape:
        raise ValueError("Franka task reset mask has the wrong shape")
    reset = reset_mask.astype(mx.bool_)
    return FrankaPickPlaceTaskState(
        phase=mx.where(reset, replacement.phase, current.phase),
        phase_steps=mx.where(
            reset,
            replacement.phase_steps,
            current.phase_steps,
        ),
        grasp_steps=mx.where(
            reset,
            replacement.grasp_steps,
            current.grasp_steps,
        ),
        settle_steps=mx.where(
            reset,
            replacement.settle_steps,
            current.settle_steps,
        ),
        initial_object_position=mx.where(
            reset[:, None],
            replacement.initial_object_position,
            current.initial_object_position,
        ),
        previous_object_position=mx.where(
            reset[:, None],
            replacement.previous_object_position,
            current.previous_object_position,
        ),
    )


def _collider_range(
    collider: mx.array,
    first: int,
    count: int,
) -> mx.array:
    return (collider >= first) & (collider < first + count)


def _pair_matches(
    collider_a: mx.array,
    collider_b: mx.array,
    first: int,
    count: int,
    other: int,
) -> mx.array:
    return (
        _collider_range(collider_a, first, count)
        & (collider_b == other)
    ) | (
        _collider_range(collider_b, first, count)
        & (collider_a == other)
    )


def step_franka_pick_place_task(
    state: FrankaPickPlaceTaskState,
    sampled: SampledWorldFamilyState,
    contacts: ContactEvidence,
    *,
    physics_valid: mx.array,
    config: FrankaPickPlaceTaskConfig = FrankaPickPlaceTaskConfig(),
) -> tuple[FrankaPickPlaceTaskState, FrankaTaskEvidence]:
    """Advance evidence-based task phases without host inspection."""

    config.validate()
    environment_count = int(sampled.world.q.shape[0])
    if physics_valid.shape != (environment_count,):
        raise ValueError("physics_valid must be an environment vector")
    valid_contact = contacts.mask & physics_valid[:, None]
    collider_a = contacts.stable_ids[:, :, 0]
    collider_b = contacts.stable_ids[:, :, 1]
    left = valid_contact & _pair_matches(
        collider_a,
        collider_b,
        config.left_finger_first_collider,
        config.left_finger_collider_count,
        config.object_collider,
    )
    right = valid_contact & _pair_matches(
        collider_a,
        collider_b,
        config.right_finger_first_collider,
        config.right_finger_collider_count,
        config.object_collider,
    )
    bilateral = mx.any(left, axis=-1) & mx.any(right, axis=-1)
    grasp_steps = mx.where(
        bilateral,
        state.grasp_steps + mx.array(1, dtype=mx.uint32),
        mx.zeros_like(state.grasp_steps),
    )
    stable_grasp = grasp_steps >= config.stable_grasp_steps

    object_support = valid_contact & (
        (
            (collider_a == config.object_collider)
            & (
                (collider_b == config.ground_collider)
                | (collider_b == config.target_collider)
            )
        )
        | (
            (collider_b == config.object_collider)
            & (
                (collider_a == config.ground_collider)
                | (collider_a == config.target_collider)
            )
        )
    )
    support_contact = mx.any(object_support, axis=-1)
    object_position = sampled.world.scene_bodies.position[
        :, config.object_scene_index, :3
    ]
    object_velocity = sampled.world.scene_bodies.linear_velocity[
        :, config.object_scene_index, :3
    ]
    target_position = sampled.world.scene_bodies.position[
        :, config.target_scene_index, :3
    ]
    desired_position = target_position + mx.array(
        (0.0, 0.0, config.object_half_height),
        dtype=mx.float32,
    )
    target_delta = object_position - desired_position
    target_xy_distance = mx.sqrt(
        mx.sum(target_delta[:, :2] * target_delta[:, :2], axis=-1)
        + 1.0e-12
    )
    target_height_error = mx.abs(target_delta[:, 2])
    within_target = (
        (target_xy_distance < config.target_radius)
        & (target_height_error < config.target_height_tolerance)
    )
    lifted = (
        object_position[:, 2] - state.initial_object_position[:, 2]
        >= config.lift_height
    )
    gripper_aperture = sampled.world.q[:, 7] + sampled.world.q[:, 8]
    released = (
        (gripper_aperture >= config.release_aperture)
        & ~bilateral
    )
    object_speed = mx.sqrt(
        mx.sum(object_velocity * object_velocity, axis=-1) + 1.0e-12
    )
    settling = (
        (state.phase >= FrankaTaskPhase.release)
        &
        within_target
        & released
        & support_contact
        & (object_speed < config.settle_speed)
    )
    settle_steps = mx.where(
        settling,
        state.settle_steps + mx.array(1, dtype=mx.uint32),
        mx.zeros_like(state.settle_steps),
    )
    settled = settle_steps >= config.settle_steps

    tool_position = franka_kinematics(sampled.world.q).position
    tool_distance = mx.sqrt(
        mx.sum(
            (tool_position - object_position)
            * (tool_position - object_position),
            axis=-1,
        )
        + 1.0e-12
    )
    phase = state.phase
    next_phase = mx.where(
        (phase == FrankaTaskPhase.approach)
        & (tool_distance < config.pregrasp_distance),
        mx.array(FrankaTaskPhase.pregrasp, dtype=mx.uint32),
        phase,
    )
    next_phase = mx.where(
        (phase == FrankaTaskPhase.pregrasp)
        & (mx.any(left | right, axis=-1)),
        mx.array(FrankaTaskPhase.contact, dtype=mx.uint32),
        next_phase,
    )
    next_phase = mx.where(
        (phase == FrankaTaskPhase.contact) & bilateral,
        mx.array(FrankaTaskPhase.grasp, dtype=mx.uint32),
        next_phase,
    )
    next_phase = mx.where(
        (phase == FrankaTaskPhase.grasp) & stable_grasp,
        mx.array(FrankaTaskPhase.lift, dtype=mx.uint32),
        next_phase,
    )
    next_phase = mx.where(
        (phase == FrankaTaskPhase.lift) & lifted,
        mx.array(FrankaTaskPhase.transport, dtype=mx.uint32),
        next_phase,
    )
    next_phase = mx.where(
        (phase == FrankaTaskPhase.transport) & within_target,
        mx.array(FrankaTaskPhase.place, dtype=mx.uint32),
        next_phase,
    )
    next_phase = mx.where(
        (phase == FrankaTaskPhase.place) & released,
        mx.array(FrankaTaskPhase.release, dtype=mx.uint32),
        next_phase,
    )
    next_phase = mx.where(
        (phase == FrankaTaskPhase.release) & settling,
        mx.array(FrankaTaskPhase.settle, dtype=mx.uint32),
        next_phase,
    )
    phase = next_phase
    phase_changed = phase != state.phase
    phase_steps = mx.where(
        phase_changed,
        mx.zeros_like(state.phase_steps),
        state.phase_steps + mx.array(1, dtype=mx.uint32),
    )

    previous_target_delta = (
        state.previous_object_position - desired_position
    )
    previous_target_distance = mx.sqrt(
        mx.sum(
            previous_target_delta * previous_target_delta,
            axis=-1,
        )
        + 1.0e-12
    )
    target_distance = mx.sqrt(
        mx.sum(target_delta * target_delta, axis=-1) + 1.0e-12
    )
    task_margin = mx.minimum(
        config.target_radius - target_xy_distance,
        config.target_height_tolerance - target_height_error,
    )
    safety_margin = object_position[:, 2] + config.object_half_height
    success = (
        settled
        & physics_valid
        & (phase == FrankaTaskPhase.settle)
    )
    state_limit = safety_margin < 0.0
    failure_mask = (
        state_limit.astype(mx.uint32)
        | ((~physics_valid).astype(mx.uint32) << 1)
        | (
            (
                (phase >= FrankaTaskPhase.grasp)
                & (phase < FrankaTaskPhase.place)
                & ~bilateral
            ).astype(mx.uint32)
            << 2
        )
    )
    phase_progress = (
        phase.astype(mx.float32) - state.phase.astype(mx.float32)
    )
    reward = (
        2.0 * mx.exp(-12.0 * tool_distance)
        + 8.0 * (previous_target_distance - target_distance)
        + 0.75 * phase_progress
        + 0.25 * bilateral.astype(mx.float32)
        + 0.5 * stable_grasp.astype(mx.float32)
        + 0.75 * lifted.astype(mx.float32)
        + 1.0 * within_target.astype(mx.float32)
        + 5.0 * success.astype(mx.float32)
        - 2.0 * state_limit.astype(mx.float32)
    )
    reward = mx.where(
        physics_valid,
        reward,
        mx.zeros_like(reward),
    )
    next_state = FrankaPickPlaceTaskState(
        phase=phase,
        phase_steps=phase_steps,
        grasp_steps=grasp_steps,
        settle_steps=settle_steps,
        initial_object_position=state.initial_object_position,
        previous_object_position=object_position,
    )
    evidence = FrankaTaskEvidence(
        bilateral_contact=bilateral,
        stable_grasp=stable_grasp,
        lifted=lifted,
        within_target=within_target,
        released=released,
        support_contact=support_contact,
        settled=settled,
        success=success,
        task_margin=task_margin,
        safety_margin=safety_margin,
        reward=reward,
        failure_mask=failure_mask,
    )
    return next_state, evidence


def task_evidence_values(evidence: FrankaTaskEvidence) -> mx.array:
    """Pack deployable task evidence into an explicit float tensor."""

    return mx.stack(
        (
            evidence.bilateral_contact.astype(mx.float32),
            evidence.stable_grasp.astype(mx.float32),
            evidence.lifted.astype(mx.float32),
            evidence.within_target.astype(mx.float32),
            evidence.released.astype(mx.float32),
            evidence.support_contact.astype(mx.float32),
            evidence.settled.astype(mx.float32),
            evidence.task_margin,
            evidence.safety_margin,
        ),
        axis=-1,
    )


__all__ = [
    "AdaptedFrankaAction",
    "CartesianActionAdapterConfig",
    "FrankaActionAdapter",
    "FrankaActionMode",
    "FrankaKinematics",
    "FrankaPickPlaceTaskConfig",
    "FrankaPickPlaceTaskState",
    "FrankaTaskEvidence",
    "FrankaTaskPhase",
    "OperationalSpaceImpedanceConfig",
    "adapt_cartesian_action",
    "adapt_joint_position_action",
    "adapt_operational_space_action",
    "franka_kinematics",
    "initial_franka_pick_place_task",
    "select_franka_pick_place_task",
    "step_franka_pick_place_task",
    "task_evidence_values",
]
