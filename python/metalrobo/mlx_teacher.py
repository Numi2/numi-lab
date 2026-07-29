"""Privileged, policy-independent Franka demonstration retargeting on MLX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx

from .mlx_manipulation import (
    CartesianActionAdapterConfig,
    FrankaPickPlaceTaskState,
    FrankaTaskPhase,
    franka_kinematics,
)
from .mlx_world import SampledWorldFamilyState


class PrivilegedTeacherAction(NamedTuple):
    normalized_action: mx.array
    target_position: mx.array
    detour_active: mx.array


@dataclass(frozen=True, slots=True)
class FrankaPrivilegedTeacherConfig:
    object_scene_index: int = 0
    target_scene_index: int = 2
    clutter_scene_index: int = 3
    pregrasp_height: float = 0.10
    grasp_height: float = 0.025
    lift_clearance: float = 0.12
    transport_clearance: float = 0.14
    place_clearance: float = 0.065
    clutter_clearance: float = 0.11
    waypoint_tolerance: float = 0.025

    def validate(self) -> None:
        if (
            min(
                self.object_scene_index,
                self.target_scene_index,
                self.clutter_scene_index,
            )
            < 0
            or self.pregrasp_height <= 0.0
            or self.grasp_height <= 0.0
            or self.lift_clearance <= self.grasp_height
            or self.transport_clearance <= self.grasp_height
            or self.place_clearance <= self.grasp_height
            or self.clutter_clearance <= 0.0
            or self.waypoint_tolerance <= 0.0
        ):
            raise ValueError("privileged-teacher configuration is invalid")


def _select_vector(
    condition: mx.array,
    selected: mx.array,
    otherwise: mx.array,
) -> mx.array:
    return mx.where(condition[:, None], selected, otherwise)


def _collision_aware_waypoint(
    start: mx.array,
    goal: mx.array,
    clutter: mx.array,
    *,
    clearance: float,
) -> tuple[mx.array, mx.array]:
    path_xy = goal[:, :2] - start[:, :2]
    path_length_squared = mx.maximum(
        mx.sum(path_xy * path_xy, axis=-1),
        1.0e-8,
    )
    clutter_delta = clutter[:, :2] - start[:, :2]
    projection = mx.clip(
        mx.sum(clutter_delta * path_xy, axis=-1)
        / path_length_squared,
        0.0,
        1.0,
    )
    closest = start[:, :2] + projection[:, None] * path_xy
    separation = clutter[:, :2] - closest
    distance = mx.sqrt(
        mx.sum(separation * separation, axis=-1) + 1.0e-12
    )
    between = (projection > 0.05) & (projection < 0.95)
    blocked = between & (distance < clearance)

    cross = (
        path_xy[:, 0] * clutter_delta[:, 1]
        - path_xy[:, 1] * clutter_delta[:, 0]
    )
    side = mx.where(cross >= 0.0, -1.0, 1.0)
    path_length = mx.sqrt(path_length_squared)
    perpendicular = mx.stack(
        (
            -path_xy[:, 1] / path_length,
            path_xy[:, 0] / path_length,
        ),
        axis=-1,
    )
    waypoint_xy = (
        clutter[:, :2]
        + side[:, None] * clearance * perpendicular
    )
    waypoint_z = mx.maximum(
        mx.maximum(start[:, 2], goal[:, 2]),
        clutter[:, 2] + clearance,
    )
    waypoint = mx.concatenate(
        (waypoint_xy, waypoint_z[:, None]),
        axis=-1,
    )
    return waypoint, blocked


def privileged_franka_teacher_action(
    sampled: SampledWorldFamilyState,
    task: FrankaPickPlaceTaskState,
    *,
    config: FrankaPrivilegedTeacherConfig = (
        FrankaPrivilegedTeacherConfig()
    ),
    action_config: CartesianActionAdapterConfig = (
        CartesianActionAdapterConfig()
    ),
) -> PrivilegedTeacherAction:
    """Retarget one pick-place demonstration segment per environment.

    The teacher uses privileged object, target, clutter, and task-phase state.
    It computes a deterministic clearance waypoint when the direct Cartesian
    segment intersects clutter. Every emitted action still has to complete the
    real contact task through physics before a demonstration is retained.
    """

    config.validate()
    action_config.validate()
    scene = sampled.world.scene_bodies.position
    if int(scene.shape[1]) <= config.clutter_scene_index:
        raise ValueError("privileged teacher requires the Franka scene cohort")
    object_position = scene[:, config.object_scene_index, :3]
    target_position = scene[:, config.target_scene_index, :3]
    clutter_position = scene[:, config.clutter_scene_index, :3]
    tool_position = franka_kinematics(sampled.world.q).position
    vertical = mx.array((0.0, 0.0, 1.0), dtype=mx.float32)

    approach = (
        object_position + config.pregrasp_height * vertical[None, :]
    )
    grasp = object_position + config.grasp_height * vertical[None, :]
    lift = (
        task.initial_object_position
        + config.lift_clearance * vertical[None, :]
    )
    transport = (
        target_position + config.transport_clearance * vertical[None, :]
    )
    place = target_position + config.place_clearance * vertical[None, :]

    phase = task.phase
    desired = approach
    desired = _select_vector(
        phase == FrankaTaskPhase.pregrasp,
        grasp,
        desired,
    )
    desired = _select_vector(
        (phase == FrankaTaskPhase.contact)
        | (phase == FrankaTaskPhase.grasp),
        grasp,
        desired,
    )
    desired = _select_vector(
        phase == FrankaTaskPhase.lift,
        lift,
        desired,
    )
    desired = _select_vector(
        phase == FrankaTaskPhase.transport,
        transport,
        desired,
    )
    desired = _select_vector(
        phase >= FrankaTaskPhase.place,
        place,
        desired,
    )

    waypoint, blocked = _collision_aware_waypoint(
        tool_position,
        desired,
        clutter_position,
        clearance=config.clutter_clearance,
    )
    detour_phase = (
        (phase == FrankaTaskPhase.approach)
        | (phase == FrankaTaskPhase.lift)
        | (phase == FrankaTaskPhase.transport)
    )
    detour_active = blocked & detour_phase
    waypoint_distance = mx.sqrt(
        mx.sum(
            (tool_position - waypoint)
            * (tool_position - waypoint),
            axis=-1,
        )
        + 1.0e-12
    )
    detour_active &= waypoint_distance > config.waypoint_tolerance
    desired = _select_vector(detour_active, waypoint, desired)

    translation = mx.clip(
        (desired - tool_position) / action_config.translation_scale,
        -1.0,
        1.0,
    )
    rotation = mx.zeros_like(translation)
    gripper_open = (
        (phase < FrankaTaskPhase.contact)
        | (phase >= FrankaTaskPhase.release)
    )
    gripper = mx.where(
        gripper_open,
        mx.ones(phase.shape, dtype=mx.float32),
        -mx.ones(phase.shape, dtype=mx.float32),
    )
    return PrivilegedTeacherAction(
        normalized_action=mx.concatenate(
            (translation, rotation, gripper[:, None]),
            axis=-1,
        ),
        target_position=desired,
        detour_active=detour_active,
    )


__all__ = [
    "FrankaPrivilegedTeacherConfig",
    "PrivilegedTeacherAction",
    "privileged_franka_teacher_action",
]
