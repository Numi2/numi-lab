"""Pure-array MLX state and device-resident world stepping.

Both free-motion ABA and contact-capable simulation encode into MLX's active
Metal command encoder. Contact worlds carry every mutable manifold and convex
cache explicitly, so ``mx.compile`` sees a pure state transition rather than a
hidden native singleton.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import mlx.core as mx

from ._mlx_ext import (  # type: ignore[attr-defined]
    MLXCompiledWorld,
    MetalWorldCapacityProfile,
    aba_step as _aba_step,
    compile_world,
    compile_world_pack,
    world_family_state as _world_family_state,
    world_step as _world_step,
)


class SceneBodyState(NamedTuple):
    """Environment-major semantic state for dynamic/kinematic scene bodies."""

    position: mx.array
    orientation: mx.array
    linear_velocity: mx.array
    angular_velocity: mx.array


class RodState(NamedTuple):
    """Explicit environment-major deforming-thread state."""

    position: mx.array
    velocity: mx.array
    twist: mx.array
    twist_rate: mx.array


class SolverCache(NamedTuple):
    """Explicit device arrays carried between pure physics calls."""

    manifold_headers: mx.array
    manifold_points: mx.array
    manifold_counts: mx.array
    pair_cache: mx.array
    rod_witnesses: mx.array


class TactileState(NamedTuple):
    """Explicit canonical tactile history carried through compiled rollouts."""

    previous_depth_m: mx.array
    previous_validity: mx.array
    previous_object_shape_ids: mx.array
    previous_tangential_motion: mx.array
    target_local_anchor: mx.array
    frame_index: mx.array
    time_seconds: mx.array


class ActuatorState(NamedTuple):
    """Explicit deterministic transmission and per-environment calibration."""

    effective_position_target: mx.array
    profile_values: mx.array


class WorldState(NamedTuple):
    """MLX PyTree for one environment-major world batch."""

    q: mx.array
    v: mx.array
    scene_bodies: SceneBodyState
    rods: RodState
    solver_cache: SolverCache
    tactile: TactileState
    actuators: ActuatorState


class ScenarioState(NamedTuple):
    """Explicit sampled scenario provenance carried beside world state."""

    headers: mx.array
    values: mx.array
    identities: mx.array


class WorldPhysicalParameters(NamedTuple):
    """Per-environment body and controller parameters from WorldProgram."""

    body_values: mx.array
    body_identities: mx.array
    controller_values: mx.array
    controller_identities: mx.array


class SampledWorldFamilyState(NamedTuple):
    """One pure MLX reset artifact with no hidden simulator state."""

    world: WorldState
    scenarios: ScenarioState
    parameters: WorldPhysicalParameters


class ControllerDelayState(NamedTuple):
    """Explicit device-resident command history for sampled controller delay."""

    history: mx.array


class ContactEvidence(NamedTuple):
    """Fixed-capacity device evidence; ``mask`` selects valid contact slots."""

    values: mx.array
    stable_ids: mx.array
    counts: mx.array
    mask: mx.array


class TactileSummary(NamedTuple):
    """Named physical reductions; no packed channel indices are public."""

    pose_position_and_timestamp: mx.array
    pose_orientation: mx.array
    net_force_and_contact_area: mx.array
    net_torque_and_maximum_depth: mx.array
    centroid_local_and_mean_depth: mx.array
    centroid_world_and_active_count: mx.array
    center_of_pressure_local_and_force_weight: mx.array
    center_of_pressure_world_and_contact_count: mx.array
    tangential_motion_and_friction: mx.array
    statistics_and_identity: mx.array


class ObjectContactPointSet(NamedTuple):
    """Fixed-capacity, masked object-local tactile geometry labels.

    These are geometric sample labels for field-estimator training. Exact
    solver points, forces, and impulses remain in ``ContactEvidence``.
    """

    object_shape_ids: mx.array
    object_local_point_and_depth: mx.array
    object_local_sensor_normal: mx.array
    mask: mx.array


class TactileObservation(NamedTuple):
    """Canonical metric tactile observation published beside physics state."""

    penetration_depth_m: mx.array
    depth_velocity_m_s: mx.array
    tangential_motion: mx.array
    validity: mx.array
    object_shape_ids: mx.array
    summary: TactileSummary
    object_contacts: ObjectContactPointSet
    status: mx.array


class StepOutput(NamedTuple):
    """Fixed-shape output suitable for ``mx.compile`` and lazy rollouts."""

    next_state: WorldState
    observations: mx.array
    contacts: ContactEvidence
    tactile: TactileObservation
    body_states: mx.array
    sensors: mx.array
    status: mx.array
    physics_error: mx.array
    acceleration: mx.array


class SampledWorldStepOutput(NamedTuple):
    """Physics result plus explicit scenario/parameter and delay state."""

    next_state: SampledWorldFamilyState
    delay_state: ControllerDelayState
    applied_actions: mx.array
    physics: StepOutput


def _broadcast_scene_default(
    values: list[float],
    environment_count: int,
    scene_body_count: int,
) -> mx.array:
    value = mx.array(values, dtype=mx.float32).reshape(
        (1, scene_body_count, 4)
    )
    return mx.broadcast_to(
        value,
        (environment_count, scene_body_count, 4),
    )


def _initial_rod_state(
    world: MLXCompiledWorld,
    environment_count: int,
) -> RodState:
    node_count = int(world.rod_node_count)
    edge_count = int(world.rod_edge_count)
    if node_count:
        position = mx.broadcast_to(
            mx.array(
                world.default_rod_positions,
                dtype=mx.float32,
            ).reshape((1, node_count, 4)),
            (environment_count, node_count, 4),
        )
        velocity = mx.broadcast_to(
            mx.array(
                world.default_rod_velocities,
                dtype=mx.float32,
            ).reshape((1, node_count, 4)),
            (environment_count, node_count, 4),
        )
    else:
        position = mx.zeros(
            (environment_count, 0, 4),
            dtype=mx.float32,
        )
        velocity = mx.zeros(
            (environment_count, 0, 4),
            dtype=mx.float32,
        )
    if edge_count:
        twist = mx.broadcast_to(
            mx.array(
                world.default_rod_twists,
                dtype=mx.float32,
            ).reshape((1, edge_count)),
            (environment_count, edge_count),
        )
        twist_rate = mx.broadcast_to(
            mx.array(
                world.default_rod_twist_rates,
                dtype=mx.float32,
            ).reshape((1, edge_count)),
            (environment_count, edge_count),
        )
    else:
        twist = mx.zeros(
            (environment_count, 0),
            dtype=mx.float32,
        )
        twist_rate = mx.zeros(
            (environment_count, 0),
            dtype=mx.float32,
        )
    return RodState(position, velocity, twist, twist_rate)


def _initial_tactile_state(
    world: MLXCompiledWorld,
    environment_count: int,
) -> TactileState:
    sample_count = int(world.tactile_sample_count)
    if sample_count:
        dense_shape = (environment_count, sample_count)
        motion_shape = (environment_count, sample_count, 4)
        clock_shape = (environment_count,)
    else:
        dense_shape = (environment_count, 0)
        motion_shape = (environment_count, 0, 4)
        clock_shape = (environment_count, 0)
    return TactileState(
        previous_depth_m=mx.zeros(
            dense_shape,
            dtype=mx.float32,
        ),
        previous_validity=mx.zeros(
            dense_shape,
            dtype=mx.uint32,
        ),
        previous_object_shape_ids=mx.full(
            dense_shape,
            0xFFFFFFFF,
            dtype=mx.uint32,
        ),
        previous_tangential_motion=mx.zeros(
            motion_shape,
            dtype=mx.float32,
        ),
        target_local_anchor=mx.zeros(
            motion_shape,
            dtype=mx.float32,
        ),
        frame_index=mx.zeros(
            clock_shape,
            dtype=mx.uint64,
        ),
        time_seconds=mx.zeros(
            clock_shape,
            dtype=mx.float32,
        ),
    )


def _initial_actuator_state(
    world: MLXCompiledWorld,
    environment_count: int,
) -> ActuatorState:
    velocity_count = int(world.nv)
    return ActuatorState(
        effective_position_target=mx.broadcast_to(
            mx.array(
                world.default_actuator_targets,
                dtype=mx.float32,
            ).reshape((1, velocity_count)),
            (environment_count, velocity_count),
        ),
        profile_values=mx.broadcast_to(
            mx.array(
                world.actuator_profile_values,
                dtype=mx.float32,
            ).reshape((1, velocity_count, 7)),
            (environment_count, velocity_count, 7),
        ),
    )


def materialize_correlated_actuator_profiles(
    world: MLXCompiledWorld,
    state: ActuatorState,
    profile_multiplier_bank: mx.array,
    selected_profiles: mx.array,
    *,
    reset_mask: mx.array | None = None,
) -> ActuatorState:
    """Select one correlated actuator profile per environment at reset.

    The bank has shape ``[profile, dof, 6]`` and multiplies, in order,
    torque constant, current limit, no-load speed, efficiency, backlash, and
    delay. Stall torque is then re-cooked as torque constant times current
    limit. Inactive/unidentified DoFs retain their neutral execution records.
    The selected values live in explicit MLX state and are consumed without a
    solver-loop profile branch.
    """

    environment_count = int(state.profile_values.shape[0])
    velocity_count = int(world.nv)
    if state.profile_values.shape != (
        environment_count,
        velocity_count,
        7,
    ):
        raise ValueError("actuator state does not match the compiled world")
    if (
        profile_multiplier_bank.ndim != 3
        or profile_multiplier_bank.shape[1:] != (
            velocity_count,
            6,
        )
        or int(profile_multiplier_bank.shape[0]) <= 0
    ):
        raise ValueError(
            "actuator profile bank must have shape [profile,dof,6]"
        )
    if selected_profiles.shape != (environment_count,):
        raise ValueError(
            "selected actuator profiles must have shape [environment]"
        )
    selected = mx.take(
        profile_multiplier_bank.astype(mx.float32),
        selected_profiles.astype(mx.uint32),
        axis=0,
    )
    base = mx.broadcast_to(
        mx.array(
            world.actuator_profile_values,
            dtype=mx.float32,
        ).reshape((1, velocity_count, 7)),
        (environment_count, velocity_count, 7),
    )
    positive = mx.maximum(selected[:, :, :3], 1.0e-6)
    motor = base[:, :, :3] * positive
    efficiency = mx.clip(
        base[:, :, 3:4]
        * mx.maximum(selected[:, :, 3:4], 1.0e-6),
        1.0e-6,
        1.0,
    )
    transmission = base[:, :, 4:6] * mx.maximum(
        selected[:, :, 4:6],
        0.0,
    )
    candidate = mx.concatenate(
        (
            motor,
            efficiency,
            transmission,
            motor[:, :, 0:1] * motor[:, :, 1:2],
        ),
        axis=-1,
    )
    active = (
        mx.array(
            world.actuator_profile_flags,
            dtype=mx.uint32,
        ).reshape((1, velocity_count, 1))
        & mx.array(1, dtype=mx.uint32)
    ) != 0
    candidate = mx.where(active, candidate, base)
    if reset_mask is not None:
        if reset_mask.shape != (environment_count,):
            raise ValueError(
                "actuator reset mask must have shape [environment]"
            )
        candidate = mx.where(
            reset_mask.astype(mx.bool_)[:, None, None],
            candidate,
            state.profile_values,
        )
    return ActuatorState(
        effective_position_target=state.effective_position_target,
        profile_values=candidate,
    )


def initial_state(
    world: MLXCompiledWorld,
    environment_count: int,
) -> WorldState:
    """Create compiled defaults and empty persistent caches as MLX arrays."""

    if environment_count <= 0:
        raise ValueError("environment_count must be positive")
    if environment_count > world.environment_capacity:
        raise ValueError(
            "environment_count exceeds the world's compiled capacity"
        )
    q = mx.broadcast_to(
        mx.array(world.default_q, dtype=mx.float32),
        (environment_count, world.nq),
    )
    v = mx.broadcast_to(
        mx.array(world.default_v, dtype=mx.float32),
        (environment_count, world.nv),
    )

    scene_count = int(world.scene_body_count)
    if scene_count:
        scene = SceneBodyState(
            position=_broadcast_scene_default(
                world.default_scene_positions,
                environment_count,
                scene_count,
            ),
            orientation=_broadcast_scene_default(
                world.default_scene_orientations,
                environment_count,
                scene_count,
            ),
            linear_velocity=_broadcast_scene_default(
                world.default_scene_linear_velocities,
                environment_count,
                scene_count,
            ),
            angular_velocity=_broadcast_scene_default(
                world.default_scene_angular_velocities,
                environment_count,
                scene_count,
            ),
        )
    else:
        empty_scene = mx.zeros(
            (environment_count, 0, 4),
            dtype=mx.float32,
        )
        scene = SceneBodyState(
            empty_scene,
            empty_scene,
            empty_scene,
            empty_scene,
        )

    if world.contact_supported:
        manifold_count = int(world.manifold_capacity)
        solver_cache = SolverCache(
            manifold_headers=mx.zeros(
                (
                    environment_count,
                    manifold_count,
                    int(world.manifold_header_words),
                ),
                dtype=mx.uint32,
            ),
            manifold_points=mx.zeros(
                (
                    environment_count,
                    manifold_count,
                    int(world.manifold_point_capacity),
                    int(world.manifold_point_words),
                ),
                dtype=mx.uint32,
            ),
            manifold_counts=mx.zeros(
                (environment_count,),
                dtype=mx.uint32,
            ),
            pair_cache=mx.zeros(
                (
                    environment_count,
                    int(world.pair_cache_capacity),
                    int(world.pair_cache_words),
                ),
                dtype=mx.uint32,
            ),
            rod_witnesses=mx.zeros(
                (
                    environment_count,
                    int(world.rod_witness_capacity),
                    int(world.rod_witness_words),
                ),
                dtype=mx.uint32,
            ),
        )
    else:
        empty_u32 = mx.zeros(
            (environment_count, 0),
            dtype=mx.uint32,
        )
        solver_cache = SolverCache(
            empty_u32,
            empty_u32,
            empty_u32,
            empty_u32,
            empty_u32,
        )
    return WorldState(
        q=q,
        v=v,
        scene_bodies=scene,
        rods=_initial_rod_state(world, environment_count),
        solver_cache=solver_cache,
        tactile=_initial_tactile_state(
            world,
            environment_count,
        ),
        actuators=_initial_actuator_state(
            world,
            environment_count,
        ),
    )

def sampled_state_from_world_family(
    world: MLXCompiledWorld,
    family: Any,
    *,
    stream: mx.Stream | mx.Device | None = None,
) -> SampledWorldFamilyState:
    """Import reset, scenario, and physical state on MLX's active encoder.

    ``family`` is a sampled ``FrankaPickPlaceWorldFamily``. Its private Metal
    reset buffers are retained by the lazy MLX primitive until evaluation, so
    this path performs no NumPy conversion or CPU readback.
    """

    layout = family.layout
    environment_count = int(layout.active_instance_count)
    if environment_count <= 0:
        raise ValueError("sample the world family before importing its state")
    expected = (
        int(world.nq),
        int(world.nv),
        # The family and parameter buffers address the complete EngineModel.
        # This remains stable if CompiledWorld.body_count changes from
        # articulation-local to whole-world semantics.
        int(world.model_body_count),
        int(world.scene_body_count),
    )
    actual = (
        int(layout.nq),
        int(layout.nv),
        int(layout.body_count),
        int(layout.scene_body_count),
    )
    compiled_pack_hash = int(world.authored_pack_hash)
    sampled_pack_hash = int(family.authored_pack_hash)
    if compiled_pack_hash != sampled_pack_hash:
        raise ValueError(
            "sampled world family and compiled MLX physics must use "
            "the same authored pack"
        )
    if actual != expected:
        raise ValueError(
            "world-family topology does not match the compiled MLX world"
        )
    buffers = family.device_buffers
    generation = int(family.stats.sample_count)
    (
        q,
        v,
        position,
        orientation,
        linear_velocity,
        angular_velocity,
        scenario_headers,
        scenario_values,
        scenario_identities,
        body_values,
        body_identities,
        controller_values,
        controller_identities,
    ) = _world_family_state(
        world,
        int(buffers.reset_q),
        int(buffers.reset_v),
        int(buffers.reset_scene_bodies),
        int(buffers.scenario_headers),
        int(buffers.scenario_values),
        int(buffers.body_parameters),
        int(buffers.controller_parameters),
        environment_count,
        int(layout.variation_count),
        int(layout.body_count),
        int(layout.articulation_count),
        generation,
        sampled_pack_hash,
        stream=stream,
    )

    manifold_count = int(world.manifold_capacity)
    solver_cache = SolverCache(
        manifold_headers=mx.zeros(
            (
                environment_count,
                manifold_count,
                int(world.manifold_header_words),
            ),
            dtype=mx.uint32,
        ),
        manifold_points=mx.zeros(
            (
                environment_count,
                manifold_count,
                int(world.manifold_point_capacity),
                int(world.manifold_point_words),
            ),
            dtype=mx.uint32,
        ),
        manifold_counts=mx.zeros(
            (environment_count,),
            dtype=mx.uint32,
        ),
        pair_cache=mx.zeros(
            (
                environment_count,
                int(world.pair_cache_capacity),
                int(world.pair_cache_words),
            ),
            dtype=mx.uint32,
        ),
        rod_witnesses=mx.zeros(
            (
                environment_count,
                int(world.rod_witness_capacity),
                int(world.rod_witness_words),
            ),
            dtype=mx.uint32,
        ),
    )
    return SampledWorldFamilyState(
        world=WorldState(
            q=q,
            v=v,
            scene_bodies=SceneBodyState(
                position,
                orientation,
                linear_velocity,
                angular_velocity,
            ),
            rods=_initial_rod_state(world, environment_count),
            solver_cache=solver_cache,
            tactile=_initial_tactile_state(
                world,
                environment_count,
            ),
            actuators=_initial_actuator_state(
                world,
                environment_count,
            ),
        ),
        scenarios=ScenarioState(
            headers=scenario_headers,
            values=scenario_values,
            identities=scenario_identities,
        ),
        parameters=WorldPhysicalParameters(
            body_values=body_values,
            body_identities=body_identities,
            controller_values=controller_values,
            controller_identities=controller_identities,
        ),
    )


def initial_state_from_world_family(
    world: MLXCompiledWorld,
    family: Any,
    *,
    stream: mx.Stream | mx.Device | None = None,
) -> WorldState:
    """Compatibility view returning only the explicit physics reset state."""

    return sampled_state_from_world_family(
        world,
        family,
        stream=stream,
    ).world


def _select_reset(
    reset_mask: mx.array,
    reset_value: mx.array,
    current_value: mx.array,
) -> mx.array:
    shape = (int(reset_mask.shape[0]),) + (1,) * (
        current_value.ndim - 1
    )
    return mx.where(
        reset_mask.astype(mx.bool_).reshape(shape),
        reset_value,
        current_value,
    )


def reset_sampled_world_family(
    current: SampledWorldFamilyState,
    replacement: SampledWorldFamilyState,
    reset_mask: mx.array,
) -> SampledWorldFamilyState:
    """Select independent family resets entirely as MLX array operations."""

    return SampledWorldFamilyState(
        world=WorldState(
            q=_select_reset(
                reset_mask,
                replacement.world.q,
                current.world.q,
            ),
            v=_select_reset(
                reset_mask,
                replacement.world.v,
                current.world.v,
            ),
            scene_bodies=SceneBodyState(
                *(
                    _select_reset(reset_mask, new, old)
                    for new, old in zip(
                        replacement.world.scene_bodies,
                        current.world.scene_bodies,
                        strict=True,
                    )
                )
            ),
            rods=RodState(
                *(
                    _select_reset(reset_mask, new, old)
                    for new, old in zip(
                        replacement.world.rods,
                        current.world.rods,
                        strict=True,
                    )
                )
            ),
            solver_cache=SolverCache(
                *(
                    _select_reset(reset_mask, new, old)
                    for new, old in zip(
                        replacement.world.solver_cache,
                        current.world.solver_cache,
                        strict=True,
                    )
                )
            ),
            tactile=TactileState(
                *(
                    _select_reset(reset_mask, new, old)
                    for new, old in zip(
                        replacement.world.tactile,
                        current.world.tactile,
                        strict=True,
                    )
                )
            ),
            actuators=ActuatorState(
                *(
                    _select_reset(reset_mask, new, old)
                    for new, old in zip(
                        replacement.world.actuators,
                        current.world.actuators,
                        strict=True,
                    )
                )
            ),
        ),
        scenarios=ScenarioState(
            *(
                _select_reset(reset_mask, new, old)
                for new, old in zip(
                    replacement.scenarios,
                    current.scenarios,
                    strict=True,
                )
            )
        ),
        parameters=WorldPhysicalParameters(
            *(
                _select_reset(reset_mask, new, old)
                for new, old in zip(
                    replacement.parameters,
                    current.parameters,
                    strict=True,
                )
            )
        ),
    )


def step(
    world: MLXCompiledWorld,
    state: WorldState,
    actions: mx.array,
    *,
    reset_mask: mx.array | None = None,
    reset_state: WorldState | None = None,
    body_parameters: mx.array | None = None,
    controller_parameters: mx.array | None = None,
    stream: mx.Stream | mx.Device | None = None,
) -> StepOutput:
    """Advance one transactional control step without host synchronization.

    Autodiff and ``vmap`` through physics are intentionally unsupported.
    Environment batching is the native first array dimension.
    """

    q = state.q
    v = state.v
    scene = state.scene_bodies
    rods = state.rods
    cache = state.solver_cache
    tactile_state = state.tactile
    actuator_state = state.actuators
    tactile_reset_mask = (
        reset_mask.astype(mx.uint32)
        if reset_mask is not None
        else mx.zeros((int(q.shape[0]),), dtype=mx.uint32)
    )
    if reset_mask is not None:
        replacement = (
            reset_state
            if reset_state is not None
            else initial_state(world, int(q.shape[0]))
        )
        q = _select_reset(reset_mask, replacement.q, q)
        v = _select_reset(reset_mask, replacement.v, v)
        scene = SceneBodyState(
            *(
                _select_reset(reset_mask, new, old)
                for new, old in zip(
                    replacement.scene_bodies,
                    scene,
                    strict=True,
                )
            )
        )
        rods = RodState(
            *(
                _select_reset(reset_mask, new, old)
                for new, old in zip(
                    replacement.rods,
                    rods,
                    strict=True,
                )
            )
        )
        cache = SolverCache(
            *(
                _select_reset(reset_mask, new, old)
                for new, old in zip(
                    replacement.solver_cache,
                    cache,
                    strict=True,
                )
            )
        )
        tactile_state = TactileState(
            *(
                _select_reset(reset_mask, new, old)
                for new, old in zip(
                    replacement.tactile,
                    tactile_state,
                    strict=True,
                )
            )
        )
        actuator_state = ActuatorState(
            *(
                _select_reset(reset_mask, new, old)
                for new, old in zip(
                    replacement.actuators,
                    actuator_state,
                    strict=True,
                )
            )
        )

    environment_count = int(q.shape[0])
    if body_parameters is None:
        body_parameters = mx.ones(
            (
                environment_count,
                int(world.model_body_count),
                4,
            ),
            dtype=mx.float32,
        )
    if controller_parameters is None:
        controller_parameters = mx.broadcast_to(
            mx.array(
                [1.0, 1.0, 0.0, 1.0],
                dtype=mx.float32,
            ).reshape((1, 1, 4)),
            (
                environment_count,
                int(world.articulation_count),
                4,
            ),
        )
    if world.actuation_mode == "implicit_position":
        half_backlash = 0.5 * actuator_state.profile_values[:, :, 4]
        actions = mx.clip(
            actuator_state.effective_position_target,
            actions - half_backlash,
            actions + half_backlash,
        )
        next_actuator_state = ActuatorState(
            effective_position_target=actions,
            profile_values=actuator_state.profile_values,
        )
    else:
        next_actuator_state = actuator_state
    if world.contact_supported:
        tactile_timestamp = (
            tactile_state.time_seconds +
            float(world.control_timestep)
        )
        (
            next_q,
            next_v,
            next_position,
            next_orientation,
            next_linear_velocity,
            next_angular_velocity,
            next_manifold_headers,
            next_manifold_points,
            next_manifold_counts,
            next_pair_cache,
            next_rod_position,
            next_rod_velocity,
            next_rod_twist,
            next_rod_twist_rate,
            next_rod_witnesses,
            acceleration,
            status,
            contact_values,
            contact_ids,
            contact_counts,
            contact_mask,
            tactile_depth,
            tactile_depth_velocity,
            tactile_motion,
            tactile_validity,
            tactile_object_ids,
            tactile_anchor,
            tactile_pose_position,
            tactile_pose_orientation,
            tactile_force,
            tactile_torque,
            tactile_centroid_local,
            tactile_centroid_world,
            tactile_cop_local,
            tactile_cop_world,
            tactile_motion_summary,
            tactile_statistics,
            tactile_status,
            tactile_object_local_points,
            tactile_object_local_normals,
            tactile_object_contact_mask,
            body_states,
        ) = _world_step(
            world,
            q,
            v,
            actions,
            scene.position,
            scene.orientation,
            scene.linear_velocity,
            scene.angular_velocity,
            cache.manifold_headers,
            cache.manifold_points,
            cache.manifold_counts,
            cache.pair_cache,
            rods.position,
            rods.velocity,
            rods.twist,
            rods.twist_rate,
            cache.rod_witnesses,
            body_parameters,
            controller_parameters,
            tactile_state.previous_depth_m,
            tactile_state.previous_validity,
            tactile_state.previous_object_shape_ids,
            tactile_state.previous_tangential_motion,
            tactile_state.target_local_anchor,
            tactile_state.frame_index,
            tactile_timestamp,
            tactile_reset_mask,
            actuator_state.profile_values,
            stream=stream,
        )
        next_tactile_state = TactileState(
            previous_depth_m=tactile_depth,
            previous_validity=tactile_validity,
            previous_object_shape_ids=tactile_object_ids,
            previous_tangential_motion=tactile_motion,
            target_local_anchor=tactile_anchor,
            frame_index=tactile_state.frame_index + 1,
            time_seconds=tactile_timestamp,
        )
        next_state = WorldState(
            q=next_q,
            v=next_v,
            scene_bodies=SceneBodyState(
                next_position,
                next_orientation,
                next_linear_velocity,
                next_angular_velocity,
            ),
            rods=RodState(
                next_rod_position,
                next_rod_velocity,
                next_rod_twist,
                next_rod_twist_rate,
            ),
            solver_cache=SolverCache(
                next_manifold_headers,
                next_manifold_points,
                next_manifold_counts,
                next_pair_cache,
                next_rod_witnesses,
            ),
            tactile=next_tactile_state,
            actuators=next_actuator_state,
        )
        contacts = ContactEvidence(
            contact_values,
            contact_ids,
            contact_counts,
            contact_mask.astype(mx.bool_),
        )
        tactile = TactileObservation(
            penetration_depth_m=tactile_depth,
            depth_velocity_m_s=tactile_depth_velocity,
            tangential_motion=tactile_motion,
            validity=tactile_validity,
            object_shape_ids=tactile_object_ids,
            summary=TactileSummary(
                pose_position_and_timestamp=tactile_pose_position,
                pose_orientation=tactile_pose_orientation,
                net_force_and_contact_area=tactile_force,
                net_torque_and_maximum_depth=tactile_torque,
                centroid_local_and_mean_depth=tactile_centroid_local,
                centroid_world_and_active_count=tactile_centroid_world,
                center_of_pressure_local_and_force_weight=tactile_cop_local,
                center_of_pressure_world_and_contact_count=tactile_cop_world,
                tangential_motion_and_friction=tactile_motion_summary,
                statistics_and_identity=tactile_statistics,
            ),
            object_contacts=ObjectContactPointSet(
                object_shape_ids=tactile_object_ids,
                object_local_point_and_depth=(
                    tactile_object_local_points
                ),
                object_local_sensor_normal=(
                    tactile_object_local_normals
                ),
                mask=tactile_object_contact_mask.astype(
                    mx.bool_
                ),
            ),
            status=tactile_status,
        )
    else:
        next_q, next_v, acceleration, status = _aba_step(
            world,
            q,
            v,
            actions,
            stream=stream,
        )
        next_state = WorldState(
            q=next_q,
            v=next_v,
            scene_bodies=scene,
            rods=rods,
            solver_cache=cache,
            tactile=tactile_state,
            actuators=next_actuator_state,
        )
        contacts = ContactEvidence(
            mx.zeros(
                (environment_count, 0, 16),
                dtype=mx.float32,
            ),
            mx.zeros(
                (environment_count, 0, 4),
                dtype=mx.uint32,
            ),
            mx.zeros(
                (environment_count,),
                dtype=mx.uint32,
            ),
            mx.zeros(
                (environment_count, 0),
                dtype=mx.bool_,
            ),
        )
        sensor_count = int(world.tactile_sensor_count)
        sample_count = int(world.tactile_sample_count)
        empty_dense = mx.zeros(
            (environment_count, sample_count),
            dtype=mx.float32,
        )
        empty_motion = mx.zeros(
            (environment_count, sample_count, 4),
            dtype=mx.float32,
        )
        empty_summary = mx.zeros(
            (environment_count, sensor_count, 4),
            dtype=mx.float32,
        )
        tactile = TactileObservation(
            penetration_depth_m=empty_dense,
            depth_velocity_m_s=empty_dense,
            tangential_motion=empty_motion,
            validity=mx.zeros(
                (environment_count, sample_count),
                dtype=mx.uint32,
            ),
            object_shape_ids=mx.full(
                (environment_count, sample_count),
                0xFFFFFFFF,
                dtype=mx.uint32,
            ),
            summary=TactileSummary(
                *([empty_summary] * 9),
                mx.zeros(
                    (environment_count, sensor_count, 4),
                    dtype=mx.uint32,
                ),
            ),
            object_contacts=ObjectContactPointSet(
                object_shape_ids=mx.full(
                    (environment_count, sample_count),
                    0xFFFFFFFF,
                    dtype=mx.uint32,
                ),
                object_local_point_and_depth=empty_motion,
                object_local_sensor_normal=empty_motion,
                mask=mx.zeros(
                    (environment_count, sample_count),
                    dtype=mx.bool_,
                ),
            ),
            status=mx.zeros(
                (environment_count, sensor_count, 8),
                dtype=mx.uint32,
            ),
        )
        body_states = mx.zeros(
            (environment_count, 0, 32),
            dtype=mx.uint32,
        )

    if world.floating_root or world.contact_supported:
        valid_contacts = contacts.mask.astype(mx.float32)
        normal_load = (
            mx.sum(
                contacts.values[:, :, 7] * valid_contacts,
                axis=-1,
                keepdims=True,
            )
            / float(world.control_timestep)
        )
        active_contact_count = mx.sum(
            valid_contacts,
            axis=-1,
            keepdims=True,
        )
        sensors = mx.concatenate(
            (
                (
                    acceleration[:, :6]
                    if world.floating_root
                    else mx.zeros(
                        (environment_count, 6),
                        dtype=mx.float32,
                    )
                ),
                normal_load,
                active_contact_count,
            ),
            axis=-1,
        )
    else:
        sensors = mx.zeros(
            (environment_count, 0),
            dtype=mx.float32,
        )

    return StepOutput(
        next_state=next_state,
        observations=mx.concatenate(
            (next_state.q, next_state.v),
            axis=-1,
        ),
        contacts=contacts,
        tactile=tactile,
        body_states=body_states,
        sensors=sensors,
        status=status,
        physics_error=status[:, 0] != 0,
        acceleration=acceleration,
    )


def initial_controller_delay_state(
    environment_count: int,
    action_count: int,
    *,
    maximum_delay_steps: int,
) -> ControllerDelayState:
    """Allocate a fixed command ring carried explicitly through MLX graphs."""

    if (
        environment_count <= 0
        or action_count <= 0
        or maximum_delay_steps < 0
    ):
        raise ValueError("controller-delay dimensions are invalid")
    return ControllerDelayState(
        history=mx.zeros(
            (
                environment_count,
                maximum_delay_steps + 1,
                action_count,
            ),
            dtype=mx.float32,
        )
    )


def step_sampled_world_family(
    world: MLXCompiledWorld,
    state: SampledWorldFamilyState,
    actions: mx.array,
    delay_state: ControllerDelayState,
    *,
    control_period_seconds: float,
    stream: mx.Stream | mx.Device | None = None,
) -> SampledWorldStepOutput:
    """Advance a sampled world with causal physics and controller variation.

    Mass/inertia, friction, restitution, damping, gains, and damping gains are
    consumed by ``world_step``. This adapter additionally turns the sampled
    controller and authored per-DoF actuator latency into an explicit
    per-environment command delay.
    """

    if control_period_seconds <= 0.0:
        raise ValueError("control_period_seconds must be positive")
    environment_count = int(state.world.q.shape[0])
    if (
        actions.ndim != 2
        or int(actions.shape[0]) != environment_count
        or delay_state.history.ndim != 3
        or int(delay_state.history.shape[0]) != environment_count
        or int(delay_state.history.shape[2]) != int(actions.shape[1])
    ):
        raise ValueError("actions and controller delay state do not match")
    next_history = mx.concatenate(
        (
            actions.astype(mx.float32)[:, None, :],
            delay_state.history[:, :-1, :],
        ),
        axis=1,
    )
    maximum_delay = int(next_history.shape[1]) - 1
    if int(state.parameters.controller_values.shape[1]) > 0:
        latency = state.parameters.controller_values[:, 0, 2]
    else:
        latency = mx.zeros(
            (environment_count,),
            dtype=mx.float32,
        )
    latency = (
        latency[:, None] +
        state.world.actuators.profile_values[:, :, 5]
    )
    delay_steps = mx.clip(
        mx.floor(
            latency / float(control_period_seconds) + 0.5
        ),
        0.0,
        float(maximum_delay),
    ).astype(mx.uint32)
    applied_actions = mx.take_along_axis(
        next_history,
        delay_steps[:, None, :],
        axis=1,
    )[:, 0, :]
    physics = step(
        world,
        state.world,
        applied_actions,
        body_parameters=state.parameters.body_values,
        controller_parameters=state.parameters.controller_values,
        stream=stream,
    )
    return SampledWorldStepOutput(
        next_state=SampledWorldFamilyState(
            world=physics.next_state,
            scenarios=state.scenarios,
            parameters=state.parameters,
        ),
        delay_state=ControllerDelayState(next_history),
        applied_actions=applied_actions,
        physics=physics,
    )


__all__ = [
    "ActuatorState",
    "ContactEvidence",
    "ControllerDelayState",
    "MLXCompiledWorld",
    "MetalWorldCapacityProfile",
    "ObjectContactPointSet",
    "RodState",
    "SceneBodyState",
    "ScenarioState",
    "SampledWorldFamilyState",
    "SampledWorldStepOutput",
    "SolverCache",
    "StepOutput",
    "TactileObservation",
    "TactileState",
    "TactileSummary",
    "WorldState",
    "WorldPhysicalParameters",
    "compile_world",
    "compile_world_pack",
    "initial_state",
    "initial_controller_delay_state",
    "initial_state_from_world_family",
    "materialize_correlated_actuator_profiles",
    "reset_sampled_world_family",
    "sampled_state_from_world_family",
    "step",
    "step_sampled_world_family",
]
