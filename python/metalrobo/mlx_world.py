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
    world_family_state as _world_family_state,
    world_step as _world_step,
)


class SceneBodyState(NamedTuple):
    """Environment-major semantic state for dynamic/kinematic scene bodies."""

    position: mx.array
    orientation: mx.array
    linear_velocity: mx.array
    angular_velocity: mx.array


class SolverCache(NamedTuple):
    """Explicit device arrays carried between pure physics calls."""

    manifold_headers: mx.array
    manifold_points: mx.array
    manifold_counts: mx.array
    pair_cache: mx.array


class WorldState(NamedTuple):
    """MLX PyTree for one environment-major world batch."""

    q: mx.array
    v: mx.array
    scene_bodies: SceneBodyState
    solver_cache: SolverCache


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


class StepOutput(NamedTuple):
    """Fixed-shape output suitable for ``mx.compile`` and lazy rollouts."""

    next_state: WorldState
    observations: mx.array
    contacts: ContactEvidence
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
        )
    return WorldState(
        q=q,
        v=v,
        scene_bodies=scene,
        solver_cache=solver_cache,
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
        int(world.body_count) + int(world.scene_body_count),
        int(world.scene_body_count),
    )
    actual = (
        int(layout.nq),
        int(layout.nv),
        int(layout.body_count),
        int(layout.scene_body_count),
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
            solver_cache=solver_cache,
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
    cache = state.solver_cache
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
    if world.contact_supported:
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
            acceleration,
            status,
            contact_values,
            contact_ids,
            contact_counts,
            contact_mask,
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
            body_parameters,
            controller_parameters,
            stream=stream,
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
            solver_cache=SolverCache(
                next_manifold_headers,
                next_manifold_points,
                next_manifold_counts,
                next_pair_cache,
            ),
        )
        contacts = ContactEvidence(
            contact_values,
            contact_ids,
            contact_counts,
            contact_mask.astype(mx.bool_),
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
            solver_cache=cache,
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
    controller latency into an explicit per-environment command delay.
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
    delay_steps = mx.clip(
        mx.floor(
            latency / float(control_period_seconds) + 0.5
        ),
        0.0,
        float(maximum_delay),
    ).astype(mx.uint32)
    applied_actions = mx.take_along_axis(
        next_history,
        delay_steps[:, None, None],
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
    "ContactEvidence",
    "ControllerDelayState",
    "MLXCompiledWorld",
    "MetalWorldCapacityProfile",
    "SceneBodyState",
    "ScenarioState",
    "SampledWorldFamilyState",
    "SampledWorldStepOutput",
    "SolverCache",
    "StepOutput",
    "WorldState",
    "WorldPhysicalParameters",
    "compile_world",
    "initial_state",
    "initial_controller_delay_state",
    "initial_state_from_world_family",
    "reset_sampled_world_family",
    "sampled_state_from_world_family",
    "step",
    "step_sampled_world_family",
]
