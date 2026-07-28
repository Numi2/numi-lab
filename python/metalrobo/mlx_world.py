"""Pure-array MLX state and device-resident world stepping.

Both free-motion ABA and contact-capable simulation encode into MLX's active
Metal command encoder. Contact worlds carry every mutable manifold and convex
cache explicitly, so ``mx.compile`` sees a pure state transition rather than a
hidden native singleton.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from ._mlx_ext import (  # type: ignore[attr-defined]
    MLXCompiledWorld,
    MetalWorldCapacityProfile,
    aba_step as _aba_step,
    compile_world,
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


def step(
    world: MLXCompiledWorld,
    state: WorldState,
    actions: mx.array,
    *,
    reset_mask: mx.array | None = None,
    reset_state: WorldState | None = None,
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

    return StepOutput(
        next_state=next_state,
        observations=mx.concatenate(
            (next_state.q, next_state.v),
            axis=-1,
        ),
        contacts=contacts,
        sensors=mx.zeros(
            (environment_count, 0),
            dtype=mx.float32,
        ),
        status=status,
        physics_error=status[:, 0] != 0,
        acceleration=acceleration,
    )


__all__ = [
    "ContactEvidence",
    "MLXCompiledWorld",
    "MetalWorldCapacityProfile",
    "SceneBodyState",
    "SolverCache",
    "StepOutput",
    "WorldState",
    "compile_world",
    "initial_state",
    "step",
]
