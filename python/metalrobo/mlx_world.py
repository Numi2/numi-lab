"""Pure-array MLX state and stepping API.

The native extension encodes physics into MLX's active Metal command encoder.
It never creates, commits, or waits on a second command buffer. The current
primitive closes the device-native free-motion ABA path for Franka and G1;
contact cache fields are explicit and empty until the contact encoder adapter
is promoted to this API.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from ._mlx_ext import (  # type: ignore[attr-defined]
    MLXCompiledWorld,
    aba_step as _aba_step,
    compile_world,
)


class SolverCache(NamedTuple):
    """Explicit device arrays carried between pure physics calls."""

    manifold_headers: mx.array
    manifold_points: mx.array
    manifold_counts: mx.array
    warm_start: mx.array


class WorldState(NamedTuple):
    """MLX PyTree for one environment-major world batch."""

    q: mx.array
    v: mx.array
    scene_bodies: mx.array
    solver_cache: SolverCache


class StepOutput(NamedTuple):
    """Fixed-shape output suitable for ``mx.compile`` and lazy rollouts."""

    next_state: WorldState
    observations: mx.array
    contacts: mx.array
    sensors: mx.array
    status: mx.array
    physics_error: mx.array
    acceleration: mx.array


def initial_state(
    world: MLXCompiledWorld,
    environment_count: int,
) -> WorldState:
    """Create compiled defaults entirely as MLX arrays."""

    if environment_count <= 0:
        raise ValueError("environment_count must be positive")
    q = mx.broadcast_to(
        mx.array(world.default_q, dtype=mx.float32),
        (environment_count, world.nq),
    )
    v = mx.broadcast_to(
        mx.array(world.default_v, dtype=mx.float32),
        (environment_count, world.nv),
    )
    empty_float = mx.zeros((environment_count, 0), dtype=mx.float32)
    empty_u32 = mx.zeros((environment_count, 0), dtype=mx.uint32)
    return WorldState(
        q=q,
        v=v,
        scene_bodies=empty_float,
        solver_cache=SolverCache(
            manifold_headers=empty_u32,
            manifold_points=empty_float,
            manifold_counts=empty_u32,
            warm_start=empty_float,
        ),
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
    """Advance one transactional control step without a host synchronization.

    Autodiff and ``vmap`` through physics are intentionally unsupported.
    Environment batching is the first array dimension.
    """

    if state.scene_bodies.shape[-1] != 0:
        raise ValueError(
            "the current MLX primitive is free-motion ABA only; "
            "scene-body/contact state is not silently ignored"
        )
    q = state.q
    v = state.v
    if reset_mask is not None:
        replacement = (
            reset_state
            if reset_state is not None
            else initial_state(world, int(q.shape[0]))
        )
        mask = reset_mask.astype(mx.bool_)[:, None]
        q = mx.where(mask, replacement.q, q)
        v = mx.where(mask, replacement.v, v)
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
        scene_bodies=state.scene_bodies,
        solver_cache=state.solver_cache,
    )
    environment_count = int(q.shape[0])
    return StepOutput(
        next_state=next_state,
        observations=mx.concatenate((next_q, next_v), axis=-1),
        contacts=mx.zeros(
            (environment_count, 0, 16),
            dtype=mx.float32,
        ),
        sensors=mx.zeros(
            (environment_count, 0),
            dtype=mx.float32,
        ),
        status=status,
        physics_error=status[:, 0] != 0,
        acceleration=acceleration,
    )


__all__ = [
    "MLXCompiledWorld",
    "SolverCache",
    "StepOutput",
    "WorldState",
    "compile_world",
    "initial_state",
    "step",
]
