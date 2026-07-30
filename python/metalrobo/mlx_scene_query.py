"""MLX-native rigid state and geometric scene queries."""

from __future__ import annotations

from typing import Any, NamedTuple

import mlx.core as mx

from ._mlx_ext import (  # type: ignore[attr-defined]
    MLXCompiledWorld,
    materialize_body_states as _materialize_body_states,
    scene_raycast as _scene_raycast,
)


BODY_STATE_WORDS = 32
INVALID_ID = 0xFFFFFFFF

_INCLUDE_DISABLED = 1 << 0
_FORCE_TWO_SIDED = 1 << 1
_FACE_FORWARD_NORMAL = 1 << 2


class SceneRaycastResult(NamedTuple):
    """Metric ray hits with stable physics identities.

    ``identities[..., :]`` stores shape, body, material, and geometric
    feature IDs. Misses have ``distance_m == -1`` and all identity channels
    set to ``INVALID_ID``.
    """

    distance_m: mx.array
    point_world: mx.array
    normal_world: mx.array
    identities: mx.array
    validity: mx.array


def materialize_body_states(
    world: MLXCompiledWorld,
    state: Any,
    *,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    """Materialize one authoritative rigid-state tensor on the GPU.

    Articulated poses and velocities are derived from ``q``/``v`` while
    standalone rigid bodies come from ``state.scene_bodies``. The returned
    uint32 tensor uses native ``MRBodyStateGPU`` records without host
    conversion and can be shared by visual observations and geometric
    sensors. A tactile ``StepOutput.body_states`` already contains the same
    geometric state and should be reused instead of dispatching this helper.
    """

    scene = state.scene_bodies
    return _materialize_body_states(
        world,
        state.q,
        state.v,
        scene.position,
        scene.orientation,
        scene.linear_velocity,
        scene.angular_velocity,
        stream=stream,
    )[0]


def _ray_batch(
    value: mx.array,
    environment_count: int,
    label: str,
) -> mx.array:
    array = value.astype(mx.float32)
    if array.ndim == 2:
        if int(array.shape[-1]) not in (3, 4):
            raise ValueError(f"{label} must end in xyz or xyzw")
        array = mx.broadcast_to(
            array[None, ...],
            (environment_count,) + tuple(array.shape),
        )
    elif array.ndim == 3:
        if (
            int(array.shape[0]) != environment_count
            or int(array.shape[-1]) not in (3, 4)
        ):
            raise ValueError(
                f"{label} must have shape (environment, ray, 3|4)"
            )
    else:
        raise ValueError(
            f"{label} must have shape (ray, 3|4) or "
            "(environment, ray, 3|4)"
        )
    if int(array.shape[-1]) == 3:
        array = mx.concatenate(
            (
                array,
                mx.zeros(
                    tuple(array.shape[:-1]) + (1,),
                    dtype=mx.float32,
                ),
            ),
            axis=-1,
        )
    return array


def _scalar_plane(
    value: float | int | mx.array,
    shape: tuple[int, int],
    dtype: mx.Dtype,
    label: str,
) -> mx.array:
    if isinstance(value, (float, int)):
        return mx.full(shape, value, dtype=dtype)
    array = value.astype(dtype)
    if array.ndim == 1 and int(array.shape[0]) == shape[1]:
        array = array[None, :]
    try:
        return mx.broadcast_to(array, shape)
    except ValueError as error:
        raise ValueError(
            f"{label} must be scalar, per-ray, or per-environment/ray"
        ) from error


def scene_raycast(
    world: MLXCompiledWorld,
    body_states: mx.array,
    origins: mx.array,
    directions: mx.array,
    *,
    maximum_distance_m: float | mx.array = 100.0,
    collision_group: int | mx.array = INVALID_ID,
    collision_mask: int | mx.array = INVALID_ID,
    excluded_body: int | mx.array = INVALID_ID,
    include_disabled: bool = False,
    two_sided: bool = True,
    face_forward_normals: bool = True,
    stream: mx.Stream | mx.Device | None = None,
) -> SceneRaycastResult:
    """Cast arbitrary batched rays without leaving MLX's Metal command stream.

    Rays can be shared across environments with shape ``(ray, 3|4)`` or
    supplied per environment. Directions need not be normalized. Collision
    group/mask filtering follows the same two-way bit contract as physics.
    """

    if body_states.dtype != mx.uint32 or body_states.ndim != 3:
        raise ValueError(
            "body_states must be the uint32 tensor returned by "
            "materialize_body_states"
        )
    environment_count = int(body_states.shape[0])
    expected_body_shape = (
        environment_count,
        int(world.model_body_count),
        BODY_STATE_WORDS,
    )
    if tuple(body_states.shape) != expected_body_shape:
        raise ValueError(
            f"body_states must have shape {expected_body_shape}"
        )
    origin_batch = _ray_batch(
        origins,
        environment_count,
        "origins",
    )
    direction_batch = _ray_batch(
        directions,
        environment_count,
        "directions",
    )
    if tuple(origin_batch.shape) != tuple(direction_batch.shape):
        raise ValueError("origins and directions must have matching rays")
    ray_count = int(origin_batch.shape[1])
    scalar_shape = (environment_count, ray_count)
    maximum = _scalar_plane(
        maximum_distance_m,
        scalar_shape,
        mx.float32,
        "maximum_distance_m",
    )
    groups = _scalar_plane(
        collision_group,
        scalar_shape,
        mx.uint32,
        "collision_group",
    )
    masks = _scalar_plane(
        collision_mask,
        scalar_shape,
        mx.uint32,
        "collision_mask",
    )
    exclusions = _scalar_plane(
        excluded_body,
        scalar_shape,
        mx.uint32,
        "excluded_body",
    )
    flags = (
        (_INCLUDE_DISABLED if include_disabled else 0)
        | (_FORCE_TWO_SIDED if two_sided else 0)
        | (_FACE_FORWARD_NORMAL if face_forward_normals else 0)
    )
    option_flags = mx.full(
        scalar_shape,
        flags,
        dtype=mx.uint32,
    )
    options = mx.stack(
        (groups, masks, exclusions, option_flags),
        axis=-1,
    )
    distance, point, normal, identities, validity = _scene_raycast(
        world,
        body_states,
        origin_batch,
        direction_batch,
        maximum,
        options,
        stream=stream,
    )
    return SceneRaycastResult(
        distance_m=distance,
        point_world=point,
        normal_world=normal,
        identities=identities,
        validity=validity.astype(mx.bool_),
    )


__all__ = [
    "BODY_STATE_WORDS",
    "INVALID_ID",
    "SceneRaycastResult",
    "materialize_body_states",
    "scene_raycast",
]
