"""MLX-native rigid state and geometric scene queries."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from ._mlx_ext import (  # type: ignore[attr-defined]
    MLXCompiledWorld,
    materialize_body_states as _materialize_body_states,
    scene_raycast as _scene_raycast,
    scene_raycast_pattern as _scene_raycast_pattern,
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


class RayPattern(NamedTuple):
    """Graph-static rays expressed in authored parent-body frames.

    A parent body of ``INVALID_ID`` means the corresponding origin and
    direction are already world-relative. Other rays follow that body's
    current pose independently in every environment.
    """

    parent_bodies: mx.array
    origins_local_m: mx.array
    directions_local: mx.array


def make_ray_pattern(
    parent_body: int | npt.ArrayLike,
    origins_local_m: npt.ArrayLike,
    directions_local: npt.ArrayLike,
) -> RayPattern:
    """Cook an immutable mounted-ray pattern for reuse by compiled graphs."""

    origins = np.asarray(origins_local_m, dtype=np.float32)
    directions = np.asarray(directions_local, dtype=np.float32)
    if (
        origins.ndim != 2
        or directions.ndim != 2
        or origins.shape != directions.shape
        or origins.shape[0] == 0
        or origins.shape[1] not in (3, 4)
    ):
        raise ValueError(
            "ray-pattern origins and directions must have matching "
            "nonempty shape (ray, 3|4)"
        )
    if not np.all(np.isfinite(origins)) or not np.all(
        np.isfinite(directions)
    ):
        raise ValueError("ray-pattern geometry must be finite")
    direction_norms = np.linalg.norm(
        directions[:, :3],
        axis=1,
        keepdims=True,
    )
    if np.any(direction_norms <= 1.0e-8):
        raise ValueError("ray-pattern directions must be nonzero")
    directions = directions.copy()
    directions[:, :3] /= direction_norms
    if origins.shape[1] == 3:
        origins = np.concatenate(
            (
                origins,
                np.zeros((origins.shape[0], 1), dtype=np.float32),
            ),
            axis=1,
        )
        directions = np.concatenate(
            (
                directions,
                np.zeros(
                    (directions.shape[0], 1),
                    dtype=np.float32,
                ),
            ),
            axis=1,
        )

    source_parents = np.asarray(parent_body)
    if source_parents.ndim == 0:
        source_parents = np.full(
            (origins.shape[0],),
            source_parents,
            dtype=np.int64,
        )
    if (
        source_parents.shape != (origins.shape[0],)
        or source_parents.dtype.kind not in "iu"
        or np.any(source_parents < 0)
        or np.any(source_parents > INVALID_ID)
    ):
        raise ValueError(
            "parent_body must be one uint32-compatible body per ray"
        )
    return RayPattern(
        parent_bodies=mx.array(
            source_parents.astype(np.uint32, copy=False)
        ),
        origins_local_m=mx.array(origins),
        directions_local=mx.array(directions),
    )


def make_grid_ray_pattern(
    parent_body: int,
    *,
    size_m: tuple[float, float],
    resolution: tuple[int, int],
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> RayPattern:
    """Cook an XY grid in one body frame, ordered row-major."""

    width, height = (float(value) for value in size_m)
    columns, rows = (int(value) for value in resolution)
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width < 0.0
        or height < 0.0
        or columns <= 0
        or rows <= 0
        or (columns > 1 and width == 0.0)
        or (rows > 1 and height == 0.0)
    ):
        raise ValueError(
            "grid size must be finite/nonnegative and repeated axes "
            "must have positive extent"
        )
    offset = np.asarray(origin_m, dtype=np.float32)
    ray_direction = np.asarray(direction, dtype=np.float32)
    if offset.shape != (3,) or ray_direction.shape != (3,):
        raise ValueError("grid origin and direction must be xyz vectors")
    x_values = np.linspace(
        -0.5 * width,
        0.5 * width,
        columns,
        dtype=np.float32,
    )
    y_values = np.linspace(
        -0.5 * height,
        0.5 * height,
        rows,
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    origins = np.stack(
        (
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.zeros((rows * columns,), dtype=np.float32),
        ),
        axis=-1,
    )
    origins += offset
    directions = np.broadcast_to(
        ray_direction,
        origins.shape,
    )
    return make_ray_pattern(parent_body, origins, directions)


def make_lidar_ray_pattern(
    parent_body: int,
    *,
    horizontal_fov_degrees: tuple[float, float] = (-180.0, 180.0),
    vertical_fov_degrees: tuple[float, float] = (-15.0, 15.0),
    horizontal_samples: int = 360,
    vertical_channels: int = 16,
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RayPattern:
    """Cook a body-frame LiDAR pattern with +X forward and +Z up."""

    horizontal_samples = int(horizontal_samples)
    vertical_channels = int(vertical_channels)
    horizontal = tuple(
        float(value) for value in horizontal_fov_degrees
    )
    vertical = tuple(float(value) for value in vertical_fov_degrees)
    if (
        horizontal_samples <= 0
        or vertical_channels <= 0
        or not all(math.isfinite(value) for value in horizontal + vertical)
        or horizontal[1] <= horizontal[0]
        or vertical[1] < vertical[0]
        or horizontal[1] - horizontal[0] > 360.0
        or vertical[0] < -90.0
        or vertical[1] > 90.0
    ):
        raise ValueError("LiDAR FOV and sample counts are invalid")
    full_circle = math.isclose(
        horizontal[1] - horizontal[0],
        360.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )
    azimuth_degrees = (
        np.asarray(
            [0.5 * (horizontal[0] + horizontal[1])],
            dtype=np.float32,
        )
        if horizontal_samples == 1
        else np.linspace(
            horizontal[0],
            horizontal[1],
            horizontal_samples,
            endpoint=not full_circle,
            dtype=np.float32,
        )
    )
    elevation_degrees = (
        np.asarray(
            [0.5 * (vertical[0] + vertical[1])],
            dtype=np.float32,
        )
        if vertical_channels == 1
        else np.linspace(
            vertical[0],
            vertical[1],
            vertical_channels,
            dtype=np.float32,
        )
    )
    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)
    azimuth_grid, elevation_grid = np.meshgrid(
        azimuth,
        elevation,
    )
    cos_elevation = np.cos(elevation_grid)
    directions = np.stack(
        (
            cos_elevation * np.cos(azimuth_grid),
            cos_elevation * np.sin(azimuth_grid),
            np.sin(elevation_grid),
        ),
        axis=-1,
    ).reshape((-1, 3))
    origin = np.asarray(origin_m, dtype=np.float32)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("LiDAR origin must be a finite xyz vector")
    origins = np.broadcast_to(origin, directions.shape)
    return make_ray_pattern(parent_body, origins, directions)


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


def scene_raycast_pattern(
    world: MLXCompiledWorld,
    body_states: mx.array,
    pattern: RayPattern,
    *,
    maximum_distance_m: float | mx.array = 100.0,
    collision_group: int | mx.array = INVALID_ID,
    collision_mask: int | mx.array = INVALID_ID,
    excluded_body: int | mx.array | None = None,
    exclude_parent: bool = True,
    include_disabled: bool = False,
    two_sided: bool = True,
    face_forward_normals: bool = True,
    stream: mx.Stream | mx.Device | None = None,
) -> SceneRaycastResult:
    """Cast a mounted pattern without constructing world-space ray tensors.

    Pattern geometry is shared across environments. The Metal kernel reads
    each environment's current parent-body pose, transforms that ray, applies
    collision filtering, and traverses the complete dynamic scene in one
    dispatch. Parent-body geometry is excluded by default.
    """

    if body_states.dtype != mx.uint32 or body_states.ndim != 3:
        raise ValueError(
            "body_states must be the uint32 tensor returned by "
            "materialize_body_states or StepOutput.body_states"
        )
    if not isinstance(pattern, RayPattern):
        raise TypeError("pattern must be a RayPattern")
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
    if pattern.parent_bodies.dtype != mx.uint32:
        raise ValueError("pattern parent bodies must be uint32")
    ray_count = int(pattern.parent_bodies.shape[0])
    pattern_shape = (ray_count, 4)
    if (
        pattern.parent_bodies.ndim != 1
        or ray_count <= 0
        or pattern.origins_local_m.dtype != mx.float32
        or tuple(pattern.origins_local_m.shape) != pattern_shape
        or pattern.directions_local.dtype != mx.float32
        or tuple(pattern.directions_local.shape) != pattern_shape
    ):
        raise ValueError(
            "pattern arrays must have shapes (ray,) and (ray, 4)"
        )
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
    if excluded_body is None:
        exclusions = (
            mx.broadcast_to(
                pattern.parent_bodies[None, :],
                scalar_shape,
            )
            if exclude_parent
            else mx.full(
                scalar_shape,
                INVALID_ID,
                dtype=mx.uint32,
            )
        )
    else:
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
    distance, point, normal, identities, validity = (
        _scene_raycast_pattern(
            world,
            body_states,
            pattern.parent_bodies,
            pattern.origins_local_m,
            pattern.directions_local,
            maximum,
            options,
            stream=stream,
        )
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
    "RayPattern",
    "SceneRaycastResult",
    "make_grid_ray_pattern",
    "make_lidar_ray_pattern",
    "make_ray_pattern",
    "materialize_body_states",
    "scene_raycast",
    "scene_raycast_pattern",
]
