"""MLX-native visual observations on MetalRobo's active command stream."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from ._mlx_ext import visual_observation as _visual_observation
from .hybrid_renderer import HybridObservationRenderer
from .worlds import FrankaPickPlaceWorldFamily


BODY_STATE_WORDS = 32


class VisualObservation(NamedTuple):
    """Policy RGB-D plus synchronized simulation truth."""

    rgb: mx.array
    depth: mx.array
    segmentation: mx.array
    identities: mx.array
    normals: mx.array
    motion: mx.array
    validity: mx.array


def visual_observation(
    renderer: HybridObservationRenderer,
    worlds: FrankaPickPlaceWorldFamily,
    current_body_states: mx.array,
    previous_body_states: mx.array | None = None,
    *,
    frame_index: int = 0,
    sensor_sequence: int = 0,
    camera_index: int = 0,
    stream: mx.Stream | mx.Device | None = None,
) -> VisualObservation:
    """Encode one lazy visual frame directly into MLX-owned device arrays.

    ``current_body_states`` and ``previous_body_states`` are uint32 arrays
    with shape ``(environments, bodies, 32)``. Their storage is the native
    128-byte ``MRBodyStateGPU`` record produced by a MetalRobo physics stage.
    The primitive appends all rendering work to MLX's active compute encoder;
    it does not copy through renderer-owned images, commit, wait, or read back.
    """

    if renderer.renderer_profile != "sensor_fast":
        raise ValueError("MLX visual observation requires sensor_fast")
    if renderer._bindings.path != worlds._bindings.path:
        raise ValueError(
            "renderer and world family must use the same native library"
        )
    renderer_layout = renderer.layout
    world_layout = worlds.layout
    shape = (
        int(world_layout.active_instance_count),
        int(renderer_layout.body_count),
        BODY_STATE_WORDS,
    )
    if shape[0] <= 0:
        raise ValueError("sample the world family before rendering")
    if (
        current_body_states.dtype != mx.uint32
        or tuple(current_body_states.shape) != shape
    ):
        raise ValueError(
            f"current_body_states must be uint32 with shape {shape}"
        )
    previous = (
        current_body_states
        if previous_body_states is None
        else previous_body_states
    )
    if previous.dtype != mx.uint32 or tuple(previous.shape) != shape:
        raise ValueError(
            f"previous_body_states must be uint32 with shape {shape}"
        )
    if not 0 <= camera_index < int(
        world_layout.sensor_count_per_instance
    ):
        raise ValueError("camera_index is outside the sampled sensor range")
    if frame_index < 0 or frame_index > (1 << 64) - 1:
        raise ValueError("frame_index must fit in a uint64")
    if sensor_sequence < 0 or sensor_sequence > (1 << 32) - 1:
        raise ValueError("sensor_sequence must fit in a uint32")

    outputs = _visual_observation(
        int(renderer._require_open()),
        int(worlds._require_open()),
        current_body_states,
        previous,
        int(frame_index),
        int(sensor_sequence),
        int(camera_index),
        stream=stream,
    )
    return VisualObservation(*outputs)


__all__ = [
    "BODY_STATE_WORDS",
    "VisualObservation",
    "visual_observation",
]
