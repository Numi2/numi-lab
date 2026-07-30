"""MLX-native visual observations on MetalRobo's active command stream."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from ._mlx_abi import ENGINE_ABI_VERSION as _ENGINE_ABI_VERSION
from ._mlx_ext import (  # type: ignore[attr-defined]
    MLXCompiledWorld,
    visual_observation as _visual_observation,
)
from .hybrid_renderer import HybridObservationRenderer
from .worlds import PackedWorldFamily

del _ENGINE_ABI_VERSION


BODY_STATE_WORDS = 32
_MANDATORY_CHANNELS = frozenset(("rgb", "depth", "validity"))
_OPTIONAL_CHANNEL_BITS = {
    "segmentation": 1 << 0,
    "identities": 1 << 1,
    "normals": 1 << 2,
    "motion": 1 << 3,
}


class VisualObservation(NamedTuple):
    """Policy RGB-D plus statically selected simulation truth.

    Omitted optional fields retain their names but have zero spatial extent,
    so policy code never depends on packed channel indices.
    """

    rgb: mx.array
    depth: mx.array
    segmentation: mx.array
    identities: mx.array
    normals: mx.array
    motion: mx.array
    validity: mx.array


def visual_observation(
    world: MLXCompiledWorld,
    renderer: HybridObservationRenderer,
    worlds: PackedWorldFamily,
    current_body_states: mx.array,
    previous_body_states: mx.array | None = None,
    *,
    frame_index: int = 0,
    sensor_sequence: int = 0,
    camera_index: int = 0,
    channels: tuple[str, ...] = ("rgb", "depth", "validity"),
    stream: mx.Stream | mx.Device | None = None,
) -> VisualObservation:
    """Encode one lazy visual frame directly into MLX-owned device arrays.

    ``world`` and ``worlds`` must have been compiled independently from the
    same explicit MRWorldPack. The artifact hash is checked before graph
    construction so matching dimensions can never hide a mismatched scene.

    ``current_body_states`` and ``previous_body_states`` are uint32 arrays
    with shape ``(environments, bodies, 32)``. Their storage is the native
    128-byte ``MRBodyStateGPU`` record produced by a MetalRobo physics stage.
    The primitive appends all rendering work to MLX's active compute encoder;
    it does not copy through renderer-owned images, commit, wait, or read back.
    """

    if renderer.renderer_profile != "sensor_fast":
        raise ValueError("MLX visual observation requires sensor_fast")
    if not renderer.graph_only:
        raise ValueError(
            "MLX visual observation requires graph_only=True so the "
            "renderer does not retain duplicate observation planes"
        )
    if renderer._bindings.path != worlds._bindings.path:
        raise ValueError(
            "renderer and world family must use the same native library"
        )
    if not isinstance(worlds, PackedWorldFamily):
        raise TypeError(
            "MLX visual observations require an explicit PackedWorldFamily"
        )
    compiled_pack_hash = int(world.authored_pack_hash)
    sampled_pack_hash = int(worlds.authored_pack_hash)
    if (
        compiled_pack_hash == 0
        or sampled_pack_hash == 0
        or compiled_pack_hash != sampled_pack_hash
    ):
        raise ValueError(
            "compiled physics and sampled visual worlds must use the "
            "same explicit authored pack"
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
    if not isinstance(channels, tuple) or any(
        not isinstance(channel, str) for channel in channels
    ):
        raise TypeError("channels must be a tuple of channel names")
    selected_channels = frozenset(channels)
    known_channels = _MANDATORY_CHANNELS | frozenset(
        _OPTIONAL_CHANNEL_BITS
    )
    if len(selected_channels) != len(channels):
        raise ValueError("visual observation channels must be unique")
    if (
        not _MANDATORY_CHANNELS.issubset(selected_channels)
        or not selected_channels.issubset(known_channels)
    ):
        raise ValueError(
            "visual observations require rgb, depth, and validity; "
            "optional channels are segmentation, identities, normals, "
            "and motion"
        )
    output_mask = sum(
        bit
        for channel, bit in _OPTIONAL_CHANNEL_BITS.items()
        if channel in selected_channels
    )

    outputs = _visual_observation(
        world,
        int(renderer._require_open()),
        int(worlds._require_open()),
        current_body_states,
        previous,
        int(frame_index),
        int(sensor_sequence),
        int(camera_index),
        int(output_mask),
        stream=stream,
    )
    return VisualObservation(*outputs)


__all__ = [
    "BODY_STATE_WORDS",
    "VisualObservation",
    "visual_observation",
]
