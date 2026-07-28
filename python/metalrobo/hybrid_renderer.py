"""Metal-native Gaussian RGB-D and segmentation observations."""

from __future__ import annotations

import ctypes as ct
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .native import (
    MetalRoboError,
    _decode,
    _load_bindings,
    resolve_metallib_path,
)
from .worlds import FrankaPickPlaceWorldFamily

_FLOAT4 = (np.float32, (4,))
_UINT4 = (np.uint32, (4,))

HYBRID_GAUSSIAN_DTYPE = np.dtype(
    [
        ("mean_and_opacity", _FLOAT4),
        ("scale_and_importance", _FLOAT4),
        ("orientation", _FLOAT4),
        ("color_and_emission", _FLOAT4),
        ("binding", _UINT4),
    ],
    align=True,
)

GAUSSIAN_ASSET_LOCAL = 0
GAUSSIAN_BODY_LOCAL = 1
GAUSSIAN_WORLD = 2
INVALID_INDEX = np.uint32(0xFFFFFFFF)


@dataclass(frozen=True, slots=True)
class HybridRendererLayout:
    capacity: int
    active_environment_count: int
    width: int
    height: int
    tile_count_x: int
    tile_count_y: int
    gaussian_count: int
    maximum_gaussians_per_tile: int
    retained_private_bytes: int
    last_render_milliseconds: float


@dataclass(frozen=True, slots=True)
class HybridObservationDeviceBuffers:
    """Borrowed ``id<MTLBuffer>`` addresses for MLX graph composition."""

    rgb: int
    depth: int
    segmentation: int
    projected_gaussians: int
    tile_overflow_counts: int


@dataclass(frozen=True, slots=True)
class HybridObservationSnapshot:
    """Stable host copy created only by an explicit renderer readback."""

    rgb: npt.NDArray[np.float32]
    depth: npt.NDArray[np.float32]
    segmentation: npt.NDArray[np.uint32]


def make_asset_gaussians(
    means: npt.ArrayLike,
    scales: npt.ArrayLike,
    colors: npt.ArrayLike,
    asset_indices: npt.ArrayLike,
    semantic_labels: npt.ArrayLike,
    *,
    opacity: float | npt.ArrayLike = 1.0,
    importance: float | npt.ArrayLike = 1.0,
    orientations: npt.ArrayLike | None = None,
) -> npt.NDArray[np.void]:
    """Pack asset-local Gaussian attributes into the native scene ABI."""

    mean_array = np.asarray(means, dtype=np.float32)
    scale_array = np.asarray(scales, dtype=np.float32)
    color_array = np.asarray(colors, dtype=np.float32)
    assets = np.asarray(asset_indices, dtype=np.uint32)
    semantics = np.asarray(semantic_labels, dtype=np.uint32)
    if mean_array.ndim != 2 or mean_array.shape[1] != 3:
        raise ValueError("means must have shape [gaussian, 3]")
    count = mean_array.shape[0]
    if scale_array.shape != (count, 3):
        raise ValueError("scales must match means with shape [gaussian, 3]")
    if color_array.shape != (count, 3):
        raise ValueError("colors must match means with shape [gaussian, 3]")
    if assets.shape != (count,) or semantics.shape != (count,):
        raise ValueError("asset_indices and semantic_labels must have shape [gaussian]")
    opacity_array = np.broadcast_to(
        np.asarray(opacity, dtype=np.float32),
        (count,),
    )
    importance_array = np.broadcast_to(
        np.asarray(importance, dtype=np.float32),
        (count,),
    )
    orientation_array = (
        np.broadcast_to(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (count, 4),
        )
        if orientations is None
        else np.asarray(orientations, dtype=np.float32)
    )
    if orientation_array.shape != (count, 4):
        raise ValueError("orientations must have shape [gaussian, 4]")
    if (
        not np.all(np.isfinite(mean_array))
        or not np.all(np.isfinite(scale_array))
        or not np.all(np.isfinite(color_array))
        or not np.all(np.isfinite(opacity_array))
        or not np.all(np.isfinite(importance_array))
        or not np.all(np.isfinite(orientation_array))
        or np.any(scale_array <= 0.0)
        or np.any(opacity_array < 0.0)
        or np.any(opacity_array > 1.0)
    ):
        raise ValueError("Gaussian attributes must be finite and physical")

    packed = np.zeros((count,), dtype=HYBRID_GAUSSIAN_DTYPE)
    packed["mean_and_opacity"][:, :3] = mean_array
    packed["mean_and_opacity"][:, 3] = opacity_array
    packed["scale_and_importance"][:, :3] = scale_array
    packed["scale_and_importance"][:, 3] = importance_array
    orientation_norm = np.linalg.norm(
        orientation_array,
        axis=1,
        keepdims=True,
    )
    if np.any(orientation_norm <= 1.0e-8):
        raise ValueError("Gaussian orientations must be nonzero quaternions")
    packed["orientation"] = orientation_array / orientation_norm
    packed["color_and_emission"][:, :3] = color_array
    packed["binding"][:, 0] = assets
    packed["binding"][:, 1] = INVALID_INDEX
    packed["binding"][:, 2] = semantics
    packed["binding"][:, 3] = GAUSSIAN_ASSET_LOCAL
    return packed


class HybridObservationRenderer:
    """Render sampled world families without host-side environment work."""

    def __init__(
        self,
        gaussians: npt.ArrayLike,
        *,
        asset_count: int,
        capacity: int = 256,
        width: int = 160,
        height: int = 120,
        library_path: str | os.PathLike[str] | None = None,
        metallib_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not 1 <= int(asset_count) <= np.iinfo(np.uint32).max:
            raise ValueError("asset_count must fit in a nonzero uint32")
        for name, value in (
            ("capacity", capacity),
            ("width", width),
            ("height", height),
        ):
            if not 1 <= int(value) <= np.iinfo(np.uint32).max:
                raise ValueError(f"{name} must fit in a nonzero uint32")
        packed = np.ascontiguousarray(
            gaussians,
            dtype=HYBRID_GAUSSIAN_DTYPE,
        )
        if packed.ndim != 1 or packed.size == 0:
            raise ValueError("gaussians must be a nonempty one-dimensional array")

        self._bindings = _load_bindings(library_path)
        self.library_path: Path = self._bindings.path
        self.metallib_path = resolve_metallib_path(
            metallib_path,
            library_path=self.library_path,
        )
        encoded_metallib = (
            os.fsencode(self.metallib_path) if self.metallib_path is not None else None
        )
        self._handle = self._bindings.lib.mr_hybrid_renderer_create(
            ct.c_void_p(int(packed.ctypes.data)),
            ct.c_size_t(packed.size),
            ct.c_uint32(asset_count),
            ct.c_uint32(capacity),
            ct.c_uint32(width),
            ct.c_uint32(height),
            encoded_metallib,
        )
        if not self._handle:
            raise MetalRoboError(
                f"Could not create hybrid renderer: {self._bindings.last_error()}"
            )

    def _require_open(self) -> ct.c_void_p:
        handle = getattr(self, "_handle", None)
        if not handle:
            raise MetalRoboError("HybridObservationRenderer is closed")
        return handle

    @property
    def device_name(self) -> str:
        return _decode(
            self._bindings.lib.mr_hybrid_renderer_device_name(self._require_open())
        )

    @property
    def layout(self) -> HybridRendererLayout:
        native = self._bindings.lib.mr_hybrid_renderer_layout(self._require_open())
        return HybridRendererLayout(
            capacity=int(native.capacity),
            active_environment_count=int(native.active_environment_count),
            width=int(native.width),
            height=int(native.height),
            tile_count_x=int(native.tile_count_x),
            tile_count_y=int(native.tile_count_y),
            gaussian_count=int(native.gaussian_count),
            maximum_gaussians_per_tile=int(native.maximum_gaussians_per_tile),
            retained_private_bytes=int(native.retained_private_bytes),
            last_render_milliseconds=float(native.last_render_milliseconds),
        )

    @property
    def device_buffers(self) -> HybridObservationDeviceBuffers:
        handle = self._require_open()
        addresses = tuple(
            int(
                self._bindings.lib.mr_hybrid_renderer_native_buffer(
                    handle,
                    ct.c_uint32(kind),
                )
                or 0
            )
            for kind in range(5)
        )
        if not all(addresses):
            raise MetalRoboError("Hybrid renderer Metal buffers are unavailable")
        return HybridObservationDeviceBuffers(*addresses)

    def render(
        self,
        worlds: FrankaPickPlaceWorldFamily,
        environment_count: int | None = None,
        *,
        camera_index: int = 0,
    ) -> None:
        if worlds._bindings.path != self._bindings.path:
            raise ValueError(
                "renderer and world family must use the same native library"
            )
        world_layout = worlds.layout
        count = (
            world_layout.active_instance_count
            if environment_count is None
            else int(environment_count)
        )
        if not 1 <= count <= self.layout.capacity:
            raise ValueError(
                "environment_count must be sampled and within renderer capacity"
            )
        if not 0 <= int(camera_index) < world_layout.sensor_count_per_instance:
            raise ValueError("camera_index is outside the world sensor range")
        status = self._bindings.lib.mr_hybrid_renderer_render(
            self._require_open(),
            worlds._require_open(),
            ct.c_uint32(count),
            ct.c_uint32(camera_index),
        )
        if status != 0:
            raise MetalRoboError(
                f"Could not render observations: {self._bindings.last_error()}"
            )

    def snapshot(self) -> HybridObservationSnapshot:
        handle = self._require_open()
        if self._bindings.lib.mr_hybrid_renderer_readback(handle) != 0:
            raise MetalRoboError(
                f"Could not read back observations: {self._bindings.last_error()}"
            )
        layout = self.layout
        shape = (
            layout.active_environment_count,
            layout.height,
            layout.width,
        )
        pixel_count = int(np.prod(shape, dtype=np.int64))
        rgb = (
            np.ctypeslib.as_array(
                self._bindings.lib.mr_hybrid_renderer_rgb(handle),
                shape=(pixel_count * 4,),
            )
            .reshape((*shape, 4))
            .copy()
        )
        depth = (
            np.ctypeslib.as_array(
                self._bindings.lib.mr_hybrid_renderer_depth(handle),
                shape=(pixel_count,),
            )
            .reshape(shape)
            .copy()
        )
        segmentation = (
            np.ctypeslib.as_array(
                self._bindings.lib.mr_hybrid_renderer_segmentation(handle),
                shape=(pixel_count,),
            )
            .reshape(shape)
            .copy()
        )
        for array in (rgb, depth, segmentation):
            array.setflags(write=False)
        return HybridObservationSnapshot(rgb, depth, segmentation)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._bindings.lib.mr_hybrid_renderer_destroy(handle)
            self._handle = None

    def __enter__(self) -> HybridObservationRenderer:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
