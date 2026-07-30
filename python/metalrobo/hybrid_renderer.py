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
FRAME_VALID = np.uint32(1 << 0)
DEPTH_VALID = np.uint32(1 << 1)
GEOMETRY_VALID = np.uint32(1 << 2)


def _uint32(
    value: object,
    *,
    name: str,
    nonzero: bool = False,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    minimum = 1 if nonzero else 0
    if not minimum <= result <= np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit in a uint32")
    return result


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
    maximum_mesh_triangles_per_tile: int
    mesh_vertex_count: int
    mesh_triangle_count: int
    mesh_primitive_count: int
    mesh_instance_count: int
    mesh_index_count: int
    material_count: int
    texture_count: int
    light_count: int
    body_count: int
    sensor_binding_count: int
    shadow_layer_capacity: int
    ray_instance_count: int
    shadow_workspace_bytes: int
    acceleration_structure_bytes: int
    retained_private_bytes: int
    last_render_milliseconds: float


@dataclass(frozen=True, slots=True)
class HybridObservationDeviceBuffers:
    """Borrowed ``id<MTLBuffer>`` addresses for native graph bridges."""

    rgb: int
    depth: int
    segmentation: int
    projected_gaussians: int
    tile_overflow_counts: int
    identities: int
    normals: int
    motion: int
    validity: int


@dataclass(frozen=True, slots=True)
class VisualFrameMetadata:
    dimensions: tuple[int, int, int, int]
    identity: tuple[int, int, int, int]
    timing: tuple[float, float, float, float]
    contract: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class HybridObservationSnapshot:
    """Stable host copy created only by an explicit renderer readback."""

    rgb: npt.NDArray[np.float32]
    depth: npt.NDArray[np.float32]
    segmentation: npt.NDArray[np.uint32]
    identities: npt.NDArray[np.uint32]
    normals: npt.NDArray[np.float32]
    motion: npt.NDArray[np.float32]
    validity: npt.NDArray[np.uint32]
    metadata: VisualFrameMetadata

    @property
    def frame_validity(self) -> npt.NDArray[np.bool_]:
        result = (self.validity & FRAME_VALID) != 0
        result.setflags(write=False)
        return result

    @property
    def depth_validity(self) -> npt.NDArray[np.bool_]:
        result = (self.validity & DEPTH_VALID) != 0
        result.setflags(write=False)
        return result

    @property
    def geometry_validity(self) -> npt.NDArray[np.bool_]:
        result = (self.validity & GEOMETRY_VALID) != 0
        result.setflags(write=False)
        return result

    @property
    def metric_depth(self) -> npt.NDArray[np.float32]:
        result = np.where(
            self.depth_validity,
            self.depth,
            0.0,
        ).astype(np.float32, copy=False)
        result.setflags(write=False)
        return result


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
    source_assets = np.asarray(asset_indices)
    source_semantics = np.asarray(semantic_labels)
    if mean_array.ndim != 2 or mean_array.shape[1] != 3:
        raise ValueError("means must have shape [gaussian, 3]")
    count = mean_array.shape[0]
    if scale_array.shape != (count, 3):
        raise ValueError("scales must match means with shape [gaussian, 3]")
    if color_array.shape != (count, 3):
        raise ValueError("colors must match means with shape [gaussian, 3]")
    if source_assets.shape != (count,) or source_semantics.shape != (count,):
        raise ValueError("asset_indices and semantic_labels must have shape [gaussian]")
    if (
        source_assets.dtype.kind not in "iu"
        or source_semantics.dtype.kind not in "iu"
        or (source_assets.size and np.any(source_assets < 0))
        or (source_semantics.size and np.any(source_semantics <= 0))
        or np.any(source_assets > np.iinfo(np.uint32).max)
        or np.any(source_semantics >= np.iinfo(np.uint32).max)
    ):
        raise ValueError("asset indices and nonzero semantic labels must fit in uint32")
    assets = np.asarray(source_assets, dtype=np.uint32)
    semantics = np.asarray(source_semantics, dtype=np.uint32)
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
        or np.any(color_array < 0.0)
        or np.any(opacity_array < 0.0)
        or np.any(opacity_array > 1.0)
        or np.any(importance_array < 0.0)
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
        gaussians: npt.ArrayLike | None = None,
        *,
        asset_count: int,
        body_count: int = 0,
        visual_pack: str | os.PathLike[str] | None = None,
        environment_pack: str | os.PathLike[str] | None = None,
        visual_asset_index: int = 0,
        semantic_id: int = 1,
        instance_id: int = 1,
        light_rig: str = "studio_key",
        renderer_profile: str = "sensor_fast",
        capacity: int = 256,
        width: int = 160,
        height: int = 120,
        library_path: str | os.PathLike[str] | None = None,
        metallib_path: str | os.PathLike[str] | None = None,
    ) -> None:
        asset_count = _uint32(
            asset_count,
            name="asset_count",
            nonzero=True,
        )
        if asset_count >= int(INVALID_INDEX):
            raise ValueError("asset_count must leave room for valid instance ids")
        body_count = _uint32(body_count, name="body_count")
        visual_asset_index = _uint32(
            visual_asset_index,
            name="visual_asset_index",
        )
        if visual_asset_index >= asset_count:
            raise ValueError("visual_asset_index must be inside asset_count")
        semantic_id = _uint32(
            semantic_id,
            name="semantic_id",
            nonzero=True,
        )
        instance_id = _uint32(
            instance_id,
            name="instance_id",
            nonzero=True,
        )
        if semantic_id == int(INVALID_INDEX) or instance_id == int(INVALID_INDEX):
            raise ValueError("semantic_id and instance_id must be valid identities")
        if light_rig not in {"studio_key", "indoor_area"}:
            raise ValueError("light_rig must be 'studio_key' or 'indoor_area'")
        if renderer_profile not in {"sensor_fast", "sensor_reference"}:
            raise ValueError(
                "renderer_profile must be 'sensor_fast' or 'sensor_reference'"
            )
        if renderer_profile == "sensor_reference" and body_count == 0:
            raise ValueError(
                "sensor_reference requires body_count for its motion layout"
            )
        dimensions: dict[str, int] = {}
        for name, value in (
            ("capacity", capacity),
            ("width", width),
            ("height", height),
        ):
            dimensions[name] = _uint32(
                value,
                name=name,
                nonzero=True,
            )
        capacity = dimensions["capacity"]
        width = dimensions["width"]
        height = dimensions["height"]
        packed = (
            np.empty((0,), dtype=HYBRID_GAUSSIAN_DTYPE)
            if gaussians is None
            else np.ascontiguousarray(
                gaussians,
                dtype=HYBRID_GAUSSIAN_DTYPE,
            )
        )
        if packed.ndim != 1:
            raise ValueError("gaussians must be a one-dimensional array")
        if packed.size and (
            np.any(packed["binding"][:, 0] >= asset_count)
            or np.any(packed["binding"][:, 2] == 0)
            or np.any(packed["binding"][:, 2] == INVALID_INDEX)
        ):
            raise ValueError("Gaussian asset and semantic bindings are invalid")
        resolved_visual_pack: Path | None = None
        if visual_pack is not None:
            resolved_visual_pack = Path(visual_pack).expanduser().resolve()
            if not resolved_visual_pack.is_file():
                raise FileNotFoundError(
                    f"VisualAssetPackV2 does not exist: {resolved_visual_pack}"
                )
        if packed.size == 0 and resolved_visual_pack is None:
            raise ValueError("gaussians or visual_pack must be supplied")
        resolved_environment_pack: Path | None = None
        if environment_pack is not None:
            resolved_environment_pack = (
                Path(environment_pack).expanduser().resolve()
            )
            if not resolved_environment_pack.is_file():
                raise FileNotFoundError(
                    "VisualEnvironmentPackV2 does not exist: "
                    f"{resolved_environment_pack}"
                )

        self._bindings = _load_bindings(library_path)
        self.library_path: Path = self._bindings.path
        self.metallib_path = resolve_metallib_path(
            metallib_path,
            library_path=self.library_path,
        )
        encoded_metallib = (
            os.fsencode(self.metallib_path) if self.metallib_path is not None else None
        )
        encoded_visual_pack = (
            os.fsencode(resolved_visual_pack)
            if resolved_visual_pack is not None
            else None
        )
        encoded_environment_pack = (
            os.fsencode(resolved_environment_pack)
            if resolved_environment_pack is not None
            else None
        )
        self._handle = self._bindings.lib.mr_hybrid_renderer_create_v3(
            (
                ct.c_void_p(int(packed.ctypes.data))
                if packed.size
                else None
            ),
            ct.c_size_t(packed.size),
            encoded_visual_pack,
            encoded_environment_pack,
            ct.c_uint32(asset_count),
            ct.c_uint32(body_count),
            ct.c_uint32(visual_asset_index),
            ct.c_uint32(semantic_id),
            ct.c_uint32(instance_id),
            light_rig.encode("utf-8"),
            renderer_profile.encode("utf-8"),
            ct.c_uint32(capacity),
            ct.c_uint32(width),
            ct.c_uint32(height),
            encoded_metallib,
        )
        if not self._handle:
            raise MetalRoboError(
                f"Could not create hybrid renderer: {self._bindings.last_error()}"
            )
        self.visual_pack = resolved_visual_pack
        self.environment_pack = resolved_environment_pack
        self.light_rig = light_rig
        self.renderer_profile = renderer_profile

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
            maximum_mesh_triangles_per_tile=int(
                native.maximum_mesh_triangles_per_tile
            ),
            mesh_vertex_count=int(native.mesh_vertex_count),
            mesh_triangle_count=int(native.mesh_triangle_count),
            mesh_primitive_count=int(native.mesh_primitive_count),
            mesh_instance_count=int(native.mesh_instance_count),
            mesh_index_count=int(native.mesh_index_count),
            material_count=int(native.material_count),
            texture_count=int(native.texture_count),
            light_count=int(native.light_count),
            body_count=int(native.body_count),
            sensor_binding_count=int(native.sensor_binding_count),
            shadow_layer_capacity=int(native.shadow_layer_capacity),
            ray_instance_count=int(native.ray_instance_count),
            shadow_workspace_bytes=int(native.shadow_workspace_bytes),
            acceleration_structure_bytes=int(
                native.acceleration_structure_bytes
            ),
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
            for kind in range(9)
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
            else _uint32(
                environment_count,
                name="environment_count",
                nonzero=True,
            )
        )
        if not 1 <= count <= self.layout.capacity:
            raise ValueError(
                "environment_count must be sampled and within renderer capacity"
            )
        camera_index = _uint32(
            camera_index,
            name="camera_index",
        )
        if camera_index >= world_layout.sensor_count_per_instance:
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
        identities = (
            np.ctypeslib.as_array(
                self._bindings.lib.mr_hybrid_renderer_identities(handle),
                shape=(pixel_count * 4,),
            )
            .reshape((*shape, 4))
            .copy()
        )
        normals = (
            np.ctypeslib.as_array(
                self._bindings.lib.mr_hybrid_renderer_normals(handle),
                shape=(pixel_count * 4,),
            )
            .reshape((*shape, 4))
            .copy()
        )
        motion = (
            np.ctypeslib.as_array(
                self._bindings.lib.mr_hybrid_renderer_motion(handle),
                shape=(pixel_count * 4,),
            )
            .reshape((*shape, 4))
            .copy()
        )
        validity = (
            np.ctypeslib.as_array(
                self._bindings.lib.mr_hybrid_renderer_validity(handle),
                shape=(pixel_count,),
            )
            .reshape(shape)
            .copy()
        )
        native_metadata = self._bindings.lib.mr_hybrid_renderer_frame_metadata(handle)
        metadata = VisualFrameMetadata(
            tuple(int(value) for value in native_metadata.dimensions),
            tuple(int(value) for value in native_metadata.identity),
            tuple(float(value) for value in native_metadata.timing),
            tuple(int(value) for value in native_metadata.contract),
        )
        for array in (
            rgb,
            depth,
            segmentation,
            identities,
            normals,
            motion,
            validity,
        ):
            array.setflags(write=False)
        return HybridObservationSnapshot(
            rgb,
            depth,
            segmentation,
            identities,
            normals,
            motion,
            validity,
            metadata,
        )

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
