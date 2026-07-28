"""GPU-resident world-family compiler frontend.

The normal path calls :meth:`FrankaPickPlaceWorldFamily.sample` and hands the
borrowed Metal buffers to a native MLX primitive. ``snapshot`` is intentionally
an explicit CPU inspection/export boundary.
"""

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

_FLOAT4 = (np.float32, (4,))
_UINT4 = (np.uint32, (4,))

WORLD_INSTANCE_DTYPE = np.dtype(
    [("ranges", _UINT4), ("program", _UINT4), ("identity", _UINT4)],
    align=True,
)
WORLD_ASSET_DTYPE = np.dtype(
    [
        ("position_and_scale", _FLOAT4),
        ("orientation", _FLOAT4),
        ("linear_velocity", _FLOAT4),
        ("angular_velocity", _FLOAT4),
        ("physical", _FLOAT4),
        ("controller", _FLOAT4),
        ("identity", _UINT4),
    ],
    align=True,
)
WORLD_SENSOR_DTYPE = np.dtype(
    [
        ("position_and_focal_scale", _FLOAT4),
        ("orientation", _FLOAT4),
        ("intrinsics", _FLOAT4),
        ("distortion", _FLOAT4),
        ("noise_and_latency", _FLOAT4),
        ("identity", _UINT4),
    ],
    align=True,
)
WORLD_APPEARANCE_DTYPE = np.dtype(
    [
        ("color_and_light", _FLOAT4),
        ("material", _FLOAT4),
        ("identity", _UINT4),
    ],
    align=True,
)


@dataclass(frozen=True, slots=True)
class WorldFamilyLayout:
    capacity: int
    active_instance_count: int
    asset_count_per_instance: int
    sensor_count_per_instance: int
    appearance_count_per_instance: int
    variation_count: int
    categorical_value_count: int
    retained_private_bytes: int


@dataclass(frozen=True, slots=True)
class WorldFamilyStats:
    compile_count: int
    sample_count: int
    readback_count: int
    last_sample_milliseconds: float


@dataclass(frozen=True, slots=True)
class WorldFamilyDeviceBuffers:
    """Borrowed ``id<MTLBuffer>`` addresses for native/MLX graph composition."""

    instance_headers: int
    asset_instances: int
    sensor_instances: int
    appearance_instances: int


@dataclass(frozen=True, slots=True)
class WorldFamilySnapshot:
    """Stable host copy produced only by an explicit readback."""

    instances: npt.NDArray[np.void]
    assets: npt.NDArray[np.void]
    sensors: npt.NDArray[np.void]
    appearances: npt.NDArray[np.void]


def _stable_structured_copy(
    address: int | None,
    count: int,
    dtype: np.dtype,
    label: str,
) -> npt.NDArray[np.void]:
    byte_count = count * dtype.itemsize
    if byte_count == 0:
        result = np.empty((0,), dtype=dtype)
    else:
        if not address:
            raise MetalRoboError(
                f"Native world family returned a null {label} pointer"
            )
        raw_type = ct.c_uint8 * byte_count
        raw = np.ctypeslib.as_array(raw_type.from_address(int(address)))
        result = raw.view(dtype).reshape(count).copy()
    result.setflags(write=False)
    return result


class FrankaPickPlaceWorldFamily:
    """Compile and sample the canonical six-axis Franka world family.

    Sampling performs no per-environment Python work. The generated headers,
    assets, sensors, and appearances stay in private Metal buffers until a
    native simulation/render/policy stage consumes them.
    """

    def __init__(
        self,
        capacity: int = 4096,
        *,
        library_path: str | os.PathLike[str] | None = None,
        metallib_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not 1 <= int(capacity) <= np.iinfo(np.uint32).max:
            raise ValueError("capacity must fit in a nonzero uint32")
        self._bindings = _load_bindings(library_path)
        self.library_path: Path = self._bindings.path
        self.metallib_path = resolve_metallib_path(
            metallib_path,
            library_path=self.library_path,
        )
        encoded_metallib = (
            os.fsencode(self.metallib_path)
            if self.metallib_path is not None
            else None
        )
        self._handle = (
            self._bindings.lib.mr_create_franka_pick_place_world_family(
                ct.c_uint32(capacity),
                encoded_metallib,
            )
        )
        if not self._handle:
            raise MetalRoboError(
                "Could not create Franka world family: "
                f"{self._bindings.last_error()}"
            )

    def _require_open(self) -> ct.c_void_p:
        handle = getattr(self, "_handle", None)
        if not handle:
            raise MetalRoboError("FrankaPickPlaceWorldFamily is closed")
        return handle

    @property
    def device_name(self) -> str:
        return _decode(
            self._bindings.lib.mr_world_family_device_name(
                self._require_open()
            )
        )

    @property
    def layout(self) -> WorldFamilyLayout:
        native = self._bindings.lib.mr_world_family_layout(
            self._require_open()
        )
        return WorldFamilyLayout(
            capacity=int(native.capacity),
            active_instance_count=int(native.active_instance_count),
            asset_count_per_instance=int(native.asset_count_per_instance),
            sensor_count_per_instance=int(native.sensor_count_per_instance),
            appearance_count_per_instance=int(
                native.appearance_count_per_instance
            ),
            variation_count=int(native.variation_count),
            categorical_value_count=int(native.categorical_value_count),
            retained_private_bytes=int(native.retained_private_bytes),
        )

    @property
    def stats(self) -> WorldFamilyStats:
        native = self._bindings.lib.mr_world_family_stats(
            self._require_open()
        )
        return WorldFamilyStats(
            compile_count=int(native.compile_count),
            sample_count=int(native.sample_count),
            readback_count=int(native.readback_count),
            last_sample_milliseconds=float(
                native.last_sample_milliseconds
            ),
        )

    @property
    def device_buffers(self) -> WorldFamilyDeviceBuffers:
        handle = self._require_open()
        addresses = tuple(
            int(
                self._bindings.lib.mr_world_family_native_buffer(
                    handle,
                    ct.c_uint32(kind),
                )
                or 0
            )
            for kind in range(4)
        )
        if not all(addresses):
            raise MetalRoboError(
                "Native world-family Metal buffers are unavailable"
            )
        return WorldFamilyDeviceBuffers(*addresses)

    def sample(self, instance_count: int, *, seed: int = 1) -> None:
        if not 1 <= int(instance_count) <= self.layout.capacity:
            raise ValueError(
                "instance_count must be nonzero and within compiled capacity"
            )
        if not 0 <= int(seed) <= np.iinfo(np.uint64).max:
            raise ValueError("seed must fit in a uint64")
        status = self._bindings.lib.mr_world_family_sample(
            self._require_open(),
            ct.c_uint32(instance_count),
            ct.c_uint64(seed),
        )
        if status != 0:
            raise MetalRoboError(
                f"Could not sample world family: {self._bindings.last_error()}"
            )

    def snapshot(self) -> WorldFamilySnapshot:
        handle = self._require_open()
        status = self._bindings.lib.mr_world_family_readback(handle)
        if status != 0:
            raise MetalRoboError(
                f"Could not read back world family: "
                f"{self._bindings.last_error()}"
            )
        layout = self.layout
        environment_count = layout.active_instance_count
        return WorldFamilySnapshot(
            instances=_stable_structured_copy(
                self._bindings.lib.mr_world_family_instance_headers(
                    handle
                ),
                environment_count,
                WORLD_INSTANCE_DTYPE,
                "instance",
            ),
            assets=_stable_structured_copy(
                self._bindings.lib.mr_world_family_asset_instances(
                    handle
                ),
                environment_count * layout.asset_count_per_instance,
                WORLD_ASSET_DTYPE,
                "asset",
            ),
            sensors=_stable_structured_copy(
                self._bindings.lib.mr_world_family_sensor_instances(
                    handle
                ),
                environment_count * layout.sensor_count_per_instance,
                WORLD_SENSOR_DTYPE,
                "sensor",
            ),
            appearances=_stable_structured_copy(
                self._bindings.lib.mr_world_family_appearance_instances(
                    handle
                ),
                environment_count *
                layout.appearance_count_per_instance,
                WORLD_APPEARANCE_DTYPE,
                "appearance",
            ),
        )

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._bindings.lib.mr_world_family_destroy(handle)
            self._handle = None

    def __enter__(self) -> FrankaPickPlaceWorldFamily:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
