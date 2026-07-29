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
WORLD_SCENARIO_HEADER_DTYPE = np.dtype(
    [
        ("identity", _UINT4),
        ("provenance", _UINT4),
        ("sampling", _UINT4),
    ],
    align=True,
)
WORLD_SCENARIO_VALUE_DTYPE = np.dtype(
    [("value", _FLOAT4), ("identity", _UINT4)],
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
    asset_binding_count: int
    binding_index_count: int
    primary_articulation_index: int
    nq: int
    nv: int
    body_count: int
    scene_body_count: int
    articulation_count: int
    retained_private_bytes: int


@dataclass(frozen=True, slots=True)
class WorldFamilyStats:
    compile_count: int
    sample_count: int
    readback_count: int
    last_sample_milliseconds: float


@dataclass(frozen=True, slots=True)
class ScenarioFeature:
    id: str
    target_id: str
    axis: int
    distribution: int
    target: int
    ordinal: int
    parameters: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ScenarioSchema:
    id: str
    fingerprint: int
    features: tuple[ScenarioFeature, ...]


@dataclass(frozen=True, slots=True)
class WorldFamilyDeviceBuffers:
    """Borrowed ``id<MTLBuffer>`` addresses for native/MLX graph composition."""

    instance_headers: int
    asset_instances: int
    sensor_instances: int
    appearance_instances: int
    asset_bindings: int
    binding_indices: int
    reset_q: int
    reset_v: int
    reset_scene_bodies: int
    body_parameters: int
    controller_parameters: int
    scenario_headers: int
    scenario_values: int


@dataclass(frozen=True, slots=True)
class WorldFamilySnapshot:
    """Stable host copy produced only by an explicit readback."""

    instances: npt.NDArray[np.void]
    assets: npt.NDArray[np.void]
    sensors: npt.NDArray[np.void]
    appearances: npt.NDArray[np.void]
    scenario_headers: npt.NDArray[np.void]
    scenario_values: npt.NDArray[np.void]


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
        self._initialize(
            capacity,
            library_path=library_path,
            metallib_path=metallib_path,
            pack_path=None,
        )

    def _initialize(
        self,
        capacity: int,
        *,
        library_path: str | os.PathLike[str] | None,
        metallib_path: str | os.PathLike[str] | None,
        pack_path: str | os.PathLike[str] | None,
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
        if pack_path is None:
            handle = (
                self._bindings.lib.mr_create_franka_pick_place_world_family(
                    ct.c_uint32(capacity),
                    encoded_metallib,
                )
            )
        else:
            resolved_pack = Path(pack_path).expanduser().resolve()
            if not resolved_pack.is_file():
                raise FileNotFoundError(
                    f"MRWorldPack does not exist: {resolved_pack}"
                )
            handle = self._bindings.lib.mr_load_world_family_pack(
                os.fsencode(resolved_pack),
                ct.c_uint32(capacity),
                encoded_metallib,
            )
        self._handle = handle
        if not self._handle:
            raise MetalRoboError(
                "Could not create world family: "
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
            asset_binding_count=int(native.asset_binding_count),
            binding_index_count=int(native.binding_index_count),
            primary_articulation_index=int(
                native.primary_articulation_index
            ),
            nq=int(native.nq),
            nv=int(native.nv),
            body_count=int(native.body_count),
            scene_body_count=int(native.scene_body_count),
            articulation_count=int(native.articulation_count),
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
    def scenario_schema(self) -> ScenarioSchema:
        handle = self._require_open()
        features = []
        for index in range(self.layout.variation_count):
            native = (
                self._bindings.lib.mr_world_family_scenario_feature(
                    handle,
                    ct.c_uint32(index),
                )
            )
            features.append(
                ScenarioFeature(
                    id=_decode(
                        self._bindings.lib
                        .mr_world_family_scenario_feature_id(
                            handle,
                            ct.c_uint32(index),
                        )
                    ),
                    target_id=_decode(
                        self._bindings.lib
                        .mr_world_family_scenario_target_id(
                            handle,
                            ct.c_uint32(index),
                        )
                    ),
                    axis=int(native.axis),
                    distribution=int(native.distribution),
                    target=int(native.target),
                    ordinal=int(native.ordinal),
                    parameters=tuple(
                        float(value) for value in native.parameters
                    ),
                )
            )
        return ScenarioSchema(
            id=_decode(
                self._bindings.lib.mr_world_family_scenario_id(
                    handle
                )
            ),
            fingerprint=int(
                self._bindings.lib
                .mr_world_family_scenario_fingerprint(handle)
            ),
            features=tuple(features),
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
            for kind in range(13)
        )
        if not all(addresses):
            raise MetalRoboError(
                "Native world-family Metal buffers are unavailable"
            )
        return WorldFamilyDeviceBuffers(*addresses)

    def configure_sampling(
        self,
        *,
        alignment_fingerprint: int = 0,
        particle_quantiles: npt.ArrayLike | None = None,
        particle_weights: npt.ArrayLike | None = None,
        particle_residuals: npt.ArrayLike | None = None,
        feedback_fingerprint: int = 0,
        region_kinds: npt.ArrayLike | None = None,
        region_weights: npt.ArrayLike | None = None,
        region_bounds: npt.ArrayLike | None = None,
        broad_weight: float = 0.5,
        failure_weight: float = 0.3,
        uncertainty_weight: float = 0.2,
        alignment_jitter: float = 0.05,
    ) -> None:
        """Upload immutable alignment particles and feedback regions."""

        feature_count = self.layout.variation_count
        quantiles = np.ascontiguousarray(
            (
                np.empty((0, feature_count), dtype=np.float32)
                if particle_quantiles is None
                else particle_quantiles
            ),
            dtype=np.float32,
        )
        if quantiles.ndim != 2 or quantiles.shape[1] != feature_count:
            raise ValueError(
                "particle_quantiles must have shape [particle, feature]"
            )
        weights = np.ascontiguousarray(
            (
                np.empty((0,), dtype=np.float32)
                if particle_weights is None
                else particle_weights
            ),
            dtype=np.float32,
        )
        if weights.shape != (quantiles.shape[0],):
            raise ValueError(
                "particle_weights must have one value per particle"
            )
        residuals = np.ascontiguousarray(
            (
                np.zeros_like(weights)
                if particle_residuals is None
                else particle_residuals
            ),
            dtype=np.float32,
        )
        if residuals.shape != weights.shape:
            raise ValueError(
                "particle_residuals must have one value per particle"
            )
        kinds = np.ascontiguousarray(
            (
                np.empty((0,), dtype=np.uint32)
                if region_kinds is None
                else region_kinds
            ),
            dtype=np.uint32,
        )
        region_weight_values = np.ascontiguousarray(
            (
                np.empty((0,), dtype=np.float32)
                if region_weights is None
                else region_weights
            ),
            dtype=np.float32,
        )
        bounds = np.ascontiguousarray(
            (
                np.empty((0, feature_count, 2), dtype=np.float32)
                if region_bounds is None
                else region_bounds
            ),
            dtype=np.float32,
        )
        if (
            kinds.ndim != 1
            or region_weight_values.shape != kinds.shape
            or bounds.shape != (kinds.size, feature_count, 2)
        ):
            raise ValueError(
                "feedback regions require kinds/weights [region] and "
                "bounds [region, feature, 2]"
            )
        if kinds.size == 0:
            broad_weight = 1.0
            failure_weight = 0.0
            uncertainty_weight = 0.0
        status = (
            self._bindings.lib.mr_world_family_configure_sampling(
                self._require_open(),
                ct.c_uint64(alignment_fingerprint),
                quantiles.ctypes.data_as(ct.POINTER(ct.c_float)),
                weights.ctypes.data_as(ct.POINTER(ct.c_float)),
                residuals.ctypes.data_as(ct.POINTER(ct.c_float)),
                ct.c_uint32(quantiles.shape[0]),
                ct.c_uint64(feedback_fingerprint),
                kinds.ctypes.data_as(ct.POINTER(ct.c_uint32)),
                region_weight_values.ctypes.data_as(
                    ct.POINTER(ct.c_float)
                ),
                bounds.ctypes.data_as(ct.POINTER(ct.c_float)),
                ct.c_uint32(kinds.size),
                ct.c_float(broad_weight),
                ct.c_float(failure_weight),
                ct.c_float(uncertainty_weight),
                ct.c_float(alignment_jitter),
            )
        )
        if status != 0:
            raise MetalRoboError(
                "Could not configure adaptive world sampling: "
                f"{self._bindings.last_error()}"
            )

    def sample(
        self,
        instance_count: int,
        *,
        seed: int = 1,
        mode: str = "coverage",
        episode_counter: int = 0,
    ) -> None:
        if not 1 <= int(instance_count) <= self.layout.capacity:
            raise ValueError(
                "instance_count must be nonzero and within compiled capacity"
            )
        if not 0 <= int(seed) <= np.iinfo(np.uint64).max:
            raise ValueError("seed must fit in a uint64")
        modes = {
            "coverage": 0,
            "curriculum": 1,
            "replay": 2,
        }
        if mode not in modes:
            raise ValueError(
                "mode must be 'coverage', 'curriculum', or 'replay'"
            )
        if not 0 <= int(episode_counter) <= np.iinfo(np.uint64).max:
            raise ValueError("episode_counter must fit in a uint64")
        if (
            int(episode_counter) + int(instance_count) - 1
            > np.iinfo(np.uint64).max
        ):
            raise ValueError(
                "per-environment episode counters overflow uint64"
            )
        status = self._bindings.lib.mr_world_family_sample_ex(
            self._require_open(),
            ct.c_uint32(instance_count),
            ct.c_uint64(seed),
            ct.c_uint32(modes[mode]),
            ct.c_uint64(episode_counter),
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
            scenario_headers=_stable_structured_copy(
                self._bindings.lib.mr_world_family_scenario_headers(
                    handle
                ),
                environment_count,
                WORLD_SCENARIO_HEADER_DTYPE,
                "scenario header",
            ),
            scenario_values=_stable_structured_copy(
                self._bindings.lib.mr_world_family_scenario_values(
                    handle
                ),
                environment_count * layout.variation_count,
                WORLD_SCENARIO_VALUE_DTYPE,
                "scenario value",
            ).reshape(
                environment_count,
                layout.variation_count,
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


class PackedWorldFamily(FrankaPickPlaceWorldFamily):
    """Load and sample a reusable ``MRWorldPack`` on the Apple GPU."""

    def __init__(
        self,
        pack_path: str | os.PathLike[str],
        capacity: int = 4096,
        *,
        library_path: str | os.PathLike[str] | None = None,
        metallib_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._initialize(
            capacity,
            library_path=library_path,
            metallib_path=metallib_path,
            pack_path=pack_path,
        )
