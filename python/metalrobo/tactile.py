"""Apple-native metric tactile observations and calibration contracts.

Simulation produces the canonical depth map directly from geometry. A real
vision-based tactile sensor may use a learned translator, but its output must
obey the same metric schema and fingerprint checks defined here.
"""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from .native import (
    MetalRoboError,
    _decode,
    _load_bindings,
    resolve_metallib_path,
)

FloatArray = npt.NDArray[np.float32]
UInt32Array = npt.NDArray[np.uint32]

TACTILE_ABI_VERSION = 1
TACTILE_DEPTH = 0
TACTILE_DEPTH_VELOCITY = 1
TACTILE_VALIDITY = 2
TACTILE_OBJECT_SHAPE_IDS = 3
TACTILE_DEBUG_HITS = 4
TACTILE_SUMMARIES = 5
TACTILE_STATUSES = 6

SAMPLE_VALID = np.uint32(1 << 0)
CONTACT_VALID = np.uint32(1 << 1)
DEPTH_SATURATED = np.uint32(1 << 2)
FILTERED_TARGET = np.uint32(1 << 3)

_DENSE_CHANNELS = (
    "penetration_depth_m",
    "validity_bits",
    "object_shape_id",
    "depth_velocity_m_per_s",
)
_SUMMARY_CHANNELS = (
    "sensor_pose",
    "timestamp_s",
    "net_force_n",
    "net_torque_nm",
    "center_of_pressure_sensor_m",
    "center_of_pressure_force_weight_n",
    "geometric_contact_centroid_m",
    "contact_area_m2",
    "maximum_depth_m",
    "mean_depth_m",
)
_FINGERPRINT = re.compile(r"0x[0-9a-f]{16}")


def _positive_uint32(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 0 < result <= np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must be a positive uint32")
    return result


def _nonnegative_uint32(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must be a uint32")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _pointer(value: int | None, name: str, *, optional: bool = False) -> ct.c_void_p:
    if value is None and optional:
        return ct.c_void_p()
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer native pointer")
    address = int(value)
    if address <= 0:
        raise ValueError(f"{name} must be a nonzero native pointer")
    return ct.c_void_p(address)


@dataclass(frozen=True, slots=True)
class TactileSensorContract:
    id: str
    width: int
    height: int
    surface_kind: int
    maximum_depth_m: float
    active_threshold_m: float
    shell_thickness_m: float
    update_period_steps: int
    parent_body: int
    backing_shape: int


@dataclass(frozen=True, slots=True)
class TactileObservationContract:
    abi_version: int
    fingerprint: str
    primary_representation: str
    dense_channels: tuple[str, ...]
    summary_channels: tuple[str, ...]
    sensors: tuple[TactileSensorContract, ...]
    raw: Mapping[str, Any]

    @classmethod
    def from_json(cls, payload: str | bytes) -> "TactileObservationContract":
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise ValueError("tactile observation contract must be an object")
        if document.get("schema") != "metalrobo.tactile_observation":
            raise ValueError("not a MetalRobo tactile observation contract")
        abi_version = int(document.get("abi_version", -1))
        if abi_version != TACTILE_ABI_VERSION:
            raise ValueError(
                f"unsupported tactile ABI {abi_version}; "
                f"expected {TACTILE_ABI_VERSION}"
            )
        fingerprint = document.get("fingerprint")
        if not isinstance(fingerprint, str) or _FINGERPRINT.fullmatch(
            fingerprint
        ) is None:
            raise ValueError("tactile contract fingerprint is not canonical")
        if document.get("primary_representation") != "metric_normal_penetration":
            raise ValueError("tactile primary representation is not metric depth")
        fixed_fields = {
            "depth_unit": "m",
            "velocity_unit": "m/s",
            "force_unit": "N",
            "torque_unit": "N*m",
            "layout": "environment_sensor_row_major",
            "wrench_source": "physics_solver_impulse_over_explicit_interval",
            "center_of_pressure_definition": (
                "solver_force_magnitude_weighted_contact_position"
            ),
            "real_sensor_input": "camera_to_metric_depth_translator",
        }
        for name, expected in fixed_fields.items():
            if document.get(name) != expected:
                raise ValueError(
                    f"tactile contract {name} must be {expected!r}"
                )
        dense_channels = tuple(document.get("dense_channels", ()))
        summary_channels = tuple(document.get("summary_channels", ()))
        if dense_channels != _DENSE_CHANNELS:
            raise ValueError("tactile dense-channel contract is incompatible")
        if summary_channels != _SUMMARY_CHANNELS:
            raise ValueError("tactile summary-channel contract is incompatible")
        if document.get("simulation_translator_required") is not False:
            raise ValueError("simulation must not depend on a tactile translator")

        raw_sensors = document.get("sensors")
        if not isinstance(raw_sensors, list) or not raw_sensors:
            raise ValueError("tactile contract contains no sensors")
        parsed_sensors: list[TactileSensorContract] = []
        sensor_ids: set[str] = set()
        for index, sensor in enumerate(raw_sensors):
            if not isinstance(sensor, dict):
                raise ValueError(f"tactile sensor {index} must be an object")
            sensor_id = sensor.get("id")
            if (
                not isinstance(sensor_id, str)
                or not sensor_id
                or sensor_id in sensor_ids
            ):
                raise ValueError("tactile sensor IDs must be nonempty and unique")
            sensor_ids.add(sensor_id)
            surface_kind = _nonnegative_uint32(
                sensor.get("surface_kind"),
                "sensor surface kind",
            )
            if surface_kind > 2:
                raise ValueError("sensor surface kind is unsupported")
            maximum_depth = _finite_float(
                sensor.get("maximum_depth_m"),
                "sensor maximum depth",
            )
            active_threshold = _finite_float(
                sensor.get("active_threshold_m"),
                "sensor active threshold",
            )
            shell_thickness = _finite_float(
                sensor.get("shell_thickness_m"),
                "sensor shell thickness",
            )
            if (
                maximum_depth <= 0.0
                or active_threshold < 0.0
                or active_threshold > maximum_depth
                or shell_thickness + 1.0e-7 < maximum_depth
            ):
                raise ValueError(
                    "sensor depth, threshold, and shell are physically invalid"
                )
            parsed_sensors.append(
                TactileSensorContract(
                    id=sensor_id,
                    width=_positive_uint32(
                        sensor.get("width"),
                        "sensor width",
                    ),
                    height=_positive_uint32(
                        sensor.get("height"),
                        "sensor height",
                    ),
                    surface_kind=surface_kind,
                    maximum_depth_m=maximum_depth,
                    active_threshold_m=active_threshold,
                    shell_thickness_m=shell_thickness,
                    update_period_steps=_positive_uint32(
                        sensor.get("update_period_steps"),
                        "sensor update period",
                    ),
                    parent_body=_nonnegative_uint32(
                        sensor.get("parent_body"),
                        "sensor parent body",
                    ),
                    backing_shape=_nonnegative_uint32(
                        sensor.get("backing_shape"),
                        "sensor backing shape",
                    ),
                )
            )
        sensors = tuple(parsed_sensors)
        if not sensors:
            raise ValueError("tactile contract contains no sensors")
        return cls(
            abi_version=abi_version,
            fingerprint=fingerprint,
            primary_representation=str(document["primary_representation"]),
            dense_channels=dense_channels,
            summary_channels=summary_channels,
            sensors=sensors,
            raw=document,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            dict(self.raw),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        ) + ("\n" if indent is not None else "")

    @property
    def total_sample_count(self) -> int:
        return sum(sensor.width * sensor.height for sensor in self.sensors)


@dataclass(frozen=True, slots=True)
class TactileLayout:
    capacity: int
    active_environment_count: int
    body_count: int
    shape_count: int
    sensor_count: int
    sample_count: int
    target_count: int
    contact_capacity_per_environment: int
    query_backend: int
    hardware_ray_queries_available: bool
    retained_bytes: int
    bytes_per_environment: int
    last_observe_milliseconds: float


@dataclass(frozen=True, slots=True)
class TactileDeviceBuffers:
    """Borrowed ``id<MTLBuffer>`` addresses for graph composition."""

    penetration_depth: int
    depth_velocity: int
    validity: int
    object_shape_ids: int
    # Zero in the default headless configuration.
    debug_hits: int
    summaries: int
    statuses: int


@dataclass(frozen=True, slots=True)
class TactileObservationSnapshot:
    penetration_depth_m: FloatArray
    depth_velocity_m_per_s: FloatArray
    validity: UInt32Array
    object_shape_ids: UInt32Array
    pose_position_and_timestamp: FloatArray
    pose_orientation: FloatArray
    net_force_and_contact_area: FloatArray
    net_torque_and_maximum_depth: FloatArray
    centroid_local_and_mean_depth: FloatArray
    centroid_world_and_active_count: FloatArray
    center_of_pressure_local_and_force_weight: FloatArray
    center_of_pressure_world_and_contact_count: FloatArray
    statistics_and_identity: UInt32Array

    @property
    def contact_mask(self) -> npt.NDArray[np.bool_]:
        result = (self.validity & CONTACT_VALID) != 0
        result.setflags(write=False)
        return result


class FrankaTactileObservation:
    """Persistent native tactile context for the canonical Franka fingertips."""

    def __init__(
        self,
        capacity: int,
        *,
        contact_capacity_per_environment: int = 128,
        library_path: str | os.PathLike[str] | None = None,
        metallib_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._bindings = _load_bindings(library_path)
        resolved_metallib = resolve_metallib_path(
            metallib_path,
            library_path=self._bindings.path,
        )
        self._handle = self._bindings.lib.mr_tactile_create_franka(
            _positive_uint32(capacity, "capacity"),
            _positive_uint32(
                contact_capacity_per_environment,
                "contact_capacity_per_environment",
            ),
            (
                os.fsencode(resolved_metallib)
                if resolved_metallib is not None
                else None
            ),
        )
        if not self._handle:
            raise MetalRoboError(self._bindings.last_error())
        try:
            self._contract = TactileObservationContract.from_json(
                _decode(
                    self._bindings.lib.mr_tactile_observation_metadata_json(
                        self._handle
                    )
                )
            )
        except BaseException:
            self.close()
            raise

    @property
    def contract(self) -> TactileObservationContract:
        return self._contract

    @property
    def device_name(self) -> str:
        self._require_open()
        return _decode(self._bindings.lib.mr_tactile_device_name(self._handle))

    @property
    def layout(self) -> TactileLayout:
        self._require_open()
        raw = self._bindings.lib.mr_tactile_layout(self._handle)
        return TactileLayout(
            capacity=int(raw.capacity),
            active_environment_count=int(raw.active_environment_count),
            body_count=int(raw.body_count),
            shape_count=int(raw.shape_count),
            sensor_count=int(raw.sensor_count),
            sample_count=int(raw.sample_count),
            target_count=int(raw.target_count),
            contact_capacity_per_environment=int(
                raw.contact_capacity_per_environment
            ),
            query_backend=int(raw.query_backend),
            hardware_ray_queries_available=bool(
                raw.hardware_ray_queries_available
            ),
            retained_bytes=int(raw.retained_bytes),
            bytes_per_environment=int(raw.bytes_per_environment),
            last_observe_milliseconds=float(raw.last_observe_milliseconds),
        )

    @property
    def device_buffers(self) -> TactileDeviceBuffers:
        self._require_open()
        addresses = tuple(
            int(self._bindings.lib.mr_tactile_native_buffer(self._handle, kind) or 0)
            for kind in range(7)
        )
        if any(
            address == 0
            for index, address in enumerate(addresses)
            if index != TACTILE_DEBUG_HITS
        ):
            raise MetalRoboError(self._bindings.last_error())
        return TactileDeviceBuffers(*addresses)

    def encode_device(
        self,
        *,
        body_states: int,
        environment_count: int,
        observation_timestep_seconds: float,
        frame_index: int,
        timestamp_seconds: float,
        metal_compute_command_encoder: int,
        contacts: int | None = None,
        contact_counts: int | None = None,
        reset_mask: int | None = None,
        contact_impulse_timestep_seconds: float = 0.0,
    ) -> None:
        """Encode without allocation, command commit, synchronization, or readback."""

        self._require_open()
        layout = self.layout
        environments = _positive_uint32(
            environment_count, "environment_count"
        )
        if environments > layout.capacity:
            raise ValueError("environment_count exceeds compiled tactile capacity")
        if (contacts is None) != (contact_counts is None):
            raise ValueError("contacts and contact_counts must be supplied together")
        if observation_timestep_seconds <= 0.0:
            raise ValueError("observation_timestep_seconds must be positive")
        if contacts is not None and contact_impulse_timestep_seconds <= 0.0:
            raise ValueError(
                "contact_impulse_timestep_seconds must be positive "
                "when solver contacts are supplied"
            )
        status = self._bindings.lib.mr_tactile_encode(
            self._handle,
            _pointer(body_states, "body_states"),
            _pointer(contacts, "contacts", optional=True),
            _pointer(contact_counts, "contact_counts", optional=True),
            _pointer(reset_mask, "reset_mask", optional=True),
            environments,
            layout.body_count,
            (
                layout.contact_capacity_per_environment
                if contacts is not None
                else 0
            ),
            float(observation_timestep_seconds),
            float(contact_impulse_timestep_seconds),
            int(frame_index),
            float(timestamp_seconds),
            _pointer(
                metal_compute_command_encoder,
                "metal_compute_command_encoder",
            ),
        )
        if status != 0:
            raise MetalRoboError(self._bindings.last_error())

    def readback(self) -> TactileObservationSnapshot:
        """Return stable host copies for inspection; never call in the RL hot path."""

        self._require_open()
        if self._bindings.lib.mr_tactile_readback(self._handle) != 0:
            raise MetalRoboError(self._bindings.last_error())
        layout = self.layout
        dense_shape = (
            layout.active_environment_count,
            layout.sample_count,
        )
        summary_shape = (
            layout.active_environment_count,
            layout.sensor_count,
        )
        dense_count = int(np.prod(dense_shape, dtype=np.int64))
        summary_count = int(np.prod(summary_shape, dtype=np.int64))

        def floats(name: str) -> FloatArray:
            pointer = getattr(self._bindings.lib, name)(self._handle)
            if not pointer:
                raise MetalRoboError(f"{name} returned no readback")
            result = np.ctypeslib.as_array(pointer, shape=(dense_count,)).copy()
            result.shape = dense_shape
            result.setflags(write=False)
            return result

        def uints(name: str) -> UInt32Array:
            pointer = getattr(self._bindings.lib, name)(self._handle)
            if not pointer:
                raise MetalRoboError(f"{name} returned no readback")
            result = np.ctypeslib.as_array(pointer, shape=(dense_count,)).copy()
            result.shape = dense_shape
            result.setflags(write=False)
            return result

        summaries = self._bindings.lib.mr_tactile_summaries(self._handle)
        if not summaries:
            raise MetalRoboError("native tactile summaries are unavailable")

        def summary_float(field: str) -> FloatArray:
            result = np.empty((summary_count, 4), dtype=np.float32)
            for index in range(summary_count):
                result[index] = tuple(getattr(summaries[index], field))
            result.shape = (*summary_shape, 4)
            result.setflags(write=False)
            return result

        statistics = np.empty((summary_count, 4), dtype=np.uint32)
        for index in range(summary_count):
            statistics[index] = tuple(
                summaries[index].statistics_and_identity
            )
        statistics.shape = (*summary_shape, 4)
        statistics.setflags(write=False)

        return TactileObservationSnapshot(
            penetration_depth_m=floats("mr_tactile_depth"),
            depth_velocity_m_per_s=floats("mr_tactile_depth_velocity"),
            validity=uints("mr_tactile_validity"),
            object_shape_ids=uints("mr_tactile_object_shape_ids"),
            pose_position_and_timestamp=summary_float(
                "pose_position_and_timestamp"
            ),
            pose_orientation=summary_float("pose_orientation"),
            net_force_and_contact_area=summary_float(
                "net_force_and_contact_area"
            ),
            net_torque_and_maximum_depth=summary_float(
                "net_torque_and_maximum_depth"
            ),
            centroid_local_and_mean_depth=summary_float(
                "centroid_local_and_mean_depth"
            ),
            centroid_world_and_active_count=summary_float(
                "centroid_world_and_active_count"
            ),
            center_of_pressure_local_and_force_weight=summary_float(
                "center_of_pressure_local_and_force_weight"
            ),
            center_of_pressure_world_and_contact_count=summary_float(
                "center_of_pressure_world_and_contact_count"
            ),
            statistics_and_identity=statistics,
        )

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._bindings.lib.mr_tactile_destroy(handle)
            self._handle = None

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None):
            raise RuntimeError("Franka tactile observation context is closed")

    def __enter__(self) -> "FrankaTactileObservation":
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class TactileCalibrationRecord:
    raw_frame_uri: str
    target_depth_uri: str
    sensor_id: str
    indenter_asset_id: str
    timestamp_s: float
    indentation_depth_m: float
    indenter_pose_sensor_xyzw: tuple[float, float, float, float, float, float, float]
    validity_uri: str | None = None
    force_torque_uri: str | None = None
    schema: str = "metalrobo.tactile_calibration"

    def validate(self) -> None:
        if (
            self.schema != "metalrobo.tactile_calibration"
            or not self.raw_frame_uri
            or not self.target_depth_uri
            or not self.sensor_id
            or not self.indenter_asset_id
            or not np.isfinite(self.timestamp_s)
            or not np.isfinite(self.indentation_depth_m)
            or self.indentation_depth_m < 0.0
            or len(self.indenter_pose_sensor_xyzw) != 7
            or not np.all(np.isfinite(self.indenter_pose_sensor_xyzw))
            or not np.isclose(
                np.linalg.norm(self.indenter_pose_sensor_xyzw[3:]),
                1.0,
                atol=1.0e-5,
            )
            or (
                self.validity_uri is not None
                and not self.validity_uri
            )
            or (
                self.force_torque_uri is not None
                and not self.force_torque_uri
            )
        ):
            raise ValueError("tactile calibration record is invalid")


def append_calibration_record(
    manifest: str | os.PathLike[str],
    record: TactileCalibrationRecord,
) -> None:
    """Append one paired real-frame/metric-depth calibration record."""

    record.validate()
    path = Path(manifest).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False))
        stream.write("\n")


def save_tactile_checkpoint_contract(
    checkpoint_directory: str | os.PathLike[str],
    contract: TactileObservationContract,
) -> Path:
    """Atomically pin the exact tactile schema beside a policy checkpoint."""

    directory = Path(checkpoint_directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "tactile_observation.json"
    temporary = directory / "tactile_observation.json.tmp"
    temporary.write_text(contract.to_json(), encoding="utf-8")
    temporary.replace(target)

    return target


def load_tactile_checkpoint_contract(
    checkpoint_directory: str | os.PathLike[str],
    *,
    expected_fingerprint: str | None = None,
) -> TactileObservationContract:
    path = Path(checkpoint_directory).expanduser() / "tactile_observation.json"
    contract = TactileObservationContract.from_json(
        path.read_text(encoding="utf-8")
    )
    if (
        expected_fingerprint is not None
        and contract.fingerprint != expected_fingerprint
    ):
        raise ValueError(
            "policy tactile fingerprint mismatch: "
            f"{contract.fingerprint} != {expected_fingerprint}"
        )
    return contract


def write_depth_preview_pgm(
    path: str | os.PathLike[str],
    depth_m: npt.ArrayLike,
    *,
    maximum_depth_m: float,
) -> Path:
    """Write an opt-in 16-bit preview; metric arrays remain authoritative."""

    if not np.isfinite(maximum_depth_m) or maximum_depth_m <= 0.0:
        raise ValueError("maximum_depth_m must be positive")
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or not np.all(np.isfinite(depth)):
        raise ValueError("depth preview must be one finite 2D map")
    quantized = np.rint(
        np.clip(depth, 0.0, maximum_depth_m)
        / maximum_depth_m
        * np.float32(65535.0)
    ).astype(">u2")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(f"P5\n{depth.shape[1]} {depth.shape[0]}\n65535\n".encode())
        stream.write(quantized.tobytes(order="C"))
    return output


__all__ = [
    "CONTACT_VALID",
    "DEPTH_SATURATED",
    "FILTERED_TARGET",
    "FrankaTactileObservation",
    "SAMPLE_VALID",
    "TACTILE_ABI_VERSION",
    "TactileCalibrationRecord",
    "TactileDeviceBuffers",
    "TactileLayout",
    "TactileObservationContract",
    "TactileObservationSnapshot",
    "TactileSensorContract",
    "append_calibration_record",
    "load_tactile_checkpoint_contract",
    "save_tactile_checkpoint_contract",
    "write_depth_preview_pgm",
]
