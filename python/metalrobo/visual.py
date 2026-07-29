"""Versioned visual observations, perception providers, and episode streams.

The module deliberately defines data contracts rather than a particular
vision model. Simulated renderer output and physical RGB-D capture use the
same :class:`VisualFrameBatchV1`; simulation-only labels remain in the
parallel :class:`VisualTruthBatchV1`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum, IntFlag
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol

import numpy as np
import numpy.typing as npt

VISUAL_FRAME_BATCH_VERSION = 1
PERCEPTION_PROVIDER_VERSION = 1
VISUAL_EPISODE_STREAM_VERSION = 1


class VisualFrameSource(IntEnum):
    SIMULATION = 0
    CAPTURE = 1
    REPLAY = 2


class VisualCoordinateFrame(IntEnum):
    PIXEL = 0
    CAMERA = 1
    ROBOT_BASE = 2
    WORLD = 3
    OBJECT = 4


class VisualModality(IntFlag):
    RGB = 1 << 0
    DEPTH = 1 << 1
    DEPTH_VALIDITY = 1 << 2
    NORMAL = 1 << 3
    MOTION = 1 << 4
    SEMANTIC = 1 << 5
    INSTANCE = 1 << 6
    LINK = 1 << 7
    KEYPOINT = 1 << 8
    FEATURE = 1 << 9
    OBJECT_POSE = 1 << 10


class PerceptionCapability(IntFlag):
    DENSE_DEPTH = 1 << 0
    SEMANTIC = 1 << 1
    INSTANCE = 1 << 2
    TRACKING = 1 << 3
    OBJECT_POSE = 1 << 4
    KEYPOINT = 1 << 5
    DENSE_FEATURE = 1 << 6
    EMBEDDING = 1 << 7


class ObservationProfileV1(str, Enum):
    RAW_RGBD = "raw_rgbd"
    RGB_XYZ = "rgb_xyz"
    OBJECT_CENTRIC = "object_centric"
    DENSE_FEATURES = "dense_features"
    COMPACT_LATENT = "compact_latent"


FloatArray = npt.NDArray[np.float32]
DoubleArray = npt.NDArray[np.float64]
UIntArray = npt.NDArray[np.uint32]
ByteArray = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]


def _as_array(
    value: npt.ArrayLike,
    dtype: npt.DTypeLike,
    *,
    name: str,
) -> np.ndarray[Any, Any]:
    result = np.ascontiguousarray(value, dtype=dtype)
    if result.dtype.kind == "f" and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    result.setflags(write=False)
    return result


def _as_bool_array(
    value: npt.ArrayLike,
    *,
    name: str,
) -> BoolArray:
    source = np.asarray(value)
    if source.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean values")
    return _as_array(source, np.bool_, name=name)


def _as_unsigned_array(
    value: npt.ArrayLike,
    dtype: npt.DTypeLike,
    *,
    name: str,
) -> np.ndarray[Any, Any]:
    source = np.asarray(value)
    target = np.dtype(dtype)
    if (
        target.kind != "u"
        or source.dtype.kind not in "iu"
        or (source.size and np.any(source < 0))
        or (source.size and np.any(source > np.iinfo(target).max))
    ):
        raise ValueError(f"{name} must contain exact {target.name} values")
    return _as_array(source, target, name=name)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(value: int, *, name: str) -> int:
    return _integer(
        value,
        name=name,
        minimum=1,
        maximum=(1 << 64) - 1,
    )


def _integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its supported range")
    return result


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is invalid")
    return result


def _array_schema(value: np.ndarray[Any, Any]) -> dict[str, Any]:
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape[1:]),
    }


_VISUAL_MODALITY_MASK = sum(int(value) for value in VisualModality)
_PERCEPTION_CAPABILITY_MASK = sum(int(value) for value in PerceptionCapability)


def _visual_modality(
    value: VisualModality | int,
    *,
    name: str,
    single: bool = False,
) -> VisualModality:
    raw = _integer(value, name=name, minimum=1)
    if raw <= 0 or raw & ~_VISUAL_MODALITY_MASK:
        raise ValueError(f"{name} contains unsupported visual modalities")
    if single and raw & (raw - 1):
        raise ValueError(f"{name} must identify exactly one modality")
    return VisualModality(raw)


def _perception_capabilities(
    value: PerceptionCapability | int,
) -> PerceptionCapability:
    raw = _integer(value, name="capabilities", minimum=1)
    if raw <= 0 or raw & ~_PERCEPTION_CAPABILITY_MASK:
        raise ValueError("perception capabilities contain unsupported flags")
    return PerceptionCapability(raw)


def _device_buffers(value: Mapping[str, int]) -> Mapping[str, int]:
    result = dict(value)
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(address, (bool, np.bool_))
        or not isinstance(address, (int, np.integer))
        or int(address) <= 0
        or int(address) > np.iinfo(np.uintp).max
        for name, address in result.items()
    ):
        raise ValueError(
            "device buffers require nonempty names and positive integer handles"
        )
    return MappingProxyType({name: int(address) for name, address in result.items()})


def _undistort_normalized(
    distorted_x: FloatArray,
    distorted_y: FloatArray,
    distortion: FloatArray,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Invert MetalRobo's Brown-Conrady projection deterministically."""

    k1, k2, p1, p2 = (float(value) for value in distortion)
    distorted_x64 = np.asarray(distorted_x, dtype=np.float64)
    distorted_y64 = np.asarray(distorted_y, dtype=np.float64)
    x = np.array(distorted_x64, copy=True)
    y = np.array(distorted_y64, copy=True)
    projectable = np.ones(x.shape, dtype=np.bool_)
    for _ in range(8):
        radius_squared = x * x + y * y
        radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
        projectable &= np.isfinite(radial) & (np.abs(radial) > 1.0e-8)
        tangential_x = 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x)
        tangential_y = p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y
        safe_radial = np.where(projectable, radial, 1.0)
        x = (distorted_x64 - tangential_x) / safe_radial
        y = (distorted_y64 - tangential_y) / safe_radial
        projectable &= np.isfinite(x) & np.isfinite(y)
    radius_squared = x * x + y * y
    radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
    projected_x = x * radial + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x)
    projected_y = y * radial + p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y
    projectable &= (
        np.isfinite(projected_x)
        & np.isfinite(projected_y)
        & (np.abs(projected_x - distorted_x64) <= 1.0e-4)
        & (np.abs(projected_y - distorted_y64) <= 1.0e-4)
    )
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        projectable,
    )


@dataclass(frozen=True, slots=True)
class VisualCameraFrameV1:
    sensor_id: str
    intrinsics: FloatArray
    distortion: FloatArray
    camera_to_base: FloatArray
    capture_timestamp_seconds: float
    frame_age_seconds: float
    exposure_seconds: float
    shutter_readout_seconds: float
    frame_index: int
    sensor_sequence: int
    valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise ValueError("sensor_id cannot be empty")
        if not isinstance(self.valid, (bool, np.bool_)):
            raise ValueError("camera validity must be boolean")
        intrinsics = _as_array(
            self.intrinsics,
            np.float32,
            name="intrinsics",
        )
        distortion = _as_array(
            self.distortion,
            np.float32,
            name="distortion",
        )
        camera_to_base = _as_array(
            self.camera_to_base,
            np.float32,
            name="camera_to_base",
        )
        if intrinsics.shape != (4,) or np.any(intrinsics[:2] <= 0):
            raise ValueError("intrinsics must be finite [fx, fy, cx, cy]")
        if distortion.shape != (4,):
            raise ValueError("distortion must have shape [4]")
        if camera_to_base.shape != (4, 4):
            raise ValueError("camera_to_base must have shape [4, 4]")
        if not np.allclose(
            camera_to_base[3],
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            atol=1.0e-5,
        ):
            raise ValueError("camera_to_base is not homogeneous")
        rotation = camera_to_base[:3, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3, dtype=np.float32),
            rtol=0.0,
            atol=1.0e-4,
        ) or not np.isclose(
            np.linalg.det(rotation),
            1.0,
            rtol=0.0,
            atol=1.0e-4,
        ):
            raise ValueError("camera_to_base rotation is not rigid")
        capture_timestamp_seconds = _number(
            self.capture_timestamp_seconds,
            name="capture_timestamp_seconds",
        )
        frame_age_seconds = _number(
            self.frame_age_seconds,
            name="frame_age_seconds",
            minimum=0.0,
        )
        exposure_seconds = _number(
            self.exposure_seconds,
            name="exposure_seconds",
            minimum=0.0,
        )
        shutter_readout_seconds = _number(
            self.shutter_readout_seconds,
            name="shutter_readout_seconds",
            minimum=0.0,
        )
        frame_index = _integer(
            self.frame_index,
            name="frame_index",
            maximum=(1 << 64) - 1,
        )
        sensor_sequence = _integer(
            self.sensor_sequence,
            name="sensor_sequence",
            maximum=(1 << 32) - 1,
        )
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "camera_to_base", camera_to_base)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "sensor_sequence", sensor_sequence)
        object.__setattr__(
            self,
            "capture_timestamp_seconds",
            capture_timestamp_seconds,
        )
        object.__setattr__(
            self,
            "frame_age_seconds",
            frame_age_seconds,
        )
        object.__setattr__(
            self,
            "exposure_seconds",
            exposure_seconds,
        )
        object.__setattr__(
            self,
            "shutter_readout_seconds",
            shutter_readout_seconds,
        )
        object.__setattr__(self, "valid", bool(self.valid))


@dataclass(frozen=True, slots=True)
class VisualBatchProvenanceV1:
    episode_twin_fingerprint: int
    scenario_fingerprint: int
    renderer_fingerprint: int
    sensor_profile_fingerprint: int
    calibration_fingerprint: int

    def __post_init__(self) -> None:
        for name in (
            "episode_twin_fingerprint",
            "scenario_fingerprint",
            "renderer_fingerprint",
            "sensor_profile_fingerprint",
            "calibration_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _fingerprint(getattr(self, name), name=name),
            )


@dataclass(frozen=True, slots=True)
class VisualFrameBatchV1:
    source: VisualFrameSource
    provenance: VisualBatchProvenanceV1
    cameras: tuple[VisualCameraFrameV1, ...]
    rgb_linear: FloatArray
    depth_meters: FloatArray
    depth_validity: BoolArray
    device_buffers: Mapping[str, int] = field(default_factory=dict)
    schema_version: int = VISUAL_FRAME_BATCH_VERSION

    def __post_init__(self) -> None:
        schema_version = _integer(
            self.schema_version,
            name="schema_version",
            maximum=VISUAL_FRAME_BATCH_VERSION,
        )
        if schema_version != VISUAL_FRAME_BATCH_VERSION:
            raise ValueError("unsupported VisualFrameBatch version")
        try:
            source = VisualFrameSource(
                _integer(
                    self.source,
                    name="source",
                    maximum=int(VisualFrameSource.REPLAY),
                )
            )
        except ValueError as error:
            raise ValueError("unsupported visual frame source") from error
        if not isinstance(self.provenance, VisualBatchProvenanceV1):
            raise TypeError("provenance must be VisualBatchProvenanceV1")
        cameras = tuple(self.cameras)
        if not all(isinstance(camera, VisualCameraFrameV1) for camera in cameras):
            raise TypeError("cameras must contain VisualCameraFrameV1")
        rgb = _as_array(self.rgb_linear, np.float32, name="rgb_linear")
        depth = np.ascontiguousarray(self.depth_meters, dtype=np.float32)
        validity = _as_bool_array(
            self.depth_validity,
            name="depth_validity",
        )
        if (
            rgb.ndim != 5
            or rgb.shape[-1] != 4
            or any(dimension <= 0 for dimension in rgb.shape[:-1])
        ):
            raise ValueError(
                "rgb_linear must have shape [environment, view, height, width, 4]"
            )
        expected = rgb.shape[:-1]
        if depth.shape != expected or validity.shape != expected:
            raise ValueError("depth and validity must match RGB dimensions")
        if len(cameras) != rgb.shape[0] * rgb.shape[1]:
            raise ValueError("camera metadata must be environment-view major")
        if np.any(validity & (~np.isfinite(depth) | (depth <= 0.0))):
            raise ValueError("valid depth samples must be positive and finite")
        depth = np.where(validity, depth, 0.0).astype(
            np.float32,
            copy=False,
        )
        reference = cameras[0]
        sensor_ids = [cameras[view].sensor_id for view in range(rgb.shape[1])]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("sensor ids must be unique across views")
        for environment in range(rgb.shape[0]):
            environment_reference = cameras[environment * rgb.shape[1]]
            observation_timestamp_seconds = (
                environment_reference.capture_timestamp_seconds
                + environment_reference.frame_age_seconds
            )
            for view in range(rgb.shape[1]):
                camera = cameras[environment * rgb.shape[1] + view]
                if camera.sensor_id != sensor_ids[view]:
                    raise ValueError("sensor ids must be stable across environments")
                if camera.frame_index != reference.frame_index:
                    raise ValueError("camera frames must share the batch frame index")
                if not np.isclose(
                    camera.capture_timestamp_seconds + camera.frame_age_seconds,
                    observation_timestamp_seconds,
                    rtol=0.0,
                    atol=1.0e-6,
                ):
                    raise ValueError(
                        "camera capture times and frame ages are not aligned"
                    )
                if not camera.valid and np.any(validity[environment, view]):
                    raise ValueError("an invalid camera cannot publish valid depth")
        depth.setflags(write=False)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "rgb_linear", rgb)
        object.__setattr__(self, "depth_meters", depth)
        object.__setattr__(self, "depth_validity", validity)
        object.__setattr__(
            self,
            "device_buffers",
            _device_buffers(self.device_buffers),
        )

    @property
    def environment_count(self) -> int:
        return int(self.rgb_linear.shape[0])

    @property
    def view_count(self) -> int:
        return int(self.rgb_linear.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb_linear.shape[2])

    @property
    def width(self) -> int:
        return int(self.rgb_linear.shape[3])

    def camera(self, environment: int, view: int) -> VisualCameraFrameV1:
        if not 0 <= environment < self.environment_count:
            raise IndexError("environment index is out of range")
        if not 0 <= view < self.view_count:
            raise IndexError("view index is out of range")
        return self.cameras[environment * self.view_count + view]


@dataclass(frozen=True, slots=True)
class VisualTruthBatchV1:
    coordinate_frame: VisualCoordinateFrame
    normals: FloatArray
    motion: FloatArray
    semantic_ids: UIntArray
    instance_ids: UIntArray
    link_ids: UIntArray
    visibility: FloatArray
    occlusion: ByteArray
    object_poses: DoubleArray
    link_poses: DoubleArray
    keypoints: DoubleArray
    contacts: DoubleArray
    frame_index: int
    timestamp_seconds: float
    device_buffers: Mapping[str, int] = field(default_factory=dict)
    schema_version: int = VISUAL_FRAME_BATCH_VERSION

    def __post_init__(self) -> None:
        schema_version = _integer(
            self.schema_version,
            name="schema_version",
            maximum=VISUAL_FRAME_BATCH_VERSION,
        )
        if schema_version != VISUAL_FRAME_BATCH_VERSION:
            raise ValueError("unsupported VisualTruthBatch version")
        try:
            coordinate_frame = VisualCoordinateFrame(
                _integer(
                    self.coordinate_frame,
                    name="coordinate_frame",
                    maximum=int(VisualCoordinateFrame.OBJECT),
                )
            )
        except ValueError as error:
            raise ValueError("unsupported visual truth coordinate frame") from error
        frame_index = _integer(
            self.frame_index,
            name="truth frame_index",
            maximum=(1 << 64) - 1,
        )
        timestamp_seconds = _number(
            self.timestamp_seconds,
            name="truth timestamp",
        )
        normals = _as_array(self.normals, np.float32, name="normals")
        motion = _as_array(self.motion, np.float32, name="motion")
        semantic = _as_unsigned_array(
            self.semantic_ids,
            np.uint32,
            name="semantic_ids",
        )
        instance = _as_unsigned_array(
            self.instance_ids,
            np.uint32,
            name="instance_ids",
        )
        link = _as_unsigned_array(
            self.link_ids,
            np.uint32,
            name="link_ids",
        )
        visibility = _as_array(
            self.visibility,
            np.float32,
            name="visibility",
        )
        occlusion = _as_unsigned_array(
            self.occlusion,
            np.uint8,
            name="occlusion",
        )
        if normals.ndim != 5 or normals.shape[-1] != 4:
            raise ValueError("normals must have shape [E, V, H, W, 4]")
        if motion.shape != normals.shape:
            raise ValueError("motion must match normal shape")
        dense_shape = normals.shape[:-1]
        if any(dimension <= 0 for dimension in dense_shape):
            raise ValueError("visual truth dimensions must be positive")
        for name, value in (
            ("semantic_ids", semantic),
            ("instance_ids", instance),
            ("link_ids", link),
            ("visibility", visibility),
            ("occlusion", occlusion),
        ):
            if value.shape != dense_shape:
                raise ValueError(f"{name} must match dense truth shape")
        if np.any((visibility < 0.0) | (visibility > 1.0)):
            raise ValueError("visibility must be in [0, 1]")
        expected_occlusion = np.rint((1.0 - visibility) * 255.0).astype(np.int16)
        if np.any(np.abs(occlusion.astype(np.int16) - expected_occlusion) > 1):
            raise ValueError("visibility and occlusion must be complementary")
        packed = {}
        for name, value, width in (
            ("object_poses", self.object_poses, 11),
            ("link_poses", self.link_poses, 11),
            ("keypoints", self.keypoints, 8),
            ("contacts", self.contacts, 12),
        ):
            array = _as_array(value, np.float64, name=name)
            if array.ndim != 3 or array.shape[0] != dense_shape[0]:
                raise ValueError(f"{name} must be environment-major")
            if array.shape[-1] != width:
                raise ValueError(f"{name} record width must be {width}")
            identities = array[..., -4:]
            if np.any(
                (identities < 0.0)
                | (identities > np.iinfo(np.uint32).max)
                | (identities != np.floor(identities))
            ):
                raise ValueError(f"{name} identity fields must be exact uint32 values")
            packed[name] = array
        for name in ("object_poses", "link_poses"):
            orientations = packed[name][..., 3:7]
            if orientations.size and not np.allclose(
                np.linalg.norm(orientations, axis=-1),
                1.0,
                rtol=0.0,
                atol=1.0e-4,
            ):
                raise ValueError(f"{name} contains a nonunit orientation")
        object_identities = packed["object_poses"][..., -4:-2]
        if object_identities.size and np.any(
            (object_identities == 0.0) | (object_identities == np.iinfo(np.uint32).max)
        ):
            raise ValueError("object poses require semantic and instance identities")
        link_identities = packed["link_poses"][..., -2]
        if link_identities.size and np.any(link_identities == np.iinfo(np.uint32).max):
            raise ValueError("link poses require link identities")
        keypoint_identities = packed["keypoints"][..., -4:-2]
        if keypoint_identities.size and np.any(
            (keypoint_identities == 0.0)
            | (keypoint_identities == np.iinfo(np.uint32).max)
        ):
            raise ValueError("keypoints require semantic and instance identities")
        contact_identities = packed["contacts"][..., -4:-2]
        contact_normals = packed["contacts"][..., 4:7]
        if packed["contacts"].size and (
            np.any(packed["contacts"][..., 3] < 0.0)
            or np.any(
                (contact_identities == 0.0)
                | (contact_identities == np.iinfo(np.uint32).max)
            )
            or not np.allclose(
                np.linalg.norm(contact_normals, axis=-1),
                1.0,
                rtol=0.0,
                atol=1.0e-4,
            )
        ):
            raise ValueError("contact annotations are invalid")
        if np.any(
            (packed["keypoints"][..., 3] < 0.0) | (packed["keypoints"][..., 3] > 1.0)
        ):
            raise ValueError("keypoint visibility must be in [0, 1]")
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "motion", motion)
        object.__setattr__(self, "semantic_ids", semantic)
        object.__setattr__(self, "instance_ids", instance)
        object.__setattr__(self, "link_ids", link)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "occlusion", occlusion)
        for name, value in packed.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "timestamp_seconds", timestamp_seconds)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "device_buffers",
            _device_buffers(self.device_buffers),
        )


@dataclass(frozen=True, slots=True)
class PerceptionProviderDescriptorV1:
    provider_id: str
    content_hash: str
    input_modalities: VisualModality
    capabilities: PerceptionCapability
    temporal_window: int = 1
    accepts_device_buffers: bool = False
    schema_version: int = PERCEPTION_PROVIDER_VERSION

    def __post_init__(self) -> None:
        schema_version = _integer(
            self.schema_version,
            name="schema_version",
            maximum=PERCEPTION_PROVIDER_VERSION,
        )
        temporal_window = _integer(
            self.temporal_window,
            name="temporal_window",
            minimum=1,
            maximum=(1 << 32) - 1,
        )
        input_modalities = _visual_modality(
            self.input_modalities,
            name="input_modalities",
        )
        capabilities = _perception_capabilities(self.capabilities)
        available_inputs = (
            VisualModality.RGB | VisualModality.DEPTH | VisualModality.DEPTH_VALIDITY
        )
        if (
            schema_version != PERCEPTION_PROVIDER_VERSION
            or not isinstance(self.provider_id, str)
            or not self.provider_id
            or not isinstance(self.content_hash, str)
            or not self.content_hash
            or not isinstance(self.accepts_device_buffers, (bool, np.bool_))
            or input_modalities & ~available_inputs
        ):
            raise ValueError("perception provider descriptor is invalid")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "temporal_window", temporal_window)
        object.__setattr__(self, "input_modalities", input_modalities)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "accepts_device_buffers",
            bool(self.accepts_device_buffers),
        )


@dataclass(frozen=True, slots=True)
class PerceptionTensorV1:
    tensor_id: str
    modality: VisualModality
    coordinate_frame: VisualCoordinateFrame
    values: np.ndarray[Any, Any]
    timestamp_seconds: float
    confidence: FloatArray | float = 1.0
    valid: BoolArray | bool = True
    device_buffers: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tensor_id, str) or not self.tensor_id:
            raise ValueError("perception tensor identity is invalid")
        modality = _visual_modality(
            self.modality,
            name="perception tensor modality",
            single=True,
        )
        try:
            coordinate_frame = VisualCoordinateFrame(
                _integer(
                    self.coordinate_frame,
                    name="coordinate_frame",
                    maximum=int(VisualCoordinateFrame.OBJECT),
                )
            )
        except ValueError as error:
            raise ValueError(
                "unsupported perception tensor coordinate frame"
            ) from error
        timestamp_seconds = _number(
            self.timestamp_seconds,
            name="perception tensor timestamp",
        )
        values = np.ascontiguousarray(self.values)
        if values.dtype.kind == "f":
            values = np.ascontiguousarray(values, dtype=np.float32)
        elif values.dtype.kind == "b":
            values = np.ascontiguousarray(values, dtype=np.uint8)
        elif values.dtype.kind in "iu":
            if values.size and (
                np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max)
            ):
                raise ValueError("perception integer tensor exceeds uint32")
            values = np.ascontiguousarray(values, dtype=np.uint32)
        else:
            raise ValueError("perception tensor values are unsupported")
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise ValueError("perception tensor contains nonfinite values")
        confidence = _as_array(
            self.confidence,
            np.float32,
            name="confidence",
        )
        valid = _as_bool_array(self.valid, name="valid")
        if np.ndim(self.confidence) == 0:
            confidence = confidence.reshape(())
        if np.ndim(self.valid) == 0:
            valid = valid.reshape(())
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("perception confidence must be in [0, 1]")
        allowed_metadata_shapes = {
            (),
            values.shape,
            values.shape[:-1],
        }
        if (
            confidence.shape not in allowed_metadata_shapes
            or valid.shape not in allowed_metadata_shapes
        ):
            raise ValueError(
                "perception confidence and validity must be scalar or "
                "aligned with tensor records"
            )
        device_buffers = _device_buffers(self.device_buffers)
        if values.size == 0 and device_buffers:
            raise ValueError("an empty perception tensor cannot expose device storage")
        values.setflags(write=False)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "timestamp_seconds", timestamp_seconds)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(
            self,
            "device_buffers",
            device_buffers,
        )


@dataclass(frozen=True, slots=True)
class PerceptionResultBatchV1:
    provider_id: str
    provider_content_hash: str
    frame_index: int
    timestamp_seconds: float
    tensors: tuple[PerceptionTensorV1, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.provider_id, str)
            or not self.provider_id
            or not isinstance(self.provider_content_hash, str)
            or not self.provider_content_hash
        ):
            raise ValueError("perception result identity is invalid")
        frame_index = _integer(
            self.frame_index,
            name="perception frame_index",
            maximum=(1 << 64) - 1,
        )
        timestamp_seconds = _number(
            self.timestamp_seconds,
            name="perception result timestamp",
        )
        tensors = tuple(self.tensors)
        if not all(isinstance(tensor, PerceptionTensorV1) for tensor in tensors):
            raise TypeError("perception results must contain PerceptionTensorV1")
        ids = [tensor.tensor_id for tensor in tensors]
        if len(ids) != len(set(ids)):
            raise ValueError("perception tensor ids must be unique")
        if any(
            not np.isclose(
                tensor.timestamp_seconds,
                timestamp_seconds,
                rtol=0.0,
                atol=1.0e-6,
            )
            for tensor in tensors
        ):
            raise ValueError(
                "perception tensor timestamps must match their result batch"
            )
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "timestamp_seconds", timestamp_seconds)
        object.__setattr__(self, "tensors", tensors)

    def first(self, modality: VisualModality) -> PerceptionTensorV1 | None:
        return next(
            (
                tensor
                for tensor in self.tensors
                if tensor.modality & modality and np.all(tensor.valid)
            ),
            None,
        )


class PerceptionProviderV1(Protocol):
    @property
    def descriptor(self) -> PerceptionProviderDescriptorV1: ...

    def infer(self, frames: VisualFrameBatchV1) -> PerceptionResultBatchV1: ...


@dataclass(slots=True)
class CallablePerceptionProviderV1:
    """Adapter for Core ML, MLX, YOLO-family, or remote provider callables."""

    descriptor: PerceptionProviderDescriptorV1
    function: Callable[[VisualFrameBatchV1], PerceptionResultBatchV1]

    def infer(self, frames: VisualFrameBatchV1) -> PerceptionResultBatchV1:
        result = self.function(frames)
        if not isinstance(result, PerceptionResultBatchV1):
            raise TypeError("provider callable must return PerceptionResultBatchV1")
        reference = frames.cameras[0]
        if (
            result.provider_id != self.descriptor.provider_id
            or result.provider_content_hash != self.descriptor.content_hash
            or result.frame_index != reference.frame_index
            or not np.isclose(
                result.timestamp_seconds,
                reference.capture_timestamp_seconds,
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            raise ValueError(
                "provider output provenance or frame synchronization is invalid"
            )
        capability_by_modality = {
            VisualModality.DEPTH: PerceptionCapability.DENSE_DEPTH,
            VisualModality.DEPTH_VALIDITY: PerceptionCapability.DENSE_DEPTH,
            VisualModality.SEMANTIC: PerceptionCapability.SEMANTIC,
            VisualModality.INSTANCE: PerceptionCapability.INSTANCE,
            VisualModality.MOTION: PerceptionCapability.TRACKING,
            VisualModality.OBJECT_POSE: PerceptionCapability.OBJECT_POSE,
            VisualModality.KEYPOINT: PerceptionCapability.KEYPOINT,
        }
        for tensor in result.tensors:
            if tensor.modality == VisualModality.FEATURE:
                supported = self.descriptor.capabilities & (
                    PerceptionCapability.DENSE_FEATURE | PerceptionCapability.EMBEDDING
                )
            else:
                required = capability_by_modality.get(tensor.modality)
                supported = (
                    required is not None and self.descriptor.capabilities & required
                )
            if not supported:
                raise ValueError("provider emitted a capability it did not advertise")
        return result


@dataclass(frozen=True, slots=True)
class PolicyObservationBatchV1:
    profile: ObservationProfileV1
    deployable: FloatArray
    privileged: FloatArray
    frame_index: int
    timestamp_seconds: float

    def __post_init__(self) -> None:
        try:
            profile = ObservationProfileV1(self.profile)
        except ValueError as error:
            raise ValueError("unsupported observation profile") from error
        deployable = _as_array(
            self.deployable,
            np.float32,
            name="deployable",
        )
        privileged = _as_array(
            self.privileged,
            np.float32,
            name="privileged",
        )
        if (
            deployable.ndim != 2
            or privileged.ndim != 2
            or deployable.shape[0] == 0
            or deployable.shape[1] == 0
            or privileged.shape[0] != deployable.shape[0]
        ):
            raise ValueError(
                "policy observations must be aligned environment-major matrices"
            )
        frame_index = _integer(
            self.frame_index,
            name="policy frame_index",
            maximum=(1 << 64) - 1,
        )
        timestamp_seconds = _number(
            self.timestamp_seconds,
            name="policy timestamp",
        )
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "deployable", deployable)
        object.__setattr__(self, "privileged", privileged)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "timestamp_seconds", timestamp_seconds)


class PolicyObservationAssemblerV1:
    """Assemble deployable and privileged observations without mixing them."""

    def assemble(
        self,
        frames: VisualFrameBatchV1,
        *,
        profile: ObservationProfileV1,
        perception: PerceptionResultBatchV1 | None = None,
        proprioception: npt.ArrayLike = (),
        previous_actions: npt.ArrayLike = (),
        task_commands: npt.ArrayLike = (),
        privileged_state: npt.ArrayLike = (),
    ) -> PolicyObservationBatchV1:
        try:
            profile = ObservationProfileV1(profile)
        except ValueError as error:
            raise ValueError("unsupported observation profile") from error
        environment_count = frames.environment_count

        def side(value: npt.ArrayLike, name: str) -> FloatArray:
            array = _as_array(value, np.float32, name=name)
            if array.size == 0:
                array = np.empty((environment_count, 0), dtype=np.float32)
            if array.ndim != 2 or array.shape[0] != environment_count:
                raise ValueError(f"{name} must have shape [environment, width]")
            return array

        proprio = side(proprioception, "proprioception")
        previous = side(previous_actions, "previous_actions")
        commands = side(task_commands, "task_commands")
        privileged = side(privileged_state, "privileged_state")

        rgb = frames.rgb_linear[..., :3]
        valid = frames.depth_validity[..., None]
        depth = np.where(valid, frames.depth_meters[..., None], 0.0)
        if profile == ObservationProfileV1.RAW_RGBD:
            visual = np.concatenate(
                (rgb, depth, valid.astype(np.float32)),
                axis=-1,
            )
        elif profile == ObservationProfileV1.RGB_XYZ:
            xyz = np.zeros((*frames.depth_meters.shape, 3), dtype=np.float32)
            xyz_valid = np.array(valid, copy=True)
            y, x = np.indices((frames.height, frames.width), dtype=np.float32)
            x += 0.5
            y += 0.5
            for environment in range(environment_count):
                for view in range(frames.view_count):
                    camera = frames.camera(environment, view)
                    fx, fy, cx, cy = camera.intrinsics
                    z = frames.depth_meters[environment, view]
                    normalized_x, normalized_y, projectable = _undistort_normalized(
                        (x - cx) / fx,
                        (y - cy) / fy,
                        camera.distortion,
                    )
                    camera_points = np.stack(
                        (
                            normalized_x * z,
                            normalized_y * z,
                            z,
                            np.ones_like(z),
                        ),
                        axis=-1,
                    )
                    base_points = (camera_points @ camera.camera_to_base.T)[..., :3]
                    point_valid = valid[environment, view] & projectable[..., None]
                    xyz_valid[environment, view] = point_valid
                    xyz[environment, view] = np.where(
                        point_valid,
                        base_points,
                        0.0,
                    )
            visual = np.concatenate(
                (rgb, xyz, xyz_valid.astype(np.float32)),
                axis=-1,
            )
        else:
            if perception is None:
                raise ValueError("selected profile requires perception results")
            reference = frames.cameras[0]
            if perception.frame_index != reference.frame_index or not np.isclose(
                perception.timestamp_seconds,
                reference.capture_timestamp_seconds,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "perception result does not match the visual frame batch"
                )
            modality = (
                VisualModality.OBJECT_POSE
                if profile == ObservationProfileV1.OBJECT_CENTRIC
                else VisualModality.FEATURE
            )
            selected = perception.first(modality)
            if selected is None:
                raise ValueError("perception result lacks the requested modality")
            if selected.values.dtype != np.dtype(np.float32):
                raise ValueError(
                    "policy perception tensors must contain float32 values"
                )
            values = np.asarray(selected.values)
            if values.ndim == 0 or values.shape[0] != environment_count:
                raise ValueError("perception tensor is not environment-major")
            visual = values
        flattened = np.ascontiguousarray(
            visual.reshape(environment_count, -1),
            dtype=np.float32,
        )
        deployable = np.concatenate(
            (flattened, proprio, previous, commands),
            axis=1,
        )
        deployable.setflags(write=False)
        privileged.setflags(write=False)
        camera = frames.cameras[0]
        return PolicyObservationBatchV1(
            profile=profile,
            deployable=deployable,
            privileged=privileged,
            frame_index=camera.frame_index,
            timestamp_seconds=camera.capture_timestamp_seconds,
        )


@dataclass(frozen=True, slots=True)
class VisualEpisodeProvenanceV1:
    episode_id: str
    source: VisualFrameSource
    episode_twin_fingerprint: int
    world_family_fingerprint: int
    scenario_fingerprint: int
    renderer_fingerprint: int
    visual_scene_fingerprint: int
    sensor_profile_fingerprint: int
    calibration_fingerprint: int
    physics_fingerprint: int
    visual_asset_hashes: tuple[str, ...] = ()
    nominal_rate_hz: float = 15.0

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("visual episode provenance is invalid")
        try:
            source = VisualFrameSource(
                _integer(
                    self.source,
                    name="source",
                    maximum=int(VisualFrameSource.REPLAY),
                )
            )
        except ValueError as error:
            raise ValueError("unsupported visual episode source") from error
        nominal_rate_hz = _number(
            self.nominal_rate_hz,
            name="nominal_rate_hz",
            minimum=np.finfo(np.float64).tiny,
        )
        if not isinstance(self.visual_asset_hashes, (tuple, list)):
            raise ValueError("visual asset hashes must be a sequence")
        visual_asset_hashes = tuple(self.visual_asset_hashes)
        for name in (
            "episode_twin_fingerprint",
            "world_family_fingerprint",
            "scenario_fingerprint",
            "renderer_fingerprint",
            "visual_scene_fingerprint",
            "sensor_profile_fingerprint",
            "calibration_fingerprint",
            "physics_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _fingerprint(getattr(self, name), name=name),
            )
        if any(
            not isinstance(value, str) or not value for value in visual_asset_hashes
        ) or len(visual_asset_hashes) != len(set(visual_asset_hashes)):
            raise ValueError("visual asset hashes must be nonempty and unique")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "nominal_rate_hz", nominal_rate_hz)
        object.__setattr__(
            self,
            "visual_asset_hashes",
            visual_asset_hashes,
        )


@dataclass(frozen=True, slots=True)
class VisualEpisodeStepV1:
    frames: VisualFrameBatchV1
    proprioception: FloatArray
    action: FloatArray
    task_command: FloatArray
    reward: FloatArray
    privileged_state: FloatArray
    scenario_key: npt.NDArray[np.uint64]
    events: UIntArray
    truth: VisualTruthBatchV1 | None = None
    physics_state: Mapping[str, npt.ArrayLike] = field(default_factory=dict)


def _deterministic_npz(arrays: Mapping[str, np.ndarray[Any, Any]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array_bytes = io.BytesIO()
            np.lib.format.write_array(
                array_bytes,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
            )
            entry = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            entry.compress_type = zipfile.ZIP_STORED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, array_bytes.getvalue())
    return output.getvalue()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(value)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class VisualEpisodeWriterV1:
    """Content-addressed deterministic chunks for online or captured episodes."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        provenance: VisualEpisodeProvenanceV1,
        *,
        chunk_steps: int = 64,
    ) -> None:
        chunk_steps = _integer(
            chunk_steps,
            name="chunk_steps",
            minimum=1,
        )
        if not isinstance(provenance, VisualEpisodeProvenanceV1):
            raise TypeError("provenance must be VisualEpisodeProvenanceV1")
        self.root = Path(root).expanduser().resolve()
        self.provenance = provenance
        self.chunk_steps = chunk_steps
        self._buffer: dict[str, list[np.ndarray[Any, Any]]] = {}
        self._chunks: list[dict[str, Any]] = []
        self._step_count = 0
        self._layout: dict[str, Any] | None = None
        self._finalized = False
        self._last_frame_index: int | None = None
        self._last_camera_timestamps: DoubleArray | None = None
        self._last_sensor_sequences: UIntArray | None = None
        self._scenario_keys: npt.NDArray[np.uint64] | None = None

    def _step_arrays(
        self,
        step: VisualEpisodeStepV1,
    ) -> dict[str, np.ndarray[Any, Any]]:
        frames = step.frames
        if not isinstance(frames, VisualFrameBatchV1):
            raise TypeError("episode frames must be VisualFrameBatchV1")
        if step.truth is not None and not isinstance(
            step.truth,
            VisualTruthBatchV1,
        ):
            raise TypeError("episode truth must be VisualTruthBatchV1")
        environment_count = frames.environment_count
        if frames.source != self.provenance.source:
            raise ValueError("frame source does not match episode provenance")
        frame_provenance = frames.provenance
        expected_provenance = (
            self.provenance.episode_twin_fingerprint,
            self.provenance.scenario_fingerprint,
            self.provenance.renderer_fingerprint,
            self.provenance.sensor_profile_fingerprint,
            self.provenance.calibration_fingerprint,
        )
        observed_provenance = (
            frame_provenance.episode_twin_fingerprint,
            frame_provenance.scenario_fingerprint,
            frame_provenance.renderer_fingerprint,
            frame_provenance.sensor_profile_fingerprint,
            frame_provenance.calibration_fingerprint,
        )
        if observed_provenance != expected_provenance:
            raise ValueError("frame provenance does not match episode provenance")

        def packed(
            value: npt.ArrayLike,
            name: str,
            dtype: npt.DTypeLike,
            *,
            matrix: bool = True,
        ) -> np.ndarray[Any, Any]:
            source = np.asarray(value)
            target_dtype = np.dtype(dtype)
            if target_dtype.kind == "u" and (
                source.dtype.kind not in "iu" or (source.size and np.any(source < 0))
            ):
                raise ValueError(f"{name} must contain unsigned integers")
            array = np.ascontiguousarray(source, dtype=target_dtype)
            if array.ndim == 0:
                raise ValueError(f"{name} must be environment-major")
            if array.ndim == 1:
                array = array[:, None]
            if matrix and array.ndim != 2:
                raise ValueError(f"{name} must have shape [environment, width]")
            if array.shape[0] != environment_count:
                raise ValueError(f"{name} must be environment-major")
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains nonfinite values")
            return array

        intrinsics = np.stack(
            [camera.intrinsics for camera in frames.cameras],
        ).reshape(environment_count, frames.view_count, 4)
        distortion = np.stack(
            [camera.distortion for camera in frames.cameras],
        ).reshape(environment_count, frames.view_count, 4)
        camera_to_base = np.stack(
            [camera.camera_to_base for camera in frames.cameras],
        ).reshape(environment_count, frames.view_count, 4, 4)
        capture = np.asarray(
            [camera.capture_timestamp_seconds for camera in frames.cameras],
            dtype=np.float64,
        ).reshape(environment_count, frames.view_count)
        frame_age = np.asarray(
            [camera.frame_age_seconds for camera in frames.cameras],
            dtype=np.float32,
        ).reshape(environment_count, frames.view_count)
        exposure = np.asarray(
            [camera.exposure_seconds for camera in frames.cameras],
            dtype=np.float32,
        ).reshape(environment_count, frames.view_count)
        shutter_readout = np.asarray(
            [camera.shutter_readout_seconds for camera in frames.cameras],
            dtype=np.float32,
        ).reshape(environment_count, frames.view_count)
        frame_index = np.asarray(
            [camera.frame_index for camera in frames.cameras],
            dtype=np.uint64,
        ).reshape(environment_count, frames.view_count)
        sensor_sequence = np.asarray(
            [camera.sensor_sequence for camera in frames.cameras],
            dtype=np.uint32,
        ).reshape(environment_count, frames.view_count)
        frame_valid = np.asarray(
            [camera.valid for camera in frames.cameras],
            dtype=np.bool_,
        ).reshape(environment_count, frames.view_count)
        arrays: dict[str, np.ndarray[Any, Any]] = {
            "frame/rgb_linear": np.asarray(frames.rgb_linear),
            "frame/depth_meters": np.asarray(frames.depth_meters),
            "frame/depth_validity": np.asarray(frames.depth_validity),
            "frame/intrinsics": intrinsics,
            "frame/distortion": distortion,
            "frame/camera_to_base": camera_to_base,
            "frame/capture_timestamp_seconds": capture,
            "frame/frame_age_seconds": frame_age,
            "frame/exposure_seconds": exposure,
            "frame/shutter_readout_seconds": shutter_readout,
            "frame/frame_index": frame_index,
            "frame/sensor_sequence": sensor_sequence,
            "frame/valid": frame_valid,
            "telemetry/proprioception": packed(
                step.proprioception,
                "proprioception",
                np.float32,
            ),
            "telemetry/action": packed(step.action, "action", np.float32),
            "telemetry/task_command": packed(
                step.task_command,
                "task_command",
                np.float32,
            ),
            "telemetry/reward": packed(step.reward, "reward", np.float32),
            "supervision/privileged_state": packed(
                step.privileged_state,
                "privileged_state",
                np.float32,
            ),
            "provenance/scenario_key": packed(
                step.scenario_key,
                "scenario_key",
                np.uint64,
            ),
            "provenance/scenario_fingerprint": np.full(
                (environment_count, 1),
                frame_provenance.scenario_fingerprint,
                dtype=np.uint64,
            ),
            "events/flags": packed(step.events, "events", np.uint32),
        }
        if np.any(arrays["provenance/scenario_key"] == 0):
            raise ValueError("scenario keys must be nonzero")
        for name in (
            "telemetry/reward",
            "provenance/scenario_key",
            "events/flags",
        ):
            if arrays[name].shape != (environment_count, 1):
                raise ValueError(f"{name} must have one value per environment")
        if step.truth is not None:
            truth = step.truth
            reference = frames.cameras[0]
            if (
                truth.normals.shape[:-1] != frames.depth_meters.shape
                or truth.frame_index != reference.frame_index
                or not np.isclose(
                    truth.timestamp_seconds,
                    reference.capture_timestamp_seconds,
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ):
                raise ValueError("truth does not match frame dimensions or timing")
            arrays.update(
                {
                    "truth/normals": np.asarray(truth.normals),
                    "truth/motion": np.asarray(truth.motion),
                    "truth/semantic_ids": np.asarray(truth.semantic_ids),
                    "truth/instance_ids": np.asarray(truth.instance_ids),
                    "truth/link_ids": np.asarray(truth.link_ids),
                    "truth/visibility": np.asarray(truth.visibility),
                    "truth/occlusion": np.asarray(truth.occlusion),
                    "truth/object_poses": np.asarray(truth.object_poses),
                    "truth/link_poses": np.asarray(truth.link_poses),
                    "truth/keypoints": np.asarray(truth.keypoints),
                    "truth/contacts": np.asarray(truth.contacts),
                    "truth/coordinate_frame": np.full(
                        (environment_count, 1),
                        int(truth.coordinate_frame),
                        dtype=np.uint32,
                    ),
                }
            )
        for name, value in step.physics_state.items():
            if not isinstance(name, str) or not name or "/" in name:
                raise ValueError("physics state names must be simple identifiers")
            arrays[f"physics/{name}"] = packed(
                value,
                f"physics/{name}",
                np.float32,
                matrix=False,
            )
        return arrays

    def append(self, step: VisualEpisodeStepV1) -> None:
        if self._finalized:
            raise RuntimeError("visual episode writer is finalized")
        if not isinstance(step, VisualEpisodeStepV1):
            raise TypeError("step must be VisualEpisodeStepV1")
        arrays = self._step_arrays(step)
        reference = step.frames.cameras[0]
        camera_timestamps = np.asarray(
            [camera.capture_timestamp_seconds for camera in step.frames.cameras],
            dtype=np.float64,
        ).reshape(
            step.frames.environment_count,
            step.frames.view_count,
        )
        sensor_sequences = np.asarray(
            [camera.sensor_sequence for camera in step.frames.cameras],
            dtype=np.uint32,
        ).reshape(
            step.frames.environment_count,
            step.frames.view_count,
        )
        scenario_keys = arrays["provenance/scenario_key"]
        if (
            self._last_frame_index is not None
            and reference.frame_index <= self._last_frame_index
        ):
            raise ValueError("visual episode frame indices must increase strictly")
        if self._last_camera_timestamps is not None and np.any(
            camera_timestamps < self._last_camera_timestamps
        ):
            raise ValueError("visual episode camera timestamps must be monotonic")
        if self._last_sensor_sequences is not None and np.any(
            sensor_sequences < self._last_sensor_sequences
        ):
            raise ValueError("visual episode sensor sequences must be monotonic")
        if (
            self._last_sensor_sequences is not None
            and self._last_camera_timestamps is not None
            and np.any(
                (sensor_sequences == self._last_sensor_sequences)
                & ~np.isclose(
                    camera_timestamps,
                    self._last_camera_timestamps,
                    rtol=0.0,
                    atol=1.0e-9,
                )
            )
        ):
            raise ValueError("a reused sensor sequence changed capture timestamp")
        if self._scenario_keys is not None and not np.array_equal(
            scenario_keys,
            self._scenario_keys,
        ):
            raise ValueError("visual episode scenario keys changed")
        sensor_ids = [
            camera.sensor_id for camera in step.frames.cameras[: step.frames.view_count]
        ]
        schema = {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
            for name, value in sorted(arrays.items())
        }
        if self._layout is None:
            self._layout = {
                "arrays": schema,
                "sensor_ids": sensor_ids,
                "environment_count": step.frames.environment_count,
                "view_count": step.frames.view_count,
                "height": step.frames.height,
                "width": step.frames.width,
            }
        elif schema != self._layout["arrays"]:
            raise ValueError("visual episode step layout changed")
        elif sensor_ids != self._layout["sensor_ids"]:
            raise ValueError("visual episode sensor identities changed")
        for name, value in arrays.items():
            self._buffer.setdefault(name, []).append(value)
        self._last_frame_index = reference.frame_index
        self._last_camera_timestamps = np.array(
            camera_timestamps,
            copy=True,
        )
        self._last_sensor_sequences = np.array(
            sensor_sequences,
            copy=True,
        )
        if self._scenario_keys is None:
            self._scenario_keys = np.array(
                scenario_keys,
                copy=True,
            )
        if len(next(iter(self._buffer.values()))) >= self.chunk_steps:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        arrays = {
            name: np.stack(values, axis=0) for name, values in self._buffer.items()
        }
        data = _deterministic_npz(arrays)
        content_hash = _sha256(data)
        ordinal = len(self._chunks)
        relative = Path("chunks") / (f"{ordinal:06d}-{content_hash[:16]}.npz")
        _atomic_write(self.root / relative, data)
        count = int(next(iter(arrays.values())).shape[0])
        self._chunks.append(
            {
                "path": relative.as_posix(),
                "sha256": content_hash,
                "step_start": self._step_count,
                "step_count": count,
                "arrays": {
                    name: _array_schema(value) for name, value in sorted(arrays.items())
                },
            }
        )
        self._step_count += count
        self._buffer.clear()

    def finalize(self) -> Path:
        if self._finalized:
            return self.root / "manifest.json"
        self._flush()
        if self._step_count == 0 or self._layout is None:
            raise ValueError("visual episode cannot be empty")
        manifest: dict[str, Any] = {
            "schema_version": VISUAL_EPISODE_STREAM_VERSION,
            "provenance": {
                **asdict(self.provenance),
                "source": int(self.provenance.source),
            },
            "layout": self._layout,
            "step_count": self._step_count,
            "chunks": self._chunks,
        }
        manifest["content_hash"] = _sha256(_canonical_json(manifest))
        path = self.root / "manifest.json"
        _atomic_write(path, _canonical_json(manifest) + b"\n")
        self._finalized = True
        return path

    def __enter__(self) -> "VisualEpisodeWriterV1":
        return self

    def __exit__(
        self,
        exception_type: object,
        *_: object,
    ) -> None:
        if exception_type is None:
            self.finalize()


class VisualEpisodeReaderV1:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = (
            self.root
            if self.root.name == "manifest.json"
            else self.root / "manifest.json"
        )
        if manifest_path.name == "manifest.json":
            self.root = manifest_path.parent
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(manifest, dict):
            raise ValueError("visual episode manifest must be an object")
        self.manifest = manifest
        if self.manifest.get("schema_version") != VISUAL_EPISODE_STREAM_VERSION:
            raise ValueError("unsupported visual episode stream version")
        expected = self.manifest.get("content_hash")
        unsigned = dict(self.manifest)
        unsigned.pop("content_hash", None)
        if expected != _sha256(_canonical_json(unsigned)):
            raise ValueError("visual episode manifest hash does not match")
        self._validate_manifest()
        self.manifest = _freeze_json(self.manifest)

    @staticmethod
    def _array_spec(
        value: object,
        *,
        name: str,
    ) -> tuple[np.dtype[Any], tuple[int, ...]]:
        if not isinstance(value, dict) or set(value) != {"dtype", "shape"}:
            raise ValueError(f"invalid array descriptor: {name}")
        dtype_value = value["dtype"]
        shape_value = value["shape"]
        if not isinstance(dtype_value, str) or not isinstance(
            shape_value,
            list,
        ):
            raise ValueError(f"invalid array descriptor: {name}")
        try:
            dtype = np.dtype(dtype_value)
        except TypeError as error:
            raise ValueError(f"invalid array dtype: {name}") from error
        if dtype.hasobject:
            raise ValueError(f"object arrays are not supported: {name}")
        if any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            for dimension in shape_value
        ):
            raise ValueError(f"invalid array shape: {name}")
        return dtype, tuple(shape_value)

    @staticmethod
    def _positive_integer(value: object, *, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _validate_manifest(self) -> None:
        required_manifest = {
            "schema_version",
            "provenance",
            "layout",
            "step_count",
            "chunks",
            "content_hash",
        }
        if set(self.manifest) != required_manifest:
            raise ValueError("visual episode manifest fields are invalid")
        provenance_value = self.manifest["provenance"]
        provenance_fields = {
            field_name for field_name in VisualEpisodeProvenanceV1.__dataclass_fields__
        }
        if (
            not isinstance(provenance_value, dict)
            or set(provenance_value) != provenance_fields
            or not isinstance(
                provenance_value.get("visual_asset_hashes"),
                list,
            )
        ):
            raise ValueError("visual episode provenance fields are invalid")
        try:
            self.provenance = VisualEpisodeProvenanceV1(
                **{
                    **provenance_value,
                    "visual_asset_hashes": tuple(
                        provenance_value["visual_asset_hashes"]
                    ),
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError("visual episode provenance is invalid") from error

        layout = self.manifest["layout"]
        required_layout = {
            "arrays",
            "sensor_ids",
            "environment_count",
            "view_count",
            "height",
            "width",
        }
        if not isinstance(layout, dict) or set(layout) != required_layout:
            raise ValueError("visual episode layout fields are invalid")
        environment_count = self._positive_integer(
            layout["environment_count"],
            name="environment_count",
        )
        view_count = self._positive_integer(
            layout["view_count"],
            name="view_count",
        )
        height = self._positive_integer(layout["height"], name="height")
        width = self._positive_integer(layout["width"], name="width")
        sensor_ids = layout["sensor_ids"]
        if (
            not isinstance(sensor_ids, list)
            or len(sensor_ids) != view_count
            or any(not isinstance(value, str) or not value for value in sensor_ids)
            or len(sensor_ids) != len(set(sensor_ids))
        ):
            raise ValueError("visual episode sensor ids are invalid")
        arrays = layout["arrays"]
        if not isinstance(arrays, dict) or not arrays:
            raise ValueError("visual episode array layout is empty")
        self._array_specs = {
            name: self._array_spec(spec, name=name)
            for name, spec in arrays.items()
            if isinstance(name, str) and name
        }
        if len(self._array_specs) != len(arrays):
            raise ValueError("visual episode array names are invalid")

        required_shapes: dict[
            str,
            tuple[npt.DTypeLike, tuple[int, ...]],
        ] = {
            "frame/rgb_linear": (
                np.float32,
                (environment_count, view_count, height, width, 4),
            ),
            "frame/depth_meters": (
                np.float32,
                (environment_count, view_count, height, width),
            ),
            "frame/depth_validity": (
                np.bool_,
                (environment_count, view_count, height, width),
            ),
            "frame/intrinsics": (
                np.float32,
                (environment_count, view_count, 4),
            ),
            "frame/distortion": (
                np.float32,
                (environment_count, view_count, 4),
            ),
            "frame/camera_to_base": (
                np.float32,
                (environment_count, view_count, 4, 4),
            ),
            "frame/capture_timestamp_seconds": (
                np.float64,
                (environment_count, view_count),
            ),
            "frame/frame_age_seconds": (
                np.float32,
                (environment_count, view_count),
            ),
            "frame/exposure_seconds": (
                np.float32,
                (environment_count, view_count),
            ),
            "frame/shutter_readout_seconds": (
                np.float32,
                (environment_count, view_count),
            ),
            "frame/frame_index": (
                np.uint64,
                (environment_count, view_count),
            ),
            "frame/sensor_sequence": (
                np.uint32,
                (environment_count, view_count),
            ),
            "frame/valid": (
                np.bool_,
                (environment_count, view_count),
            ),
            "provenance/scenario_fingerprint": (
                np.uint64,
                (environment_count, 1),
            ),
        }
        for name, (dtype, shape) in required_shapes.items():
            if name not in self._array_specs:
                raise ValueError(f"visual episode is missing {name}")
            actual_dtype, actual_shape = self._array_specs[name]
            if actual_dtype != np.dtype(dtype) or actual_shape != shape:
                raise ValueError(f"visual episode layout mismatch: {name}")
        for name in (
            "telemetry/proprioception",
            "telemetry/action",
            "telemetry/task_command",
            "telemetry/reward",
            "supervision/privileged_state",
            "provenance/scenario_key",
            "events/flags",
        ):
            if name not in self._array_specs:
                raise ValueError(f"visual episode is missing {name}")
            _, shape = self._array_specs[name]
            if len(shape) != 2 or shape[0] != environment_count:
                raise ValueError(
                    f"visual episode array is not environment-major: {name}"
                )
        required_matrix_dtypes = {
            "telemetry/proprioception": np.float32,
            "telemetry/action": np.float32,
            "telemetry/task_command": np.float32,
            "telemetry/reward": np.float32,
            "supervision/privileged_state": np.float32,
            "provenance/scenario_key": np.uint64,
            "events/flags": np.uint32,
        }
        for name, dtype in required_matrix_dtypes.items():
            if self._array_specs[name][0] != np.dtype(dtype):
                raise ValueError(f"visual episode array dtype mismatch: {name}")
        for name in (
            "telemetry/reward",
            "provenance/scenario_key",
            "events/flags",
        ):
            if self._array_specs[name][1] != (environment_count, 1):
                raise ValueError(f"visual episode array width mismatch: {name}")
        truth_arrays = {
            "truth/normals",
            "truth/motion",
            "truth/semantic_ids",
            "truth/instance_ids",
            "truth/link_ids",
            "truth/visibility",
            "truth/occlusion",
            "truth/object_poses",
            "truth/link_poses",
            "truth/keypoints",
            "truth/contacts",
            "truth/coordinate_frame",
        }
        present_truth = truth_arrays.intersection(self._array_specs)
        if present_truth and present_truth != truth_arrays:
            raise ValueError("visual truth arrays are incomplete")
        if present_truth and self._array_specs["truth/coordinate_frame"] != (
            np.dtype(np.uint32),
            (environment_count, 1),
        ):
            raise ValueError("visual truth coordinate frame is invalid")
        if present_truth:
            dense_truth_specs = {
                "truth/normals": (
                    np.float32,
                    (environment_count, view_count, height, width, 4),
                ),
                "truth/motion": (
                    np.float32,
                    (environment_count, view_count, height, width, 4),
                ),
                "truth/semantic_ids": (
                    np.uint32,
                    (environment_count, view_count, height, width),
                ),
                "truth/instance_ids": (
                    np.uint32,
                    (environment_count, view_count, height, width),
                ),
                "truth/link_ids": (
                    np.uint32,
                    (environment_count, view_count, height, width),
                ),
                "truth/visibility": (
                    np.float32,
                    (environment_count, view_count, height, width),
                ),
                "truth/occlusion": (
                    np.uint8,
                    (environment_count, view_count, height, width),
                ),
            }
            for name, (dtype, shape) in dense_truth_specs.items():
                if self._array_specs[name] != (np.dtype(dtype), shape):
                    raise ValueError(f"visual truth layout mismatch: {name}")
            for name, width_value in (
                ("truth/object_poses", 11),
                ("truth/link_poses", 11),
                ("truth/keypoints", 8),
                ("truth/contacts", 12),
            ):
                dtype, shape = self._array_specs[name]
                if (
                    dtype != np.dtype(np.float64)
                    or len(shape) != 3
                    or shape[0] != environment_count
                    or shape[2] != width_value
                ):
                    raise ValueError(f"visual truth record layout mismatch: {name}")
        known_arrays = set(required_shapes) | set(required_matrix_dtypes) | truth_arrays
        for name, (dtype, shape) in self._array_specs.items():
            if name in known_arrays:
                continue
            physics_name = (
                name.removeprefix("physics/") if name.startswith("physics/") else ""
            )
            if (
                not physics_name
                or "/" in physics_name
                or dtype != np.dtype(np.float32)
                or len(shape) < 2
                or shape[0] != environment_count
            ):
                raise ValueError(f"visual episode array is unsupported: {name}")

        step_count = self._positive_integer(
            self.manifest["step_count"],
            name="step_count",
        )
        chunks = self.manifest["chunks"]
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("visual episode chunks are invalid")
        expected_start = 0
        for ordinal, chunk in enumerate(chunks):
            required_chunk = {
                "path",
                "sha256",
                "step_start",
                "step_count",
                "arrays",
            }
            if not isinstance(chunk, dict) or set(chunk) != required_chunk:
                raise ValueError("visual episode chunk fields are invalid")
            chunk_count = self._positive_integer(
                chunk["step_count"],
                name="chunk step_count",
            )
            if (
                not isinstance(chunk["step_start"], int)
                or isinstance(chunk["step_start"], bool)
                or chunk["step_start"] != expected_start
            ):
                raise ValueError("visual episode chunks are not contiguous")
            relative = chunk["path"]
            expected_prefix = f"{ordinal:06d}-"
            if (
                not isinstance(relative, str)
                or not relative.startswith(f"chunks/{expected_prefix}")
                or not relative.endswith(".npz")
                or len(relative) != len("chunks/000000-0000000000000000.npz")
            ):
                raise ValueError("visual episode chunk path is invalid")
            content_hash = chunk["sha256"]
            if (
                not isinstance(content_hash, str)
                or len(content_hash) != 64
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in content_hash
                )
                or relative[len(f"chunks/{expected_prefix}") : -len(".npz")].lower()
                != content_hash[:16].lower()
            ):
                raise ValueError("visual episode chunk hash is invalid")
            if chunk["arrays"] != arrays:
                raise ValueError("visual episode chunk layout changed")
            expected_start += chunk_count
        if expected_start != step_count:
            raise ValueError("visual episode step count does not match chunks")

    def chunks(self) -> Iterator[Mapping[str, np.ndarray[Any, Any]]]:
        last_frame_index: int | None = None
        last_timestamp_seconds: float | None = None
        last_camera_timestamps: np.ndarray[Any, Any] | None = None
        last_sensor_sequences: np.ndarray[Any, Any] | None = None
        scenario_keys: np.ndarray[Any, Any] | None = None
        for chunk in self.manifest["chunks"]:
            path = (self.root / chunk["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("visual episode chunk escapes its root")
            data = path.read_bytes()
            if _sha256(data) != chunk["sha256"]:
                raise ValueError(f"visual episode chunk hash mismatch: {path}")
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)) or set(
                    archive.files
                ) != set(self._array_specs):
                    raise ValueError(f"visual episode chunk fields mismatch: {path}")
                arrays: dict[str, np.ndarray[Any, Any]] = {}
                for name, (dtype, shape) in self._array_specs.items():
                    value = np.asarray(archive[name])
                    expected_shape = (chunk["step_count"], *shape)
                    if value.dtype != dtype or value.shape != expected_shape:
                        raise ValueError(f"visual episode chunk array mismatch: {name}")
                    value.setflags(write=False)
                    arrays[name] = value
                frame_indices = arrays["frame/frame_index"]
                capture_timestamps = arrays["frame/capture_timestamp_seconds"]
                sensor_sequences = arrays["frame/sensor_sequence"]
                for step in range(chunk["step_count"]):
                    frame_index = int(frame_indices[step, 0, 0])
                    timestamp_seconds = float(capture_timestamps[step, 0, 0])
                    camera_timestamps = capture_timestamps[step]
                    if (
                        np.any(frame_indices[step] != frame_index)
                        or not np.all(np.isfinite(camera_timestamps))
                        or (
                            last_frame_index is not None
                            and frame_index <= last_frame_index
                        )
                        or (
                            last_timestamp_seconds is not None
                            and timestamp_seconds < last_timestamp_seconds
                        )
                        or (
                            last_camera_timestamps is not None
                            and np.any(camera_timestamps < last_camera_timestamps)
                        )
                        or (
                            last_sensor_sequences is not None
                            and np.any(sensor_sequences[step] < last_sensor_sequences)
                        )
                        or (
                            last_sensor_sequences is not None
                            and last_camera_timestamps is not None
                            and np.any(
                                (sensor_sequences[step] == last_sensor_sequences)
                                & ~np.isclose(
                                    camera_timestamps,
                                    last_camera_timestamps,
                                    rtol=0.0,
                                    atol=1.0e-9,
                                )
                            )
                        )
                    ):
                        raise ValueError("visual episode frame timing is invalid")
                    last_frame_index = frame_index
                    last_timestamp_seconds = timestamp_seconds
                    last_camera_timestamps = np.array(
                        camera_timestamps,
                        copy=True,
                    )
                    last_sensor_sequences = np.array(
                        sensor_sequences[step],
                        copy=True,
                    )
                if np.any(
                    arrays["provenance/scenario_fingerprint"]
                    != self.provenance.scenario_fingerprint
                ) or np.any(arrays["provenance/scenario_key"] == 0):
                    raise ValueError("visual episode scenario provenance is invalid")
                chunk_scenario_keys = arrays["provenance/scenario_key"]
                reference_scenario_keys = (
                    chunk_scenario_keys[0] if scenario_keys is None else scenario_keys
                )
                if np.any(chunk_scenario_keys != reference_scenario_keys[None, ...]):
                    raise ValueError("visual episode scenario keys changed")
                scenario_keys = np.array(
                    reference_scenario_keys,
                    copy=True,
                )
                depth = arrays["frame/depth_meters"]
                depth_validity = arrays["frame/depth_validity"]
                frame_valid = arrays["frame/valid"][
                    ...,
                    None,
                    None,
                ]
                frame_age = arrays["frame/frame_age_seconds"]
                observation_timestamps = capture_timestamps + frame_age
                if (
                    np.any(depth_validity & (~np.isfinite(depth) | (depth <= 0.0)))
                    or np.any(~depth_validity & (depth != 0.0))
                    or np.any(~frame_valid & depth_validity)
                    or not np.all(np.isfinite(arrays["frame/rgb_linear"]))
                    or np.any(frame_age < 0.0)
                    or not np.allclose(
                        observation_timestamps,
                        observation_timestamps[..., :1],
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                    or np.any(arrays["frame/exposure_seconds"] < 0.0)
                    or np.any(arrays["frame/shutter_readout_seconds"] < 0.0)
                ):
                    raise ValueError("visual episode camera samples are invalid")
                intrinsics = arrays["frame/intrinsics"]
                camera_to_base = arrays["frame/camera_to_base"]
                rotations = camera_to_base[..., :3, :3]
                if (
                    not np.all(np.isfinite(intrinsics))
                    or not np.all(np.isfinite(arrays["frame/distortion"]))
                    or np.any(intrinsics[..., :2] <= 0.0)
                    or not np.all(np.isfinite(camera_to_base))
                    or not np.allclose(
                        camera_to_base[..., 3, :],
                        np.asarray(
                            [0.0, 0.0, 0.0, 1.0],
                            dtype=np.float32,
                        ),
                        rtol=0.0,
                        atol=1.0e-5,
                    )
                    or not np.allclose(
                        np.swapaxes(rotations, -1, -2) @ rotations,
                        np.eye(3, dtype=np.float32),
                        rtol=0.0,
                        atol=1.0e-4,
                    )
                    or not np.allclose(
                        np.linalg.det(rotations),
                        1.0,
                        rtol=0.0,
                        atol=1.0e-4,
                    )
                ):
                    raise ValueError("visual episode calibration is invalid")
                if "truth/coordinate_frame" in arrays and np.any(
                    arrays["truth/coordinate_frame"] > int(VisualCoordinateFrame.OBJECT)
                ):
                    raise ValueError("visual truth coordinate frame is invalid")
                if "truth/visibility" in arrays and (
                    np.any(arrays["truth/visibility"] < 0.0)
                    or np.any(arrays["truth/visibility"] > 1.0)
                    or not all(
                        np.all(np.isfinite(arrays[name]))
                        for name in (
                            "truth/normals",
                            "truth/motion",
                            "truth/visibility",
                            "truth/object_poses",
                            "truth/link_poses",
                            "truth/keypoints",
                            "truth/contacts",
                        )
                    )
                ):
                    raise ValueError("visual truth samples are invalid")
                if "truth/visibility" in arrays:
                    expected_occlusion = np.rint(
                        (1.0 - arrays["truth/visibility"]) * 255.0
                    ).astype(np.int16)
                    if np.any(
                        np.abs(
                            arrays["truth/occlusion"].astype(np.int16)
                            - expected_occlusion
                        )
                        > 1
                    ):
                        raise ValueError(
                            "visual truth visibility and occlusion disagree"
                        )
                if "truth/object_poses" in arrays:
                    for name in (
                        "truth/object_poses",
                        "truth/link_poses",
                        "truth/keypoints",
                        "truth/contacts",
                    ):
                        identities = arrays[name][..., -4:]
                        if np.any(
                            (identities < 0.0)
                            | (identities > np.iinfo(np.uint32).max)
                            | (identities != np.floor(identities))
                        ):
                            raise ValueError("visual truth identities are invalid")
                    for name in (
                        "truth/object_poses",
                        "truth/link_poses",
                    ):
                        orientations = arrays[name][..., 3:7]
                        if orientations.size and not np.allclose(
                            np.linalg.norm(
                                orientations,
                                axis=-1,
                            ),
                            1.0,
                            rtol=0.0,
                            atol=1.0e-4,
                        ):
                            raise ValueError("visual truth orientation is invalid")
                    object_identities = arrays["truth/object_poses"][
                        ...,
                        -4:-2,
                    ]
                    keypoint_identities = arrays["truth/keypoints"][
                        ...,
                        -4:-2,
                    ]
                    link_identities = arrays["truth/link_poses"][
                        ...,
                        -2,
                    ]
                    contact_identities = arrays["truth/contacts"][
                        ...,
                        -4:-2,
                    ]
                    contact_normals = arrays["truth/contacts"][
                        ...,
                        4:7,
                    ]
                    if (
                        np.any(
                            (object_identities == 0.0)
                            | (object_identities == np.iinfo(np.uint32).max)
                        )
                        or np.any(
                            (keypoint_identities == 0.0)
                            | (keypoint_identities == np.iinfo(np.uint32).max)
                        )
                        or np.any(link_identities == np.iinfo(np.uint32).max)
                        or np.any(arrays["truth/contacts"][..., 3] < 0.0)
                        or np.any(
                            (contact_identities == 0.0)
                            | (contact_identities == np.iinfo(np.uint32).max)
                        )
                        or (
                            contact_normals.size
                            and not np.allclose(
                                np.linalg.norm(
                                    contact_normals,
                                    axis=-1,
                                ),
                                1.0,
                                rtol=0.0,
                                atol=1.0e-4,
                            )
                        )
                    ):
                        raise ValueError("visual truth semantic identities are invalid")
                    keypoint_visibility = arrays["truth/keypoints"][..., 3]
                    if np.any(
                        (keypoint_visibility < 0.0) | (keypoint_visibility > 1.0)
                    ):
                        raise ValueError("visual keypoint visibility is invalid")
                yield MappingProxyType(arrays)


def _linear_rgb_to_uint8(value: npt.ArrayLike) -> ByteArray:
    linear = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.asarray(np.rint(srgb * 255.0), dtype=np.uint8)


def export_lerobot_v3(
    stream: str | os.PathLike[str] | VisualEpisodeReaderV1,
    output_root: str | os.PathLike[str],
    *,
    repo_id: str,
    robot_type: str = "metalrobo",
    task: str = "MetalRobo visual episode",
    include_depth: bool = True,
) -> Path:
    """Export a stream through LeRobot's v3 writer when LeRobot is installed."""

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise ImportError(
            "LeRobot >= 0.4 is required for the optional v3 exporter"
        ) from error
    reader = (
        stream
        if isinstance(stream, VisualEpisodeReaderV1)
        else VisualEpisodeReaderV1(stream)
    )
    layout = reader.manifest["layout"]
    sensor_ids = layout["sensor_ids"]
    environment_count = int(layout["environment_count"])
    height = int(layout["height"])
    width = int(layout["width"])
    sample = next(reader.chunks(), None)
    if sample is None:
        raise ValueError("visual episode contains no chunks")
    state_width = int(sample["telemetry/proprioception"].shape[-1])
    action_width = int(sample["telemetry/action"].shape[-1])
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_width,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_width,),
            "names": None,
        },
    }
    for sensor_id in sensor_ids:
        features[f"observation.images.{sensor_id}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
        if include_depth:
            features[f"observation.depth.{sensor_id}"] = {
                "dtype": "float32",
                "shape": (height, width, 1),
                "names": ["height", "width", "channel"],
            }
    destination = Path(output_root).expanduser().resolve()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=destination,
        fps=float(reader.manifest["provenance"]["nominal_rate_hz"]),
        robot_type=robot_type,
        features=features,
        use_videos=True,
    )
    for environment in range(environment_count):
        for chunk in reader.chunks():
            step_count = int(chunk["frame/rgb_linear"].shape[0])
            for step in range(step_count):
                frame: dict[str, Any] = {
                    "observation.state": chunk["telemetry/proprioception"][
                        step, environment
                    ],
                    "action": chunk["telemetry/action"][step, environment],
                    "task": task,
                }
                for view, sensor_id in enumerate(sensor_ids):
                    frame[f"observation.images.{sensor_id}"] = _linear_rgb_to_uint8(
                        chunk["frame/rgb_linear"][
                            step,
                            environment,
                            view,
                            ...,
                            :3,
                        ]
                    )
                    if include_depth:
                        depth = chunk["frame/depth_meters"][
                            step,
                            environment,
                            view,
                        ]
                        depth_validity = chunk["frame/depth_validity"][
                            step, environment, view
                        ]
                        frame[f"observation.depth.{sensor_id}"] = np.where(
                            depth_validity,
                            depth,
                            0.0,
                        )[..., None].astype(
                            np.float32,
                            copy=False,
                        )
                dataset.add_frame(frame)
        dataset.save_episode()
    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    return destination


__all__ = [
    "CallablePerceptionProviderV1",
    "ObservationProfileV1",
    "PerceptionCapability",
    "PerceptionProviderDescriptorV1",
    "PerceptionProviderV1",
    "PerceptionResultBatchV1",
    "PerceptionTensorV1",
    "PolicyObservationAssemblerV1",
    "PolicyObservationBatchV1",
    "VisualBatchProvenanceV1",
    "VisualCameraFrameV1",
    "VisualCoordinateFrame",
    "VisualEpisodeProvenanceV1",
    "VisualEpisodeReaderV1",
    "VisualEpisodeStepV1",
    "VisualEpisodeWriterV1",
    "VisualFrameBatchV1",
    "VisualFrameSource",
    "VisualModality",
    "VisualTruthBatchV1",
    "export_lerobot_v3",
]
