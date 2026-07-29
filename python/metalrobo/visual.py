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


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(value: int, *, name: str) -> int:
    result = int(value)
    if not 0 < result < 1 << 64:
        raise ValueError(f"{name} must be a nonzero uint64")
    return result


def _array_schema(value: np.ndarray[Any, Any]) -> dict[str, Any]:
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape[1:]),
    }


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
        if not self.sensor_id:
            raise ValueError("sensor_id cannot be empty")
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
        for name, value in (
            ("capture_timestamp_seconds", self.capture_timestamp_seconds),
            ("frame_age_seconds", self.frame_age_seconds),
            ("exposure_seconds", self.exposure_seconds),
            ("shutter_readout_seconds", self.shutter_readout_seconds),
        ):
            if not np.isfinite(value) or (
                name != "capture_timestamp_seconds" and value < 0.0
            ):
                raise ValueError(f"{name} is invalid")
        if not 0 <= int(self.frame_index) < 1 << 64:
            raise ValueError("frame_index must fit in a uint64")
        if not 0 <= int(self.sensor_sequence) < 1 << 32:
            raise ValueError("sensor_sequence must fit in a uint32")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "camera_to_base", camera_to_base)


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
        if self.schema_version != VISUAL_FRAME_BATCH_VERSION:
            raise ValueError("unsupported VisualFrameBatch version")
        rgb = _as_array(self.rgb_linear, np.float32, name="rgb_linear")
        depth = _as_array(
            self.depth_meters,
            np.float32,
            name="depth_meters",
        )
        validity = _as_array(
            self.depth_validity,
            np.bool_,
            name="depth_validity",
        )
        if rgb.ndim != 5 or rgb.shape[-1] != 4:
            raise ValueError(
                "rgb_linear must have shape [environment, view, height, width, 4]"
            )
        expected = rgb.shape[:-1]
        if depth.shape != expected or validity.shape != expected:
            raise ValueError("depth and validity must match RGB dimensions")
        if len(self.cameras) != rgb.shape[0] * rgb.shape[1]:
            raise ValueError("camera metadata must be environment-view major")
        if not all(
            isinstance(address, int) and address > 0
            for address in self.device_buffers.values()
        ):
            raise ValueError("device buffer handles must be positive integers")
        object.__setattr__(self, "rgb_linear", rgb)
        object.__setattr__(self, "depth_meters", depth)
        object.__setattr__(self, "depth_validity", validity)
        object.__setattr__(
            self,
            "device_buffers",
            dict(self.device_buffers),
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
    object_poses: FloatArray
    link_poses: FloatArray
    keypoints: FloatArray
    contacts: FloatArray
    frame_index: int
    timestamp_seconds: float
    device_buffers: Mapping[str, int] = field(default_factory=dict)
    schema_version: int = VISUAL_FRAME_BATCH_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VISUAL_FRAME_BATCH_VERSION:
            raise ValueError("unsupported VisualTruthBatch version")
        normals = _as_array(self.normals, np.float32, name="normals")
        motion = _as_array(self.motion, np.float32, name="motion")
        semantic = _as_array(
            self.semantic_ids,
            np.uint32,
            name="semantic_ids",
        )
        instance = _as_array(
            self.instance_ids,
            np.uint32,
            name="instance_ids",
        )
        link = _as_array(self.link_ids, np.uint32, name="link_ids")
        visibility = _as_array(
            self.visibility,
            np.float32,
            name="visibility",
        )
        occlusion = _as_array(
            self.occlusion,
            np.uint8,
            name="occlusion",
        )
        if normals.ndim != 5 or normals.shape[-1] != 4:
            raise ValueError("normals must have shape [E, V, H, W, 4]")
        if motion.shape != normals.shape:
            raise ValueError("motion must match normal shape")
        dense_shape = normals.shape[:-1]
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
        packed = {}
        for name, value, width in (
            ("object_poses", self.object_poses, 11),
            ("link_poses", self.link_poses, 11),
            ("keypoints", self.keypoints, 8),
            ("contacts", self.contacts, 12),
        ):
            array = _as_array(value, np.float32, name=name)
            if array.ndim != 3 or array.shape[0] != dense_shape[0]:
                raise ValueError(f"{name} must be environment-major")
            if array.shape[-1] != width:
                raise ValueError(f"{name} record width must be {width}")
            packed[name] = array
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "motion", motion)
        object.__setattr__(self, "semantic_ids", semantic)
        object.__setattr__(self, "instance_ids", instance)
        object.__setattr__(self, "link_ids", link)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "occlusion", occlusion)
        for name, value in packed.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "device_buffers",
            dict(self.device_buffers),
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
        if (
            self.schema_version != PERCEPTION_PROVIDER_VERSION
            or not self.provider_id
            or not self.content_hash
            or not self.input_modalities
            or not self.capabilities
            or self.temporal_window <= 0
        ):
            raise ValueError("perception provider descriptor is invalid")


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
        if not self.tensor_id or not self.modality:
            raise ValueError("perception tensor identity is invalid")
        values = np.ascontiguousarray(self.values)
        if values.size == 0 or values.dtype.kind not in "fiu b".replace(" ", ""):
            raise ValueError("perception tensor values are unsupported")
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise ValueError("perception tensor contains nonfinite values")
        confidence = _as_array(
            self.confidence,
            np.float32,
            name="confidence",
        )
        valid = _as_array(self.valid, np.bool_, name="valid")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("perception confidence must be in [0, 1]")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(
            self,
            "device_buffers",
            dict(self.device_buffers),
        )


@dataclass(frozen=True, slots=True)
class PerceptionResultBatchV1:
    provider_id: str
    provider_content_hash: str
    frame_index: int
    timestamp_seconds: float
    tensors: tuple[PerceptionTensorV1, ...]

    def __post_init__(self) -> None:
        if not self.provider_id or not self.provider_content_hash:
            raise ValueError("perception result identity is invalid")
        ids = [tensor.tensor_id for tensor in self.tensors]
        if len(ids) != len(set(ids)):
            raise ValueError("perception tensor ids must be unique")

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
        if (
            result.provider_id != self.descriptor.provider_id
            or result.provider_content_hash != self.descriptor.content_hash
        ):
            raise ValueError("provider output provenance does not match descriptor")
        return result


@dataclass(frozen=True, slots=True)
class PolicyObservationBatchV1:
    profile: ObservationProfileV1
    deployable: FloatArray
    privileged: FloatArray
    frame_index: int
    timestamp_seconds: float


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
        if profile is ObservationProfileV1.RAW_RGBD:
            visual = np.concatenate(
                (rgb, depth, valid.astype(np.float32)),
                axis=-1,
            )
        elif profile is ObservationProfileV1.RGB_XYZ:
            xyz = np.zeros((*frames.depth_meters.shape, 3), dtype=np.float32)
            y, x = np.indices((frames.height, frames.width), dtype=np.float32)
            x += 0.5
            y += 0.5
            for environment in range(environment_count):
                for view in range(frames.view_count):
                    camera = frames.camera(environment, view)
                    fx, fy, cx, cy = camera.intrinsics
                    z = frames.depth_meters[environment, view]
                    camera_points = np.stack(
                        (
                            (x - cx) / fx * z,
                            (y - cy) / fy * z,
                            z,
                            np.ones_like(z),
                        ),
                        axis=-1,
                    )
                    base_points = (
                        camera_points @ camera.camera_to_base.T
                    )[..., :3]
                    xyz[environment, view] = np.where(
                        valid[environment, view],
                        base_points,
                        0.0,
                    )
            visual = np.concatenate(
                (rgb, xyz, valid.astype(np.float32)),
                axis=-1,
            )
        else:
            if perception is None:
                raise ValueError("selected profile requires perception results")
            modality = (
                VisualModality.OBJECT_POSE
                if profile is ObservationProfileV1.OBJECT_CENTRIC
                else VisualModality.FEATURE
            )
            selected = perception.first(modality)
            if selected is None:
                raise ValueError("perception result lacks the requested modality")
            values = np.asarray(selected.values, dtype=np.float32)
            if values.shape[0] != environment_count:
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
    renderer_fingerprint: int
    visual_scene_fingerprint: int
    sensor_profile_fingerprint: int
    calibration_fingerprint: int
    physics_fingerprint: int
    visual_asset_hashes: tuple[str, ...] = ()
    nominal_rate_hz: float = 15.0

    def __post_init__(self) -> None:
        if not self.episode_id or not np.isfinite(self.nominal_rate_hz):
            raise ValueError("visual episode provenance is invalid")
        if self.nominal_rate_hz <= 0.0:
            raise ValueError("nominal_rate_hz must be positive")
        for name in (
            "episode_twin_fingerprint",
            "world_family_fingerprint",
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
        if any(not value for value in self.visual_asset_hashes):
            raise ValueError("visual asset hashes cannot be empty")


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
        if chunk_steps <= 0:
            raise ValueError("chunk_steps must be positive")
        self.root = Path(root).expanduser().resolve()
        self.provenance = provenance
        self.chunk_steps = int(chunk_steps)
        self._buffer: dict[str, list[np.ndarray[Any, Any]]] = {}
        self._chunks: list[dict[str, Any]] = []
        self._step_count = 0
        self._layout: dict[str, Any] | None = None
        self._finalized = False

    def _step_arrays(
        self,
        step: VisualEpisodeStepV1,
    ) -> dict[str, np.ndarray[Any, Any]]:
        frames = step.frames
        environment_count = frames.environment_count
        if frames.source is not self.provenance.source:
            raise ValueError("frame source does not match episode provenance")
        frame_provenance = frames.provenance
        expected_provenance = (
            self.provenance.episode_twin_fingerprint,
            self.provenance.renderer_fingerprint,
            self.provenance.sensor_profile_fingerprint,
            self.provenance.calibration_fingerprint,
        )
        observed_provenance = (
            frame_provenance.episode_twin_fingerprint,
            frame_provenance.renderer_fingerprint,
            frame_provenance.sensor_profile_fingerprint,
            frame_provenance.calibration_fingerprint,
        )
        if observed_provenance != expected_provenance:
            raise ValueError(
                "frame provenance does not match episode provenance"
            )

        def packed(value: npt.ArrayLike, name: str, dtype: npt.DTypeLike) -> np.ndarray:
            array = np.ascontiguousarray(value, dtype=dtype)
            if array.ndim == 1:
                array = array[:, None]
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
            "events/flags": packed(step.events, "events", np.uint32),
        }
        if step.truth is not None:
            truth = step.truth
            if truth.normals.shape[:-1] != frames.depth_meters.shape:
                raise ValueError("truth does not match frame dimensions")
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
                }
            )
        for name, value in step.physics_state.items():
            if not name or "/" in name:
                raise ValueError("physics state names must be simple identifiers")
            arrays[f"physics/{name}"] = packed(
                value,
                f"physics/{name}",
                np.float32,
            )
        return arrays

    def append(self, step: VisualEpisodeStepV1) -> None:
        if self._finalized:
            raise RuntimeError("visual episode writer is finalized")
        arrays = self._step_arrays(step)
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
                "sensor_ids": [
                    camera.sensor_id
                    for camera in step.frames.cameras[
                        : step.frames.view_count
                    ]
                ],
                "environment_count": step.frames.environment_count,
                "view_count": step.frames.view_count,
                "height": step.frames.height,
                "width": step.frames.width,
            }
        elif schema != self._layout["arrays"]:
            raise ValueError("visual episode step layout changed")
        for name, value in arrays.items():
            self._buffer.setdefault(name, []).append(value)
        if len(next(iter(self._buffer.values()))) >= self.chunk_steps:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        arrays = {
            name: np.stack(values, axis=0)
            for name, values in self._buffer.items()
        }
        data = _deterministic_npz(arrays)
        content_hash = _sha256(data)
        ordinal = len(self._chunks)
        relative = Path("chunks") / (
            f"{ordinal:06d}-{content_hash[:16]}.npz"
        )
        _atomic_write(self.root / relative, data)
        count = int(next(iter(arrays.values())).shape[0])
        self._chunks.append(
            {
                "path": relative.as_posix(),
                "sha256": content_hash,
                "step_start": self._step_count,
                "step_count": count,
                "arrays": {
                    name: _array_schema(value)
                    for name, value in sorted(arrays.items())
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
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != VISUAL_EPISODE_STREAM_VERSION:
            raise ValueError("unsupported visual episode stream version")
        expected = self.manifest.get("content_hash")
        unsigned = dict(self.manifest)
        unsigned.pop("content_hash", None)
        if expected != _sha256(_canonical_json(unsigned)):
            raise ValueError("visual episode manifest hash does not match")

    def chunks(self) -> Iterator[Mapping[str, np.ndarray[Any, Any]]]:
        for chunk in self.manifest["chunks"]:
            path = self.root / chunk["path"]
            data = path.read_bytes()
            if _sha256(data) != chunk["sha256"]:
                raise ValueError(f"visual episode chunk hash mismatch: {path}")
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                yield {
                    name: np.asarray(archive[name])
                    for name in archive.files
                }


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
    chunks = list(reader.chunks())
    if not chunks:
        raise ValueError("visual episode contains no chunks")
    sample = chunks[0]
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
        fps=float(
            reader.manifest["provenance"]["nominal_rate_hz"]
        ),
        robot_type=robot_type,
        features=features,
        use_videos=True,
    )
    for environment in range(environment_count):
        for chunk in chunks:
            step_count = int(chunk["frame/rgb_linear"].shape[0])
            for step in range(step_count):
                frame: dict[str, Any] = {
                    "observation.state": chunk[
                        "telemetry/proprioception"
                    ][step, environment],
                    "action": chunk["telemetry/action"][step, environment],
                    "task": task,
                }
                for view, sensor_id in enumerate(sensor_ids):
                    frame[f"observation.images.{sensor_id}"] = (
                        _linear_rgb_to_uint8(
                            chunk["frame/rgb_linear"][
                                step,
                                environment,
                                view,
                                ...,
                                :3,
                            ]
                        )
                    )
                    if include_depth:
                        frame[f"observation.depth.{sensor_id}"] = (
                            chunk["frame/depth_meters"][
                                step,
                                environment,
                                view,
                                ...,
                                None,
                            ]
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
