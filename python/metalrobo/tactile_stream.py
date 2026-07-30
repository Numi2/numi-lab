"""Physical tactile stream contracts and Sharpa Tacmap decoding.

The canonical MetalRobo tactile observation describes metric simulation
output.  This module deliberately describes the *input* side of a measured
dataset.  A physical stream is not promoted to the canonical metric contract
until its units, coordinate frames, mosaic layout, and calibration have been
verified and fingerprinted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt


PHYSICAL_TACTILE_STREAM_SCHEMA = (
    "metalrobo.physical_tactile_stream"
)
PHYSICAL_TACTILE_STREAM_FORMAT = 1
ORIGAMI_REPOSITORY = (
    "SharpaIT/Robotic_Origami_Challenge"
)
ORIGAMI_LICENSE = "CC-BY-4.0"

WRENCH_AXES = ("fx", "fy", "fz", "tx", "ty", "tz")
WAVE_FINGERTIP_ORDER = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_little",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_little",
)
WAVE_HAND_JOINT_ORDER = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "little_CMC",
    "little_MCP_FE",
    "little_MCP_AA",
    "little_PIP",
    "little_DIP",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_CANONICAL_FINGERPRINT = re.compile(r"0x[0-9a-f]{16}")


def origami_joint_names() -> tuple[str, ...]:
    """Return the card-defined 65D state/action order."""

    return (
        *(f"left_arm_j{index}" for index in range(7)),
        *(f"left_hand_j{index}" for index in range(22)),
        *(f"right_arm_j{index}" for index in range(7)),
        *(f"right_hand_j{index}" for index in range(22)),
        *(f"motor_j{index}" for index in range(7)),
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class VectorStream:
    key: str
    names: tuple[str, ...]
    semantics: str
    verified: bool

    def validate(self) -> None:
        if (
            not self.key
            or not self.names
            or len(set(self.names)) != len(self.names)
            or any(not name for name in self.names)
            or not self.semantics
        ):
            raise ValueError("vector stream contract is invalid")


@dataclass(frozen=True, slots=True)
class MosaicCrop:
    sensor: str
    x: int
    y: int
    width: int
    height: int

    def validate(
        self,
        *,
        frame_width: int,
        frame_height: int,
    ) -> None:
        if (
            not self.sensor
            or self.x < 0
            or self.y < 0
            or self.width <= 0
            or self.height <= 0
            or self.x + self.width > frame_width
            or self.y + self.height > frame_height
        ):
            raise ValueError("tactile mosaic crop is invalid")


@dataclass(frozen=True, slots=True)
class VideoStream:
    key: str
    kind: str
    width: int
    height: int
    channels: int
    encoding: str
    layout_verified: bool
    crops: tuple[MosaicCrop, ...] = ()

    def validate(
        self,
        *,
        sensor_order: Sequence[str],
    ) -> None:
        kinds = {"scene", "tactile_raw", "tactile_deformation"}
        encodings = {
            "rgb",
            "unverified_deformation_video",
            "sharpa_tacmap_uint8",
            "metric_depth_m",
        }
        if (
            not self.key
            or self.kind not in kinds
            or self.width <= 0
            or self.height <= 0
            or self.channels <= 0
            or self.encoding not in encodings
        ):
            raise ValueError("video stream contract is invalid")
        if self.kind == "scene":
            if self.encoding != "rgb" or self.crops:
                raise ValueError(
                    "scene videos must be uncropped RGB streams"
                )
        crop_sensors = [crop.sensor for crop in self.crops]
        if len(set(crop_sensors)) != len(crop_sensors):
            raise ValueError("video stream contains duplicate crops")
        for crop in self.crops:
            crop.validate(
                frame_width=self.width,
                frame_height=self.height,
            )
            if crop.sensor not in sensor_order:
                raise ValueError(
                    "video crop names an unknown tactile sensor"
                )
        if self.layout_verified and tuple(crop_sensors) != tuple(
            sensor_order
        ):
            raise ValueError(
                "verified tactile mosaics require every sensor in "
                "contract order"
            )
        if not self.layout_verified and self.crops:
            raise ValueError(
                "unverified tactile mosaics cannot publish crops"
            )
        if (
            self.encoding in {
                "sharpa_tacmap_uint8",
                "metric_depth_m",
            }
            and (
                self.kind != "tactile_deformation"
                or not self.layout_verified
            )
        ):
            raise ValueError(
                "metric deformation decoding requires a verified layout"
            )


@dataclass(frozen=True, slots=True)
class TactileStreamContract:
    """Fingerprint-bound physical stream description."""

    source_repository: str
    source_revision: str
    source_license: str
    rate_hz: float
    state: VectorStream
    action: VectorStream
    wrench_key: str
    sensors: tuple[str, ...]
    wrench_axes: tuple[str, ...]
    force_unit: str
    torque_unit: str
    wrench_frame: str
    wrench_verified: bool
    videos: tuple[VideoStream, ...]
    canonical_target_fingerprint: str | None = None
    schema: str = PHYSICAL_TACTILE_STREAM_SCHEMA
    format_version: int = PHYSICAL_TACTILE_STREAM_FORMAT

    def validate(self) -> None:
        if (
            self.schema != PHYSICAL_TACTILE_STREAM_SCHEMA
            or self.format_version
            != PHYSICAL_TACTILE_STREAM_FORMAT
            or not self.source_repository
            or _GIT_REVISION.fullmatch(self.source_revision) is None
            or not self.source_license
            or not math.isfinite(self.rate_hz)
            or self.rate_hz <= 0.0
        ):
            raise ValueError("physical tactile stream header is invalid")
        self.state.validate()
        self.action.validate()
        if (
            not self.wrench_key
            or not self.sensors
            or len(set(self.sensors)) != len(self.sensors)
            or any(not sensor for sensor in self.sensors)
            or tuple(self.wrench_axes) != WRENCH_AXES
            or self.force_unit not in {"N", "unverified"}
            or self.torque_unit not in {"N*m", "unverified"}
            or not self.wrench_frame
        ):
            raise ValueError("physical wrench stream is invalid")
        if self.wrench_verified and (
            self.force_unit != "N"
            or self.torque_unit != "N*m"
            or self.wrench_frame == "unverified"
        ):
            raise ValueError(
                "verified wrench streams require SI units and a frame"
            )
        keys = [video.key for video in self.videos]
        if len(set(keys)) != len(keys):
            raise ValueError("physical stream contains duplicate videos")
        for video in self.videos:
            video.validate(sensor_order=self.sensors)
        if (
            self.canonical_target_fingerprint is not None
            and _CANONICAL_FINGERPRINT.fullmatch(
                self.canonical_target_fingerprint
            )
            is None
        ):
            raise ValueError(
                "canonical tactile target fingerprint is invalid"
            )
        metric_videos = [
            video
            for video in self.videos
            if video.encoding
            in {"sharpa_tacmap_uint8", "metric_depth_m"}
        ]
        if self.canonical_target_fingerprint is not None and (
            not metric_videos
            or not self.wrench_verified
        ):
            raise ValueError(
                "canonical alignment requires verified metric and "
                "wrench streams"
            )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._payload(include_fingerprint=False))
        ).hexdigest()

    @property
    def canonically_aligned(self) -> bool:
        return self.canonical_target_fingerprint is not None

    def _payload(self, *, include_fingerprint: bool) -> dict[str, Any]:
        def vector(value: VectorStream) -> dict[str, Any]:
            payload = asdict(value)
            payload["names"] = list(value.names)
            return payload

        videos: list[dict[str, Any]] = []
        for video in self.videos:
            payload = asdict(video)
            payload["crops"] = [
                asdict(crop) for crop in video.crops
            ]
            videos.append(payload)
        result: dict[str, Any] = {
            "schema": self.schema,
            "format_version": self.format_version,
            "source": {
                "repository": self.source_repository,
                "revision": self.source_revision,
                "license": self.source_license,
            },
            "rate_hz": self.rate_hz,
            "state": vector(self.state),
            "action": vector(self.action),
            "wrench": {
                "key": self.wrench_key,
                "sensors": list(self.sensors),
                "axes": list(self.wrench_axes),
                "force_unit": self.force_unit,
                "torque_unit": self.torque_unit,
                "frame": self.wrench_frame,
                "verified": self.wrench_verified,
            },
            "videos": videos,
            "canonical_target_fingerprint": (
                self.canonical_target_fingerprint
            ),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return self._payload(include_fingerprint=True)

    def to_json(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(
            destination.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "TactileStreamContract":
        if not isinstance(payload, dict):
            raise ValueError("physical tactile stream must be an object")
        source = payload.get("source")
        wrench = payload.get("wrench")
        if not isinstance(source, dict) or not isinstance(wrench, dict):
            raise ValueError("physical tactile stream sections are missing")

        def vector(name: str) -> VectorStream:
            value = payload.get(name)
            if not isinstance(value, dict):
                raise ValueError(f"{name} stream is missing")
            return VectorStream(
                key=str(value.get("key", "")),
                names=tuple(value.get("names", ())),
                semantics=str(value.get("semantics", "")),
                verified=bool(value.get("verified", False)),
            )

        video_values = payload.get("videos")
        if not isinstance(video_values, list):
            raise ValueError("physical video streams are missing")
        videos: list[VideoStream] = []
        for value in video_values:
            if not isinstance(value, dict):
                raise ValueError("physical video stream is invalid")
            crops = value.get("crops", [])
            if not isinstance(crops, list):
                raise ValueError("physical video crops are invalid")
            if any(not isinstance(crop, dict) for crop in crops):
                raise ValueError("physical video crop is invalid")
            videos.append(
                VideoStream(
                    key=str(value.get("key", "")),
                    kind=str(value.get("kind", "")),
                    width=int(value.get("width", 0)),
                    height=int(value.get("height", 0)),
                    channels=int(value.get("channels", 0)),
                    encoding=str(value.get("encoding", "")),
                    layout_verified=bool(
                        value.get("layout_verified", False)
                    ),
                    crops=tuple(
                        MosaicCrop(
                            sensor=str(crop.get("sensor", "")),
                            x=int(crop.get("x", -1)),
                            y=int(crop.get("y", -1)),
                            width=int(crop.get("width", 0)),
                            height=int(crop.get("height", 0)),
                        )
                        for crop in crops
                    ),
                )
            )
        result = cls(
            source_repository=str(source.get("repository", "")),
            source_revision=str(source.get("revision", "")),
            source_license=str(source.get("license", "")),
            rate_hz=float(payload.get("rate_hz", 0.0)),
            state=vector("state"),
            action=vector("action"),
            wrench_key=str(wrench.get("key", "")),
            sensors=tuple(wrench.get("sensors", ())),
            wrench_axes=tuple(wrench.get("axes", ())),
            force_unit=str(wrench.get("force_unit", "")),
            torque_unit=str(wrench.get("torque_unit", "")),
            wrench_frame=str(wrench.get("frame", "")),
            wrench_verified=bool(wrench.get("verified", False)),
            videos=tuple(videos),
            canonical_target_fingerprint=payload.get(
                "canonical_target_fingerprint"
            ),
            schema=str(payload.get("schema", "")),
            format_version=int(payload.get("format_version", 0)),
        )
        result.validate()
        fingerprint = payload.get("fingerprint")
        if (
            not isinstance(fingerprint, str)
            or _SHA256.fullmatch(fingerprint) is None
            or fingerprint != result.fingerprint
        ):
            raise ValueError(
                "physical tactile stream fingerprint mismatch"
            )
        return result

    @classmethod
    def from_json(
        cls,
        path: str | os.PathLike[str],
    ) -> "TactileStreamContract":
        payload = json.loads(
            Path(path).expanduser().read_text(encoding="utf-8")
        )
        return cls.from_dict(payload)


def make_origami_stream_contract(
    revision: str,
) -> TactileStreamContract:
    """Create the conservative contract supported by the public card.

    The card names vector order and image dimensions, but does not establish
    wrench units/frames, command semantics, mosaic crops, or metric depth
    calibration.  Those fields therefore remain explicitly unverified.
    """

    names = origami_joint_names()
    videos = (
        VideoStream(
            "observation.images.head_left",
            "scene",
            480,
            480,
            3,
            "rgb",
            False,
        ),
        VideoStream(
            "observation.images.head_right",
            "scene",
            480,
            480,
            3,
            "rgb",
            False,
        ),
        VideoStream(
            "observation.images.wrist_left",
            "scene",
            480,
            480,
            3,
            "rgb",
            False,
        ),
        VideoStream(
            "observation.images.wrist_right",
            "scene",
            480,
            480,
            3,
            "rgb",
            False,
        ),
        VideoStream(
            "observation.images.tactile_deform",
            "tactile_deformation",
            1200,
            480,
            3,
            "unverified_deformation_video",
            False,
        ),
        VideoStream(
            "observation.images.tactile_raw",
            "tactile_raw",
            1600,
            480,
            3,
            "rgb",
            False,
        ),
    )
    result = TactileStreamContract(
        source_repository=ORIGAMI_REPOSITORY,
        source_revision=revision,
        source_license=ORIGAMI_LICENSE,
        rate_hz=30.0,
        state=VectorStream(
            "observation.state",
            names,
            "joint_space_state",
            True,
        ),
        action=VectorStream(
            "action",
            names,
            "unverified_joint_space_command",
            False,
        ),
        wrench_key="observation.tactile",
        sensors=WAVE_FINGERTIP_ORDER,
        wrench_axes=WRENCH_AXES,
        force_unit="unverified",
        torque_unit="unverified",
        wrench_frame="unverified",
        wrench_verified=False,
        videos=videos,
    )
    result.validate()
    return result


def with_verified_origami_layout(
    contract: TactileStreamContract,
    *,
    raw_crops: Sequence[MosaicCrop],
    deformation_crops: Sequence[MosaicCrop],
    deformation_encoding: str,
    force_unit: str,
    torque_unit: str,
    wrench_frame: str,
    action_semantics: str,
    canonical_target_fingerprint: str | None = None,
) -> TactileStreamContract:
    """Return a promoted contract after inspecting pinned source data."""

    videos: list[VideoStream] = []
    for video in contract.videos:
        if video.kind == "tactile_raw":
            video = replace(
                video,
                layout_verified=True,
                crops=tuple(raw_crops),
            )
        elif video.kind == "tactile_deformation":
            video = replace(
                video,
                encoding=deformation_encoding,
                layout_verified=True,
                crops=tuple(deformation_crops),
            )
        videos.append(video)
    promoted = replace(
        contract,
        action=replace(
            contract.action,
            semantics=action_semantics,
            verified=True,
        ),
        force_unit=force_unit,
        torque_unit=torque_unit,
        wrench_frame=wrench_frame,
        wrench_verified=True,
        videos=tuple(videos),
        canonical_target_fingerprint=canonical_target_fingerprint,
    )
    promoted.validate()
    return promoted


def unpack_tactile_mosaic(
    frame: npt.ArrayLike,
    stream: VideoStream,
    *,
    sensor_order: Sequence[str],
) -> npt.NDArray[np.uint8]:
    """Crop one verified mosaic into contract-ordered fingertip images."""

    stream.validate(sensor_order=sensor_order)
    if not stream.layout_verified or not stream.crops:
        raise ValueError("tactile mosaic layout is not verified")
    image = np.asarray(frame)
    expected = (stream.height, stream.width, stream.channels)
    if image.shape != expected:
        raise ValueError(
            f"tactile mosaic has shape {image.shape}; expected {expected}"
        )
    crops = [
        image[
            crop.y : crop.y + crop.height,
            crop.x : crop.x + crop.width,
            :,
        ]
        for crop in stream.crops
    ]
    shape = crops[0].shape
    if any(crop.shape != shape for crop in crops):
        raise ValueError(
            "one tactile mosaic stream requires equal crop dimensions"
        )
    return np.ascontiguousarray(np.stack(crops), dtype=np.uint8)


def decode_sharpa_tacmap_uint8(
    quantized: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Invert Sharpa Tacmap's piecewise uint8 deformation quantizer.

    Values 0..100 have 5 micrometre resolution.  Values above 100 have
    30 micrometre resolution.  Code 255 is a saturated lower bound and is
    therefore returned with validity false rather than presented as an exact
    depth.
    """

    source = np.asarray(quantized)
    if source.dtype != np.uint8:
        raise ValueError("Sharpa deformation input must be uint8")
    values = source.astype(np.float32)
    shallow = values <= 100.0
    depth = np.where(
        shallow,
        values * 5.0e-6,
        5.0e-4 + (values - 100.0) * 3.0e-5,
    ).astype(np.float32)
    valid = source < np.uint8(255)
    return (
        np.ascontiguousarray(depth),
        np.ascontiguousarray(valid),
    )


__all__ = [
    "MosaicCrop",
    "ORIGAMI_LICENSE",
    "ORIGAMI_REPOSITORY",
    "PHYSICAL_TACTILE_STREAM_FORMAT",
    "PHYSICAL_TACTILE_STREAM_SCHEMA",
    "TactileStreamContract",
    "VectorStream",
    "VideoStream",
    "WAVE_FINGERTIP_ORDER",
    "WAVE_HAND_JOINT_ORDER",
    "WRENCH_AXES",
    "decode_sharpa_tacmap_uint8",
    "make_origami_stream_contract",
    "origami_joint_names",
    "unpack_tactile_mosaic",
    "with_verified_origami_layout",
]
