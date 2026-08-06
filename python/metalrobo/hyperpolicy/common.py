"""Shared numeric and artifact primitives for ARDY hyper-policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

HYPER_BASE_FORMAT = "numi.hyper-base-policy.v1"
MOTION_POLICY_FORMAT = "numi.ardy-hyper-policy.v1"
MOTION_BUNDLE_FORMAT = "numi.motion-policy-bundle.v1"
_ACTIVATION_IDENTITY = 0
_ACTIVATION_ELU = 3
_EPSILON = 1.0e-8

class MotionEventKind(IntEnum):
    """Provider-neutral physical phase landmarks.

    Contact predictions only declare intended guards.  Measured contact remains
    authoritative when :class:`EventSynchronizedPhaseTracker` advances them.
    """

    start = 1
    support_change = 2
    contact_on = 3
    contact_off = 4
    takeoff = 5
    apex = 6
    landing = 7
    stop = 8


@dataclass(frozen=True, slots=True)
class MotionEvent:
    phase: float
    frame: int
    kind: MotionEventKind
    required_contact_on_mask: int = 0
    required_contact_off_mask: int = 0
    confidence: float = 1.0
    minimum_dwell_steps: int = 1

    def validate(self, frame_count: int, contact_count: int) -> None:
        contact_mask = (1 << contact_count) - 1
        if (
            not math.isfinite(self.phase)
            or not 0.0 <= self.phase <= 1.0
            or not 0 <= self.frame < frame_count
            or self.minimum_dwell_steps <= 0
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
            or self.required_contact_on_mask & ~contact_mask
            or self.required_contact_off_mask & ~contact_mask
            or self.required_contact_on_mask & self.required_contact_off_mask
        ):
            raise ValueError("motion event contract is invalid")



def event_safe_tangents(
    phases: np.ndarray,
    values: np.ndarray,
    *,
    event_phases: Sequence[float] = (),
) -> np.ndarray:
    """Fritsch-Carlson tangents with zero derivative at physical events."""

    x = np.asarray(phases, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or y.shape[0] != x.size:
        raise ValueError("event-safe tangent table dimensions are invalid")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.all(
        np.diff(x) > 0.0
    ):
        raise ValueError("event-safe tangent input is invalid")
    original_shape = y.shape
    flat = y.reshape(x.size, -1)
    h = np.diff(x)
    slope = np.diff(flat, axis=0) / h[:, None]
    tangent = np.zeros_like(flat)
    tangent[0] = slope[0]
    tangent[-1] = slope[-1]
    if x.size > 2:
        left = slope[:-1]
        right = slope[1:]
        same_sign = left * right > 0.0
        w1 = (2.0 * h[1:] + h[:-1])[:, None]
        w2 = (h[1:] + 2.0 * h[:-1])[:, None]
        denominator = np.zeros_like(left)
        np.divide(w1, left, out=denominator, where=np.abs(left) > _EPSILON)
        temporary = np.zeros_like(right)
        np.divide(w2, right, out=temporary, where=np.abs(right) > _EPSILON)
        denominator += temporary
        harmonic = np.zeros_like(left)
        np.divide(
            w1 + w2,
            denominator,
            out=harmonic,
            where=np.abs(denominator) > _EPSILON,
        )
        tangent[1:-1] = np.where(same_sign, harmonic, 0.0)

    # Clamp endpoints and interval derivatives to avoid cubic overshoot even
    # when a learned coefficient program is locally non-monotone.
    for interval in range(x.size - 1):
        delta = slope[interval]
        zero = np.abs(delta) <= _EPSILON
        tangent[interval, zero] = 0.0
        tangent[interval + 1, zero] = 0.0
        alpha = np.zeros_like(delta)
        beta = np.zeros_like(delta)
        np.divide(
            tangent[interval],
            delta,
            out=alpha,
            where=~zero,
        )
        np.divide(
            tangent[interval + 1],
            delta,
            out=beta,
            where=~zero,
        )
        negative = (alpha < 0.0) | (beta < 0.0)
        tangent[interval, negative] = 0.0
        tangent[interval + 1, negative] = 0.0
        magnitude = np.square(alpha) + np.square(beta)
        excessive = magnitude > 9.0
        if np.any(excessive):
            scale = np.ones_like(magnitude)
            scale[excessive] = 3.0 / np.sqrt(magnitude[excessive])
            tangent[interval, excessive] = (
                scale[excessive] * alpha[excessive] * delta[excessive]
            )
            tangent[interval + 1, excessive] = (
                scale[excessive] * beta[excessive] * delta[excessive]
            )

    for event_phase in event_phases:
        index = int(np.argmin(np.abs(x - float(event_phase))))
        if abs(float(x[index]) - float(event_phase)) <= 1.0e-4:
            tangent[index] = 0.0
    return tangent.reshape(original_shape).astype(np.float32)


def evaluate_event_safe_cubic(
    phases: np.ndarray,
    values: np.ndarray,
    tangents: np.ndarray,
    phase: float,
) -> np.ndarray:
    x = np.asarray(phases, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    m = np.asarray(tangents, dtype=np.float64)
    if (
        x.ndim != 1
        or x.size < 2
        or y.shape != m.shape
        or y.shape[0] != x.size
        or not math.isfinite(phase)
    ):
        raise ValueError("event-safe cubic evaluation contract is invalid")
    value = float(np.clip(phase, x[0], x[-1]))
    index = int(np.searchsorted(x, value, side="right") - 1)
    index = int(np.clip(index, 0, x.size - 2))
    width = x[index + 1] - x[index]
    u = (value - x[index]) / width
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return (
        h00 * y[index]
        + h10 * width * m[index]
        + h01 * y[index + 1]
        + h11 * width * m[index + 1]
    ).astype(np.float32)



def _canonicalize_heading(
    root_quaternion: np.ndarray,
    root_position: np.ndarray,
    tracked_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_matrix = _quaternion_to_matrix(root_quaternion[:1])[0]
    yaw = math.atan2(float(initial_matrix[1, 0]), float(initial_matrix[0, 0]))
    inverse_yaw = np.asarray(
        [0.0, 0.0, math.sin(-0.5 * yaw), math.cos(-0.5 * yaw)],
        dtype=np.float64,
    )
    canonical_rotation = _quaternion_multiply(
        np.broadcast_to(inverse_yaw, root_quaternion.shape),
        root_quaternion,
    )
    canonical_rotation = _normalize_quaternion(canonical_rotation)
    rotation = _quaternion_to_matrix(inverse_yaw[None])[0]
    origin = root_position[0].copy()
    origin[2] = 0.0
    canonical_position = (root_position - origin) @ rotation.T
    if tracked_position.shape[1]:
        canonical_tracked = (
            tracked_position - origin[None, None, :]
        ) @ rotation.T
    else:
        canonical_tracked = tracked_position.copy()
    return canonical_rotation, canonical_position, canonical_tracked


def _resample_linear(
    source_time: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    flat = source.reshape(source.shape[0], -1)
    result = np.empty((target_time.size, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(target_time, source_time, flat[:, column])
    return result.reshape((target_time.size,) + source.shape[1:])


def _resample_cubic_limited(
    source_time: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.shape[0] < 3:
        return _resample_linear(source_time, source, target_time)
    flat = source.reshape(source.shape[0], -1)
    derivative = _finite_difference(flat, source_time)
    result = np.empty((target_time.size, flat.shape[1]), dtype=np.float64)
    interval = np.searchsorted(source_time, target_time, side="right") - 1
    interval = np.clip(interval, 0, source_time.size - 2)
    for output, (time, index) in enumerate(zip(target_time, interval, strict=True)):
        width = source_time[index + 1] - source_time[index]
        u = (time - source_time[index]) / width
        u2 = u * u
        u3 = u2 * u
        value = (
            (2.0 * u3 - 3.0 * u2 + 1.0) * flat[index]
            + (u3 - 2.0 * u2 + u) * width * derivative[index]
            + (-2.0 * u3 + 3.0 * u2) * flat[index + 1]
            + (u3 - u2) * width * derivative[index + 1]
        )
        lower = np.minimum(flat[index], flat[index + 1])
        upper = np.maximum(flat[index], flat[index + 1])
        result[output] = np.clip(value, lower, upper)
    return result.reshape((target_time.size,) + source.shape[1:])


def _resample_quaternion(
    source_time: np.ndarray,
    quaternion: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    source = _normalize_quaternion(np.asarray(quaternion, dtype=np.float64))
    interval = np.searchsorted(source_time, target_time, side="right") - 1
    interval = np.clip(interval, 0, source_time.size - 2)
    result = np.empty((target_time.size, 4), dtype=np.float64)
    for output, (time, index) in enumerate(zip(target_time, interval, strict=True)):
        width = source_time[index + 1] - source_time[index]
        fraction = (time - source_time[index]) / width
        result[output] = _quaternion_slerp(
            source[index], source[index + 1], float(fraction)
        )
    return _normalize_quaternion(result)


def _quaternion_slerp(first: np.ndarray, second: np.ndarray, t: float) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(
            ((1.0 - t) * left + t * right)[None]
        )[0]
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (
        math.sin((1.0 - t) * angle) / sine * left
        + math.sin(t * angle) / sine * right
    )


def _finite_difference(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    result = np.empty_like(value)
    result[0] = (value[1] - value[0]) / (t[1] - t[0])
    result[-1] = (value[-1] - value[-2]) / (t[-1] - t[-2])
    if value.shape[0] > 2:
        denominator = (t[2:] - t[:-2]).reshape(
            (-1,) + (1,) * (value.ndim - 1)
        )
        result[1:-1] = (value[2:] - value[:-2]) / denominator
    return result


def _quaternion_angular_velocity(
    quaternion: np.ndarray, time: np.ndarray
) -> np.ndarray:
    q = _normalize_quaternion(np.asarray(quaternion, dtype=np.float64))
    result = np.zeros((q.shape[0], 3), dtype=np.float64)
    for index in range(q.shape[0] - 1):
        relative = _quaternion_multiply(
            _quaternion_conjugate(q[index])[None], q[index + 1][None]
        )[0]
        vector = _quaternion_log(relative[None])[0]
        result[index] = vector / (time[index + 1] - time[index])
    result[-1] = result[-2]
    if q.shape[0] > 2:
        result[1:-1] = 0.5 * (result[:-2] + result[1:-1])
    return result


def _quaternion_log(quaternion: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(np.asarray(quaternion, dtype=np.float64))
    q = np.where(q[:, 3:4] < 0.0, -q, q)
    vector = q[:, :3]
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(q[:, 3:4], -1.0, 1.0))
    scale = np.divide(
        angle,
        norm,
        out=np.full_like(angle, 2.0),
        where=norm > _EPSILON,
    )
    return vector * scale


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[..., :3] *= -1.0
    return value


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(np.asarray(first), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(second), -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm < _EPSILON):
        raise ValueError("quaternion has zero norm")
    return value / norm


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(quaternion)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _sample_rows_linear(
    phases: np.ndarray,
    values: np.ndarray,
    target_phases: np.ndarray,
) -> np.ndarray:
    x = np.asarray(phases, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_phases, dtype=np.float64)
    flat = y.reshape(y.shape[0], -1)
    result = np.empty((target.size, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(target, x, flat[:, column])
    return result.reshape((target.size,) + y.shape[1:]).astype(np.float32)


def _array_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    header = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return header + b"\0" + np.ascontiguousarray(array).tobytes(order="C")


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(_array_bytes(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _event_record(event: MotionEvent) -> dict[str, Any]:
    return {
        "phase": event.phase,
        "frame": event.frame,
        "kind": int(event.kind),
        "required_contact_on_mask": event.required_contact_on_mask,
        "required_contact_off_mask": event.required_contact_off_mask,
        "confidence": event.confidence,
        "minimum_dwell_steps": event.minimum_dwell_steps,
    }


def _event_from_record(value: Mapping[str, Any]) -> MotionEvent:
    return MotionEvent(
        phase=float(value["phase"]),
        frame=int(value["frame"]),
        kind=MotionEventKind(int(value["kind"])),
        required_contact_on_mask=int(value["required_contact_on_mask"]),
        required_contact_off_mask=int(value["required_contact_off_mask"]),
        confidence=float(value["confidence"]),
        minimum_dwell_steps=int(value["minimum_dwell_steps"]),
    )


class _atomic_directory:
    def __init__(self, target: Path) -> None:
        self.target = target
        self.staging: Path | None = None

    def __enter__(self) -> Path:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.target.name}.", dir=self.target.parent
            )
        )
        return self.staging

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self.staging is not None
        if exc_type is not None:
            shutil.rmtree(self.staging, ignore_errors=True)
            return
        backup = self.target.with_name(f".{self.target.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        if self.target.exists():
            os.replace(self.target, backup)
        try:
            os.replace(self.staging, self.target)
        except BaseException:
            if backup.exists() and not self.target.exists():
                os.replace(backup, self.target)
            raise
        finally:
            if backup.exists():
                shutil.rmtree(backup)


def _write_array_files(
    directory: Path,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in sorted(arrays):
        path = directory / f"{name}.npy"
        with path.open("wb") as stream:
            np.save(stream, np.asarray(arrays[name]), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        result[name] = _sha256_file(path)
    return result


def _verify_file_hashes(directory: Path, hashes: Mapping[str, str]) -> None:
    for name, expected in hashes.items():
        path = directory / f"{name}.npy"
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"artifact array {name!r} failed authentication")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
