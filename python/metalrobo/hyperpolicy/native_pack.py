"""Canonical MRLEARN HyperPolicyPack v1 serialization.

The byte contract matches ``writeHyperPolicyPack`` in the native runtime.  It
lets the offline MLX compiler publish the exact artifact consumed by C++/Metal
without importing the simulator or using an ad-hoc directory at deployment.
"""

from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile
from typing import Sequence

import numpy as np

from .base import HyperBasePolicy
from .generated import GeneratedMotionPolicy

_MAGIC = b"MRLEARN\0"
_VERSION = 1
_KIND = 9
_HEADER = struct.Struct("<8sIIQQ")
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211


def _hash(payload: bytes | bytearray | memoryview) -> int:
    value = _FNV_OFFSET
    for byte in payload:
        value ^= int(byte)
        value = (value * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def u32(self, value: int) -> None:
        self.data += struct.pack("<I", int(value))

    def u64(self, value: int) -> None:
        self.data += struct.pack("<Q", int(value))

    def f32(self, value: float) -> None:
        self.data += struct.pack("<f", float(value))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u64(len(encoded))
        self.data += encoded

    def vector(self, values: np.ndarray, dtype: str) -> None:
        value = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
        self.u64(value.size)
        self.data += value.tobytes(order="C")


class _Reader:
    def __init__(self, payload: memoryview) -> None:
        self.payload = payload
        self.offset = 0

    def take(self, count: int) -> memoryview:
        if count < 0 or self.offset + count > len(self.payload):
            raise ValueError("HyperPolicyPack is truncated")
        result = self.payload[self.offset : self.offset + count]
        self.offset += count
        return result

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def string(self) -> str:
        return bytes(self.take(self.u64())).decode("utf-8")

    def vector(self, dtype: str) -> np.ndarray:
        count = self.u64()
        scalar = np.dtype(dtype)
        return np.frombuffer(
            self.take(count * scalar.itemsize), dtype=scalar, count=count
        ).copy()


def _source_word(fingerprint: str) -> int:
    if len(fingerprint) != 64:
        raise ValueError("source motion fingerprint must be SHA-256")
    value = int.from_bytes(bytes.fromhex(fingerprint)[:8], "little")
    return value or 1


def _contact_masks(contact_modes: np.ndarray) -> np.ndarray:
    modes = np.asarray(contact_modes, dtype=np.uint32)
    if modes.ndim != 2 or modes.shape[1] > 32:
        raise ValueError("contact mode table must be [frames, <=32]")
    bits = np.arange(modes.shape[1], dtype=np.uint32)
    return np.sum((modes != 0).astype(np.uint32) << bits[None], axis=1, dtype=np.uint32)


def write_native_hyper_policy_pack(
    path: str | Path,
    *,
    hyper_base: HyperBasePolicy,
    motion_policy: GeneratedMotionPolicy,
    contact_group_indices: Sequence[int],
    policy_id: str | None = None,
    revision: int | None = None,
    action_log_standard_deviation: np.ndarray | None = None,
    maximum_phase_advance_per_step: float = 0.04,
    phase_alignment_blend: float = 0.35,
    phase_alignment_huber_delta: float = 2.0,
) -> int:
    """Publish one deployable HyperPolicyPack and return its FNV content hash."""

    base = hyper_base.with_fingerprint()
    policy = motion_policy.with_fingerprint()
    base.validate(require_fingerprint=True)
    policy.validate(hyper_base=base, require_fingerprint=True)
    contact_groups = np.asarray(contact_group_indices, dtype=np.uint32)
    if contact_groups.shape != (policy.contact_modes.shape[1],):
        raise ValueError("one task contact-group index is required per track")
    log_std = (
        np.empty(0, dtype=np.float32)
        if action_log_standard_deviation is None
        else np.asarray(action_log_standard_deviation, dtype=np.float32)
    )
    if log_std.size not in (0, base.action_count):
        raise ValueError("action log-standard-deviation width is invalid")
    if log_std.size and np.any((log_std < -5.0) | (log_std > 2.0)):
        raise ValueError("action log standard deviations must be in [-5, 2]")

    writer = _Writer()
    writer.string(policy_id or policy.id)
    writer.u64(revision or policy.revision)
    writer.u64(1)
    writer.u64(base.world_fingerprint)
    writer.u64(base.task_fingerprint)
    writer.u64(base.observation_fingerprint)
    writer.u64(base.action_fingerprint)
    writer.u64(_source_word(policy.source_motion_fingerprint))
    writer.vector(base.observation_mean, "<f4")
    writer.vector(base.observation_inverse_standard_deviation, "<f4")
    writer.u64(len(base.layers))
    for layer in base.layers:
        writer.u32(layer.input_count)
        writer.u32(layer.output_count)
        writer.u32(layer.rank)
        writer.u32(layer.activation)
        writer.vector(layer.weight, "<f4")
        writer.vector(layer.bias, "<f4")
        writer.vector(layer.adapter_down, "<f4")
        writer.vector(layer.adapter_up, "<f4")
        writer.vector(layer.adapter_bias_basis, "<f4")
    writer.vector(base.coefficient_limits, "<f4")
    writer.vector(base.action_bias, "<f4")
    writer.vector(base.action_scale, "<f4")
    writer.vector(log_std, "<f4")
    writer.f32(base.observation_clip)
    writer.f32(base.action_clip)
    writer.vector(policy.knot_phases, "<f4")
    writer.vector(policy.coefficient_knots, "<f4")
    writer.vector(policy.coefficient_tangents, "<f4")
    writer.vector(policy.authority_knots, "<f4")
    writer.vector(policy.authority_tangents, "<f4")
    writer.vector(policy.phase_rate_knots, "<f4")
    writer.vector(policy.phase_rate_tangents, "<f4")
    writer.vector(policy.reference_phases, "<f4")
    writer.vector(policy.reference_actions, "<f4")
    writer.u32(policy.reference_signature.shape[1])
    writer.vector(policy.reference_signature, "<f4")
    writer.vector(policy.signature_weights, "<f4")
    writer.u32(policy.contact_modes.shape[1])
    writer.vector(_contact_masks(policy.contact_modes), "<u4")
    writer.vector(contact_groups, "<u4")
    writer.u64(len(policy.events))
    for event in policy.events:
        writer.f32(event.phase)
        writer.f32(event.confidence)
        writer.u32(event.required_contact_on_mask)
        writer.u32(event.required_contact_off_mask)
        writer.u32(event.minimum_dwell_steps)
        writer.u32(int(event.kind))
    writer.vector(policy.action_lower, "<f4")
    writer.vector(policy.action_upper, "<f4")
    writer.vector(policy.maximum_action_rate, "<f4")
    writer.f32(maximum_phase_advance_per_step)
    writer.f32(phase_alignment_blend)
    writer.f32(phase_alignment_huber_delta)
    writer.f32(1.0 / policy.frames_per_second)

    payload = bytes(writer.data)
    content_hash = _hash(payload)
    header = _HEADER.pack(_MAGIC, _VERSION, _KIND, len(payload), content_hash)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(header)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return content_hash


def inspect_native_hyper_policy_pack(path: str | Path) -> dict[str, object]:
    """Authenticate and return the structural identity of a native artifact."""

    data = Path(path).read_bytes()
    if len(data) < _HEADER.size:
        raise ValueError("HyperPolicyPack header is truncated")
    magic, version, kind, payload_size, content_hash = _HEADER.unpack_from(data)
    payload = memoryview(data)[_HEADER.size :]
    if (
        magic != _MAGIC
        or version != _VERSION
        or kind != _KIND
        or payload_size != len(payload)
        or content_hash != _hash(payload)
    ):
        raise ValueError("HyperPolicyPack header or fingerprint is invalid")
    reader = _Reader(payload)
    identifier = reader.string()
    revision = reader.u64()
    contract = {
        "version": reader.u64(),
        "world": reader.u64(),
        "task": reader.u64(),
        "observation": reader.u64(),
        "action": reader.u64(),
    }
    source_motion = reader.u64()
    observation_count = reader.vector("<f4").size
    reader.vector("<f4")
    layer_count = reader.u64()
    ranks: list[int] = []
    action_count = 0
    for _ in range(layer_count):
        input_count = reader.u32()
        output_count = reader.u32()
        rank = reader.u32()
        reader.u32()
        for dtype in ("<f4",) * 5:
            reader.vector(dtype)
        ranks.append(rank)
        action_count = output_count
        if input_count == 0 or output_count == 0:
            raise ValueError("HyperPolicyPack contains an empty layer")
    return {
        "id": identifier,
        "revision": revision,
        "contract": contract,
        "source_motion_fingerprint_word": source_motion,
        "observation_count": observation_count,
        "action_count": action_count,
        "layer_count": layer_count,
        "adapter_ranks": ranks,
        "content_hash": content_hash,
        "payload_bytes": payload_size,
    }
