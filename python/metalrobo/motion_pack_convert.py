"""Convert authorized tracked-link NPZ clips into a compact MotionPack.

This is a one-time authoring tool. It does not import PAC-MAN's PyTorch,
MuJoCo, or Warp runtime; the NPZ files already contain tracked-link poses.
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from .ardy_interaction_convert import read_interaction_pack
from .g1_motion_retarget import G1Kinematics

_HEADER = struct.Struct("<8sIIQQ")
_MOTION_VERSION = 1
_MOTION_KIND = 4
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _strings(values: tuple[str, ...]) -> bytes:
    return struct.pack("<Q", len(values)) + b"".join(_string(value) for value in values)


def _float_vector(values: np.ndarray) -> bytes:
    packed = np.asarray(values, dtype="<f4").reshape(-1)
    return struct.pack("<Q", packed.size) + packed.tobytes()


def _content_hash(payload: bytes) -> int:
    result = _FNV_OFFSET
    for value in payload:
        result ^= value
        result = (result * _FNV_PRIME) & ((1 << 64) - 1)
    return result or 1


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-8
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _clip_features(
    source: Path,
    body_order: tuple[str, ...],
    anchor_body: str,
    tracked_bodies: tuple[str, ...],
) -> tuple[float, np.ndarray]:
    with np.load(source, allow_pickle=False) as archive:
        positions = np.asarray(archive["body_pos_w"], dtype=np.float32)
        orientations = np.asarray(archive["body_quat_w"], dtype=np.float32)
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    if (
        positions.ndim != 3
        or positions.shape[2] != 3
        or orientations.shape != positions.shape[:2] + (4,)
        or positions.shape[1] != len(body_order)
        or positions.shape[0] < 2
        or not np.isfinite(positions).all()
        or not np.isfinite(orientations).all()
        or not np.isfinite(fps)
        or fps <= 0.0
    ):
        raise ValueError(f"{source}: tracked-link arrays are invalid")
    indices = {name: index for index, name in enumerate(body_order)}
    try:
        anchor_index = indices[anchor_body]
        tracked_indices = [indices[name] for name in tracked_bodies]
    except KeyError as error:
        raise ValueError(f"{source}: unknown body {error.args[0]}") from error
    rotations = _rotation_matrix_wxyz(orientations)
    anchor_rotation = rotations[:, anchor_index]
    anchor_inverse = np.swapaxes(anchor_rotation, 1, 2)
    relative_position = np.einsum(
        "fij,fbj->fbi",
        anchor_inverse,
        positions[:, tracked_indices] - positions[:, anchor_index, None],
    )
    relative_rotation = np.einsum(
        "fij,fbjk->fbik",
        anchor_inverse,
        rotations[:, tracked_indices],
    )
    orientation6 = relative_rotation[..., :2].reshape(
        positions.shape[0], len(tracked_indices), 6
    )
    return fps, np.concatenate((relative_position, orientation6), axis=2).reshape(
        positions.shape[0], -1
    )


def _interaction_clip_features(
    interaction_pack: Path,
    urdf: Path,
    clip_id: str | None,
    anchor_body: str,
    tracked_bodies: tuple[str, ...],
) -> tuple[str, str, str, str, float, np.ndarray]:
    """Turn visual G1 joint intent into root-invariant tracked-link features."""

    pack = read_interaction_pack(interaction_pack)
    clips = [clip for clip in pack.clips if clip_id is None or clip.id == clip_id]
    if len(clips) != 1:
        raise ValueError("InteractionPack clip identity is missing or ambiguous")
    clip = clips[0]
    model = G1Kinematics.from_urdf(urdf)
    indices = {name: index for index, name in enumerate(pack.joint_names)}
    try:
        canonical = np.asarray(
            [indices[name] for name in model.joint_indices], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(
            f"InteractionPack is missing G1 joint {error.args[0]}"
        ) from error
    features = np.empty(
        (clip.joint_targets.shape[0], 9 * len(tracked_bodies)),
        dtype=np.float32,
    )
    for frame, authored in enumerate(clip.joint_targets):
        poses = model.forward(np.asarray(authored[canonical], dtype=np.float64))
        if anchor_body not in poses or any(name not in poses for name in tracked_bodies):
            raise ValueError("motion anchor or tracked G1 link is unresolved")
        anchor = poses[anchor_body]
        inverse = anchor[:3, :3].T
        for body, name in enumerate(tracked_bodies):
            pose = poses[name]
            relative_position = inverse @ (
                pose[:3, 3] - anchor[:3, 3]
            )
            relative_rotation = inverse @ pose[:3, :3]
            base = 9 * body
            features[frame, base : base + 3] = relative_position
            features[frame, base + 3 : base + 9] = (
                relative_rotation[:, :2].reshape(6)
            )
    return (
        pack.source_repository,
        pack.source_revision,
        pack.license_name,
        clip.id,
        clip.frames_per_second,
        features,
    )


def convert(arguments: argparse.Namespace) -> None:
    tracked = tuple(arguments.tracked_bodies.split(","))
    clips: list[tuple[str, float, np.ndarray]] = []
    source_repository = arguments.source_repository
    source_revision = arguments.source_revision
    license_name = arguments.license
    if arguments.interaction_pack is not None:
        if arguments.urdf is None:
            raise ValueError("--interaction-pack requires --urdf")
        (
            inferred_repository,
            inferred_revision,
            inferred_license,
            clip_id,
            fps,
            features,
        ) = _interaction_clip_features(
            arguments.interaction_pack,
            arguments.urdf,
            arguments.clip,
            arguments.anchor_body,
            tracked,
        )
        source_repository = source_repository or inferred_repository
        source_revision = source_revision or inferred_revision
        license_name = license_name or inferred_license
        clips.append((clip_id, fps, features))
    else:
        if arguments.input is None or arguments.body_order is None:
            raise ValueError("tracked-link NPZ input requires --input and --body-order")
        body_order = tuple(arguments.body_order.split(","))
        sources = sorted(arguments.input.rglob("*.npz"))
        if not sources or not body_order:
            raise ValueError("input clips and body tables must be nonempty")
        for source in sources:
            fps, features = _clip_features(
                source, body_order, arguments.anchor_body, tracked
            )
            clips.append((source.stem, fps, features))
    if not tracked or not source_repository or not source_revision or not license_name:
        raise ValueError("motion provenance and tracked bodies must be nonempty")
    payload = b"".join(
        (
            _string(arguments.id),
            _string(source_repository),
            _string(source_revision),
            _string(license_name),
            _string(arguments.anchor_body),
            _strings(tracked),
            struct.pack("<I", 9 * len(tracked)),
            struct.pack("<Q", len(clips)),
            b"".join(
                _string(identifier) + struct.pack("<f", fps) + _float_vector(features)
                for identifier, fps, features in clips
            ),
        )
    )
    header = _HEADER.pack(
        b"MRLEARN\0",
        _MOTION_VERSION,
        _MOTION_KIND,
        len(payload),
        _content_hash(payload),
    )
    target = arguments.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(header)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(
        f"wrote {target} clips={len(clips)} "
        f"features={9 * len(tracked)} bytes={target.stat().st_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", type=Path)
    sources.add_argument("--interaction-pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--source-repository")
    parser.add_argument("--source-revision")
    parser.add_argument("--license")
    parser.add_argument("--body-order")
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--clip")
    parser.add_argument("--anchor-body", required=True)
    parser.add_argument("--tracked-bodies", required=True)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
