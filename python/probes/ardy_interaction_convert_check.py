#!/usr/bin/env python3
"""Focused ARDY -> InteractionPack contract check."""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metalrobo.ardy_interaction_convert import _content_hash, convert


class Reader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def unsigned(self) -> int:
        value = struct.unpack_from("<Q", self.payload, self.offset)[0]
        self.offset += 8
        return value

    def string(self) -> str:
        count = self.unsigned()
        value = self.payload[self.offset : self.offset + count].decode("utf-8")
        self.offset += count
        return value

    def vector(self, dtype: str) -> np.ndarray:
        count = self.unsigned()
        item_size = np.dtype(dtype).itemsize
        values = np.frombuffer(
            self.payload,
            dtype=dtype,
            count=count,
            offset=self.offset,
        ).copy()
        self.offset += count * item_size
        return values


def arguments(root: Path, qpos: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        motion_npz=root / "ardy.npz",
        qpos_csv=qpos,
        output=output,
        id="converter-check",
        clip_id="contact-shift",
        desired_outcome=None,
        source_repository="https://github.com/nv-tlabs/ardy",
        source_revision="probe-revision",
        license="Apache-2.0",
        left_contact_group="left_foot_contact",
        right_contact_group="right_foot_contact",
        counterpart="locomotion_ground",
        contact_confidence=0.5,
        joint_limit_margin=1.0e-4,
        loop=False,
    )


def expect_rejected(
    root: Path,
    qpos: np.ndarray,
    name: str,
    expected: str,
) -> None:
    csv = root / f"{name}.csv"
    output = root / f"{name}.interactionpack"
    np.savetxt(csv, qpos, delimiter=",")
    try:
        convert(arguments(root, csv, output))
    except ValueError as error:
        if expected not in str(error):
            raise AssertionError(f"unexpected rejection: {error}") from error
    else:
        raise AssertionError(f"{name} trajectory was accepted")
    if output.exists():
        raise AssertionError(f"{name} rejection published a partial artifact")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="metalrobo-ardy-check-") as directory:
        root = Path(directory)
        frame_count = 4
        qpos = np.zeros((frame_count, 36), dtype=np.float32)
        qpos[:, 2] = 0.78
        qpos[:, 3] = 1.0  # ARDY wxyz identity.
        qpos[:, 8] = np.asarray((0.0, 0.001, 0.002, 0.003), dtype=np.float32)
        contacts = np.asarray(
            (
                (1, 1, 1, 1),
                (1, 0, 1, 0),
                (1, 0, 0, 0),
                (0, 0, 0, 0),
            ),
            dtype=np.float32,
        )
        np.savez(
            root / "ardy.npz",
            foot_contacts=contacts,
            fps=np.asarray((25.0,), dtype=np.float32),
            text=np.asarray(("shift support and release",)),
        )
        csv = root / "ardy.csv"
        output = root / "ardy.interactionpack"
        np.savetxt(csv, qpos, delimiter=",")
        convert(arguments(root, csv, output))

        artifact = output.read_bytes()
        magic, version, kind, payload_size, content_hash = struct.unpack_from(
            "<8sIIQQ", artifact, 0
        )
        payload = artifact[32:]
        if (
            magic != b"MRLEARN\0"
            or version != 1
            or kind != 5
            or payload_size != len(payload)
            or content_hash != _content_hash(payload)
        ):
            raise AssertionError("InteractionPack header or content hash is invalid")

        reader = Reader(payload)
        identity = tuple(reader.string() for _ in range(5))
        joint_count = reader.unsigned()
        joints = tuple(reader.string() for _ in range(joint_count))
        track_count = reader.unsigned()
        tracks = tuple(
            (reader.string(), reader.string(), reader.string())
            for _ in range(track_count)
        )
        clip_count = reader.unsigned()
        clip_id = reader.string()
        desired_outcome = reader.string()
        fps, frames, loop = struct.unpack_from("<fII", payload, reader.offset)
        reader.offset += 12
        root_targets = reader.vector("<f4")
        joint_targets = reader.vector("<f4")
        modes = reader.vector("<u4")
        masks = reader.vector("<u4")
        flags = reader.vector("<u4")
        confidence = reader.vector("<f4")
        targets = reader.vector("<f4")
        tolerances = reader.vector("<f4")
        if (
            identity[0] != "converter-check"
            or identity[4] != "metalrobo_z_up_x_forward_xyzw"
            or len(joints) != 29
            or tracks[0][2] != "locomotion_ground"
            or clip_count != 1
            or clip_id != "contact-shift"
            or desired_outcome != "shift support and release"
            or fps != 25.0
            or frames != frame_count
            or loop != 0
            or root_targets.shape != (frame_count * 7,)
            or not np.array_equal(root_targets[3:7], (0.0, 0.0, 0.0, 1.0))
            or joint_targets.shape != (frame_count * 29,)
            or not np.array_equal(modes, (2, 2, 2, 2, 2, 0, 0, 0))
            or np.any(masks)
            or not np.all(flags == 1)
            or not np.all(confidence == 0.5)
            or np.any(targets)
            or np.any(tolerances)
            or reader.offset != len(payload)
        ):
            raise AssertionError("converted InteractionPack semantics changed")

        outside_limit = qpos.copy()
        outside_limit[0, 7] = 100.0
        expect_rejected(root, outside_limit, "position-limit", "joint target")
        outside_velocity = qpos.copy()
        outside_velocity[1:, 7] = 2.0
        expect_rejected(root, outside_velocity, "velocity-limit", "velocity limit")
        invalid_quaternion = qpos.copy()
        invalid_quaternion[0, 3] = 2.0
        expect_rejected(
            root,
            invalid_quaternion,
            "quaternion-norm",
            "quaternion norm",
        )
        invalid_contacts = contacts.copy()
        invalid_contacts[0, 0] = 0.5
        np.savez(
            root / "ardy.npz",
            foot_contacts=invalid_contacts,
            fps=np.asarray((25.0,), dtype=np.float32),
            text=np.asarray(("shift support and release",)),
        )
        contact_output = root / "nonbinary-contact.interactionpack"
        try:
            convert(arguments(root, csv, contact_output))
        except ValueError as error:
            if "contacts" not in str(error):
                raise AssertionError(f"unexpected rejection: {error}") from error
        else:
            raise AssertionError("non-binary ARDY contact was accepted")
        if contact_output.exists():
            raise AssertionError("contact rejection published a partial artifact")
        print(
            "ardy_interaction_convert_check passed "
            f"content_hash={content_hash} frames={frames} contacts={track_count}"
        )


if __name__ == "__main__":
    main()
