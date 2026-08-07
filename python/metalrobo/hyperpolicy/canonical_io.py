"""Authenticated storage for canonical ARDY hyper-policy training inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import (
    _atomic_directory,
    _canonical_json,
    _event_from_record,
    _event_record,
    _verify_file_hashes,
    _write_array_files,
)
from .motion import CanonicalARDYMotion

_FORMAT = "numi.ardy-canonical-motion.v1"


def write_canonical_motion(
    motion: CanonicalARDYMotion,
    directory: str | Path,
) -> Path:
    motion.validate()
    target = Path(directory)
    arrays = {
        "phases": motion.phases,
        "root_position_quaternion_xyzw": (motion.root_position_quaternion_xyzw),
        "joint_positions": motion.joint_positions,
        "joint_velocities": motion.joint_velocities,
        "joint_accelerations": motion.joint_accelerations,
        "contact_modes": motion.contact_modes,
        "contact_confidence": motion.contact_confidence,
        "features": motion.features,
        "reference_signature": motion.reference_signature,
        "signature_weights": motion.signature_weights,
        "knot_phases": motion.knot_phases,
        "knot_event_features": motion.knot_event_features,
    }
    with _atomic_directory(target) as staging:
        files = _write_array_files(staging, arrays)
        manifest = {
            "format": _FORMAT,
            "source_fingerprint": motion.source_fingerprint,
            "prompt": motion.prompt,
            "frames_per_second": motion.frames_per_second,
            "feature_schema": list(motion.feature_schema),
            "tracked_link_names": list(motion.tracked_link_names),
            "events": [_event_record(event) for event in motion.events],
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
    return target


def read_canonical_motion(directory: str | Path) -> CanonicalARDYMotion:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("format") != _FORMAT:
        raise ValueError("canonical ARDY motion format is unsupported")
    _verify_file_hashes(source, manifest["files"])
    arrays = {
        name: np.load(source / f"{name}.npy", allow_pickle=False)
        for name in manifest["files"]
    }
    motion = CanonicalARDYMotion(
        source_fingerprint=str(manifest["source_fingerprint"]),
        prompt=str(manifest["prompt"]),
        frames_per_second=float(manifest["frames_per_second"]),
        phases=np.asarray(arrays["phases"], dtype=np.float32),
        root_position_quaternion_xyzw=np.asarray(
            arrays["root_position_quaternion_xyzw"], dtype=np.float32
        ),
        joint_positions=np.asarray(arrays["joint_positions"], dtype=np.float32),
        joint_velocities=np.asarray(arrays["joint_velocities"], dtype=np.float32),
        joint_accelerations=np.asarray(arrays["joint_accelerations"], dtype=np.float32),
        contact_modes=np.asarray(arrays["contact_modes"], dtype=np.uint32),
        contact_confidence=np.asarray(arrays["contact_confidence"], dtype=np.float32),
        features=np.asarray(arrays["features"], dtype=np.float32),
        feature_schema=tuple(str(value) for value in manifest["feature_schema"]),
        reference_signature=np.asarray(arrays["reference_signature"], dtype=np.float32),
        signature_weights=np.asarray(arrays["signature_weights"], dtype=np.float32),
        events=tuple(_event_from_record(value) for value in manifest["events"]),
        knot_phases=np.asarray(arrays["knot_phases"], dtype=np.float32),
        knot_event_features=np.asarray(arrays["knot_event_features"], dtype=np.float32),
        tracked_link_names=tuple(
            str(value) for value in manifest.get("tracked_link_names", ())
        ),
    )
    motion.validate()
    return motion
