"""Canonical robot-space ARDY motion construction and event extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np

from .common import (
    MotionEvent,
    MotionEventKind,
    _array_sha256,
    _canonical_json,
    _canonicalize_heading,
    _finite_difference,
    _quaternion_angular_velocity,
    _quaternion_log,
    _quaternion_to_matrix,
    _resample_cubic_limited,
    _resample_linear,
    _resample_quaternion,
    _sample_rows_linear,
)
from .motion_events import (
    _adaptive_knot_phases,
    _event_frame_features,
    _extract_motion_events,
    _phase_fourier_features,
    _select_tracked_links,
)


@dataclass(frozen=True, slots=True)
class CanonicalARDYMotion:
    """Deterministic, robot-space representation consumed by the hypernetwork."""

    source_fingerprint: str
    prompt: str
    frames_per_second: float
    phases: np.ndarray
    root_position_quaternion_xyzw: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_accelerations: np.ndarray
    contact_modes: np.ndarray
    contact_confidence: np.ndarray
    features: np.ndarray
    feature_schema: tuple[str, ...]
    reference_signature: np.ndarray
    signature_weights: np.ndarray
    events: tuple[MotionEvent, ...]
    knot_phases: np.ndarray
    knot_event_features: np.ndarray
    tracked_link_names: tuple[str, ...] = ()

    @property
    def frame_count(self) -> int:
        return int(self.phases.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.joint_positions.shape[1])

    @property
    def contact_count(self) -> int:
        return int(self.contact_modes.shape[1])

    @property
    def duration_seconds(self) -> float:
        return (self.frame_count - 1) / self.frames_per_second

    def validate(self) -> None:
        frames = self.frame_count
        joints = self.joint_count
        contacts = self.contact_count
        if (
            not self.source_fingerprint
            or len(self.source_fingerprint) != 64
            or not math.isfinite(self.frames_per_second)
            or self.frames_per_second <= 0.0
            or frames < 2
            or joints <= 0
            or contacts <= 0
            or self.phases.shape != (frames,)
            or self.root_position_quaternion_xyzw.shape != (frames, 7)
            or self.joint_velocities.shape != (frames, joints)
            or self.joint_accelerations.shape != (frames, joints)
            or self.contact_confidence.shape != (frames, contacts)
            or self.features.shape != (frames, len(self.feature_schema))
            or self.reference_signature.shape[0] != frames
            or self.signature_weights.shape != (self.reference_signature.shape[1],)
            or self.knot_phases.ndim != 1
            or self.knot_phases.size < 2
            or self.knot_event_features.shape[0] != self.knot_phases.size
        ):
            raise ValueError("canonical ARDY motion dimensions are invalid")
        finite_tables = (
            self.phases,
            self.root_position_quaternion_xyzw,
            self.joint_positions,
            self.joint_velocities,
            self.joint_accelerations,
            self.contact_confidence,
            self.features,
            self.reference_signature,
            self.signature_weights,
            self.knot_phases,
            self.knot_event_features,
        )
        if not all(np.isfinite(value).all() for value in finite_tables):
            raise ValueError("canonical ARDY motion contains non-finite values")
        if (
            not np.all(np.diff(self.phases) > 0.0)
            or abs(float(self.phases[0])) > 1.0e-7
            or abs(float(self.phases[-1]) - 1.0) > 1.0e-7
            or not np.all(np.diff(self.knot_phases) > 0.0)
            or abs(float(self.knot_phases[0])) > 1.0e-7
            or abs(float(self.knot_phases[-1]) - 1.0) > 1.0e-7
            or np.any(self.contact_confidence < 0.0)
            or np.any(self.contact_confidence > 1.0)
            or np.any(self.signature_weights <= 0.0)
        ):
            raise ValueError("canonical ARDY motion ranges are invalid")
        if not np.issubdtype(self.contact_modes.dtype, np.integer):
            raise ValueError("contact modes must be integer-valued")
        quaternions = self.root_position_quaternion_xyzw[:, 3:]
        if np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0)) > 1.0e-4:
            raise ValueError("root quaternions are not normalized")
        previous = -1.0
        for event in self.events:
            event.validate(frames, contacts)
            if event.phase < previous:
                raise ValueError("motion events are not phase ordered")
            previous = event.phase

    @classmethod
    def from_native_g1(
        cls,
        arrays: Mapping[str, np.ndarray],
        evidence: Mapping[str, Any],
        *,
        target_frames_per_second: float = 50.0,
        maximum_knot_interval_seconds: float = 0.20,
        maximum_knot_count: int = 64,
        maximum_tracked_links: int = 10,
    ) -> "CanonicalARDYMotion":
        """Build the exact hypernetwork input from ``native_g1_mechanism``.

        No physics fields are invented.  ARDY contact scores remain predicted
        intent and are consumed only as reference features and event guards.
        """

        root = np.asarray(arrays["root_position_quaternion_xyzw"], dtype=np.float64)
        joints = np.asarray(arrays["joint_positions"], dtype=np.float64)
        raw_contacts = np.asarray(arrays["foot_contacts"], dtype=np.uint8)
        raw_contact_scores = np.asarray(arrays["foot_contact_scores"], dtype=np.float64)
        link_names_array = np.asarray(
            arrays.get("link_names", np.empty(0, dtype=np.str_))
        )
        link_transforms = np.asarray(
            arrays.get(
                "link_position_quaternion_xyzw",
                np.empty((root.shape[0], 0, 7), dtype=np.float32),
            ),
            dtype=np.float64,
        )
        source_fps = float(evidence["fps"])
        if (
            root.ndim != 2
            or root.shape[1] != 7
            or joints.ndim != 2
            or joints.shape[0] != root.shape[0]
            or raw_contacts.shape != (root.shape[0], 4)
            or raw_contact_scores.shape != raw_contacts.shape
            or source_fps <= 0.0
            or target_frames_per_second <= 0.0
            or maximum_knot_interval_seconds <= 0.0
            or maximum_knot_count < 2
            or maximum_tracked_links < 0
        ):
            raise ValueError("native G1 ARDY arrays are invalid")
        if link_transforms.shape != (root.shape[0], link_names_array.size, 7):
            raise ValueError("G1 link transform table disagrees with link names")
        if not all(
            np.isfinite(table).all()
            for table in (root, joints, raw_contact_scores, link_transforms)
        ):
            raise ValueError("native G1 ARDY arrays contain non-finite values")

        source_duration = (root.shape[0] - 1) / source_fps
        target_frame_count = max(
            2,
            int(round(source_duration * target_frames_per_second)) + 1,
        )
        source_time = np.linspace(0.0, source_duration, root.shape[0])
        target_time = np.linspace(0.0, source_duration, target_frame_count)

        root_position = _resample_cubic_limited(source_time, root[:, :3], target_time)
        root_rotation = _resample_quaternion(source_time, root[:, 3:], target_time)
        joint_positions = _resample_cubic_limited(source_time, joints, target_time)
        contact_probabilities = np.stack(
            (
                np.max(raw_contact_scores[:, :2], axis=1),
                np.max(raw_contact_scores[:, 2:], axis=1),
            ),
            axis=1,
        )
        contact_probabilities = _resample_linear(
            source_time, contact_probabilities, target_time
        )
        contact_probabilities = np.clip(contact_probabilities, 0.0, 1.0)
        contact_modes = (contact_probabilities > 0.5).astype(np.uint32)
        contact_confidence = np.clip(
            2.0 * np.abs(contact_probabilities - 0.5), 0.0, 1.0
        )

        tracked_indices = _select_tracked_links(
            tuple(str(value) for value in link_names_array.tolist()),
            maximum_tracked_links,
        )
        tracked_names = tuple(str(link_names_array[index]) for index in tracked_indices)
        if tracked_indices:
            tracked_position = np.empty(
                (target_frame_count, len(tracked_indices), 3),
                dtype=np.float64,
            )
            for output_index, link_index in enumerate(tracked_indices):
                tracked_position[:, output_index] = _resample_cubic_limited(
                    source_time,
                    link_transforms[:, link_index, :3],
                    target_time,
                )
        else:
            tracked_position = np.empty((target_frame_count, 0, 3), dtype=np.float64)

        root_rotation, root_position, tracked_position = _canonicalize_heading(
            root_rotation, root_position, tracked_position
        )
        root = np.concatenate((root_position, root_rotation), axis=1)
        joint_velocity = _finite_difference(joint_positions, target_time)
        joint_acceleration = _finite_difference(joint_velocity, target_time)
        root_linear_velocity = _finite_difference(root_position, target_time)
        root_linear_acceleration = _finite_difference(root_linear_velocity, target_time)
        root_angular_velocity = _quaternion_angular_velocity(root_rotation, target_time)
        tracked_relative = tracked_position - root_position[:, None, :]
        tracked_velocity = _finite_difference(tracked_relative, target_time)
        phases = np.linspace(0.0, 1.0, target_frame_count, dtype=np.float64)

        events = _extract_motion_events(
            phases=phases,
            joint_velocity=joint_velocity,
            root_linear_velocity=root_linear_velocity,
            contact_modes=contact_modes,
            contact_confidence=contact_confidence,
        )
        knot_phases = _adaptive_knot_phases(
            phases=phases,
            events=events,
            duration_seconds=source_duration,
            maximum_interval_seconds=maximum_knot_interval_seconds,
            maximum_knot_count=maximum_knot_count,
        )
        event_frame_features = _event_frame_features(target_frame_count, events)
        knot_event_features = _sample_rows_linear(
            phases, event_frame_features, knot_phases
        )

        rotation_6d = _quaternion_to_matrix(root_rotation)[:, :, :2].reshape(
            target_frame_count, 6
        )
        phase_features = _phase_fourier_features(phases, harmonics=4)

        feature_parts: list[np.ndarray] = [
            joint_positions,
            joint_velocity,
            joint_acceleration,
            rotation_6d,
            root_linear_velocity,
            root_angular_velocity,
            root_linear_acceleration,
            root_position[:, 2:3],
            contact_probabilities,
            contact_confidence,
            contact_modes.astype(np.float64),
            phase_features,
            event_frame_features,
        ]
        feature_schema: list[str] = []
        feature_schema.extend(
            f"joint.position.{index}" for index in range(joints.shape[1])
        )
        feature_schema.extend(
            f"joint.velocity.{index}" for index in range(joints.shape[1])
        )
        feature_schema.extend(
            f"joint.acceleration.{index}" for index in range(joints.shape[1])
        )
        feature_schema.extend(f"root.rotation6d.{index}" for index in range(6))
        feature_schema.extend(f"root.linear_velocity.{axis}" for axis in "xyz")
        feature_schema.extend(f"root.angular_velocity.{axis}" for axis in "xyz")
        feature_schema.extend(f"root.linear_acceleration.{axis}" for axis in "xyz")
        feature_schema.append("root.height")
        feature_schema.extend(("contact.left.probability", "contact.right.probability"))
        feature_schema.extend(("contact.left.confidence", "contact.right.confidence"))
        feature_schema.extend(("contact.left.mode", "contact.right.mode"))
        feature_schema.extend(
            f"phase.fourier.{index}" for index in range(phase_features.shape[1])
        )
        feature_schema.extend(f"event.{kind.name}" for kind in MotionEventKind)
        if tracked_relative.shape[1]:
            feature_parts.extend(
                (
                    tracked_relative.reshape(target_frame_count, -1),
                    tracked_velocity.reshape(target_frame_count, -1),
                )
            )
            for name in tracked_names:
                feature_schema.extend(
                    f"link.{name}.relative_position.{axis}" for axis in "xyz"
                )
            for name in tracked_names:
                feature_schema.extend(
                    f"link.{name}.relative_velocity.{axis}" for axis in "xyz"
                )
        features = np.concatenate(feature_parts, axis=1).astype(np.float32)

        root_rotation_error = _quaternion_log(root_rotation)
        reference_signature = np.concatenate(
            (
                joint_positions,
                joint_velocity,
                root_rotation_error,
                root_linear_velocity,
                root_angular_velocity,
                contact_probabilities,
            ),
            axis=1,
        ).astype(np.float32)
        signature_weights = np.concatenate(
            (
                np.full(joints.shape[1], 1.0, dtype=np.float32),
                np.full(joints.shape[1], 0.20, dtype=np.float32),
                np.full(3, 2.0, dtype=np.float32),
                np.full(3, 0.50, dtype=np.float32),
                np.full(3, 0.75, dtype=np.float32),
                np.full(2, 4.0, dtype=np.float32),
            )
        )

        source_record = {
            "arrays_fingerprint": str(
                evidence.get("source_motion", {}).get("arrays_fingerprint", "")
            ),
            "model_revision": str(
                evidence.get("source_motion", {}).get("model_revision", "")
            ),
            "converter_revision": str(
                evidence.get("converter", {}).get("revision", "")
            ),
            "fps": source_fps,
            "target_fps": target_frames_per_second,
            "root_sha256": _array_sha256(root),
            "joint_sha256": _array_sha256(joint_positions),
            "contact_sha256": _array_sha256(contact_probabilities),
        }
        source_fingerprint = hashlib.sha256(_canonical_json(source_record)).hexdigest()
        motion = cls(
            source_fingerprint=source_fingerprint,
            prompt=str(evidence.get("source_motion", {}).get("prompt", "")),
            frames_per_second=float(target_frames_per_second),
            phases=phases.astype(np.float32),
            root_position_quaternion_xyzw=root.astype(np.float32),
            joint_positions=joint_positions.astype(np.float32),
            joint_velocities=joint_velocity.astype(np.float32),
            joint_accelerations=joint_acceleration.astype(np.float32),
            contact_modes=contact_modes.astype(np.uint32),
            contact_confidence=contact_confidence.astype(np.float32),
            features=features,
            feature_schema=tuple(feature_schema),
            reference_signature=reference_signature,
            signature_weights=signature_weights,
            events=events,
            knot_phases=knot_phases.astype(np.float32),
            knot_event_features=knot_event_features.astype(np.float32),
            tracked_link_names=tracked_names,
        )
        motion.validate()
        return motion
