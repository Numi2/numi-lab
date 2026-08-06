"""Generated per-motion adapter program and deployment bundle artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .base import HyperBasePolicy
from .common import (
    MOTION_BUNDLE_FORMAT, MOTION_POLICY_FORMAT, MotionEvent,
    _array_bytes, _atomic_directory, _canonical_json, _event_from_record,
    _event_record, _verify_file_hashes, _write_array_files,
    event_safe_tangents,
)
from .motion import CanonicalARDYMotion

@dataclass(frozen=True, slots=True)
class GeneratedMotionPolicy:
    id: str
    revision: int
    hyper_base_fingerprint: str
    source_motion_fingerprint: str
    robot_fingerprint: int
    world_fingerprint: int
    frames_per_second: float
    reference_phases: np.ndarray
    reference_root: np.ndarray
    reference_joint_positions: np.ndarray
    reference_joint_velocities: np.ndarray
    reference_actions: np.ndarray
    reference_signature: np.ndarray
    signature_weights: np.ndarray
    contact_modes: np.ndarray
    knot_phases: np.ndarray
    coefficient_knots: np.ndarray
    coefficient_tangents: np.ndarray
    coefficient_uncertainty: np.ndarray
    authority_knots: np.ndarray
    authority_tangents: np.ndarray
    phase_rate_knots: np.ndarray
    phase_rate_tangents: np.ndarray
    events: tuple[MotionEvent, ...]
    action_lower: np.ndarray
    action_upper: np.ndarray
    maximum_action_rate: np.ndarray
    predicted_failure_probability: float
    predicted_out_of_distribution_score: float
    fingerprint: str = ""

    @property
    def frame_count(self) -> int:
        return int(self.reference_phases.size)

    @property
    def action_count(self) -> int:
        return int(self.reference_actions.shape[1])

    @property
    def coefficient_count(self) -> int:
        return int(self.coefficient_knots.shape[1])

    @property
    def duration_seconds(self) -> float:
        return (self.frame_count - 1) / self.frames_per_second

    def validate(
        self,
        *,
        hyper_base: HyperBasePolicy | None = None,
        require_fingerprint: bool = False,
    ) -> None:
        frames = self.frame_count
        actions = self.action_count
        knots = int(self.knot_phases.size)
        coefficients = self.coefficient_count
        if (
            not self.id
            or self.revision <= 0
            or len(self.hyper_base_fingerprint) != 64
            or len(self.source_motion_fingerprint) != 64
            or self.robot_fingerprint <= 0
            or self.world_fingerprint <= 0
            or not math.isfinite(self.frames_per_second)
            or self.frames_per_second <= 0.0
            or frames < 2
            or actions <= 0
            or knots < 2
            or coefficients <= 0
            or self.reference_root.shape != (frames, 7)
            or self.reference_joint_positions.shape != (frames, actions)
            or self.reference_joint_velocities.shape != (frames, actions)
            or self.reference_signature.shape[0] != frames
            or self.signature_weights.shape != (
                self.reference_signature.shape[1],
            )
            or self.contact_modes.shape[0] != frames
            or self.coefficient_tangents.shape != (knots, coefficients)
            or self.coefficient_uncertainty.shape != (knots, coefficients)
            or self.authority_knots.shape != (knots, actions)
            or self.authority_tangents.shape != (knots, actions)
            or self.phase_rate_knots.shape != (knots,)
            or self.phase_rate_tangents.shape != (knots,)
            or self.action_lower.shape != (actions,)
            or self.action_upper.shape != (actions,)
            or self.maximum_action_rate.shape != (actions,)
            or not math.isfinite(self.predicted_failure_probability)
            or not 0.0 <= self.predicted_failure_probability <= 1.0
            or not math.isfinite(self.predicted_out_of_distribution_score)
            or self.predicted_out_of_distribution_score < 0.0
            or (require_fingerprint and len(self.fingerprint) != 64)
        ):
            raise ValueError("generated motion policy dimensions are invalid")
        tables = (
            self.reference_phases,
            self.reference_root,
            self.reference_joint_positions,
            self.reference_joint_velocities,
            self.reference_actions,
            self.reference_signature,
            self.signature_weights,
            self.knot_phases,
            self.coefficient_knots,
            self.coefficient_tangents,
            self.coefficient_uncertainty,
            self.authority_knots,
            self.authority_tangents,
            self.phase_rate_knots,
            self.phase_rate_tangents,
            self.action_lower,
            self.action_upper,
            self.maximum_action_rate,
        )
        if not all(np.isfinite(value).all() for value in tables):
            raise ValueError("generated motion policy contains non-finite values")
        if (
            not np.all(np.diff(self.reference_phases) > 0.0)
            or not np.all(np.diff(self.knot_phases) > 0.0)
            or abs(float(self.reference_phases[0])) > 1.0e-7
            or abs(float(self.reference_phases[-1]) - 1.0) > 1.0e-7
            or abs(float(self.knot_phases[0])) > 1.0e-7
            or abs(float(self.knot_phases[-1]) - 1.0) > 1.0e-7
            or np.any(self.signature_weights <= 0.0)
            or np.any(self.coefficient_uncertainty < 0.0)
            or np.any(self.authority_knots < 0.0)
            or np.any(self.authority_knots > 1.0)
            or np.any(self.phase_rate_knots < 0.0)
            or np.any(self.action_lower >= self.action_upper)
            or np.any(self.maximum_action_rate <= 0.0)
        ):
            raise ValueError("generated motion policy ranges are invalid")
        if not np.issubdtype(self.contact_modes.dtype, np.integer):
            raise ValueError("motion policy contact modes must be integer-valued")
        for event in self.events:
            event.validate(frames, self.contact_modes.shape[1])
        if hyper_base is not None:
            hyper_base.validate(require_fingerprint=True)
            if (
                hyper_base.fingerprint != self.hyper_base_fingerprint
                or hyper_base.world_fingerprint != self.world_fingerprint
                or hyper_base.action_count != actions
                or hyper_base.coefficient_count != coefficients
            ):
                raise ValueError(
                    "generated motion policy and hyper-base contracts differ"
                )
            limits = hyper_base.coefficient_limits[None, :]
            if np.any(np.abs(self.coefficient_knots) > limits + 1.0e-5):
                raise ValueError("generated adapter coefficient exceeds base limit")

    def computed_fingerprint(self) -> str:
        self.validate(require_fingerprint=False)
        digest = hashlib.sha256()
        metadata = {
            "format": MOTION_POLICY_FORMAT,
            "id": self.id,
            "revision": int(self.revision),
            "hyper_base_fingerprint": self.hyper_base_fingerprint,
            "source_motion_fingerprint": self.source_motion_fingerprint,
            "robot_fingerprint": int(self.robot_fingerprint),
            "world_fingerprint": int(self.world_fingerprint),
            "frames_per_second": float(self.frames_per_second),
            "predicted_failure_probability": float(
                self.predicted_failure_probability
            ),
            "predicted_out_of_distribution_score": float(
                self.predicted_out_of_distribution_score
            ),
            "events": [_event_record(event) for event in self.events],
        }
        digest.update(_canonical_json(metadata))
        for value in self._array_mapping().values():
            digest.update(_array_bytes(value))
        return digest.hexdigest()

    def with_fingerprint(self) -> "GeneratedMotionPolicy":
        return GeneratedMotionPolicy(
            **{
                **{
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "fingerprint"
                },
                "fingerprint": self.computed_fingerprint(),
            }
        )

    def _array_mapping(self) -> dict[str, np.ndarray]:
        return {
            "reference_phases": self.reference_phases,
            "reference_root": self.reference_root,
            "reference_joint_positions": self.reference_joint_positions,
            "reference_joint_velocities": self.reference_joint_velocities,
            "reference_actions": self.reference_actions,
            "reference_signature": self.reference_signature,
            "signature_weights": self.signature_weights,
            "contact_modes": self.contact_modes,
            "knot_phases": self.knot_phases,
            "coefficient_knots": self.coefficient_knots,
            "coefficient_tangents": self.coefficient_tangents,
            "coefficient_uncertainty": self.coefficient_uncertainty,
            "authority_knots": self.authority_knots,
            "authority_tangents": self.authority_tangents,
            "phase_rate_knots": self.phase_rate_knots,
            "phase_rate_tangents": self.phase_rate_tangents,
            "action_lower": self.action_lower,
            "action_upper": self.action_upper,
            "maximum_action_rate": self.maximum_action_rate,
        }

    def write(self, directory: str | Path) -> Path:
        policy = self.with_fingerprint()
        target = Path(directory)
        with _atomic_directory(target) as staging:
            file_hashes = _write_array_files(staging, policy._array_mapping())
            manifest = {
                "format": MOTION_POLICY_FORMAT,
                "id": policy.id,
                "revision": policy.revision,
                "hyper_base_fingerprint": policy.hyper_base_fingerprint,
                "source_motion_fingerprint": policy.source_motion_fingerprint,
                "robot_fingerprint": policy.robot_fingerprint,
                "world_fingerprint": policy.world_fingerprint,
                "frames_per_second": policy.frames_per_second,
                "predicted_failure_probability": (
                    policy.predicted_failure_probability
                ),
                "predicted_out_of_distribution_score": (
                    policy.predicted_out_of_distribution_score
                ),
                "fingerprint": policy.fingerprint,
                "events": [_event_record(event) for event in policy.events],
                "files": file_hashes,
            }
            (staging / "manifest.json").write_bytes(
                _canonical_json(manifest) + b"\n"
            )
        return target

    @classmethod
    def read(
        cls,
        directory: str | Path,
        *,
        hyper_base: HyperBasePolicy | None = None,
    ) -> "GeneratedMotionPolicy":
        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest.get("format") != MOTION_POLICY_FORMAT:
            raise ValueError("generated motion policy format is unsupported")
        _verify_file_hashes(source, manifest["files"])
        arrays = {
            name: np.load(source / f"{name}.npy", allow_pickle=False)
            for name in manifest["files"]
        }
        policy = cls(
            id=str(manifest["id"]),
            revision=int(manifest["revision"]),
            hyper_base_fingerprint=str(manifest["hyper_base_fingerprint"]),
            source_motion_fingerprint=str(
                manifest["source_motion_fingerprint"]
            ),
            robot_fingerprint=int(manifest["robot_fingerprint"]),
            world_fingerprint=int(manifest["world_fingerprint"]),
            frames_per_second=float(manifest["frames_per_second"]),
            reference_phases=np.asarray(
                arrays["reference_phases"], dtype=np.float32
            ),
            reference_root=np.asarray(arrays["reference_root"], dtype=np.float32),
            reference_joint_positions=np.asarray(
                arrays["reference_joint_positions"], dtype=np.float32
            ),
            reference_joint_velocities=np.asarray(
                arrays["reference_joint_velocities"], dtype=np.float32
            ),
            reference_actions=np.asarray(
                arrays["reference_actions"], dtype=np.float32
            ),
            reference_signature=np.asarray(
                arrays["reference_signature"], dtype=np.float32
            ),
            signature_weights=np.asarray(
                arrays["signature_weights"], dtype=np.float32
            ),
            contact_modes=np.asarray(arrays["contact_modes"], dtype=np.uint32),
            knot_phases=np.asarray(arrays["knot_phases"], dtype=np.float32),
            coefficient_knots=np.asarray(
                arrays["coefficient_knots"], dtype=np.float32
            ),
            coefficient_tangents=np.asarray(
                arrays["coefficient_tangents"], dtype=np.float32
            ),
            coefficient_uncertainty=np.asarray(
                arrays["coefficient_uncertainty"], dtype=np.float32
            ),
            authority_knots=np.asarray(
                arrays["authority_knots"], dtype=np.float32
            ),
            authority_tangents=np.asarray(
                arrays["authority_tangents"], dtype=np.float32
            ),
            phase_rate_knots=np.asarray(
                arrays["phase_rate_knots"], dtype=np.float32
            ),
            phase_rate_tangents=np.asarray(
                arrays["phase_rate_tangents"], dtype=np.float32
            ),
            events=tuple(_event_from_record(value) for value in manifest["events"]),
            action_lower=np.asarray(arrays["action_lower"], dtype=np.float32),
            action_upper=np.asarray(arrays["action_upper"], dtype=np.float32),
            maximum_action_rate=np.asarray(
                arrays["maximum_action_rate"], dtype=np.float32
            ),
            predicted_failure_probability=float(
                manifest["predicted_failure_probability"]
            ),
            predicted_out_of_distribution_score=float(
                manifest["predicted_out_of_distribution_score"]
            ),
            fingerprint=str(manifest["fingerprint"]),
        )
        policy.validate(hyper_base=hyper_base, require_fingerprint=True)
        if policy.computed_fingerprint() != policy.fingerprint:
            raise ValueError("generated motion policy fingerprint is invalid")
        return policy



from .bundle import MotionPolicyBundle, build_generated_motion_policy
