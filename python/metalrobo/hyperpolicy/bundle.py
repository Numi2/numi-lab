"""Authenticated deployment bundle and generated-policy finalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .base import HyperBasePolicy
from .common import (
    MOTION_BUNDLE_FORMAT,
    _atomic_directory,
    _canonical_json,
    event_safe_tangents,
)
from .generated import GeneratedMotionPolicy
from .motion import CanonicalARDYMotion


@dataclass(frozen=True, slots=True)
class MotionPolicyBundle:
    hyper_base: HyperBasePolicy
    motion_policy: GeneratedMotionPolicy

    def validate(self) -> None:
        self.hyper_base.validate(require_fingerprint=True)
        self.motion_policy.validate(
            hyper_base=self.hyper_base,
            require_fingerprint=True,
        )

    def write(self, directory: str | Path) -> Path:
        base = self.hyper_base.with_fingerprint()
        motion = self.motion_policy.with_fingerprint()
        bundle = MotionPolicyBundle(base, motion)
        bundle.validate()
        target = Path(directory)
        with _atomic_directory(target) as staging:
            base.write(staging / "hyper-base")
            motion.write(staging / "motion-policy")
            manifest = {
                "format": MOTION_BUNDLE_FORMAT,
                "hyper_base_fingerprint": base.fingerprint,
                "motion_policy_fingerprint": motion.fingerprint,
            }
            fingerprint = hashlib.sha256(_canonical_json(manifest)).hexdigest()
            manifest["fingerprint"] = fingerprint
            (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        return target

    @classmethod
    def read(cls, directory: str | Path) -> "MotionPolicyBundle":
        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest.get("format") != MOTION_BUNDLE_FORMAT:
            raise ValueError("motion policy bundle format is unsupported")
        core = {
            "format": manifest["format"],
            "hyper_base_fingerprint": manifest["hyper_base_fingerprint"],
            "motion_policy_fingerprint": manifest["motion_policy_fingerprint"],
        }
        if hashlib.sha256(_canonical_json(core)).hexdigest() != manifest.get(
            "fingerprint"
        ):
            raise ValueError("motion policy bundle fingerprint is invalid")
        base = HyperBasePolicy.read(source / "hyper-base")
        motion = GeneratedMotionPolicy.read(source / "motion-policy", hyper_base=base)
        if (
            base.fingerprint != manifest["hyper_base_fingerprint"]
            or motion.fingerprint != manifest["motion_policy_fingerprint"]
        ):
            raise ValueError("motion policy bundle members do not match manifest")
        return cls(base, motion)


def build_generated_motion_policy(
    *,
    policy_id: str,
    hyper_base: HyperBasePolicy,
    motion: CanonicalARDYMotion,
    coefficient_knots: np.ndarray,
    coefficient_uncertainty: np.ndarray,
    authority_knots: np.ndarray,
    phase_rate_multiplier_knots: np.ndarray,
    robot_fingerprint: int,
    world_fingerprint: int,
    action_lower: np.ndarray,
    action_upper: np.ndarray,
    maximum_action_rate: np.ndarray,
    predicted_failure_probability: float,
    predicted_out_of_distribution_score: float,
) -> GeneratedMotionPolicy:
    """Finalize generated tables into a bounded, runtime-ready policy."""

    hyper_base = hyper_base.with_fingerprint()
    motion.validate()
    coefficients = np.asarray(coefficient_knots, dtype=np.float32)
    uncertainty = np.asarray(coefficient_uncertainty, dtype=np.float32)
    authority = np.asarray(authority_knots, dtype=np.float32)
    phase_multiplier = np.asarray(phase_rate_multiplier_knots, dtype=np.float32)
    knot_count = motion.knot_phases.size
    if (
        coefficients.shape != (knot_count, hyper_base.coefficient_count)
        or uncertainty.shape != coefficients.shape
        or authority.shape != (knot_count, hyper_base.action_count)
        or phase_multiplier.shape != (knot_count,)
    ):
        raise ValueError("generated hypernetwork tables have invalid dimensions")
    coefficients = np.clip(
        coefficients,
        -hyper_base.coefficient_limits[None, :],
        hyper_base.coefficient_limits[None, :],
    )
    uncertainty = np.maximum(uncertainty, 0.0)
    authority = np.clip(authority, 0.0, 1.0)
    nominal_phase_rate = 1.0 / max(motion.duration_seconds, 1.0e-6)
    phase_rate = np.clip(
        phase_multiplier * nominal_phase_rate,
        0.0,
        1.5 * nominal_phase_rate,
    )
    event_phases = [event.phase for event in motion.events]
    coefficient_tangents = event_safe_tangents(
        motion.knot_phases,
        coefficients,
        event_phases=event_phases,
    )
    authority_tangents = event_safe_tangents(
        motion.knot_phases,
        authority,
        event_phases=event_phases,
    )
    phase_rate_tangents = event_safe_tangents(
        motion.knot_phases,
        phase_rate[:, None],
        event_phases=event_phases,
    )[:, 0]

    reference_actions = (
        motion.joint_positions - hyper_base.action_bias[None, :]
    ) / hyper_base.action_scale[None, :]
    if np.any(np.abs(reference_actions) > hyper_base.action_clip + 1.0e-5):
        raise ValueError("ARDY reference lies outside the hyper-base action contract")
    generated = GeneratedMotionPolicy(
        id=policy_id,
        revision=1,
        hyper_base_fingerprint=hyper_base.fingerprint,
        source_motion_fingerprint=motion.source_fingerprint,
        robot_fingerprint=robot_fingerprint,
        world_fingerprint=world_fingerprint,
        frames_per_second=motion.frames_per_second,
        reference_phases=motion.phases,
        reference_root=motion.root_position_quaternion_xyzw,
        reference_joint_positions=motion.joint_positions,
        reference_joint_velocities=motion.joint_velocities,
        reference_actions=reference_actions.astype(np.float32),
        reference_signature=motion.reference_signature,
        signature_weights=motion.signature_weights,
        contact_modes=motion.contact_modes,
        knot_phases=motion.knot_phases,
        coefficient_knots=coefficients,
        coefficient_tangents=coefficient_tangents,
        coefficient_uncertainty=uncertainty,
        authority_knots=authority,
        authority_tangents=authority_tangents,
        phase_rate_knots=phase_rate.astype(np.float32),
        phase_rate_tangents=phase_rate_tangents.astype(np.float32),
        events=motion.events,
        action_lower=np.asarray(action_lower, dtype=np.float32),
        action_upper=np.asarray(action_upper, dtype=np.float32),
        maximum_action_rate=np.asarray(maximum_action_rate, dtype=np.float32),
        predicted_failure_probability=float(
            np.clip(predicted_failure_probability, 0.0, 1.0)
        ),
        predicted_out_of_distribution_score=float(
            max(predicted_out_of_distribution_score, 0.0)
        ),
    ).with_fingerprint()
    generated.validate(hyper_base=hyper_base, require_fingerprint=True)
    return generated
