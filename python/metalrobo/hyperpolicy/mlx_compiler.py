"""Deterministic hypernetwork inference and adapter-policy compilation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import mlx.core as mx
import numpy as np

from .base import HyperBasePolicy
from .generated import GeneratedMotionPolicy, build_generated_motion_policy
from .motion import CanonicalARDYMotion
from .mlx_model import ARDYHyperNetwork

@dataclass(frozen=True, slots=True)
class GeneratedAdapterDistribution:
    coefficient_mean: np.ndarray
    coefficient_uncertainty: np.ndarray
    authority: np.ndarray
    phase_rate_multiplier: np.ndarray
    failure_probability: float
    out_of_distribution_score: float

    def candidate_coefficients(
        self,
        *,
        count: int,
        seed_material: str,
        maximum_standard_deviations: float = 1.5,
    ) -> tuple[np.ndarray, ...]:
        """Return deterministic antithetic low-discrepancy candidates.

        Candidate zero is the exact mean.  Remaining candidates use a scrambled
        Hadamard construction in coefficient space.  It gives balanced bounded
        coverage without maintaining a huge Sobol direction-number table.
        """

        if count <= 0 or maximum_standard_deviations < 0.0:
            raise ValueError("adapter candidate request is invalid")
        result = [self.coefficient_mean.astype(np.float32, copy=True)]
        if count == 1:
            return tuple(result)
        flat_count = int(self.coefficient_mean.size)
        order = 1 << max(0, (flat_count - 1).bit_length())
        digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
        rng = np.random.default_rng(
            int.from_bytes(digest[:8], "little", signed=False)
        )
        permutation = rng.permutation(order)
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=order)
        row = 1
        while len(result) < count:
            direction = _hadamard_row(order, row % order)
            direction = direction[permutation] * signs
            direction = direction[:flat_count].reshape(
                self.coefficient_mean.shape
            )
            radius_index = (row + 1) // 2
            radius = maximum_standard_deviations * min(
                1.0,
                radius_index / max(1.0, math.ceil((count - 1) / 2.0)),
            )
            sign = 1.0 if row % 2 else -1.0
            candidate = self.coefficient_mean + (
                sign * radius * direction * self.coefficient_uncertainty
            )
            result.append(candidate.astype(np.float32))
            row += 1
        return tuple(result)


class ARDYHyperPolicyCompiler:
    """Runs the trained compiler and emits authenticated deployment artifacts."""

    def __init__(
        self,
        hypernetwork: ARDYHyperNetwork,
        hyper_base: HyperBasePolicy,
    ) -> None:
        hyper_base = hyper_base.with_fingerprint()
        hyper_base.validate(require_fingerprint=True)
        configuration = hypernetwork.configuration
        if (
            configuration.action_count != hyper_base.action_count
            or configuration.coefficient_count != hyper_base.coefficient_count
        ):
            raise ValueError("hypernetwork and hyper-base contracts differ")
        self.hypernetwork = hypernetwork
        self.hyper_base = hyper_base

    def generate_distribution(
        self, motion: CanonicalARDYMotion
    ) -> GeneratedAdapterDistribution:
        motion.validate()
        configuration = self.hypernetwork.configuration
        if (
            motion.features.shape[1] != configuration.feature_count
            or motion.knot_event_features.shape[1]
            != configuration.event_feature_count
            or motion.frame_count > configuration.maximum_frames
        ):
            raise ValueError("canonical motion and hypernetwork contracts differ")
        output = self.hypernetwork(
            mx.array(motion.features[None], dtype=mx.float32),
            mx.ones((1, motion.frame_count), dtype=mx.float32),
            mx.array(motion.phases[None], dtype=mx.float32),
            mx.array(motion.knot_phases[None], dtype=mx.float32),
            mx.ones((1, motion.knot_phases.size), dtype=mx.float32),
            mx.array(
                motion.knot_event_features[None], dtype=mx.float32
            ),
        )
        mx.eval(
            output.coefficient_mean,
            output.coefficient_uncertainty,
            output.authority,
            output.phase_rate_multiplier,
            output.failure_probability,
            output.out_of_distribution_score,
        )
        return GeneratedAdapterDistribution(
            coefficient_mean=np.asarray(
                output.coefficient_mean[0], dtype=np.float32
            ),
            coefficient_uncertainty=np.asarray(
                output.coefficient_uncertainty[0], dtype=np.float32
            ),
            authority=np.asarray(output.authority[0], dtype=np.float32),
            phase_rate_multiplier=np.asarray(
                output.phase_rate_multiplier[0], dtype=np.float32
            ),
            failure_probability=float(
                output.failure_probability[0].item()
            ),
            out_of_distribution_score=float(
                output.out_of_distribution_score[0].item()
            ),
        )

    def compile(
        self,
        *,
        policy_id: str,
        motion: CanonicalARDYMotion,
        robot_fingerprint: int,
        world_fingerprint: int,
        action_lower: np.ndarray,
        action_upper: np.ndarray,
        maximum_action_rate: np.ndarray,
        coefficient_override: np.ndarray | None = None,
    ) -> GeneratedMotionPolicy:
        distribution = self.generate_distribution(motion)
        coefficients = (
            distribution.coefficient_mean
            if coefficient_override is None
            else np.asarray(coefficient_override, dtype=np.float32)
        )
        return build_generated_motion_policy(
            policy_id=policy_id,
            hyper_base=self.hyper_base,
            motion=motion,
            coefficient_knots=coefficients,
            coefficient_uncertainty=distribution.coefficient_uncertainty,
            authority_knots=distribution.authority,
            phase_rate_multiplier_knots=(
                distribution.phase_rate_multiplier
            ),
            robot_fingerprint=robot_fingerprint,
            world_fingerprint=world_fingerprint,
            action_lower=action_lower,
            action_upper=action_upper,
            maximum_action_rate=maximum_action_rate,
            predicted_failure_probability=(
                distribution.failure_probability
            ),
            predicted_out_of_distribution_score=(
                distribution.out_of_distribution_score
            ),
        )



def _hadamard_row(order: int, row: int) -> np.ndarray:
    if order <= 0 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    columns = np.arange(order, dtype=np.uint64)
    parity = np.zeros(order, dtype=np.uint8)
    value = columns & np.uint64(row)
    while np.any(value):
        parity ^= (value & np.uint64(1)).astype(np.uint8)
        value >>= np.uint64(1)
    return np.where(parity == 0, 1.0, -1.0).astype(np.float32)
