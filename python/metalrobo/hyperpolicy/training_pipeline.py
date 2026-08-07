"""Solver-rollout to specialist and hypernetwork meta-training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from .base import HyperBasePolicy
from .generated import GeneratedMotionPolicy
from .motion import CanonicalARDYMotion
from .mlx_actor import ARDYMotionConditionedPolicy, PhaseVaryingLowRankActor
from .mlx_production_model import ARDYHyperNetwork, ARDYHyperPolicyConfiguration
from .mlx_specialist import (
    MLXSpecialistAdapterLearner,
    SpecialistAdapterBatch,
    SpecialistAdapterProgram,
)
from .mlx_training import (
    HyperPolicyMetaLearner,
    HyperPolicySupervisionBatch,
)
from .mlx_bridge import export_hyper_base
from ..mlx_policy_learning import NativePolicyRollout

_PHASE_OUTCOME = "hyper_policy_phase"


@dataclass(frozen=True, slots=True)
class SpecialistTrainingResult:
    program: SpecialistAdapterProgram
    metrics: dict[str, float]
    contributing_samples: int


@dataclass(frozen=True, slots=True)
class MetaTrainingItem:
    motion: CanonicalARDYMotion
    generated_policy: GeneratedMotionPolicy
    specialist: SpecialistAdapterProgram
    rollout: NativePolicyRollout
    qualified: bool = True


def _sample_rows(
    phases: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    upper = np.searchsorted(phases, queries, side="right")
    upper = np.clip(upper, 1, phases.size - 1)
    lower = upper - 1
    width = np.maximum(phases[upper] - phases[lower], 1.0e-8)
    alpha = ((queries - phases[lower]) / width).astype(np.float32)
    return (
        (1.0 - alpha[:, None]) * values[lower] + alpha[:, None] * values[upper]
    ).astype(np.float32)


def _phase_values(rollout: NativePolicyRollout) -> np.ndarray:
    values = rollout.outcomes.get(_PHASE_OUTCOME)
    if values is None:
        raise ValueError(
            "PolicyRolloutPack is missing the exact hyper_policy_phase outcome"
        )
    result = np.asarray(values, dtype=np.float32)
    if (
        result.shape != (rollout.sample_count,)
        or not np.isfinite(result).all()
        or np.any(result < 0.0)
        or np.any(result > 1.0)
    ):
        raise ValueError("hyper-policy phase trajectory is invalid")
    return result


def _rank01(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    if source.size <= 1:
        return np.ones(source.shape, dtype=np.float32)
    order = np.argsort(source, kind="stable")
    ranks = np.empty(source.size, dtype=np.float64)
    index = 0
    while index < source.size:
        end = index + 1
        while end < source.size and source[order[end]] == source[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1)
        index = end
    ranks /= source.size - 1
    return ranks.astype(np.float32)


def _teacher_weights(rollout: NativePolicyRollout) -> np.ndarray:
    transitions = rollout.transitions
    valid = (
        (transitions["physics_error"] == 0)
        & ((transitions["done"] == 0) | (transitions["timeout"] != 0))
    ).astype(np.float32)
    reward_quality = _rank01(transitions["reward"])
    tracking = None
    for identifier in (
        "tracking_score",
        "interaction_tracking",
        "mean_tracking_score",
    ):
        if identifier in rollout.outcomes:
            tracking = np.asarray(rollout.outcomes[identifier], dtype=np.float32)
            break
    if tracking is None:
        tracking_quality = reward_quality
    else:
        finite = np.isfinite(tracking)
        if not finite.all():
            raise ValueError("rollout tracking outcome is non-finite")
        minimum = float(np.min(tracking))
        maximum = float(np.max(tracking))
        tracking_quality = (
            (tracking - minimum) / max(maximum - minimum, 1.0e-6)
        ).astype(np.float32)
    weight = valid * np.clip(
        0.10 + 0.55 * tracking_quality + 0.35 * reward_quality,
        0.0,
        1.0,
    )
    return weight.astype(np.float32)


def specialist_batch_from_rollout(
    *,
    rollout: NativePolicyRollout,
    motion_policy: GeneratedMotionPolicy,
    configuration: ARDYHyperPolicyConfiguration,
) -> SpecialistAdapterBatch:
    if rollout.teacher_actions.size == 0:
        raise ValueError("hyper-policy rollout did not publish executed teacherActions")
    phases = _phase_values(rollout)
    reference = _sample_rows(
        motion_policy.reference_phases,
        motion_policy.reference_actions,
        phases,
    )
    teacher = np.asarray(rollout.teacher_actions, dtype=np.float32).reshape(
        rollout.sample_count, rollout.action_count
    )
    if rollout.action_count != configuration.action_count:
        raise ValueError("rollout and hyper-policy action widths differ")
    weights = _teacher_weights(rollout)
    if not np.any(weights > 0.0):
        raise ValueError("rollout contains no physically valid teacher sample")
    return SpecialistAdapterBatch(
        knot_phases=motion_policy.knot_phases[None].astype(np.float32),
        knot_mask=np.ones((1, motion_policy.knot_phases.size), dtype=np.float32),
        actor_observations=np.asarray(
            rollout.actor_observations, dtype=np.float32
        ).reshape(rollout.sample_count, rollout.actor_observation_count),
        sample_motion_indices=np.zeros(rollout.sample_count, dtype=np.int32),
        sample_phases=phases,
        reference_actions=reference,
        teacher_actions=teacher,
        teacher_weights=weights,
    )


def train_specialist_from_rollout(
    *,
    rollout: NativePolicyRollout,
    motion_policy: GeneratedMotionPolicy,
    hyper_base: HyperBasePolicy,
    configuration: ARDYHyperPolicyConfiguration,
    updates: int,
    initial: SpecialistAdapterProgram | None = None,
    learning_rate: float = 2.0e-4,
) -> SpecialistTrainingResult:
    if updates <= 0:
        raise ValueError("specialist update count must be positive")
    batch = specialist_batch_from_rollout(
        rollout=rollout,
        motion_policy=motion_policy,
        configuration=configuration,
    )
    actor = PhaseVaryingLowRankActor(hyper_base)
    learner = MLXSpecialistAdapterLearner(
        actor=actor,
        hyper_base=hyper_base,
        configuration=configuration,
        motion_count=1,
        maximum_knot_count=motion_policy.knot_phases.size,
        initial=initial,
        learning_rate=learning_rate,
    )
    metrics: dict[str, float] = {}
    for _ in range(updates):
        metrics = learner.update(batch)
    return SpecialistTrainingResult(
        program=learner.export(),
        metrics=metrics,
        contributing_samples=int(np.count_nonzero(batch.teacher_weights)),
    )


def generated_program(policy: GeneratedMotionPolicy) -> SpecialistAdapterProgram:
    return SpecialistAdapterProgram(
        coefficients=policy.coefficient_knots[None].astype(np.float32),
        authority=policy.authority_knots[None].astype(np.float32),
        phase_rate_multiplier=(
            policy.phase_rate_knots * max(policy.duration_seconds, 1.0e-6)
        )[None].astype(np.float32),
    )


def write_specialist_program(
    path: str | Path,
    *,
    program: SpecialistAdapterProgram,
    motion_fingerprint: str,
    hyper_base_fingerprint: str,
    metrics: dict[str, float] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "format": "numi.ardy-specialist-program.v1",
            "motion_fingerprint": motion_fingerprint,
            "hyper_base_fingerprint": hyper_base_fingerprint,
            "metrics": metrics or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    np.savez_compressed(
        target,
        coefficients=np.asarray(program.coefficients, dtype=np.float32),
        authority=np.asarray(program.authority, dtype=np.float32),
        phase_rate_multiplier=np.asarray(
            program.phase_rate_multiplier, dtype=np.float32
        ),
        metadata=np.asarray(metadata),
    )
    return target


def read_specialist_program(
    path: str | Path,
) -> tuple[SpecialistAdapterProgram, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format") != "numi.ardy-specialist-program.v1":
            raise ValueError("specialist program format is unsupported")
        program = SpecialistAdapterProgram(
            coefficients=np.asarray(archive["coefficients"], dtype=np.float32),
            authority=np.asarray(archive["authority"], dtype=np.float32),
            phase_rate_multiplier=np.asarray(
                archive["phase_rate_multiplier"], dtype=np.float32
            ),
        )
    return program, metadata


def _pad_motion_batch(
    items: Sequence[MetaTrainingItem],
    configuration: ARDYHyperPolicyConfiguration,
) -> HyperPolicySupervisionBatch:
    if not items:
        raise ValueError("meta-training requires at least one motion")
    batch = len(items)
    maximum_frames = max(item.motion.frame_count for item in items)
    maximum_knots = max(item.motion.knot_phases.size for item in items)
    if maximum_frames > configuration.maximum_frames:
        raise ValueError("meta-training motion exceeds encoder capacity")
    motion_features = np.zeros(
        (batch, maximum_frames, configuration.feature_count),
        dtype=np.float32,
    )
    motion_valid = np.zeros((batch, maximum_frames), dtype=np.float32)
    frame_phases = np.zeros((batch, maximum_frames), dtype=np.float32)
    knot_phases = np.zeros((batch, maximum_knots), dtype=np.float32)
    knot_mask = np.zeros((batch, maximum_knots), dtype=np.float32)
    event_features = np.zeros(
        (batch, maximum_knots, configuration.event_feature_count),
        dtype=np.float32,
    )
    coefficients = np.zeros(
        (batch, maximum_knots, configuration.coefficient_count),
        dtype=np.float32,
    )
    authority = np.zeros(
        (batch, maximum_knots, configuration.action_count),
        dtype=np.float32,
    )
    phase_rate = np.zeros((batch, maximum_knots), dtype=np.float32)
    observations: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    sample_phases: list[np.ndarray] = []
    references: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    failure_targets = np.zeros(batch, dtype=np.float32)

    for index, item in enumerate(items):
        motion = item.motion
        policy = item.generated_policy
        motion.validate()
        policy.validate()
        if (
            motion.features.shape[1] != configuration.feature_count
            or motion.knot_event_features.shape[1] != configuration.event_feature_count
            or policy.action_count != configuration.action_count
            or item.specialist.coefficients.shape
            != (1, motion.knot_phases.size, configuration.coefficient_count)
            or item.specialist.authority.shape
            != (1, motion.knot_phases.size, configuration.action_count)
            or item.specialist.phase_rate_multiplier.shape
            != (1, motion.knot_phases.size)
        ):
            raise ValueError("meta-training item contracts differ")
        frames = motion.frame_count
        knots = motion.knot_phases.size
        motion_features[index, :frames] = motion.features
        motion_valid[index, :frames] = 1.0
        frame_phases[index, :frames] = motion.phases
        knot_phases[index, :knots] = motion.knot_phases
        knot_mask[index, :knots] = 1.0
        event_features[index, :knots] = motion.knot_event_features
        coefficients[index, :knots] = item.specialist.coefficients[0]
        authority[index, :knots] = item.specialist.authority[0]
        phase_rate[index, :knots] = item.specialist.phase_rate_multiplier[0]
        specialist_batch = specialist_batch_from_rollout(
            rollout=item.rollout,
            motion_policy=policy,
            configuration=configuration,
        )
        count = specialist_batch.actor_observations.shape[0]
        observations.append(specialist_batch.actor_observations)
        sample_indices.append(np.full(count, index, dtype=np.int32))
        sample_phases.append(specialist_batch.sample_phases)
        references.append(specialist_batch.reference_actions)
        teachers.append(specialist_batch.teacher_actions)
        weights.append(specialist_batch.teacher_weights)
        failure_targets[index] = 0.0 if item.qualified else 1.0

    return HyperPolicySupervisionBatch(
        motion_features=motion_features,
        motion_valid_mask=motion_valid,
        frame_phases=frame_phases,
        knot_phases=knot_phases,
        knot_mask=knot_mask,
        knot_event_features=event_features,
        specialist_coefficients=coefficients,
        specialist_authority=authority,
        specialist_phase_rate_multiplier=phase_rate,
        actor_observations=np.concatenate(observations, axis=0),
        sample_motion_indices=np.concatenate(sample_indices, axis=0),
        sample_phases=np.concatenate(sample_phases, axis=0),
        reference_actions=np.concatenate(references, axis=0),
        teacher_actions=np.concatenate(teachers, axis=0),
        sample_weights=np.concatenate(weights, axis=0),
        failure_targets=failure_targets,
    )


def meta_update(
    *,
    hypernetwork: ARDYHyperNetwork,
    hyper_base: HyperBasePolicy,
    configuration: ARDYHyperPolicyConfiguration,
    items: Sequence[MetaTrainingItem],
    updates: int,
) -> tuple[ARDYHyperNetwork, HyperBasePolicy, dict[str, float]]:
    if updates <= 0:
        raise ValueError("meta-update count must be positive")
    actor = PhaseVaryingLowRankActor(hyper_base)
    model = ARDYMotionConditionedPolicy(hypernetwork, actor)
    learner = HyperPolicyMetaLearner(model, configuration)
    batch = _pad_motion_batch(items, configuration)
    metrics: dict[str, float] = {}
    for _ in range(updates):
        metrics = learner.update(batch)
    mx.eval(model.parameters())
    return (
        model.hypernetwork,
        export_hyper_base(model.actor, hyper_base),
        metrics,
    )
