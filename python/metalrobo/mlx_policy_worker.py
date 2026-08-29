"""Persistent MLX learner worker controlled by the native Swift scheduler.

The line-delimited JSON protocol is intentionally narrow: Swift publishes one
validated PolicyRolloutPack, requests one PPO update, and installs the returned
PolicyPack revision. The worker never creates, steps, resets, or schedules a
simulator.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .mlx_policy_learning import (
    MLXMotionPrior,
    MLXMotionPriorConfiguration,
    MLXPPOConfiguration,
    MLXPolicyBatch,
    MLXPolicyLearner,
    read_motion_pack,
    read_policy_rollout_pack,
)


_PPO_RESUMABLE_SCHEDULE_FIELDS = frozenset(
    {"minibatch_size", "seed"}
)
_PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS = frozenset(
    {
        "learning_rate",
        "adaptive_learning_rate",
        "minimum_learning_rate",
        "maximum_learning_rate",
    }
)
_PPO_EXPLICIT_EXPLORATION_OVERRIDE_FIELDS = frozenset(
    {"initial_log_standard_deviation"}
)
_MOTION_RESUMABLE_SCHEDULE_FIELDS = frozenset(
    {"minibatch_size", "seed"}
)


def _emit(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(
        json.dumps(
            value,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


def _configuration(
    arguments: argparse.Namespace,
) -> MLXPPOConfiguration:
    target_kl = arguments.target_kl
    if target_kl is not None and target_kl == 0.0:
        target_kl = None
    return MLXPPOConfiguration(
        update_epochs=arguments.update_epochs,
        minibatch_size=arguments.minibatch_size,
        hidden_sizes=tuple(arguments.hidden_sizes),
        learning_rate=arguments.learning_rate,
        clip_ratio=arguments.clip_ratio,
        value_coefficient=arguments.value_coefficient,
        imagination_distillation_coefficient=(
            arguments.imagination_distillation_coefficient
        ),
        entropy_coefficient=arguments.entropy_coefficient,
        maximum_gradient_norm=arguments.maximum_gradient_norm,
        target_kl=target_kl,
        adaptive_learning_rate=not arguments.fixed_learning_rate,
        minimum_learning_rate=arguments.minimum_learning_rate,
        maximum_learning_rate=arguments.maximum_learning_rate,
        discount=arguments.discount,
        gae_lambda=arguments.gae_lambda,
        initial_log_standard_deviation=(
            arguments.initial_log_standard_deviation
        ),
        observation_clip=arguments.observation_clip,
        seed=arguments.seed,
    )


def _configuration_record(learner: MLXPolicyLearner) -> str:
    return json.dumps(
        asdict(learner.configuration),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _difficulty_balanced_retention_weights(
    retention_weights: np.ndarray,
    difficulty_bands: np.ndarray,
    *,
    priority_band: int | None = None,
    priority_factor: float = 1.0,
) -> np.ndarray:
    """Give every represented protected band equal total teacher authority."""

    weights = np.asarray(retention_weights, dtype=np.float32).reshape(-1)
    bands = np.asarray(difficulty_bands).reshape(-1)
    if bands.shape != weights.shape:
        raise ValueError(
            "retention difficulty bands disagree with the policy batch"
        )
    protected_samples = weights > 0.0
    active_bands, active_counts = np.unique(
        bands[protected_samples], return_counts=True
    )
    if active_bands.size == 0:
        raise ValueError(
            "difficulty-balanced retention selects no protected bands"
        )
    if not np.isfinite(priority_factor) or priority_factor < 1.0:
        raise ValueError("retention priority factor must be finite and at least one")
    priority_represented = (
        priority_band is not None and priority_band in active_bands
    )
    # The teacher loss is normalized by the sum of these weights. Scaling each
    # represented band down to the rarest band's total contribution therefore
    # makes every protected rung equally authoritative without allowing a
    # per-sample weight above one.
    samples_per_band = int(np.min(active_counts))
    balanced = np.zeros_like(weights)
    # A rollout is allowed to omit a low-probability protected rung. In that
    # case there is nothing to amplify for this update, so retain balanced
    # authority across the represented rungs. The held-out selector still
    # evaluates every protected band and remains the fail-closed boundary.
    maximum_factor = priority_factor if priority_represented else 1.0
    for band, count in zip(active_bands, active_counts, strict=True):
        band_factor = (
            priority_factor if priority_represented and band == priority_band
            else 1.0
        )
        balanced[np.logical_and(protected_samples, bands == band)] = (
            np.float32(samples_per_band * band_factor)
            / np.float32(count * maximum_factor)
        )
    return balanced


def _retention_policy_batch(
    batch: MLXPolicyBatch,
    reference: MLXPolicyLearner,
    *,
    chunk_size: int,
    teacher_weights: np.ndarray | None = None,
    protected_actor_only: bool = False,
    difficulty_bands: np.ndarray | None = None,
    balance_difficulty_bands: bool = False,
    priority_difficulty_band: int | None = None,
    priority_factor: float = 1.0,
    rollout_teacher_blend: float = 0.0,
) -> MLXPolicyBatch:
    """Attach frozen actor targets and optionally reserve protected samples."""

    if chunk_size <= 0:
        raise ValueError("retention target chunk size must be positive")
    rollout_teacher_samples = batch.teacher_weights > 0.0
    if np.any(rollout_teacher_samples) and rollout_teacher_blend <= 0.0:
        raise ValueError(
            "retention plus rollout teacher requires a positive teacher blend"
        )
    if (
        reference.actor_observation_count
        != int(batch.actor_observations.shape[1])
        or reference.action_count != int(batch.latents.shape[1])
    ):
        raise ValueError(
            "retention policy disagrees with the actor observation or action contract"
        )
    if teacher_weights is None:
        retention_weights = np.ones(
            int(batch.actor_observations.shape[0]),
            dtype=np.float32,
        )
    else:
        retention_weights = np.asarray(
            teacher_weights, dtype=np.float32
        ).reshape(-1)
        if retention_weights.shape != (
            int(batch.actor_observations.shape[0]),
        ) or not np.all(np.isfinite(retention_weights)) or np.any(
            retention_weights < 0.0
        ) or np.any(retention_weights > 1.0):
            raise ValueError("retention teacher weights are invalid")
        if not np.any(retention_weights > 0.0):
            raise ValueError("retention teacher weights select no samples")
    protected_samples = retention_weights > 0.0
    if balance_difficulty_bands:
        if teacher_weights is None or difficulty_bands is None:
            raise ValueError(
                "difficulty-balanced retention requires selective weights and bands"
            )
        retention_weights = _difficulty_balanced_retention_weights(
            retention_weights,
            difficulty_bands,
            priority_band=priority_difficulty_band,
            priority_factor=priority_factor,
        )
    policy_weights = batch.policy_weights
    if protected_actor_only:
        policy_weights = batch.policy_weights * (
            1.0 - protected_samples.astype(np.float32)
        )
        if not np.any(policy_weights > 0.0):
            raise ValueError(
                "protected actor-only retention selects no PPO samples"
            )
    targets: list[np.ndarray] = []
    for offset in range(0, int(batch.actor_observations.shape[0]), chunk_size):
        observations = mx.array(
            batch.actor_observations[offset : offset + chunk_size],
            dtype=mx.float32,
        )
        means = reference.model.actor_mean(observations)
        mx.eval(means)
        targets.append(np.asarray(means, dtype=np.float32))
    reference_targets = np.concatenate(targets, axis=0)
    if np.any(rollout_teacher_samples):
        reference_targets = _blend_retention_and_rollout_teacher_targets(
            reference_targets,
            batch.teacher_actions,
            batch.teacher_weights,
            rollout_teacher_blend,
        )
    return MLXPolicyBatch.from_numpy(
        actor_observations=batch.actor_observations,
        critic_observations=batch.critic_observations,
        latents=batch.latents,
        old_log_probabilities=batch.old_log_probabilities,
        old_values=batch.old_values,
        advantages=batch.advantages,
        returns=batch.returns,
        teacher_actions=reference_targets,
        teacher_weights=retention_weights,
        policy_weights=policy_weights,
    )


def _blend_retention_and_rollout_teacher_targets(
    reference_targets: np.ndarray,
    rollout_teacher_targets: np.ndarray,
    rollout_teacher_weights: np.ndarray,
    blend: float,
) -> np.ndarray:
    """Blend route corrections into a frozen actor without losing provenance."""
    reference = np.asarray(reference_targets, dtype=np.float32)
    teacher = np.asarray(rollout_teacher_targets, dtype=np.float32)
    weights = np.asarray(rollout_teacher_weights, dtype=np.float32).reshape(-1, 1)
    if reference.shape != teacher.shape or weights.shape[0] != reference.shape[0]:
        raise ValueError("retention and rollout-teacher target shapes disagree")
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("retention rollout-teacher blend must be in [0, 1]")
    return reference + weights * np.float32(blend) * (teacher - reference)


def _motion_configuration_record(
    motion_prior: MLXMotionPrior,
) -> str:
    return json.dumps(
        asdict(motion_prior.configuration),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _configuration_contract(
    record: str,
    resumable_schedule_fields: frozenset[str],
) -> dict[str, Any]:
    """Return checkpoint semantics independent of execution geometry."""

    try:
        configuration = json.loads(record)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "MLX learner state configuration metadata is invalid"
        ) from error
    if not isinstance(configuration, dict):
        raise ValueError(
            "MLX learner state configuration metadata is invalid"
        )
    return {
        key: value
        for key, value in configuration.items()
        if key not in resumable_schedule_fields
    }


def _configuration_matches(
    saved: Any,
    current: str,
    resumable_schedule_fields: frozenset[str],
) -> bool:
    if not isinstance(saved, str):
        return False
    return _configuration_contract(
        saved,
        resumable_schedule_fields,
    ) == _configuration_contract(
        current,
        resumable_schedule_fields,
    )


def _require_finite_motion_state(
    motion_prior: MLXMotionPrior,
) -> None:
    arrays = [
        value
        for _, value in tree_flatten(
            motion_prior.model.trainable_parameters()
        )
    ] + [
        value
        for _, value in tree_flatten(
            motion_prior.optimizer.state
        )
    ]
    finite = [mx.all(mx.isfinite(value)) for value in arrays]
    mx.eval(*finite)
    if not all(bool(value.item()) for value in finite):
        raise ValueError(
            "motion-prior training state contains non-finite values"
        )


def _write_learner_state(
    learner: MLXPolicyLearner,
    path: Path,
    motion_prior: MLXMotionPrior | None = None,
) -> Path:
    """Atomically checkpoint model parameters and Adam moments."""

    if not learner.policy_id:
        raise ValueError("learner state requires a policy identity")
    learner._require_finite_training_state()
    target = path.expanduser().resolve()
    if target.suffix != ".safetensors":
        raise ValueError("learner state path must end in .safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"model.{name}": value
        for name, value in tree_flatten(
            learner.model.trainable_parameters()
        )
    }
    arrays.update(
        {
            f"optimizer.{name}": value
            for name, value in tree_flatten(
                learner.optimizer.state
            )
        }
    )
    if motion_prior is not None:
        _require_finite_motion_state(motion_prior)
        arrays.update(
            {
                f"motion_model.{name}": value
                for name, value in tree_flatten(
                    motion_prior.model.trainable_parameters()
                )
            }
        )
        arrays.update(
            {
                f"motion_optimizer.{name}": value
                for name, value in tree_flatten(
                    motion_prior.optimizer.state
                )
            }
        )
    mx.eval(*arrays.values())
    metadata = {
        "format": "metalrobo.mlx-learner-state",
        "schema": "3" if motion_prior is not None else "2",
        "policy_id": learner.policy_id,
        "policy_revision": str(learner.revision),
        "actor_observation_count": str(
            learner.actor_observation_count
        ),
        "critic_observation_count": str(
            learner.critic_observation_count
        ),
        "action_count": str(learner.action_count),
        "configuration": _configuration_record(learner),
    }
    if motion_prior is not None:
        metadata.update(
            {
                "motion_pack_hash": str(
                    motion_prior.motion_pack.content_hash
                ),
                "motion_feature_count": str(
                    motion_prior.motion_pack.feature_count
                ),
                "motion_configuration": (
                    _motion_configuration_record(motion_prior)
                ),
            }
        )
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.tmp.safetensors"
    )
    try:
        mx.save_safetensors(temporary, arrays, metadata=metadata)
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _restore_learner_state(
    learner: MLXPolicyLearner,
    path: Path,
    motion_prior: MLXMotionPrior | None = None,
    ppo_resumable_schedule_fields: frozenset[str] = (
        _PPO_RESUMABLE_SCHEDULE_FIELDS
    ),
    actor_observation_extension_offset: int | None = None,
) -> bool:
    target = path.expanduser().resolve()
    if target.suffix != ".safetensors":
        raise ValueError("learner state path must end in .safetensors")
    if not target.is_file():
        return False
    loaded = mx.load(target, return_metadata=True)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("MLX learner state has no metadata")
    arrays, metadata = loaded
    if not isinstance(arrays, dict) or not isinstance(metadata, dict):
        raise ValueError("MLX learner state payload is invalid")
    expected_metadata = {
        "format": "metalrobo.mlx-learner-state",
        "schema": "3" if motion_prior is not None else "2",
        "policy_id": learner.policy_id,
        "critic_observation_count": str(
            learner.critic_observation_count
        ),
        "action_count": str(learner.action_count),
    }
    if motion_prior is not None:
        expected_metadata.update(
            {
                "motion_pack_hash": str(
                    motion_prior.motion_pack.content_hash
                ),
                "motion_feature_count": str(
                    motion_prior.motion_pack.feature_count
                ),
            }
        )
    try:
        saved_actor_observation_count = int(
            metadata["actor_observation_count"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "MLX learner actor observation contract is invalid"
        ) from error
    actor_extension_count = (
        learner.actor_observation_count -
        saved_actor_observation_count
    )
    migrate_actor_input = (
        actor_observation_extension_offset is not None and
        actor_extension_count > 0 and
        0 <= actor_observation_extension_offset <=
            saved_actor_observation_count
    )
    if (
        saved_actor_observation_count !=
            learner.actor_observation_count and
        not migrate_actor_input
    ):
        raise ValueError(
            "MLX learner actor observation contract differs from the PolicyPack"
        )
    metadata_mismatches = [
        key
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    ]
    configuration_matches = _configuration_matches(
        metadata.get("configuration"),
        _configuration_record(learner),
        ppo_resumable_schedule_fields,
    )
    motion_configuration_matches = not (
        motion_prior is not None
    ) or _configuration_matches(
            metadata.get("motion_configuration"),
            _motion_configuration_record(motion_prior),
            _MOTION_RESUMABLE_SCHEDULE_FIELDS,
        )
    ppo_mismatches: list[str] = []
    if not configuration_matches:
        saved_contract = _configuration_contract(
            metadata.get("configuration"),
            ppo_resumable_schedule_fields,
        )
        current_contract = _configuration_contract(
            _configuration_record(learner),
            ppo_resumable_schedule_fields,
        )
        ppo_mismatches = sorted(
            key
            for key in set(saved_contract) | set(current_contract)
            if saved_contract.get(key) != current_contract.get(key)
        )
    if (
        metadata_mismatches or
        not configuration_matches or
        not motion_configuration_matches
    ):
        raise ValueError(
            "MLX learner state contract differs from the PolicyPack or PPO "
            "configuration: metadata=" +
            ",".join(metadata_mismatches) +
            f", ppo={configuration_matches}" +
            (f"[{','.join(ppo_mismatches)}]" if ppo_mismatches else "") +
            ", "
            f"motion={motion_configuration_matches}"
        )
    try:
        revision = int(metadata["policy_revision"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MLX learner state revision is invalid") from error
    if revision < learner.revision:
        raise ValueError(
            "MLX learner state is older than the supplied PolicyPack"
        )

    expected_model = dict(
        tree_flatten(learner.model.trainable_parameters())
    )
    model_arrays = {
        name.removeprefix("model."): value
        for name, value in arrays.items()
        if name.startswith("model.")
    }
    if set(model_arrays) != set(expected_model):
        raise ValueError(
            "MLX learner state model topology is incompatible"
        )

    def extend_actor_input(
        name: str,
        actual: mx.array,
        expected: mx.array,
    ) -> mx.array:
        if (
            migrate_actor_input and
            "actor.layers.0.weight" in name and
            len(actual.shape) == 2 and
            tuple(actual.shape) == (
                expected.shape[0],
                expected.shape[1] - actor_extension_count,
            )
        ):
            offset = int(actor_observation_extension_offset)
            return mx.concatenate(
                (
                    actual[:, :offset],
                    mx.zeros(
                        (actual.shape[0], actor_extension_count),
                        dtype=actual.dtype,
                    ),
                    actual[:, offset:],
                ),
                axis=1,
            )
        return actual

    for name, expected in expected_model.items():
        actual = extend_actor_input(
            name,
            model_arrays[name],
            expected,
        )
        if (
            tuple(actual.shape) != tuple(expected.shape)
            or actual.dtype != expected.dtype
        ):
            raise ValueError(
                f"MLX learner state model tensor {name} is incompatible"
            )
        model_arrays[name] = actual
    learner.model.update(
        tree_unflatten(sorted(model_arrays.items())),
        strict=True,
    )

    expected_optimizer = dict(
        tree_flatten(learner.optimizer.state)
    )
    optimizer_arrays = {
        name.removeprefix("optimizer."): value
        for name, value in arrays.items()
        if name.startswith("optimizer.")
    }
    if set(optimizer_arrays) != set(expected_optimizer):
        raise ValueError(
            "MLX learner state optimizer topology is incompatible"
        )
    for name, expected in expected_optimizer.items():
        actual = extend_actor_input(
            name,
            optimizer_arrays[name],
            expected,
        )
        if (
            tuple(actual.shape) != tuple(expected.shape)
            or actual.dtype != expected.dtype
        ):
            raise ValueError(
                f"MLX learner state optimizer tensor {name} is incompatible"
            )
        optimizer_arrays[name] = actual
    learner.optimizer.state = tree_unflatten(
        sorted(optimizer_arrays.items())
    )
    if motion_prior is not None:
        expected_motion_model = dict(
            tree_flatten(
                motion_prior.model.trainable_parameters()
            )
        )
        motion_model_arrays = {
            name.removeprefix("motion_model."): value
            for name, value in arrays.items()
            if name.startswith("motion_model.")
        }
        if set(motion_model_arrays) != set(expected_motion_model):
            raise ValueError(
                "MLX learner state motion model topology is incompatible"
            )
        for name, expected in expected_motion_model.items():
            actual = motion_model_arrays[name]
            if (
                tuple(actual.shape) != tuple(expected.shape)
                or actual.dtype != expected.dtype
            ):
                raise ValueError(
                    f"MLX learner state motion tensor {name} is incompatible"
                )
        motion_prior.model.update(
            tree_unflatten(sorted(motion_model_arrays.items())),
            strict=True,
        )
        expected_motion_optimizer = dict(
            tree_flatten(motion_prior.optimizer.state)
        )
        motion_optimizer_arrays = {
            name.removeprefix("motion_optimizer."): value
            for name, value in arrays.items()
            if name.startswith("motion_optimizer.")
        }
        if set(motion_optimizer_arrays) != set(
            expected_motion_optimizer
        ):
            raise ValueError(
                "MLX learner state motion optimizer topology is incompatible"
            )
        for name, expected in expected_motion_optimizer.items():
            actual = motion_optimizer_arrays[name]
            if (
                tuple(actual.shape) != tuple(expected.shape)
                or actual.dtype != expected.dtype
            ):
                raise ValueError(
                    "MLX learner state motion optimizer tensor "
                    f"{name} is incompatible"
                )
        motion_prior.optimizer.state = tree_unflatten(
            sorted(motion_optimizer_arrays.items())
        )
    learner.revision = revision
    learner.refresh_compiled_training_state()
    mx.eval(
        learner.model.parameters(),
        learner.optimizer.state,
        *(
            (
                motion_prior.model.parameters(),
                motion_prior.optimizer.state,
            )
            if motion_prior is not None
            else ()
        ),
    )
    learner._require_finite_training_state()
    if motion_prior is not None:
        _require_finite_motion_state(motion_prior)
    return True


def _validate_rollout(
    learner: MLXPolicyLearner,
    rollout_path: Path,
    expected_task_fingerprint: int | None,
    native_library: Path,
) -> tuple[Any, int]:
    rollout = read_policy_rollout_pack(
        rollout_path,
        library_path=native_library,
    )
    if (
        rollout.actor_observation_count
        != learner.actor_observation_count
        or rollout.critic_observation_count
        != learner.critic_observation_count
        or rollout.action_count != learner.action_count
    ):
        raise ValueError(
            "PolicyRolloutPack dimensions disagree with the learner"
        )
    if rollout.policy_revision != learner.revision:
        raise ValueError(
            "PolicyRolloutPack behavior revision disagrees with the learner"
        )
    if (
        expected_task_fingerprint is not None
        and rollout.task_fingerprint != expected_task_fingerprint
    ):
        raise ValueError(
            "PolicyRolloutPack task fingerprint changed during training"
        )
    if np.any(rollout.transitions["physics_error"]):
        raise ValueError(
            "PolicyRolloutPack contains a physics failure"
        )
    return rollout, rollout.task_fingerprint


def _rollout_metrics(rollout: Any) -> dict[str, Any]:
    transitions = rollout.transitions
    rewards = transitions["reward"]
    root_height = rollout.outcome("root_height")
    tilt = rollout.outcome("tilt")
    recovery_phase_rates_by_difficulty_band: dict[str, Any] = {}
    for raw_band in np.unique(transitions["difficulty_band"]):
        band = int(raw_band)
        selected = transitions["difficulty_band"] == raw_band
        flags = transitions["impact_event_flags"][selected].astype(
            np.uint32
        )
        count = int(np.sum(selected))
        recovery_phase_rates_by_difficulty_band[str(band)] = {
            "transition_count": count,
            "mean_root_height": float(
                np.mean(root_height[selected])
            ),
            "mean_tilt": float(
                np.mean(tilt[selected])
            ),
            "brace": float(np.mean((flags & (1 << 29)) != 0)),
            "trunk_clear": float(
                np.mean((flags & (1 << 28)) != 0)
            ),
            "foot_support": float(
                np.mean((flags & (1 << 27)) != 0)
            ),
            "support_transfer": float(
                np.mean((flags & (1 << 26)) != 0)
            ),
            "rise": float(np.mean((flags & (1 << 25)) != 0)),
            "standing": float(np.mean((flags & (1 << 31)) != 0)),
            "quiet_stand": float(
                np.mean((flags & (1 << 24)) != 0)
            ),
            "restored": float(np.mean((flags & (1 << 30)) != 0)),
        }
    reasons, counts = np.unique(
        transitions["termination_reason"][
            transitions["done"].astype(bool)
        ],
        return_counts=True,
    )
    impact_metrics: list[dict[str, Any]] = []
    sequence_indices = np.unique(
        transitions["impact_sequence_index"]
    )
    for raw_sequence_index in sequence_indices:
        sequence_index = int(raw_sequence_index)
        if sequence_index == 0:
            continue
        selected = (
            transitions["impact_sequence_index"] == sequence_index
        )
        flags = transitions["impact_event_flags"][selected]
        impact_metrics.append(
            {
                "sequence_index": sequence_index,
                "active_steps": int(np.sum(selected)),
                "touch_count": int(np.sum((flags & 1) != 0)),
                "recovery_count": int(np.sum((flags & 2) != 0)),
                "miss_count": int(np.sum((flags & 4) != 0)),
                "peak_tilt": float(
                    np.max(tilt[selected])
                ),
                "minimum_root_height": float(
                    np.min(root_height[selected])
                ),
            }
        )
    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "reward_standard_deviation": float(np.std(rewards)),
        "minimum_reward": float(np.min(rewards)),
        "maximum_reward": float(np.max(rewards)),
        "mean_tracking_score": float(np.mean(rollout.outcome("tracking"))),
        "mean_root_height": float(np.mean(root_height)),
        "mean_tilt": float(np.mean(tilt)),
        "recovery_phase_rates_by_difficulty_band": (
            recovery_phase_rates_by_difficulty_band
        ),
        "done_count": int(np.sum(transitions["done"])),
        "timeout_count": int(np.sum(transitions["timeout"])),
        "termination_reason_counts": {
            str(int(reason)): int(count)
            for reason, count in zip(reasons, counts, strict=True)
        },
        "impact_sequence_metrics": impact_metrics,
    }
    metrics["outcomes"] = {
        identifier: {
            "mean": float(np.mean(values)),
            "unit": rollout.outcome_units[identifier],
            "direction": rollout.outcome_directions[identifier],
        }
        for identifier, values in rollout.outcomes.items()
    }
    compatibility = {
        "mean_task_reward": "task_reward",
        "mean_contact_reward": "contact_reward",
        "mean_dodge_link_clearance_reward":
            "projectile_clearance_reward",
        "mean_dodge_evasion_reward": "projectile_evasion_reward",
        "mean_dodge_miss_reward": "projectile_miss_reward",
        "mean_dodge_safe_stillness_reward":
            "projectile_safe_stillness_reward",
        "mean_dodge_safe_action_rate_reward":
            "projectile_safe_action_reward",
        "mean_dodge_cbf_correction_reward": "cbf_correction_reward",
        "mean_dodge_cbf_buffer_reward": "cbf_buffer_reward",
        "mean_dodge_predicted_clearance_reward":
            "projectile_predicted_clearance_reward",
    }
    metrics.update({
        key: float(np.mean(rollout.outcome(outcome)))
        for key, outcome in compatibility.items()
    })
    return metrics


def _serve(arguments: argparse.Namespace) -> int:
    learner = MLXPolicyLearner.from_policy_pack(
        arguments.policy_pack,
        _configuration(arguments),
        library_path=arguments.native_library,
    )
    _bind_contract(learner, arguments)
    retention_reference = None
    if (
        not math.isfinite(arguments.retention_rollout_teacher_blend)
        or not 0.0 <= arguments.retention_rollout_teacher_blend <= 1.0
    ):
        raise ValueError("retention rollout-teacher blend must be in [0, 1]")
    if arguments.retention_policy_pack is not None:
        if arguments.imagination_distillation_coefficient <= 0.0:
            raise ValueError(
                "retention policy requires a positive distillation coefficient"
            )
        retention_reference = MLXPolicyLearner.from_actor_policy_pack(
            arguments.retention_policy_pack,
            learner.critic_observation_count,
            learner.configuration,
            preserve_critic=False,
            actor_observation_count=learner.actor_observation_count,
            actor_observation_extension_offset=(
                arguments.actor_observation_extension_offset
            ),
            library_path=arguments.native_library,
        )
        if retention_reference.action_count != learner.action_count:
            raise ValueError(
                "retention policy disagrees with the learner action contract"
            )
    motion_prior = None
    if arguments.motion_pack is not None:
        motion_prior = MLXMotionPrior(
            read_motion_pack(
                arguments.motion_pack,
                library_path=arguments.native_library,
            ),
            MLXMotionPriorConfiguration(
                hidden_sizes=tuple(arguments.motion_hidden_sizes),
                learning_rate=arguments.motion_learning_rate,
                minibatch_size=arguments.motion_minibatch_size,
                update_epochs=arguments.motion_update_epochs,
                reward_coefficient=arguments.motion_reward_coefficient,
                activation_mode=arguments.motion_activation,
                maximum_gradient_norm=arguments.maximum_gradient_norm,
                seed=arguments.seed,
            ),
        )
    restored = _restore_learner_state(
        learner,
        arguments.restore_learner_state,
        motion_prior,
        _PPO_RESUMABLE_SCHEDULE_FIELDS |
        (_PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS
         if arguments.override_resumed_learning_rate else frozenset()) |
        (_PPO_EXPLICIT_EXPLORATION_OVERRIDE_FIELDS
         if arguments.override_resumed_exploration else frozenset()),
        arguments.actor_observation_extension_offset,
    )
    if restored and arguments.override_resumed_learning_rate:
        learner.optimizer.learning_rate = arguments.learning_rate
        learner.refresh_compiled_training_state()
        mx.eval(learner.optimizer.state)
    if restored and arguments.override_resumed_exploration:
        learner.model.log_standard_deviation = mx.full(
            (learner.action_count,),
            arguments.initial_log_standard_deviation,
            dtype=mx.float32,
        )
        optimizer_state = dict(tree_flatten(learner.optimizer.state))
        for name in (
            "log_standard_deviation.m",
            "log_standard_deviation.v",
        ):
            if name in optimizer_state:
                optimizer_state[name] = mx.zeros_like(
                    optimizer_state[name]
                )
        learner.optimizer.state = tree_unflatten(
            sorted(optimizer_state.items())
        )
        learner.refresh_compiled_training_state()
        mx.eval(
            learner.model.log_standard_deviation,
            learner.optimizer.state,
        )
    if (
        arguments.train_actor_observation_extension_count is not None
        and not arguments.train_actor_observation_extension_only
    ):
        raise ValueError(
            "extension-only actor column count requires extension-only training"
        )
    if arguments.train_actor_observation_extension_only:
        if restored:
            raise ValueError(
                "extension-only actor training requires fresh optimizer state"
            )
        if arguments.actor_observation_extension_offset is None:
            raise ValueError(
                "extension-only actor training requires an observation extension offset"
            )
        learner.train_actor_observation_extension_only(
            arguments.actor_observation_extension_offset,
            arguments.train_actor_observation_extension_count,
        )
    current_policy = learner.write_policy_pack(
        arguments.output_policy_pack,
        library_path=arguments.native_library,
    )
    current_deployment_policy = learner.write_policy_pack(
        arguments.deployment_policy_pack,
        stochastic=False,
        library_path=arguments.native_library,
    )
    incumbent_policy = learner.write_policy_pack(
        arguments.incumbent_policy_pack,
        stochastic=False,
        library_path=arguments.native_library,
    )
    expected_task_fingerprint: int | None = None
    _emit(
        sys.stdout,
        {
            "status": "ready",
            "policy_id": learner.policy_id,
            "policy_revision": learner.revision,
            "actor_observation_count":
                learner.actor_observation_count,
            "critic_observation_count":
                learner.critic_observation_count,
            "action_count": learner.action_count,
            "motion_feature_count": (
                motion_prior.motion_pack.feature_count
                if motion_prior is not None
                else 0
            ),
            "learner_state_restored": restored,
            "resumed_learning_rate_overridden": bool(
                restored and arguments.override_resumed_learning_rate
            ),
            "resumed_exploration_overridden": bool(
                restored and arguments.override_resumed_exploration
            ),
            "motion_prior_enabled": motion_prior is not None,
            "retention_policy_enabled": retention_reference is not None,
            "retention_policy_pack": (
                str(arguments.retention_policy_pack)
                if arguments.retention_policy_pack is not None
                else None
            ),
            "retention_maximum_difficulty_band": (
                arguments.retention_maximum_difficulty_band
            ),
            "retention_protected_actor_only": (
                arguments.retention_protected_actor_only
            ),
            "retention_balance_difficulty_bands": (
                arguments.retention_balance_difficulty_bands
            ),
            "retention_priority_difficulty_band": (
                arguments.retention_priority_difficulty_band
            ),
            "retention_priority_factor": arguments.retention_priority_factor,
            "motion_pack_hash": (
                motion_prior.motion_pack.content_hash
                if motion_prior is not None
                else 0
            ),
            "policy_pack": str(current_policy),
            "deployment_policy_pack":
                str(current_deployment_policy),
            "incumbent_policy_pack": str(incumbent_policy),
        },
    )
    for encoded in sys.stdin:
        try:
            request = json.loads(encoded)
            if not isinstance(request, dict):
                raise ValueError("learner request must be a JSON object")
            operation = request.get("operation")
            if operation == "close":
                _emit(
                    sys.stdout,
                    {
                        "status": "closed",
                        "policy_revision": learner.revision,
                    },
                )
                return 0
            if operation != "update":
                raise ValueError("learner operation is unsupported")
            rollout_value = request.get("rollout_pack")
            if not isinstance(rollout_value, str) or not rollout_value:
                raise ValueError("update requires rollout_pack")
            (
                rollout,
                expected_task_fingerprint,
            ) = _validate_rollout(
                learner,
                Path(rollout_value).expanduser().resolve(),
                expected_task_fingerprint,
                arguments.native_library,
            )
            revision_before = learner.revision
            learning_rewards = None
            motion_metrics: dict[str, float] = {}
            if motion_prior is not None:
                learning_rewards, motion_metrics = (
                    motion_prior.blend_rewards(rollout)
                )
            policy_batch = rollout.policy_batch(
                discount=learner.configuration.discount,
                gae_lambda=learner.configuration.gae_lambda,
                rewards=learning_rewards,
            )
            if retention_reference is not None:
                retention_weights = None
                if arguments.retention_maximum_difficulty_band is not None:
                    retention_weights = (
                        rollout.transitions["difficulty_band"].reshape(-1)
                        <= arguments.retention_maximum_difficulty_band
                    ).astype(np.float32)
                policy_batch = _retention_policy_batch(
                    policy_batch,
                    retention_reference,
                    chunk_size=learner.configuration.minibatch_size,
                    teacher_weights=retention_weights,
                    protected_actor_only=(
                        arguments.retention_protected_actor_only
                    ),
                    difficulty_bands=(
                        rollout.transitions["difficulty_band"].reshape(-1)
                    ),
                    balance_difficulty_bands=(
                        arguments.retention_balance_difficulty_bands
                    ),
                    priority_difficulty_band=(
                        arguments.retention_priority_difficulty_band
                    ),
                    priority_factor=arguments.retention_priority_factor,
                    rollout_teacher_blend=(
                        arguments.retention_rollout_teacher_blend
                    ),
                )
            metrics = learner.update(policy_batch)
            metrics.update(motion_metrics)
            artifact = learner.write_policy_pack(
                arguments.output_policy_pack,
                library_path=arguments.native_library,
            )
            deployment_artifact = learner.write_policy_pack(
                arguments.deployment_policy_pack,
                stochastic=False,
                library_path=arguments.native_library,
            )
            learner_state = _write_learner_state(
                learner,
                arguments.learner_state,
                motion_prior,
            )
            response = {
                "status": "updated",
                "policy_revision_before": revision_before,
                "policy_revision_after": learner.revision,
                "task_fingerprint": rollout.task_fingerprint,
                "behavior_policy_fingerprint":
                    rollout.policy_fingerprint,
                "samples": rollout.sample_count,
                "policy_pack": str(artifact),
                "deployment_policy_pack": str(deployment_artifact),
                "learner_state": str(learner_state),
                **_rollout_metrics(rollout),
                **metrics,
            }

            # MLX intentionally caches released Metal allocations for reuse.
            # A persistent learner sees one capacity-sized rollout per update,
            # so retaining that cache competes directly with MetalWorld's
            # private heaps in unified memory and eventually forces macOS to
            # page. Policy, optimizer, and motion-prior state remain active;
            # only the completed batch and allocator cache are released at
            # this explicit Swift/learner synchronization boundary.
            del policy_batch
            del learning_rewards
            del rollout
            gc.collect()
            cache_before_clear = int(mx.get_cache_memory())
            active_after_update = int(mx.get_active_memory())
            peak_after_update = int(mx.get_peak_memory())
            mx.clear_cache()
            response.update(
                {
                    "mlx_active_memory_bytes": active_after_update,
                    "mlx_cache_memory_bytes": int(
                        mx.get_cache_memory()
                    ),
                    "mlx_cache_released_bytes": cache_before_clear,
                    "mlx_peak_memory_bytes": peak_after_update,
                }
            )
            mx.reset_peak_memory()
            _emit(sys.stdout, response)
        except Exception as error:
            _emit(
                sys.stdout,
                {
                    "status": "error",
                    "error": str(error),
                },
            )
    return 0


def _initialize(arguments: argparse.Namespace) -> int:
    if arguments.actor_policy_pack is None:
        learner = MLXPolicyLearner(
            arguments.actor_observations,
            arguments.critic_observations,
            arguments.actions,
            _configuration(arguments),
        )
    else:
        learner = MLXPolicyLearner.from_actor_policy_pack(
            arguments.actor_policy_pack,
            arguments.critic_observations,
            _configuration(arguments),
            preserve_critic=not arguments.actor_fresh_critic,
            actor_observation_count=arguments.actor_observations,
            actor_observation_extension_mean=(
                arguments.actor_observation_extension_mean
            ),
            actor_observation_extension_inverse_standard_deviation=(
                arguments
                .actor_observation_extension_inverse_standard_deviation
            ),
            actor_observation_extension_offset=(
                arguments.actor_observation_extension_offset
            ),
            library_path=arguments.native_library,
        )
        if (
            learner.actor_observation_count != arguments.actor_observations
            or learner.action_count != arguments.actions
        ):
            raise ValueError(
                "actor initialization PolicyPack disagrees with the native "
                "task observation or action contract"
            )
    if arguments.zero_actor_output:
        learner.zero_actor_output()
    _bind_contract(learner, arguments)
    artifact = learner.write_policy_pack(
        arguments.output,
        policy_id=arguments.policy_id,
        stochastic=not arguments.deterministic,
        library_path=arguments.native_library,
    )
    _emit(
        sys.stdout,
        {
            "status": "initialized",
            "policy_id": learner.policy_id,
            "policy_revision": learner.revision,
            "actor_observation_count":
                learner.actor_observation_count,
            "critic_observation_count":
                learner.critic_observation_count,
            "action_count": learner.action_count,
            "policy_pack": str(artifact),
            "policy_pack_bytes": artifact.stat().st_size,
        },
    )
    return 0


def _add_ppo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=(512, 256, 128),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--imagination-distillation-coefficient",
        type=float,
        default=1.0,
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument(
        "--maximum-gradient-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help="use zero to disable KL-based learning-rate adaptation",
    )
    parser.add_argument(
        "--fixed-learning-rate",
        action="store_true",
    )
    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--maximum-learning-rate",
        type=float,
        default=1.0e-2,
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--initial-log-standard-deviation",
        type=float,
        default=-1.6094379124341003,
    )
    parser.add_argument("--observation-clip", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1)


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world-fingerprint", type=_positive, required=True)
    parser.add_argument("--task-fingerprint", type=_positive, required=True)
    parser.add_argument(
        "--observation-fingerprint", type=_positive, required=True
    )
    parser.add_argument("--action-fingerprint", type=_positive, required=True)


def _bind_contract(
    learner: MLXPolicyLearner,
    arguments: argparse.Namespace,
) -> None:
    requested = (
        int(arguments.world_fingerprint),
        int(arguments.task_fingerprint),
        int(arguments.observation_fingerprint),
        int(arguments.action_fingerprint),
    )
    existing = (
        learner.world_fingerprint,
        learner.task_fingerprint,
        learner.observation_fingerprint,
        learner.action_fingerprint,
    )
    explicit_observation_migration = (
        getattr(arguments, "actor_observation_extension_offset", None)
        is not None
    )
    explicit_actor_initialization = (
        getattr(arguments, "actor_policy_pack", None) is not None
    )
    if (
        learner.contract_version != 0
        and existing != requested
        and not explicit_observation_migration
        and not explicit_actor_initialization
    ):
        raise ValueError(
            "PolicyPack is bound to different world, task, observation, or action semantics"
        )
    learner.bind_contract(
        world_fingerprint=requested[0],
        task_fingerprint=requested[1],
        observation_fingerprint=requested[2],
        action_fingerprint=requested[3],
    )


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLX-only learner backend for Swift-native rollouts"
    )
    operations = parser.add_subparsers(
        dest="operation",
        required=True,
    )
    initialize = operations.add_parser(
        "initialize",
        help="publish an initial stochastic actor/critic PolicyPack",
    )
    initialize.add_argument(
        "--actor-observations",
        type=_positive,
        required=True,
    )
    initialize.add_argument(
        "--critic-observations",
        type=_positive,
        required=True,
    )
    initialize.add_argument("--actions", type=_positive, required=True)
    initialize.add_argument("--policy-id", required=True)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument(
        "--actor-policy-pack",
        type=Path,
        help=(
            "initialize from a PolicyPack actor, preserving its critic and "
            "exploration head when present"
        ),
    )
    initialize.add_argument(
        "--actor-fresh-critic",
        action="store_true",
        help=(
            "preserve only the source actor and initialize a fresh critic "
            "for the requested privileged observation contract"
        ),
    )
    initialize.add_argument(
        "--zero-actor-output",
        action="store_true",
        help="zero the final actor layer so the initial action is exactly the mechanism default",
    )
    initialize.add_argument(
        "--deterministic",
        action="store_true",
        help="write an actor-only deployment PolicyPack",
    )
    initialize.add_argument(
        "--actor-observation-extension-mean",
        type=float,
        default=0.0,
        help=(
            "normalization mean for observations appended beyond the source "
            "PolicyPack contract"
        ),
    )
    initialize.add_argument(
        "--actor-observation-extension-inverse-standard-deviation",
        type=float,
        default=1.0,
        help=(
            "normalization inverse standard deviation for observations "
            "appended beyond the source PolicyPack contract"
        ),
    )
    initialize.add_argument(
        "--actor-observation-extension-offset",
        type=int,
        help=(
            "insert zero-connected observations before this source-policy "
            "index; defaults to appending them"
        ),
    )
    initialize.add_argument(
        "--native-library",
        type=Path,
        required=True,
    )
    _add_contract_arguments(initialize)
    _add_ppo_arguments(initialize)

    serve = operations.add_parser(
        "serve",
        help="retain optimizer state while Swift schedules rollout updates",
    )
    serve.add_argument("--policy-pack", type=Path, required=True)
    serve.add_argument(
        "--output-policy-pack",
        type=Path,
        required=True,
    )
    serve.add_argument(
        "--learner-state",
        type=Path,
        required=True,
    )
    serve.add_argument(
        "--restore-learner-state",
        type=Path,
        help="immutable learner state used only for restoration",
    )
    serve.add_argument(
        "--retention-policy-pack",
        type=Path,
        help=(
            "frozen source actor evaluated on current observations and used "
            "as all-sample Huber retention targets"
        ),
    )
    serve.add_argument(
        "--retention-maximum-difficulty-band",
        type=int,
        choices=range(11),
        help=(
            "apply frozen-actor retention only to rollout samples at or "
            "below this difficulty band"
        ),
    )
    serve.add_argument(
        "--retention-protected-actor-only",
        action="store_true",
        help=(
            "exclude retention-selected samples from PPO actor and entropy "
            "losses while retaining their critic updates"
        ),
    )
    serve.add_argument(
        "--retention-balance-difficulty-bands",
        action="store_true",
        help=(
            "equalize frozen-actor loss across represented protected "
            "difficulty bands"
        ),
    )
    serve.add_argument(
        "--retention-priority-difficulty-band",
        type=int,
        choices=range(11),
        help="give one protected band additional relative retention authority",
    )
    serve.add_argument(
        "--retention-priority-factor",
        type=float,
        default=1.0,
        help="relative authority for the priority band; must be at least one",
    )
    serve.add_argument(
        "--retention-rollout-teacher-blend",
        type=float,
        default=0.0,
        help=(
            "blend rollout teacher labels into frozen-actor targets; zero "
            "retains strict source targets"
        ),
    )
    serve.add_argument(
        "--actor-observation-extension-offset",
        type=int,
        help=(
            "explicitly migrate a narrower learner checkpoint by inserting "
            "zero actor weights and Adam moments at this source index"
        ),
    )
    serve.add_argument(
        "--train-actor-observation-extension-only",
        action="store_true",
        help=(
            "freeze the inherited actor and exploration parameter while "
            "training only first-layer columns at and after the observation "
            "extension offset; the critic remains trainable"
        ),
    )
    serve.add_argument(
        "--train-actor-observation-extension-count",
        type=int,
        help=(
            "limit extension-only actor training to this many columns; "
            "defaults to the suffix after the extension offset"
        ),
    )
    serve.add_argument(
        "--override-resumed-learning-rate",
        action="store_true",
        help=(
            "restore model, Adam, and motion state while "
            "explicitly replacing the PPO learning-rate schedule"
        ),
    )
    serve.add_argument(
        "--override-resumed-exploration",
        action="store_true",
        help=(
            "restore model, critic, and motion state while "
            "explicitly replacing policy exploration and its Adam moments"
        ),
    )
    serve.add_argument(
        "--deployment-policy-pack",
        type=Path,
        required=True,
        help="latest deterministic candidate PolicyPack",
    )
    serve.add_argument(
        "--incumbent-policy-pack",
        type=Path,
        required=True,
        help=(
            "immutable deterministic pre-training PolicyPack for matched "
            "incumbent evaluation"
        ),
    )
    serve.add_argument(
        "--native-library",
        type=Path,
        required=True,
    )
    serve.add_argument(
        "--motion-pack",
        type=Path,
        help="optional expert MotionPack for threat-gated AMP reward",
    )
    serve.add_argument(
        "--motion-hidden-sizes",
        type=int,
        nargs="*",
        default=(512, 256),
    )
    serve.add_argument(
        "--motion-learning-rate", type=float, default=1.0e-3
    )
    serve.add_argument(
        "--motion-minibatch-size", type=_positive, default=4096
    )
    serve.add_argument(
        "--motion-update-epochs", type=_positive, default=2
    )
    serve.add_argument(
        "--motion-reward-coefficient", type=float, default=0.3
    )
    serve.add_argument(
        "--motion-activation",
        choices=("impact", "always"),
        default="impact",
        help=(
            "activate motion reward only during impact sequences or on every "
            "continuing transition"
        ),
    )
    _add_contract_arguments(serve)
    _add_ppo_arguments(serve)

    arguments = parser.parse_args()
    configuration = _configuration(arguments)
    configuration.validate()
    if arguments.operation == "initialize":
        return _initialize(arguments)
    if arguments.operation == "serve":
        return _serve(arguments)
    raise AssertionError(f"unhandled operation: {arguments.operation}")


if __name__ == "__main__":
    raise SystemExit(main())
