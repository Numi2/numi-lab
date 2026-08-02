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
    MLXPolicyLearner,
    read_motion_pack,
    read_policy_rollout_pack,
)


_PPO_RESUMABLE_SCHEDULE_FIELDS = frozenset(
    {"minibatch_size", "seed"}
)
_PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS = frozenset(
    {"learning_rate", "adaptive_learning_rate"}
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


def _valid_curriculum_reference(level: int, reference: int) -> bool:
    if reference == 0:
        return True
    return (
        0 <= reference <= np.iinfo(np.uint64).max
        and reference & (1 << 63) != 0
        and ((reference >> 60) & 0x3) == level
    )


def _valid_curriculum_levels(
    level_series: np.ndarray,
    initial_level: int,
) -> bool:
    return (
        level_series.size != 0
        and abs(int(level_series[0]) - initial_level) <= 1
        and not np.any(np.abs(np.diff(level_series)) > 1)
    )


def _write_learner_state(
    learner: MLXPolicyLearner,
    path: Path,
    task_curriculum_level: int,
    task_curriculum_reference_rates: int,
    motion_prior: MLXMotionPrior | None = None,
) -> Path:
    """Atomically checkpoint model parameters and Adam moments."""

    if not learner.policy_id:
        raise ValueError("learner state requires a policy identity")
    if not 0 <= task_curriculum_level <= np.iinfo(np.uint32).max:
        raise ValueError("task curriculum level exceeds uint32")
    if not 0 <= task_curriculum_reference_rates <= np.iinfo(np.uint64).max:
        raise ValueError("task curriculum reference exceeds uint64")
    if not _valid_curriculum_reference(
        task_curriculum_level,
        task_curriculum_reference_rates,
    ):
        raise ValueError("task curriculum reference disagrees with its level")
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
        "task_curriculum_level": str(task_curriculum_level),
        "task_curriculum_reference_rates": str(
            task_curriculum_reference_rates
        ),
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
) -> tuple[bool, int, int]:
    target = path.expanduser().resolve()
    if target.suffix != ".safetensors":
        raise ValueError("learner state path must end in .safetensors")
    if not target.is_file():
        return False, 0, 0
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
        "actor_observation_count": str(
            learner.actor_observation_count
        ),
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
    if any(
        metadata.get(key) != value
        for key, value in expected_metadata.items()
    ) or not _configuration_matches(
        metadata.get("configuration"),
        _configuration_record(learner),
        ppo_resumable_schedule_fields,
    ) or (
        motion_prior is not None
        and not _configuration_matches(
            metadata.get("motion_configuration"),
            _motion_configuration_record(motion_prior),
            _MOTION_RESUMABLE_SCHEDULE_FIELDS,
        )
    ):
        raise ValueError(
            "MLX learner state contract differs from the PolicyPack or PPO configuration"
        )
    try:
        revision = int(metadata["policy_revision"])
        task_curriculum_level = int(
            metadata["task_curriculum_level"]
        )
        task_curriculum_reference_rates = int(
            metadata.get("task_curriculum_reference_rates", "0")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "MLX learner state revision or task curriculum is invalid"
        ) from error
    if revision < learner.revision:
        raise ValueError(
            "MLX learner state is older than the supplied PolicyPack"
        )
    if not 0 <= task_curriculum_level <= np.iinfo(np.uint32).max:
        raise ValueError("MLX learner task curriculum exceeds uint32")
    if not 0 <= task_curriculum_reference_rates <= np.iinfo(np.uint64).max:
        raise ValueError("MLX learner task curriculum reference exceeds uint64")
    if not _valid_curriculum_reference(
        task_curriculum_level,
        task_curriculum_reference_rates,
    ):
        raise ValueError(
            "MLX learner task curriculum reference disagrees with its level"
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
    for name, expected in expected_model.items():
        actual = model_arrays[name]
        if (
            tuple(actual.shape) != tuple(expected.shape)
            or actual.dtype != expected.dtype
        ):
            raise ValueError(
                f"MLX learner state model tensor {name} is incompatible"
            )
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
        actual = optimizer_arrays[name]
        if (
            tuple(actual.shape) != tuple(expected.shape)
            or actual.dtype != expected.dtype
        ):
            raise ValueError(
                f"MLX learner state optimizer tensor {name} is incompatible"
            )
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
    return (
        True,
        task_curriculum_level,
        task_curriculum_reference_rates,
    )


def _validate_rollout(
    learner: MLXPolicyLearner,
    rollout_path: Path,
    expected_task_fingerprint: int | None,
    task_curriculum_level: int,
    native_library: Path,
) -> tuple[Any, int, int]:
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
    levels = rollout.transitions["curriculum_level"].reshape(
        rollout.control_step_count,
        rollout.environment_count,
    )
    if np.any(levels != levels[:, :1]):
        raise ValueError(
            "PolicyRolloutPack task-wide curriculum differs across environments"
        )
    level_series = levels[:, 0].astype(np.int64)
    if not _valid_curriculum_levels(level_series, task_curriculum_level):
        raise ValueError(
            "PolicyRolloutPack task curriculum changes by more than one level"
        )
    return (
        rollout,
        rollout.task_fingerprint,
        int(level_series[-1]),
    )


def _rollout_metrics(rollout: Any) -> dict[str, Any]:
    transitions = rollout.transitions
    rewards = transitions["reward"]
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
                    np.max(transitions["tilt"][selected])
                ),
                "minimum_root_height": float(
                    np.min(transitions["root_height"][selected])
                ),
            }
        )
    return {
        "mean_reward": float(np.mean(rewards)),
        "reward_standard_deviation": float(np.std(rewards)),
        "minimum_reward": float(np.min(rewards)),
        "maximum_reward": float(np.max(rewards)),
        "mean_tracking_score": float(
            np.mean(transitions["tracking_score"])
        ),
        "mean_root_height": float(
            np.mean(transitions["root_height"])
        ),
        "mean_tilt": float(np.mean(transitions["tilt"])),
        "mean_task_reward": float(
            np.mean(transitions["task_reward"])
        ),
        "mean_base_reward": float(
            np.mean(transitions["base_reward"])
        ),
        "mean_joint_velocity_reward": float(
            np.mean(transitions["joint_velocity_reward"])
        ),
        "mean_joint_acceleration_reward": float(
            np.mean(transitions["joint_acceleration_reward"])
        ),
        "mean_control_reward": float(
            np.mean(transitions["control_reward"])
        ),
        "mean_posture_reward": float(
            np.mean(transitions["posture_reward"])
        ),
        "mean_energy_reward": float(
            np.mean(transitions["energy_reward"])
        ),
        "mean_contact_reward": float(
            np.mean(transitions["contact_reward"])
        ),
        "done_count": int(np.sum(transitions["done"])),
        "timeout_count": int(np.sum(transitions["timeout"])),
        "termination_reason_counts": {
            str(int(reason)): int(count)
            for reason, count in zip(reasons, counts, strict=True)
        },
        "impact_sequence_metrics": impact_metrics,
    }


def _serve(arguments: argparse.Namespace) -> int:
    if not 0 <= arguments.initial_task_curriculum_level <= np.iinfo(
        np.uint32
    ).max:
        raise ValueError("initial task curriculum level exceeds uint32")
    if not 0 <= arguments.initial_task_curriculum_reference_rates <= np.iinfo(
        np.uint64
    ).max:
        raise ValueError("initial task curriculum reference exceeds uint64")
    learner = MLXPolicyLearner.from_policy_pack(
        arguments.policy_pack,
        _configuration(arguments),
        library_path=arguments.native_library,
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
                maximum_gradient_norm=arguments.maximum_gradient_norm,
                seed=arguments.seed,
            ),
        )
    (
        restored,
        task_curriculum_level,
        task_curriculum_reference_rates,
    ) = _restore_learner_state(
        learner,
        arguments.learner_state,
        motion_prior,
        _PPO_RESUMABLE_SCHEDULE_FIELDS |
        _PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS
        if arguments.override_resumed_learning_rate
        else _PPO_RESUMABLE_SCHEDULE_FIELDS,
    )
    if restored and arguments.override_resumed_learning_rate:
        learner.optimizer.learning_rate = arguments.learning_rate
        learner.refresh_compiled_training_state()
        mx.eval(learner.optimizer.state)
    if not restored:
        task_curriculum_level = arguments.initial_task_curriculum_level
        task_curriculum_reference_rates = (
            arguments.initial_task_curriculum_reference_rates
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
            "motion_prior_enabled": motion_prior is not None,
            "motion_pack_hash": (
                motion_prior.motion_pack.content_hash
                if motion_prior is not None
                else 0
            ),
            "task_curriculum_level": task_curriculum_level,
            "task_curriculum_reference_rates": str(
                task_curriculum_reference_rates
            ),
            "policy_pack": str(current_policy),
            "deployment_policy_pack":
                str(current_deployment_policy),
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
            reference_value = request.get(
                "task_curriculum_reference_rates"
            )
            if not isinstance(reference_value, str):
                raise ValueError(
                    "update requires a decimal task curriculum reference"
                )
            try:
                rollout_curriculum_reference_rates = int(reference_value)
            except ValueError as error:
                raise ValueError(
                    "task curriculum reference is not an integer"
                ) from error
            if not 0 <= rollout_curriculum_reference_rates <= np.iinfo(
                np.uint64
            ).max:
                raise ValueError("task curriculum reference exceeds uint64")
            (
                rollout,
                expected_task_fingerprint,
                rollout_curriculum_level,
            ) = _validate_rollout(
                learner,
                Path(rollout_value).expanduser().resolve(),
                expected_task_fingerprint,
                task_curriculum_level,
                arguments.native_library,
            )
            if not _valid_curriculum_reference(
                rollout_curriculum_level,
                rollout_curriculum_reference_rates,
            ):
                raise ValueError(
                    "rollout curriculum reference disagrees with its level"
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
                rollout_curriculum_level,
                rollout_curriculum_reference_rates,
                motion_prior,
            )
            task_curriculum_level = rollout_curriculum_level
            task_curriculum_reference_rates = (
                rollout_curriculum_reference_rates
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
                "task_curriculum_level": task_curriculum_level,
                "task_curriculum_reference_rates": str(
                    task_curriculum_reference_rates
                ),
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
            actor_observation_count=arguments.actor_observations,
            actor_observation_extension_mean=(
                arguments.actor_observation_extension_mean
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
    artifact = learner.write_policy_pack(
        arguments.output,
        policy_id=arguments.policy_id,
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
        help="initialize the actor from a deterministic PolicyPack",
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
        "--override-resumed-learning-rate",
        action="store_true",
        help=(
            "restore model, Adam, motion, and curriculum state while "
            "explicitly replacing the PPO learning-rate schedule"
        ),
    )
    serve.add_argument(
        "--deployment-policy-pack",
        type=Path,
        required=True,
        help="deterministic actor PolicyPack for evaluation and deployment",
    )
    serve.add_argument(
        "--native-library",
        type=Path,
        required=True,
    )
    serve.add_argument(
        "--initial-task-curriculum-level",
        type=int,
        default=0,
    )
    serve.add_argument(
        "--initial-task-curriculum-reference-rates",
        type=int,
        default=0,
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
