#!/usr/bin/env python3
"""Focused MLX learner and optional native artifact-cycle check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from metalrobo.mlx_policy_learning import (
    MLXPPOConfiguration,
    MLXPolicyBatch,
    MLXPolicyLearner,
    NativePolicyRollout,
    read_policy_pack,
    read_policy_rollout_pack,
)
from metalrobo.mlx_policy_worker import (
    _MOTION_RESUMABLE_SCHEDULE_FIELDS,
    _PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS,
    _PPO_RESUMABLE_SCHEDULE_FIELDS,
    _configuration_matches,
    _valid_curriculum_levels,
    _valid_curriculum_reference,
)


def check_adaptive_curriculum_contract() -> None:
    reference = (1 << 63) | (2 << 60) | 10000 | (14000 << 20)
    if not _valid_curriculum_reference(2, reference):
        raise RuntimeError("valid curriculum reference was rejected")
    if _valid_curriculum_reference(1, reference):
        raise RuntimeError("cross-level curriculum reference was accepted")
    if not _valid_curriculum_levels(
        np.asarray((1, 1, 0, 0, 1), dtype=np.int64),
        1,
    ):
        raise RuntimeError("bounded adaptive curriculum was rejected")
    if _valid_curriculum_levels(
        np.asarray((1, 3), dtype=np.int64),
        1,
    ):
        raise RuntimeError("multi-level curriculum jump was accepted")


def make_learner() -> MLXPolicyLearner:
    return MLXPolicyLearner(
        480,
        495,
        29,
        MLXPPOConfiguration(
            update_epochs=1,
            minibatch_size=32,
            hidden_sizes=(32,),
            learning_rate=1.0e-4,
            target_kl=None,
            seed=17,
        ),
    )


def check_gae_boundaries() -> None:
    transitions = np.zeros(
        2,
        dtype=[
            ("reward", "<f4"),
            ("done", "<u4"),
            ("timeout", "<u4"),
            ("physics_error", "<u4"),
            ("timeout_bootstrap_value", "<f4"),
        ],
    )
    transitions["reward"] = (1.0, 1.0)
    transitions["done"][0] = 1
    transitions["timeout"][0] = 1
    transitions["timeout_bootstrap_value"][0] = 5.0
    rollout = NativePolicyRollout(
        id="gae_boundary_check",
        task_fingerprint=1,
        policy_fingerprint=1,
        policy_revision=1,
        environment_count=1,
        control_step_count=2,
        actor_observation_count=1,
        critic_observation_count=1,
        action_count=1,
        motion_feature_count=0,
        actor_observations=np.zeros(2, dtype=np.float32),
        critic_observations=np.zeros(2, dtype=np.float32),
        motion_features=np.zeros(0, dtype=np.float32),
        latents=np.zeros(2, dtype=np.float32),
        old_log_probabilities=np.zeros(2, dtype=np.float32),
        old_values=np.asarray((2.0, 4.0), dtype=np.float32),
        bootstrap_values=np.asarray((8.0,), dtype=np.float32),
        transitions=transitions,
    )
    advantages = np.asarray(
        rollout.policy_batch(
            discount=0.99,
            gae_lambda=0.95,
        ).advantages
    )
    expected = np.asarray((3.95, 4.92), dtype=np.float32)
    if not np.allclose(advantages, expected, atol=1.0e-6):
        raise RuntimeError(
            "GAE crossed an episode boundary or bootstrapped a timeout "
            f"from the wrong state: {advantages}"
        )


def check_resumable_schedule_contracts() -> None:
    ppo = {
        "hidden_sizes": [32],
        "clip_ratio": 0.2,
        "minibatch_size": 32,
        "seed": 17,
    }
    scaled_ppo = {
        **ppo,
        "minibatch_size": 256,
        "seed": 18,
    }
    if not _configuration_matches(
        json.dumps(ppo),
        json.dumps(scaled_ppo),
        _PPO_RESUMABLE_SCHEDULE_FIELDS,
    ):
        raise RuntimeError(
            "PPO execution geometry incorrectly blocks learner resume"
        )
    incompatible_ppo = {**scaled_ppo, "clip_ratio": 0.3}
    if _configuration_matches(
        json.dumps(ppo),
        json.dumps(incompatible_ppo),
        _PPO_RESUMABLE_SCHEDULE_FIELDS,
    ):
        raise RuntimeError(
            "PPO algorithm changes incorrectly pass learner resume"
        )
    fine_tuned_ppo = {
        **scaled_ppo,
        "learning_rate": 1.0e-6,
        "adaptive_learning_rate": False,
    }
    if _configuration_matches(
        json.dumps(ppo),
        json.dumps(fine_tuned_ppo),
        _PPO_RESUMABLE_SCHEDULE_FIELDS,
    ):
        raise RuntimeError(
            "learning-rate changes passed an exact learner resume"
        )
    if not _configuration_matches(
        json.dumps(ppo),
        json.dumps(fine_tuned_ppo),
        _PPO_RESUMABLE_SCHEDULE_FIELDS |
        _PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS,
    ):
        raise RuntimeError(
            "explicit learning-rate fine-tuning was rejected"
        )
    if _configuration_matches(
        json.dumps(ppo),
        json.dumps({**fine_tuned_ppo, "clip_ratio": 0.3}),
        _PPO_RESUMABLE_SCHEDULE_FIELDS |
        _PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS,
    ):
        raise RuntimeError(
            "learning-rate override admitted another PPO algorithm change"
        )
    motion = {
        "hidden_sizes": [32],
        "reward_coefficient": 0.3,
        "minibatch_size": 64,
        "seed": 17,
    }
    scaled_motion = {
        **motion,
        "minibatch_size": 512,
        "seed": 18,
    }
    if not _configuration_matches(
        json.dumps(motion),
        json.dumps(scaled_motion),
        _MOTION_RESUMABLE_SCHEDULE_FIELDS,
    ):
        raise RuntimeError(
            "motion execution geometry incorrectly blocks learner resume"
        )


def run_native_cycle(
    *,
    collector: Path,
    metallib: Path,
    library: Path,
    environments: int,
    steps: int,
    chunk: int,
    output: Path | None,
) -> dict[str, object]:
    if not collector.is_file():
        raise FileNotFoundError(f"Swift collector is missing: {collector}")
    if not metallib.is_file():
        raise FileNotFoundError(f"Metal library is missing: {metallib}")
    if not library.is_file():
        raise FileNotFoundError(f"Native library is missing: {library}")
    learner = make_learner()
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        initial_policy = temporary / "initial.policypack"
        rollout_path = temporary / "native.rolloutpack"
        updated_policy = (
            output
            if output is not None
            else temporary / "updated.policypack"
        )
        learner.write_policy_pack(
            initial_policy,
            policy_id="mlx_native_cycle_initial",
            library_path=library,
        )
        environment = os.environ.copy()
        existing_dyld = environment.get("DYLD_LIBRARY_PATH", "")
        environment["DYLD_LIBRARY_PATH"] = (
            str(library.parent)
            if not existing_dyld
            else f"{library.parent}:{existing_dyld}"
        )
        collection = subprocess.run(
            [
                str(collector),
                "--metallib",
                str(metallib),
                "--envs",
                str(environments),
                "--steps",
                str(steps),
                "--repeats",
                "1",
                "--chunk",
                str(chunk),
                "--policy-pack",
                str(initial_policy),
                "--rollout-pack",
                str(rollout_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        collection_result = json.loads(collection.stdout)
        rollout = read_policy_rollout_pack(rollout_path)
        if (
            rollout.actor_observation_count
            != learner.actor_observation_count
            or rollout.critic_observation_count
            != learner.critic_observation_count
            or rollout.action_count != learner.action_count
            or rollout.policy_revision != learner.revision
        ):
            raise RuntimeError(
                "Native rollout contract differs from the MLX behavior policy"
            )
        batch = rollout.policy_batch()
        current_log_probabilities, _, current_values = (
            learner.model.evaluate(
                batch.actor_observations,
                batch.critic_observations,
                batch.latents,
            )
        )
        mx.eval(current_log_probabilities, current_values)
        log_probability_error = float(
            np.max(
                np.abs(
                    np.asarray(current_log_probabilities)
                    - rollout.old_log_probabilities
                )
            )
        )
        value_error = float(
            np.max(
                np.abs(
                    np.asarray(current_values)
                    - rollout.old_values
                )
            )
        )
        if log_probability_error > 3.0e-3 or value_error > 2.0e-3:
            raise RuntimeError(
                "Metal and MLX disagree on the installed behavior policy: "
                f"log_probability_error={log_probability_error}, "
                f"value_error={value_error}"
            )
        metrics = learner.update(batch)
        artifact = learner.write_policy_pack(
            updated_policy,
            policy_id="mlx_native_cycle_updated",
            library_path=library,
        )
        published = read_policy_pack(artifact)
        resumed = MLXPolicyLearner.from_policy_pack(artifact)
        resumed_log_probabilities, _, resumed_values = (
            resumed.model.evaluate(
                batch.actor_observations,
                batch.critic_observations,
                batch.latents,
            )
        )
        learned_log_probabilities, _, learned_values = (
            learner.model.evaluate(
                batch.actor_observations,
                batch.critic_observations,
                batch.latents,
            )
        )
        mx.eval(
            resumed_log_probabilities,
            resumed_values,
            learned_log_probabilities,
            learned_values,
        )
        resume_parameter_error = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(resumed_log_probabilities)
                        - np.asarray(learned_log_probabilities)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(resumed_values)
                        - np.asarray(learned_values)
                    )
                )
            ),
        )
        if (
            published.revision != learner.revision
            or resumed.revision != learner.revision
            or resume_parameter_error > 1.0e-6
        ):
            raise RuntimeError(
                "PolicyPack did not preserve the trainable policy state"
            )
        reload_run = subprocess.run(
            [
                str(collector),
                "--metallib",
                str(metallib),
                "--envs",
                str(environments),
                "--steps",
                str(min(steps, 4)),
                "--repeats",
                "1",
                "--chunk",
                str(min(chunk, steps, 4)),
                "--policy-pack",
                str(artifact),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        reload_result = json.loads(reload_run.stdout)
        if reload_result["failed_environment_steps"] != 0:
            raise RuntimeError("Updated PolicyPack failed native reload")
        return {
            "check": "mlx_native_policy_cycle",
            "samples": rollout.sample_count,
            "policy_revision_before": rollout.policy_revision,
            "policy_revision_after": learner.revision,
            "resumed_policy_revision": resumed.revision,
            "resume_parameter_max_error": resume_parameter_error,
            "metal_mlx_log_probability_max_error":
                log_probability_error,
            "metal_mlx_value_max_error": value_error,
            "mean_reward": float(
                np.mean(rollout.transitions["reward"])
            ),
            "done_count": int(
                np.sum(rollout.transitions["done"])
            ),
            "minibatch_updates": int(
                metrics["minibatch_updates"]
            ),
            "loss": metrics["loss"],
            "collector_submissions":
                collection_result["submission_count"],
            "collector_failures":
                collection_result["failed_environment_steps"],
            "reload_failures":
                reload_result["failed_environment_steps"],
            "artifact": str(artifact),
            "artifact_bytes": artifact.stat().st_size,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--collector", type=Path)
    parser.add_argument("--metallib", type=Path)
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=4)
    arguments = parser.parse_args()
    check_gae_boundaries()
    check_resumable_schedule_contracts()
    check_adaptive_curriculum_contract()

    if arguments.collector is not None:
        if arguments.metallib is None or arguments.library is None:
            parser.error(
                "--collector requires --metallib and --library"
            )
        if min(
            arguments.environments,
            arguments.steps,
            arguments.chunk,
        ) <= 0:
            parser.error(
                "--environments, --steps, and --chunk must be positive"
            )
        print(
            json.dumps(
                run_native_cycle(
                    collector=arguments.collector,
                    metallib=arguments.metallib,
                    library=arguments.library,
                    environments=arguments.environments,
                    steps=arguments.steps,
                    chunk=arguments.chunk,
                    output=arguments.output,
                ),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    actor_count = 480
    critic_count = 495
    action_count = 29
    sample_count = 64
    learner = make_learner()
    generator = np.random.default_rng(17)
    actor = mx.array(
        generator.normal(
            0.0,
            0.2,
            (sample_count, actor_count),
        ).astype(np.float32)
    )
    critic = mx.array(
        generator.normal(
            0.0,
            0.2,
            (sample_count, critic_count),
        ).astype(np.float32)
    )
    latents = mx.array(
        generator.normal(
            0.0,
            0.3,
            (sample_count, action_count),
        ).astype(np.float32)
    )
    old_log_probabilities, _, old_values = (
        learner.model.evaluate(actor, critic, latents)
    )
    advantages = mx.array(
        generator.normal(0.0, 1.0, sample_count).astype(
            np.float32
        )
    )
    returns = old_values + 0.1 * advantages
    mx.eval(
        old_log_probabilities,
        old_values,
        advantages,
        returns,
    )
    batch = MLXPolicyBatch(
        actor_observations=actor,
        critic_observations=critic,
        latents=latents,
        old_log_probabilities=old_log_probabilities,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
    )
    metrics = learner.update(batch)

    if arguments.output is None:
        temporary = tempfile.TemporaryDirectory()
        output = Path(temporary.name) / "learner.policypack"
    else:
        temporary = None
        output = arguments.output
    try:
        artifact = learner.write_policy_pack(
            output,
            policy_id="mlx_policy_learning_check",
            library_path=arguments.library,
        )
        if not artifact.is_file() or artifact.stat().st_size <= 32:
            raise RuntimeError("PolicyPack artifact was not published")
        with tempfile.TemporaryDirectory() as actor_directory:
            deployment = learner.write_policy_pack(
                Path(actor_directory) / "deployment.policypack",
                stochastic=False,
                library_path=arguments.library,
            )
            initialized = MLXPolicyLearner.from_actor_policy_pack(
                deployment,
                critic_count,
                learner.configuration,
                library_path=arguments.library,
            )
            expected_actor = learner.model.actor_mean(actor)
            initialized_actor = initialized.model.actor_mean(actor)
            mx.eval(expected_actor, initialized_actor)
            actor_initialization_error = float(
                np.max(
                    np.abs(
                        np.asarray(expected_actor) -
                        np.asarray(initialized_actor)
                    )
                )
            )
            if actor_initialization_error != 0.0:
                raise RuntimeError(
                    "deployment actor initialization changed policy output"
                )
            expanded = MLXPolicyLearner.from_actor_policy_pack(
                deployment,
                critic_count + 7,
                learner.configuration,
                actor_observation_count=actor_count + 7,
                actor_observation_extension_mean=1.0,
                library_path=arguments.library,
            )
            expanded_actor = expanded.model.actor_mean(
                mx.concatenate(
                    (actor, mx.ones((sample_count, 7))),
                    axis=1,
                )
            )
            mx.eval(expanded_actor)
            expanded_mean = np.asarray(
                expanded.model.actor_observation_mean
            )
            actor_expansion_error = float(
                np.max(
                    np.abs(
                        np.asarray(expected_actor) -
                        np.asarray(expanded_actor)
                    )
                )
            )
            if actor_expansion_error != 0.0:
                raise RuntimeError(
                    "zero-connected observation expansion changed policy output"
                )
            if not np.array_equal(expanded_mean[-7:], np.ones(7)):
                raise RuntimeError(
                    "observation expansion mean was not published"
                )
            insertion = actor_count // 2
            inserted = MLXPolicyLearner.from_actor_policy_pack(
                deployment,
                critic_count + 7,
                learner.configuration,
                actor_observation_count=actor_count + 7,
                actor_observation_extension_mean=0.25,
                actor_observation_extension_offset=insertion,
                library_path=arguments.library,
            )
            inserted_actor = inserted.model.actor_mean(
                mx.concatenate(
                    (
                        actor[:, :insertion],
                        mx.full((sample_count, 7), 0.25),
                        actor[:, insertion:],
                    ),
                    axis=1,
                )
            )
            mx.eval(inserted_actor)
            if not np.array_equal(
                np.asarray(expected_actor),
                np.asarray(inserted_actor),
            ):
                raise RuntimeError(
                    "zero-connected observation insertion changed policy output"
                )
            inserted_mean = np.asarray(
                inserted.model.actor_observation_mean
            )
            if not np.array_equal(
                inserted_mean[insertion:insertion + 7],
                np.full(7, 0.25, dtype=np.float32),
            ):
                raise RuntimeError(
                    "observation insertion mean was not published"
                )
        print(
            json.dumps(
                {
                    "check": "mlx_policy_learning",
                    "policy_revision": learner.revision,
                    "minibatch_updates": int(
                        metrics["minibatch_updates"]
                    ),
                    "loss": metrics["loss"],
                    "artifact": str(artifact),
                    "artifact_bytes": artifact.stat().st_size,
                    "actor_initialization_max_error":
                        actor_initialization_error,
                    "actor_expansion_max_error": actor_expansion_error,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
