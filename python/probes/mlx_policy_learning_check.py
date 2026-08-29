#!/usr/bin/env python3
"""Focused MLX learner and optional native artifact-cycle check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

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
    _retention_policy_batch,
    _restore_learner_state,
    _write_learner_state,
)


def make_learner() -> MLXPolicyLearner:
    learner = MLXPolicyLearner(
        480,
        495,
        29,
        MLXPPOConfiguration(
            update_epochs=1,
            minibatch_size=32,
            hidden_sizes=(32,),
            critic_hidden_sizes=(32,),
            learning_rate=1.0e-4,
            target_kl=None,
            seed=17,
        ),
    )
    learner.bind_contract(
        world_fingerprint=1,
        task_fingerprint=2,
        observation_fingerprint=3,
        action_fingerprint=4,
    )
    return learner


def check_retention_targets() -> None:
    learner = make_learner()
    sample_count = 8
    actor = np.zeros((sample_count, 480), dtype=np.float32)
    critic = np.zeros((sample_count, 495), dtype=np.float32)
    batch = MLXPolicyBatch.from_numpy(
        actor_observations=actor,
        critic_observations=critic,
        latents=np.zeros((sample_count, 29), dtype=np.float32),
        old_log_probabilities=np.zeros(sample_count, dtype=np.float32),
        old_values=np.zeros(sample_count, dtype=np.float32),
        advantages=np.zeros(sample_count, dtype=np.float32),
        returns=np.zeros(sample_count, dtype=np.float32),
    )
    anchored = _retention_policy_batch(batch, learner, chunk_size=3)
    expected = learner.model.actor_mean(mx.array(actor))
    mx.eval(expected)
    if not np.array_equal(
        anchored.teacher_actions,
        np.asarray(expected, dtype=np.float32),
    ) or not np.all(anchored.teacher_weights == 1.0):
        raise RuntimeError("retention targets do not match the frozen actor")


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


def check_realized_imagination_weighting() -> None:
    transitions = np.zeros(
        4,
        dtype=[
            ("reward", "<f4"),
            ("done", "<u4"),
            ("timeout", "<u4"),
            ("physics_error", "<u4"),
            ("timeout_bootstrap_value", "<f4"),
            ("impact_sequence_index", "<u4"),
            ("impact_event_flags", "<u4"),
        ],
    )
    transitions["impact_sequence_index"] = (1, 1, 1, 0)
    transitions["impact_event_flags"][2] = 4
    rollout = NativePolicyRollout(
        id="realized_imagination_gate",
        task_fingerprint=1,
        policy_fingerprint=1,
        policy_revision=1,
        environment_count=1,
        control_step_count=4,
        actor_observation_count=1,
        critic_observation_count=1,
        action_count=1,
        motion_feature_count=0,
        actor_observations=np.zeros(4, dtype=np.float32),
        critic_observations=np.zeros(4, dtype=np.float32),
        motion_features=np.zeros(0, dtype=np.float32),
        latents=np.zeros(4, dtype=np.float32),
        old_log_probabilities=np.zeros(4, dtype=np.float32),
        old_values=np.zeros(4, dtype=np.float32),
        bootstrap_values=np.zeros(1, dtype=np.float32),
        transitions=transitions,
        outcomes={
            "root_height": np.full(4, 0.75, dtype=np.float32),
            "tilt": np.zeros(4, dtype=np.float32),
        },
        teacher_actions=np.ones(4, dtype=np.float32),
    )
    weights = rollout.policy_batch().teacher_weights
    policy_weights = rollout.policy_batch().policy_weights
    if not np.array_equal(
        weights,
        np.asarray((1.0, 1.0, 1.0, 0.0), dtype=np.float32),
    ):
        raise RuntimeError(
            "clean native realization did not back-label its imagination window"
        )
    if not np.array_equal(
        policy_weights,
        np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
    ):
        raise RuntimeError(
            "shielded imagination was scored with an uncorrected PPO action"
        )
    transitions["physics_error"][1] = 1
    failed = replace(rollout, transitions=transitions)
    if np.any(failed.policy_batch().teacher_weights):
        raise RuntimeError(
            "failed native imagination was published as teacher truth"
        )
    if not np.array_equal(
        failed.policy_batch().policy_weights,
        policy_weights,
    ):
        raise RuntimeError(
            "failed shielded imagination re-entered the PPO actor ratio"
        )

    get_up_steps = 30
    get_up_transitions = np.zeros(get_up_steps, dtype=transitions.dtype)
    get_up_root_height = np.linspace(
        0.1, 0.72, get_up_steps, dtype=np.float32
    )
    get_up_tilt = np.linspace(
        1.4, 0.1, get_up_steps, dtype=np.float32
    )
    get_up_transitions["impact_event_flags"][5:] = 1 << 30
    get_up = replace(
        rollout,
        id="realized_get_up_gate",
        control_step_count=get_up_steps,
        actor_observations=np.zeros(get_up_steps, dtype=np.float32),
        critic_observations=np.zeros(get_up_steps, dtype=np.float32),
        latents=np.zeros(get_up_steps, dtype=np.float32),
        old_log_probabilities=np.zeros(get_up_steps, dtype=np.float32),
        old_values=np.zeros(get_up_steps, dtype=np.float32),
        transitions=get_up_transitions,
        outcomes={
            "root_height": get_up_root_height,
            "tilt": get_up_tilt,
        },
        teacher_actions=np.ones(get_up_steps, dtype=np.float32),
    ).policy_batch()
    if (
        np.any(get_up.teacher_weights <= 0.0)
        or get_up.teacher_weights[-1] <= get_up.teacher_weights[0]
        or np.any(get_up.policy_weights)
    ):
        raise RuntimeError(
            "physical get-up progress did not receive local teacher weight"
        )

    standing_transitions = np.zeros(get_up_steps, dtype=transitions.dtype)
    standing_root_height = np.linspace(
        0.1, 0.68, get_up_steps, dtype=np.float32
    )
    standing_tilt = np.linspace(
        1.4, 0.2, get_up_steps, dtype=np.float32
    )
    standing_transitions["impact_event_flags"][12:] = 1 << 31
    standing = replace(
        rollout,
        id="realized_get_up_progress",
        control_step_count=get_up_steps,
        actor_observations=np.zeros(get_up_steps, dtype=np.float32),
        critic_observations=np.zeros(get_up_steps, dtype=np.float32),
        latents=np.zeros(get_up_steps, dtype=np.float32),
        old_log_probabilities=np.zeros(get_up_steps, dtype=np.float32),
        old_values=np.zeros(get_up_steps, dtype=np.float32),
        transitions=standing_transitions,
        outcomes={
            "root_height": standing_root_height,
            "tilt": standing_tilt,
        },
        teacher_actions=np.ones(get_up_steps, dtype=np.float32),
    ).policy_batch()
    if (
        np.any(standing.teacher_weights <= 0.0)
        or standing.teacher_weights[-1] <= standing.teacher_weights[0]
        or np.any(standing.policy_weights)
    ):
        raise RuntimeError(
            "sustained physics-standing progress did not receive graded credit"
        )

    collapsed_transitions = standing_transitions.copy()
    collapsed_transitions[25:]["impact_event_flags"] = 0
    collapsed = replace(
        rollout,
        id="rejected_get_up_collapse",
        control_step_count=get_up_steps,
        actor_observations=np.zeros(get_up_steps, dtype=np.float32),
        critic_observations=np.zeros(get_up_steps, dtype=np.float32),
        latents=np.zeros(get_up_steps, dtype=np.float32),
        old_log_probabilities=np.zeros(get_up_steps, dtype=np.float32),
        old_values=np.zeros(get_up_steps, dtype=np.float32),
        transitions=collapsed_transitions,
        outcomes={
            "root_height": standing_root_height,
            "tilt": standing_tilt,
        },
        teacher_actions=np.ones(get_up_steps, dtype=np.float32),
    ).policy_batch()
    if (
        np.any(collapsed.teacher_weights[:25] <= 0.0)
        or not np.allclose(
            collapsed.teacher_weights[:25],
            standing.teacher_weights[:25],
        )
        or np.any(collapsed.policy_weights)
    ):
        raise RuntimeError(
            "later collapse erased earlier accepted physical progress"
        )

    foundation_transitions = np.zeros(4, dtype=transitions.dtype)
    foundation_root_height = np.asarray(
        (0.78, 0.77, 0.76, 0.75), dtype=np.float32
    )
    foundation_tilt = np.asarray(
        (0.01, 0.03, 0.06, 0.08), dtype=np.float32
    )
    foundation_transitions["impact_event_flags"] = 1 << 22
    foundation = replace(
        rollout,
        id="foundation_interaction_teacher",
        actor_observations=np.zeros(4, dtype=np.float32),
        critic_observations=np.zeros(4, dtype=np.float32),
        latents=np.zeros(4, dtype=np.float32),
        old_log_probabilities=np.zeros(4, dtype=np.float32),
        old_values=np.zeros(4, dtype=np.float32),
        transitions=foundation_transitions,
        outcomes={
            "root_height": foundation_root_height,
            "tilt": foundation_tilt,
        },
        teacher_actions=np.ones(4, dtype=np.float32),
    ).policy_batch()
    if (
        np.any(foundation.teacher_weights <= 0.0)
        or np.any(foundation.policy_weights)
    ):
        raise RuntimeError(
            "pure InteractionPack control was not isolated to distillation"
        )
    foundation_transitions["done"][2] = 1
    failed_foundation = replace(
        rollout,
        id="failed_foundation_interaction_teacher",
        actor_observations=np.zeros(4, dtype=np.float32),
        critic_observations=np.zeros(4, dtype=np.float32),
        latents=np.zeros(4, dtype=np.float32),
        old_log_probabilities=np.zeros(4, dtype=np.float32),
        old_values=np.zeros(4, dtype=np.float32),
        transitions=foundation_transitions,
        outcomes={
            "root_height": foundation_root_height,
            "tilt": foundation_tilt,
        },
        teacher_actions=np.ones(4, dtype=np.float32),
    ).policy_batch()
    if failed_foundation.teacher_weights[2] != 0.0:
        raise RuntimeError(
            "terminated InteractionPack control became teacher truth"
        )

    navigation_transitions = np.zeros(4, dtype=transitions.dtype)
    navigation_transitions["impact_event_flags"] = 1 << 22
    navigation_transitions["done"][2] = 1
    navigation = replace(
        rollout,
        id="qualified_navigation_teacher",
        actor_observations=np.zeros(4, dtype=np.float32),
        critic_observations=np.zeros(4, dtype=np.float32),
        latents=np.zeros(4, dtype=np.float32),
        old_log_probabilities=np.zeros(4, dtype=np.float32),
        old_values=np.zeros(4, dtype=np.float32),
        transitions=navigation_transitions,
        outcomes={
            "root_height": foundation_root_height,
            "tilt": foundation_tilt,
            "navigation_waypoints_reached": np.asarray(
                (3.0, 4.0, 5.0, 3.0), dtype=np.float32
            ),
            "navigation_completion": np.asarray(
                (0.0, 0.0, 1.0, 0.0), dtype=np.float32
            ),
        },
        teacher_actions=np.ones(4, dtype=np.float32),
    ).policy_batch()
    if (
        np.any(navigation.teacher_weights[:3] <= 0.0)
        or navigation.teacher_weights[3] != 0.0
        or np.any(navigation.policy_weights)
    ):
        raise RuntimeError(
            "successful navigation teacher was not prefix-qualified"
        )
    rejected_navigation_rollout = replace(
        rollout,
        id="rejected_navigation_teacher",
        actor_observations=np.zeros(4, dtype=np.float32),
        critic_observations=np.zeros(4, dtype=np.float32),
        latents=np.zeros(4, dtype=np.float32),
        old_log_probabilities=np.zeros(4, dtype=np.float32),
        old_values=np.zeros(4, dtype=np.float32),
        transitions=navigation_transitions,
        outcomes={
            "root_height": foundation_root_height,
            "tilt": foundation_tilt,
            "navigation_waypoints_reached": np.asarray(
                (3.0, 4.0, 4.0, 3.0), dtype=np.float32
            ),
            "navigation_completion": np.zeros(4, dtype=np.float32),
        },
        teacher_actions=np.ones(4, dtype=np.float32),
    )
    rejected_navigation = rejected_navigation_rollout.policy_batch()
    if np.any(rejected_navigation.teacher_weights):
        raise RuntimeError(
            "partial navigation route became distillation truth"
        )
    staged_navigation = rejected_navigation_rollout.policy_batch(
        navigation_teacher_minimum_waypoint=4
    )
    if (
        np.any(staged_navigation.teacher_weights[:3] <= 0.0)
        or staged_navigation.teacher_weights[3] != 0.0
        or np.any(staged_navigation.policy_weights)
    ):
        raise RuntimeError(
            "staged navigation success was not back-labelled at its explicit waypoint"
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
    check_retention_targets()
    check_realized_imagination_weighting()
    check_resumable_schedule_contracts()

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
    zero_learner = make_learner()
    zero_learner.zero_actor_output()
    zero_means = zero_learner.model.actor_mean(actor)
    mx.eval(zero_means)
    if float(np.max(np.abs(np.asarray(zero_means)))) != 0.0:
        raise RuntimeError(
            "zero-output actor initialization did not publish exact default actions"
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
        teacher_actions=np.ones(
            (sample_count, action_count),
            dtype=np.float32,
        ),
        teacher_weights=np.concatenate(
            (
                np.ones(sample_count // 4, dtype=np.float32),
                np.zeros(
                    sample_count - sample_count // 4,
                    dtype=np.float32,
                ),
            )
        ),
    )
    metrics = learner.update(batch)
    if (
        metrics.get("imagination_fraction", 0.0) <= 0.0
        or metrics.get("imagination_loss", 0.0) <= 0.0
    ):
        raise RuntimeError(
            "successful imagination did not supervise the actor"
        )

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
            expected_actor = learner.model.actor_mean(actor)
            mx.eval(expected_actor)
            learner_state = Path(actor_directory) / "learner.safetensors"
            _write_learner_state(learner, learner_state)
            resumed_expanded = MLXPolicyLearner.from_actor_policy_pack(
                artifact,
                critic_count,
                learner.configuration,
                actor_observation_count=actor_count + 7,
                actor_observation_extension_mean=1.0,
                actor_observation_extension_inverse_standard_deviation=4.0,
                library_path=arguments.library,
            )
            restored = _restore_learner_state(
                resumed_expanded,
                learner_state,
                actor_observation_extension_offset=actor_count,
            )
            resumed_actor = resumed_expanded.model.actor_mean(
                mx.concatenate(
                    (actor, mx.ones((sample_count, 7))),
                    axis=1,
                )
            )
            mx.eval(resumed_actor, resumed_expanded.optimizer.state)
            if (
                not restored or
                not np.array_equal(
                    np.asarray(expected_actor),
                    np.asarray(resumed_actor),
                )
            ):
                raise RuntimeError(
                    "expanded PPO resume changed the preserved actor"
                )
            resumed_optimizer = dict(
                tree_flatten(resumed_expanded.optimizer.state)
            )
            source_optimizer = dict(
                tree_flatten(learner.optimizer.state)
            )
            expanded_moment = np.asarray(
                resumed_optimizer["actor.layers.0.weight.m"]
            )
            source_moment = np.asarray(
                source_optimizer["actor.layers.0.weight.m"]
            )
            if (
                not np.array_equal(
                    expanded_moment[:, :actor_count],
                    source_moment,
                ) or
                np.any(expanded_moment[:, actor_count:] != 0.0)
            ):
                raise RuntimeError(
                    "expanded PPO resume did not preserve Adam moments"
                )
            deployment = learner.write_policy_pack(
                Path(actor_directory) / "deployment.policypack",
                stochastic=False,
                library_path=arguments.library,
            )
            deployed = read_policy_pack(
                deployment,
                library_path=arguments.library,
            )
            if (
                deployed.critic_layers
                or deployed.critic_observation_mean.size
                or deployed.critic_observation_inverse_standard_deviation.size
                or deployed.action_log_standard_deviation.size
            ):
                raise RuntimeError(
                    "deterministic deployment retained training-only state"
                )
            initialized = MLXPolicyLearner.from_actor_policy_pack(
                deployment,
                critic_count,
                learner.configuration,
                library_path=arguments.library,
            )
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
            fresh_critic = MLXPolicyLearner.from_actor_policy_pack(
                artifact,
                critic_count + 7,
                learner.configuration,
                preserve_critic=False,
                actor_observation_count=actor_count,
                library_path=arguments.library,
            )
            fresh_critic_actor = fresh_critic.model.actor_mean(actor)
            mx.eval(fresh_critic_actor)
            if (
                fresh_critic.critic_observation_count != critic_count + 7
                or not np.array_equal(
                    np.asarray(expected_actor),
                    np.asarray(fresh_critic_actor),
                )
            ):
                raise RuntimeError(
                    "fresh-critic actor initialization changed the actor "
                    "or retained the source critic contract"
                )
            expanded = MLXPolicyLearner.from_actor_policy_pack(
                deployment,
                critic_count + 7,
                learner.configuration,
                actor_observation_count=actor_count + 7,
                actor_observation_extension_mean=1.0,
                actor_observation_extension_inverse_standard_deviation=4.0,
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
            expanded_inverse_standard_deviation = np.asarray(
                expanded.model.actor_observation_inverse_standard_deviation
            )
            if not np.array_equal(
                expanded_inverse_standard_deviation[-7:],
                np.full(7, 4.0, dtype=np.float32),
            ):
                raise RuntimeError(
                    "observation expansion inverse standard deviation was not published"
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
            residual_observation_count = 5
            residual_width = 2 * residual_observation_count
            residual = MLXPolicyLearner.from_actor_policy_pack(
                deployment,
                critic_count,
                learner.configuration,
                actor_route_residual_observation_offset=(
                    actor_count - residual_observation_count
                ),
                actor_route_residual_observation_count=(
                    residual_observation_count
                ),
                library_path=arguments.library,
            )
            residual_actor = residual.model.actor_mean(actor)
            mx.eval(residual_actor)
            residual_initialization_error = float(
                np.max(
                    np.abs(
                        np.asarray(expected_actor) -
                        np.asarray(residual_actor)
                    )
                )
            )
            if residual_initialization_error > 1.0e-6:
                raise RuntimeError(
                    "zero-output route residual changed the source actor: "
                    f"{residual_initialization_error}"
                )
            residual_layers = [
                layer
                for layer in residual.model.actor.layers
                if hasattr(layer, "weight")
            ]
            inherited_hidden = [
                layer.output_count for layer in deployed.layers[:-1]
            ]
            if [
                int(layer.weight.shape[0])
                for layer in residual_layers[:-1]
            ] != [size + residual_width for size in inherited_hidden]:
                raise RuntimeError(
                    "route residual did not widen every hidden layer"
                )
            before_residual = [
                np.asarray(layer.weight).copy()
                for layer in residual_layers
            ]
            before_bias = [
                np.asarray(layer.bias).copy()
                for layer in residual_layers
            ]
            before_log_standard_deviation = np.asarray(
                residual.model.log_standard_deviation
            ).copy()
            residual.train_actor_residual_output_only(
                residual_width,
                output_action_indices=(0, 2),
            )
            residual_old_log_probabilities, _, residual_old_values = (
                residual.model.evaluate(actor, critic, latents)
            )
            mx.eval(
                residual_old_log_probabilities,
                residual_old_values,
            )
            residual_advantages = generator.normal(
                0.0,
                1.0,
                sample_count,
            ).astype(np.float32)
            residual.update(
                MLXPolicyBatch(
                    actor_observations=np.asarray(actor),
                    critic_observations=np.asarray(critic),
                    latents=np.asarray(latents),
                    old_log_probabilities=np.asarray(
                        residual_old_log_probabilities
                    ),
                    old_values=np.asarray(residual_old_values),
                    advantages=residual_advantages,
                    returns=np.asarray(residual_old_values) +
                        0.1 * residual_advantages,
                )
            )
            after_residual = [
                np.asarray(layer.weight)
                for layer in residual_layers
            ]
            if not all(
                np.array_equal(before, after)
                for before, after in zip(
                    before_residual[:-1],
                    after_residual[:-1],
                    strict=True,
                )
            ):
                raise RuntimeError(
                    "route residual training changed a hidden carrier layer"
                )
            output_before = before_residual[-1]
            output_after = after_residual[-1]
            if (
                not np.array_equal(
                    output_before[:, :-residual_width],
                    output_after[:, :-residual_width],
                )
                or not np.array_equal(
                    output_before[[index for index in range(action_count)
                                   if index not in (0, 2)], -residual_width:],
                    output_after[[index for index in range(action_count)
                                  if index not in (0, 2)], -residual_width:],
                )
                or np.array_equal(
                    output_before[(0, 2), -residual_width:],
                    output_after[(0, 2), -residual_width:],
                )
            ):
                raise RuntimeError(
                    "route residual training escaped its selected output tail"
                )
            if (
                not all(
                    np.array_equal(before, np.asarray(layer.bias))
                    for before, layer in zip(
                        before_bias,
                        residual_layers,
                        strict=True,
                    )
                )
                or not np.array_equal(
                    before_log_standard_deviation,
                    np.asarray(residual.model.log_standard_deviation),
                )
            ):
                raise RuntimeError(
                    "route residual training changed actor bias or exploration"
                )
            residual_network = MLXPolicyLearner.from_actor_policy_pack(
                deployment,
                critic_count,
                learner.configuration,
                actor_route_residual_observation_offset=(
                    actor_count - residual_observation_count
                ),
                actor_route_residual_observation_count=(
                    residual_observation_count
                ),
                library_path=arguments.library,
            )
            residual_network_layers = [
                layer
                for layer in residual_network.model.actor.layers
                if hasattr(layer, "weight")
            ]
            network_weights_before = [
                np.asarray(layer.weight).copy()
                for layer in residual_network_layers
            ]
            network_biases_before = [
                np.asarray(layer.bias).copy()
                for layer in residual_network_layers
            ]
            network_log_standard_deviation_before = np.asarray(
                residual_network.model.log_standard_deviation
            ).copy()
            residual_network.train_actor_residual_output_only(
                residual_width,
                output_action_indices=(0, 2),
                hidden_observation_range=(
                    actor_count - residual_observation_count,
                    residual_observation_count,
                ),
            )
            for _ in range(2):
                network_old_log_probabilities, _, network_old_values = (
                    residual_network.model.evaluate(actor, critic, latents)
                )
                mx.eval(
                    network_old_log_probabilities,
                    network_old_values,
                )
                network_advantages = generator.normal(
                    0.0,
                    1.0,
                    sample_count,
                ).astype(np.float32)
                residual_network.update(
                    MLXPolicyBatch(
                        actor_observations=np.asarray(actor),
                        critic_observations=np.asarray(critic),
                        latents=np.asarray(latents),
                        old_log_probabilities=np.asarray(
                            network_old_log_probabilities
                        ),
                        old_values=np.asarray(network_old_values),
                        advantages=network_advantages,
                        returns=np.asarray(network_old_values) +
                            0.1 * network_advantages,
                    )
                )
            network_weights_after = [
                np.asarray(layer.weight)
                for layer in residual_network_layers
            ]
            changed_hidden_block = False
            for layer_index, (before, after) in enumerate(
                zip(
                    network_weights_before[:-1],
                    network_weights_after[:-1],
                    strict=True,
                )
            ):
                allowed = np.zeros(before.shape, dtype=bool)
                if layer_index == 0:
                    allowed[
                        -residual_width:,
                        actor_count - residual_observation_count:actor_count,
                    ] = True
                else:
                    allowed[-residual_width:, -residual_width:] = True
                changed = before != after
                if np.any(np.logical_and(changed, ~allowed)):
                    raise RuntimeError(
                        "route residual network training escaped its hidden block"
                    )
                changed_hidden_block = changed_hidden_block or bool(
                    np.any(np.logical_and(changed, allowed))
                )
            if not changed_hidden_block:
                raise RuntimeError(
                    "route residual network did not update a hidden block"
                )
            network_output_allowed = np.zeros(
                network_weights_before[-1].shape,
                dtype=bool,
            )
            network_output_allowed[(0, 2), -residual_width:] = True
            network_output_changed = (
                network_weights_before[-1] != network_weights_after[-1]
            )
            if (
                np.any(
                    np.logical_and(
                        network_output_changed,
                        ~network_output_allowed,
                    )
                )
                or not np.any(
                    np.logical_and(
                        network_output_changed,
                        network_output_allowed,
                    )
                )
                or not all(
                    np.array_equal(before, np.asarray(layer.bias))
                    for before, layer in zip(
                        network_biases_before,
                        residual_network_layers,
                        strict=True,
                    )
                )
                or not np.array_equal(
                    network_log_standard_deviation_before,
                    np.asarray(
                        residual_network.model.log_standard_deviation
                    ),
                )
            ):
                raise RuntimeError(
                    "route residual network changed carrier state, bias, or exploration"
                )
            selected_indices = tuple(range(0, actor_count, 2))
            projection = selected_indices + (None, None, None)
            action_multiplier = np.linspace(
                0.2,
                0.8,
                action_count,
                dtype=np.float32,
            )
            projection_defaults = (
                deployed.effective_observation_mean
                + 0.1 /
                deployed.effective_observation_inverse_standard_deviation
            ).astype(np.float32)
            projected = (
                MLXPolicyLearner.from_projected_actor_policy_pack(
                    deployment,
                    critic_count,
                    projection,
                    learner.configuration,
                    action_multiplier=action_multiplier,
                    source_observation_defaults=projection_defaults,
                    library_path=arguments.library,
                )
            )
            source_at_mean = np.broadcast_to(
                projection_defaults,
                (sample_count, actor_count),
            ).copy()
            source_at_mean[:, selected_indices] = np.asarray(actor)[
                :, selected_indices
            ]
            projected_input = np.concatenate(
                (
                    np.asarray(actor)[:, selected_indices],
                    np.zeros((sample_count, 3), dtype=np.float32),
                ),
                axis=1,
            )
            source_mean = learner.model.actor_mean(mx.array(source_at_mean))
            projected_mean = projected.model.actor_mean(
                mx.array(projected_input)
            )
            mx.eval(source_mean, projected_mean)
            expected_action = action_multiplier * (
                deployed.effective_action_bias[None, :]
                + deployed.effective_action_scale[None, :]
                * np.asarray(source_mean)
            )
            projected_action = (
                projected.action_bias[None, :]
                + projected.action_scale[None, :]
                * np.asarray(projected_mean)
            )
            if float(
                np.max(np.abs(expected_action - projected_action))
            ) > 1.0e-5:
                raise RuntimeError(
                    "actor projection changed the selected source policy"
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
                    "actor_route_residual_initialization_max_error":
                        residual_initialization_error,
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
