"""MLX PPO over replay-aligned, adaptively sampled Franka world families."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .mlx_ppo import MLXPPOTrainer
from .mlx_manipulation import (
    FrankaPickPlaceTaskState,
    adapt_cartesian_action,
    franka_kinematics,
    initial_franka_pick_place_task,
    select_franka_pick_place_task,
    step_franka_pick_place_task,
    task_evidence_values,
)
from .mlx_r2s2r import (
    CompactedEpisodeRecords,
    EpisodeAccumulator,
    accumulate_episode_evidence,
    compact_completed_episodes,
    initial_episode_accumulator,
    reset_episode_accumulator,
)
from .mlx_world import (
    ControllerDelayState,
    MLXCompiledWorld,
    SampledWorldFamilyState,
    compile_world,
    reset_sampled_world_family,
    sampled_state_from_world_family,
    step_sampled_world_family,
)
from .mlx_world_step import WorldStepResult
from .ppo import ActorCritic, PPOConfig


class MLXFamilyRolloutState(NamedTuple):
    sampled: SampledWorldFamilyState
    delay: ControllerDelayState
    task: FrankaPickPlaceTaskState
    episode_counter: mx.array
    observations: mx.array
    episode_steps: mx.array
    accumulator: EpisodeAccumulator

    @property
    def world(self):
        return self.sampled.world


class MLXFamilyRolloutBatch(NamedTuple):
    observations: mx.array
    latents: mx.array
    old_log_probabilities: mx.array
    old_values: mx.array
    advantages: mx.array
    returns: mx.array
    rewards: mx.array
    dones: mx.array
    physics_errors: mx.array
    valid_steps: mx.array
    completed: CompactedEpisodeRecords

    def flattened(self) -> dict[str, mx.array]:
        flattened = {
            "observations": self.observations.reshape(
                (-1, self.observations.shape[-1])
            ),
            "latents": self.latents.reshape(
                (-1, self.latents.shape[-1])
            ),
            "old_log_probabilities": self.old_log_probabilities.reshape(
                (-1,)
            ),
            "old_values": self.old_values.reshape((-1,)),
            "advantages": self.advantages.reshape((-1,)),
            "returns": self.returns.reshape((-1,)),
        }
        valid = self.valid_steps.reshape((-1,)).astype(mx.bool_)
        valid_count = mx.sum(valid.astype(mx.uint32))
        mx.eval(valid_count)
        count = int(valid_count.item())
        if count == 0:
            raise RuntimeError(
                "rollout contains no physically valid policy transitions"
            )
        order = mx.argsort((~valid).astype(mx.uint32))[:count]
        return {
            name: values[order]
            for name, values in flattened.items()
        }


def _family_observations(
    sampled: SampledWorldFamilyState,
    task: FrankaPickPlaceTaskState,
) -> mx.array:
    scene = sampled.world.scene_bodies
    object_position = scene.position[:, 0, :3]
    object_velocity = scene.linear_velocity[:, 0, :3]
    target_position = scene.position[:, 2, :3]
    relative_target = target_position - object_position
    tool_position = franka_kinematics(sampled.world.q).position
    relative_tool = object_position - tool_position
    normalized_phase = (
        task.phase.astype(mx.float32) / 8.0
    )[:, None]
    return mx.concatenate(
        (
            sampled.world.q,
            sampled.world.v,
            object_position,
            object_velocity,
            target_position,
            relative_target,
            tool_position,
            relative_tool,
            normalized_phase,
        ),
        axis=-1,
    )


class MLXWorldFamilyRolloutCollector:
    """Policy -> varied controller -> contact world -> task -> compaction."""

    def __init__(
        self,
        world: MLXCompiledWorld,
        model: ActorCritic,
        family,
        *,
        gamma: float,
        gae_lambda: float,
        chunk_size: int = 16,
        maximum_episode_steps: int = 256,
        sampling_mode: str = "curriculum",
        seed: int = 1,
    ) -> None:
        if (
            family.layout.capacity < world.environment_capacity
            or chunk_size <= 0
            or maximum_episode_steps <= 0
            or sampling_mode not in {"coverage", "curriculum"}
        ):
            raise ValueError("world-family rollout configuration is invalid")
        if (
            not world.contact_supported
            or world.scene_body_count < 3
            or world.nq != world.nv
        ):
            raise ValueError(
                "family PPO requires the fixed-base Franka pick-place world"
            )
        self.world = world
        self.model = model
        self.family = family
        self.environment_count = int(world.environment_capacity)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.chunk_size = int(chunk_size)
        self.maximum_episode_steps = int(maximum_episode_steps)
        self.sampling_mode = sampling_mode
        self.seed = int(seed)
        self.sample_generation = 0
        self.episode_counter_base = 0
        maximum_latency = 0.025
        self.maximum_delay_steps = max(
            1,
            int(
                maximum_latency / float(world.control_timestep)
                + 1.5
            ),
        )
        self._compiled_step = mx.compile(
            self._policy_physics_task,
            inputs=self.model.state,
        )

    def _sample_replacement(
        self,
    ) -> tuple[SampledWorldFamilyState, mx.array]:
        episode_counter_start = self.episode_counter_base
        self.family.sample(
            self.environment_count,
            seed=self.seed + self.sample_generation,
            mode=self.sampling_mode,
            episode_counter=episode_counter_start,
        )
        self.sample_generation += 1
        self.episode_counter_base += self.environment_count
        return (
            sampled_state_from_world_family(
                self.world,
                self.family,
            ),
            mx.arange(
                episode_counter_start,
                episode_counter_start + self.environment_count,
                dtype=mx.uint32,
            ),
        )

    def _delay_for(
        self,
        sampled: SampledWorldFamilyState,
    ) -> ControllerDelayState:
        return ControllerDelayState(
            history=mx.broadcast_to(
                sampled.world.q[:, None, :],
                (
                    self.environment_count,
                    self.maximum_delay_steps + 1,
                    self.world.nv,
                ),
            )
        )

    def initial(self) -> MLXFamilyRolloutState:
        sampled, episode_counter = self._sample_replacement()
        task = initial_franka_pick_place_task(sampled)
        state = MLXFamilyRolloutState(
            sampled=sampled,
            delay=self._delay_for(sampled),
            task=task,
            episode_counter=episode_counter,
            observations=_family_observations(sampled, task),
            episode_steps=mx.zeros(
                (self.environment_count,),
                dtype=mx.uint32,
            ),
            accumulator=initial_episode_accumulator(
                self.environment_count
            ),
        )
        # Materialize the borrowed family import before the next native sample
        # reuses its private reset arena.
        mx.eval(
            state.sampled.world.q,
            state.sampled.world.scene_bodies.position,
            state.sampled.scenarios.headers,
            state.sampled.parameters.body_values,
            state.sampled.parameters.controller_values,
        )
        return state

    def _policy_physics_task(
        self,
        state: MLXFamilyRolloutState,
        replacement: SampledWorldFamilyState,
        replacement_episode_counter: mx.array,
        noise: mx.array,
    ) -> tuple[
        MLXFamilyRolloutState,
        mx.array,
        mx.array,
        mx.array,
        WorldStepResult,
    ]:
        mean, value = self.model(state.observations)
        log_std = mx.clip(self.model.log_std, -5.0, 2.0)
        latent = mean + mx.exp(log_std) * noise
        normalized_action = mx.tanh(latent)
        log_probability = self.model.log_probability(
            mean,
            log_std,
            latent,
            normalized_action,
        )
        adapted = adapt_cartesian_action(
            state.sampled.world.q,
            normalized_action,
            control_period_seconds=self.world.control_timestep,
        )
        stepped = step_sampled_world_family(
            self.world,
            state.sampled,
            adapted.joint_targets,
            state.delay,
            control_period_seconds=self.world.control_timestep,
        )
        next_sampled = stepped.next_state
        physics_valid = ~stepped.physics.physics_error
        next_task, task_evidence = step_franka_pick_place_task(
            state.task,
            next_sampled,
            stepped.physics.contacts,
            physics_valid=physics_valid,
        )
        success = task_evidence.success
        state_limit = task_evidence.safety_margin < 0.0
        next_episode_steps = (
            state.episode_steps
            + mx.array(1, dtype=mx.uint32)
        )
        horizon = (
            next_episode_steps >= self.maximum_episode_steps
        )
        done = (
            stepped.physics.physics_error
            | success
            | state_limit
            | horizon
        )
        reward = (
            task_evidence.reward
            - 0.001
            * mx.mean(normalized_action * normalized_action, axis=-1)
        )
        reward = mx.where(
            physics_valid,
            reward,
            mx.zeros_like(reward),
        )
        valid_contacts = stepped.physics.contacts.mask.astype(
            mx.float32
        )
        contact_load = (
            mx.sum(
                mx.abs(stepped.physics.contacts.values[:, :, 7])
                * valid_contacts,
                axis=-1,
            )
            / float(self.world.control_timestep)
        )
        contact_count = mx.sum(valid_contacts, axis=-1).astype(
            mx.uint32
        )
        accumulator = accumulate_episode_evidence(
            state.accumulator,
            reward=reward,
            visibility=mx.ones_like(reward),
            contact_load=contact_load,
            valid_contact_count=contact_count,
            control_period_seconds=self.world.control_timestep,
        )
        termination = mx.where(
            success,
            mx.array(0, dtype=mx.uint32),
            mx.where(
                stepped.physics.physics_error,
                mx.array(3, dtype=mx.uint32),
                mx.where(
                    horizon,
                    mx.array(1, dtype=mx.uint32),
                    mx.array(4, dtype=mx.uint32),
                ),
            ),
        )
        failure_mask = task_evidence.failure_mask
        recordable_done = done & physics_valid
        completed = compact_completed_episodes(
            accumulator,
            scenario_headers=state.sampled.scenarios.headers,
            scenario_values=state.sampled.scenarios.values,
            scenario_identities=state.sampled.scenarios.identities,
            completed=recordable_done,
            success=success,
            termination=termination,
            physics_status=stepped.physics.status[:, 0],
            failure_mask_low=failure_mask,
            failure_mask_high=mx.zeros_like(failure_mask),
            task_margin=task_evidence.task_margin,
            safety_margin=task_evidence.safety_margin,
        )
        selected = reset_sampled_world_family(
            next_sampled,
            replacement,
            done,
        )
        delay = ControllerDelayState(
            history=mx.where(
                done[:, None, None],
                self._delay_for(replacement).history,
                stepped.delay_state.history,
            )
        )
        selected_task = select_franka_pick_place_task(
            next_task,
            initial_franka_pick_place_task(replacement),
            done,
        )
        selected_episode_counter = mx.where(
            done,
            replacement_episode_counter,
            state.episode_counter,
        )
        next_steps = mx.where(
            done,
            mx.zeros_like(next_episode_steps),
            next_episode_steps,
        )
        next_state = MLXFamilyRolloutState(
            sampled=selected,
            delay=delay,
            task=selected_task,
            episode_counter=selected_episode_counter,
            observations=_family_observations(
                selected,
                selected_task,
            ),
            episode_steps=next_steps,
            accumulator=reset_episode_accumulator(
                accumulator,
                done,
            ),
        )
        result = WorldStepResult(
            next_state=selected,
            deployable_observations=next_state.observations,
            privileged_observations=mx.concatenate(
                (
                    selected.world.q,
                    selected.world.v,
                    selected.world.scene_bodies.position.reshape(
                        (self.environment_count, -1)
                    ),
                    selected.world.scene_bodies.linear_velocity.reshape(
                        (self.environment_count, -1)
                    ),
                    task_evidence_values(task_evidence),
                ),
                axis=-1,
            ),
            task_phase=next_task.phase,
            task_evidence=task_evidence_values(task_evidence),
            reward=reward,
            terminated=done,
            termination=termination,
            scenario_state=state.sampled.scenarios,
            episode_counter=state.episode_counter,
            recurrent_state=mx.zeros(
                (self.environment_count, 0),
                dtype=mx.float32,
            ),
            physics_status=stepped.physics.status[:, 0],
            valid=physics_valid,
            completed=completed,
        )
        return (
            next_state,
            latent,
            log_probability,
            value,
            result,
        )

    def collect(
        self,
        state: MLXFamilyRolloutState,
        rollout_steps: int,
    ) -> tuple[MLXFamilyRolloutState, MLXFamilyRolloutBatch]:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        replacement, replacement_episode_counter = (
            self._sample_replacement()
        )
        observations = []
        latents = []
        log_probabilities = []
        values = []
        rewards = []
        dones = []
        errors = []
        valid_steps = []
        records = []
        for index in range(rollout_steps):
            policy_observations = state.observations
            noise = mx.random.normal(
                (self.environment_count, int(self.model.log_std.shape[0]))
            )
            (
                state,
                latent,
                log_probability,
                value,
                result,
            ) = self._compiled_step(
                state,
                replacement,
                replacement_episode_counter,
                noise,
            )
            observations.append(policy_observations)
            latents.append(latent)
            log_probabilities.append(log_probability)
            values.append(value)
            rewards.append(result.reward)
            dones.append(result.terminated)
            errors.append(~result.valid)
            valid_steps.append(result.valid)
            records.append(result.completed)
            if (index + 1) % self.chunk_size == 0:
                mx.async_eval(
                    state.sampled.world.q,
                    state.observations,
                    result.reward,
                    result.completed.valid_count,
                )

        _, final_value = self.model(state.observations)
        advantages = [
            mx.zeros_like(final_value)
            for _ in range(rollout_steps)
        ]
        advantage = mx.zeros_like(final_value)
        next_value = final_value
        for index in range(rollout_steps - 1, -1, -1):
            continuing = (~dones[index]).astype(mx.float32)
            delta = (
                rewards[index]
                + self.gamma * next_value * continuing
                - values[index]
            )
            advantage = (
                delta
                + self.gamma
                * self.gae_lambda
                * continuing
                * advantage
            )
            advantages[index] = advantage
            next_value = values[index]

        stacked_records = CompactedEpisodeRecords(
            *(
                mx.stack(
                    [getattr(record, field) for record in records]
                )
                for field in CompactedEpisodeRecords._fields
            )
        )
        value_tensor = mx.stack(values)
        batch = MLXFamilyRolloutBatch(
            observations=mx.stack(observations),
            latents=mx.stack(latents),
            old_log_probabilities=mx.stack(log_probabilities),
            old_values=value_tensor,
            advantages=mx.stack(advantages),
            returns=mx.stack(advantages) + value_tensor,
            rewards=mx.stack(rewards),
            dones=mx.stack(dones),
            physics_errors=mx.stack(errors),
            valid_steps=mx.stack(valid_steps),
            completed=stacked_records,
        )
        mx.async_eval(
            batch.observations,
            batch.returns,
            batch.physics_errors,
            batch.completed.valid_count,
            state.sampled.world.q,
            state.sampled.world.v,
        )
        return state, batch


class MLXWorldFamilyPPOTrainer(MLXPPOTrainer):
    """PPO trainer whose worlds come from an aligned adaptive family."""

    def __init__(
        self,
        config: PPOConfig,
        family,
        *,
        metallib_path: str | None = None,
        rollout_chunk_size: int = 16,
        maximum_episode_steps: int = 256,
        physics_substeps: int = 4,
        sampling_mode: str = "curriculum",
    ) -> None:
        config.validate()
        self.config = config
        self.family = family
        mx.random.seed(config.seed)
        self.world = compile_world(
            "franka",
            scene="pick_place",
            environment_capacity=config.environment_count,
            solver_mode="throughput_tgs",
            actuation_mode="implicit_position",
            physics_substeps=physics_substeps,
            metallib_path=metallib_path or "",
        )
        self.task_name = "franka_pick_place_family_state_v2"
        self.observation_size = self.world.nq + self.world.nv + 19
        self.action_size = 7
        self.model = ActorCritic(
            self.observation_size,
            self.action_size,
            config.hidden_sizes,
            config.initial_log_std,
        )
        self.optimizer = optim.Adam(
            learning_rate=config.learning_rate,
            bias_correction=True,
        )
        self.optimizer.init(self.model.trainable_parameters())
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.collector = MLXWorldFamilyRolloutCollector(
            self.world,
            self.model,
            family,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            chunk_size=rollout_chunk_size,
            maximum_episode_steps=maximum_episode_steps,
            sampling_mode=sampling_mode,
            seed=config.seed,
        )
        self.rollout_state = self.collector.initial()
        self.iteration = 0
        self.environment_steps = 0
        self._loss_and_grad = nn.value_and_grad(
            self.model,
            self._loss,
        )

    def close(self) -> None:
        self.family.close()


__all__ = [
    "MLXFamilyRolloutBatch",
    "MLXFamilyRolloutState",
    "MLXWorldFamilyPPOTrainer",
    "MLXWorldFamilyRolloutCollector",
]
