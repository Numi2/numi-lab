"""End-to-end MLX rollout and PPO path for the active-encoder world.

This module deliberately targets the contact-free Franka stabilization task
implemented by :mod:`metalrobo.mlx_world`. The legacy ctypes task remains a
validation adapter; no NumPy array enters this rollout, reward, GAE, or update
path.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .mlx_world import (
    MLXCompiledWorld,
    WorldState,
    compile_world,
    initial_state,
    step,
)
from .ppo import ActorCritic, PPOConfig


class MLXRolloutBatch(NamedTuple):
    """Device-backed fixed-shape PPO rollout."""

    observations: mx.array
    latents: mx.array
    old_log_probabilities: mx.array
    old_values: mx.array
    advantages: mx.array
    returns: mx.array
    rewards: mx.array
    dones: mx.array
    physics_errors: mx.array

    def flattened(self) -> dict[str, mx.array]:
        return {
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


class MLXRolloutState(NamedTuple):
    """Explicit state carried between bounded lazy rollout chunks."""

    world: WorldState
    observations: mx.array
    episode_steps: mx.array


class MLXRolloutCollector:
    """Compiled policy -> physics -> reward collector with no host staging."""

    def __init__(
        self,
        world: MLXCompiledWorld,
        model: ActorCritic,
        environment_count: int,
        *,
        gamma: float,
        gae_lambda: float,
        chunk_size: int = 16,
        maximum_episode_steps: int = 256,
    ) -> None:
        if environment_count <= 0:
            raise ValueError("environment_count must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be positive")
        if world.solver_mode != "free_motion_aba":
            raise ValueError(
                "MLXRolloutCollector requires the active-encoder ABA world"
            )
        if world.nq != world.nv:
            raise ValueError(
                "the first MLX PPO task requires scalar fixed-base joints"
            )
        self.world = world
        self.model = model
        self.environment_count = environment_count
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.chunk_size = chunk_size
        self.maximum_episode_steps = maximum_episode_steps
        self.default_q = mx.array(world.default_q, dtype=mx.float32)
        self.default_v = mx.array(world.default_v, dtype=mx.float32)
        effort_limits = mx.array(
            world.effort_limits,
            dtype=mx.float32,
        )
        # A zero metadata limit means unactuated. Never invent authority for
        # that coordinate.
        self.effort_scale = mx.where(
            effort_limits > 0.0,
            effort_limits,
            mx.zeros_like(effort_limits),
        )
        empty = initial_state(world, environment_count)
        self.empty_scene = empty.scene_bodies
        self.empty_rods = empty.rods
        self.empty_cache = empty.solver_cache
        self.empty_tactile = empty.tactile
        self.empty_actuators = empty.actuators
        self._compiled_step = mx.compile(
            self._policy_physics_reward,
            inputs=self.model.state,
        )

    def initial(self) -> MLXRolloutState:
        world_state = initial_state(
            self.world,
            self.environment_count,
        )
        return MLXRolloutState(
            world=world_state,
            observations=mx.concatenate(
                (world_state.q, world_state.v),
                axis=-1,
            ),
            episode_steps=mx.zeros(
                (self.environment_count,),
                dtype=mx.uint32,
            ),
        )

    def _policy_physics_reward(
        self,
        q: mx.array,
        v: mx.array,
        observations: mx.array,
        episode_steps: mx.array,
        noise: mx.array,
    ) -> tuple[mx.array, ...]:
        mean, value = self.model(observations)
        log_std = mx.clip(self.model.log_std, -5.0, 2.0)
        latent = mean + mx.exp(log_std) * noise
        normalized_action = mx.tanh(latent)
        log_probability = self.model.log_probability(
            mean,
            log_std,
            latent,
            normalized_action,
        )
        effort = normalized_action * self.effort_scale
        current = WorldState(
            q=q,
            v=v,
            scene_bodies=self.empty_scene,
            rods=self.empty_rods,
            solver_cache=self.empty_cache,
            tactile=self.empty_tactile,
            actuators=self.empty_actuators,
        )
        physics = step(self.world, current, effort)

        position_error = physics.next_state.q - self.default_q
        position_cost = mx.mean(
            mx.square(position_error),
            axis=-1,
        )
        velocity_cost = mx.mean(
            mx.square(physics.next_state.v),
            axis=-1,
        )
        effort_cost = mx.mean(
            mx.square(normalized_action),
            axis=-1,
        )
        reward = (
            1.0
            - 2.0 * position_cost
            - 0.02 * velocity_cost
            - 0.002 * effort_cost
        )

        next_episode_steps = episode_steps + mx.array(
            1,
            dtype=mx.uint32,
        )
        horizon = next_episode_steps >= self.maximum_episode_steps
        state_limit = mx.max(
            mx.abs(position_error),
            axis=-1,
        ) > 2.5
        done = physics.physics_error | horizon | state_limit
        reset_mask = done[:, None]
        next_q = mx.where(
            reset_mask,
            self.default_q,
            physics.next_state.q,
        )
        next_v = mx.where(
            reset_mask,
            self.default_v,
            physics.next_state.v,
        )
        next_episode_steps = mx.where(
            done,
            mx.zeros_like(next_episode_steps),
            next_episode_steps,
        )
        next_observations = mx.concatenate(
            (next_q, next_v),
            axis=-1,
        )
        return (
            next_q,
            next_v,
            next_observations,
            next_episode_steps,
            latent,
            log_probability,
            value,
            reward,
            done,
            physics.physics_error,
        )

    def collect(
        self,
        state: MLXRolloutState,
        rollout_steps: int,
    ) -> tuple[MLXRolloutState, MLXRolloutBatch]:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        q = state.world.q
        v = state.world.v
        observations = state.observations
        episode_steps = state.episode_steps
        stored_observations: list[mx.array] = []
        stored_latents: list[mx.array] = []
        stored_log_probabilities: list[mx.array] = []
        stored_values: list[mx.array] = []
        stored_rewards: list[mx.array] = []
        stored_dones: list[mx.array] = []
        stored_errors: list[mx.array] = []

        for index in range(rollout_steps):
            policy_observations = observations
            noise = mx.random.normal(
                (
                    self.environment_count,
                    self.world.nv,
                )
            )
            (
                q,
                v,
                observations,
                episode_steps,
                latent,
                log_probability,
                value,
                reward,
                done,
                physics_error,
            ) = self._compiled_step(
                q,
                v,
                observations,
                episode_steps,
                noise,
            )
            stored_observations.append(policy_observations)
            stored_latents.append(latent)
            stored_log_probabilities.append(log_probability)
            stored_values.append(value)
            stored_rewards.append(reward)
            stored_dones.append(done)
            stored_errors.append(physics_error)

            if (index + 1) % self.chunk_size == 0:
                mx.async_eval(
                    q,
                    v,
                    observations,
                    episode_steps,
                    latent,
                    reward,
                )

        _, final_value = self.model(observations)
        advantages: list[mx.array] = [
            mx.zeros_like(final_value)
            for _ in range(rollout_steps)
        ]
        advantage = mx.zeros_like(final_value)
        next_value = final_value
        for index in range(rollout_steps - 1, -1, -1):
            continuing = (
                ~stored_dones[index]
            ).astype(mx.float32)
            delta = (
                stored_rewards[index]
                + self.gamma * next_value * continuing
                - stored_values[index]
            )
            advantage = (
                delta
                + self.gamma
                * self.gae_lambda
                * continuing
                * advantage
            )
            advantages[index] = advantage
            next_value = stored_values[index]

        values = mx.stack(stored_values)
        batch = MLXRolloutBatch(
            observations=mx.stack(stored_observations),
            latents=mx.stack(stored_latents),
            old_log_probabilities=mx.stack(
                stored_log_probabilities
            ),
            old_values=values,
            advantages=mx.stack(advantages),
            returns=mx.stack(advantages) + values,
            rewards=mx.stack(stored_rewards),
            dones=mx.stack(stored_dones),
            physics_errors=mx.stack(stored_errors),
        )
        next_state = MLXRolloutState(
            world=WorldState(
                q=q,
                v=v,
                scene_bodies=self.empty_scene,
                rods=self.empty_rods,
                solver_cache=self.empty_cache,
                tactile=self.empty_tactile,
                actuators=self.empty_actuators,
            ),
            observations=observations,
            episode_steps=episode_steps,
        )
        mx.async_eval(
            batch.observations,
            batch.returns,
            batch.physics_errors,
            next_state.world.q,
            next_state.world.v,
        )
        return next_state, batch


def _metric_values(
    metrics: dict[str, mx.array],
) -> dict[str, float]:
    """Convert only at the declared optimizer/logging boundary."""

    return {key: float(value.item()) for key, value in metrics.items()}


class MLXPPOTrainer:
    """PPO whose simulation, rollout storage, rewards, and updates stay in MLX."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        metallib_path: str | None = None,
        rollout_chunk_size: int = 16,
        maximum_episode_steps: int = 256,
        physics_substeps: int = 4,
    ) -> None:
        config.validate()
        self.config = config
        mx.random.seed(config.seed)
        self.world = compile_world(
            "franka",
            physics_substeps=physics_substeps,
            metallib_path=metallib_path or "",
        )
        self.task_name = "franka_joint_stabilization_v1"
        self.observation_size = self.world.nq + self.world.nv
        self.action_size = self.world.nv
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
        self.collector = MLXRolloutCollector(
            self.world,
            self.model,
            config.environment_count,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            chunk_size=rollout_chunk_size,
            maximum_episode_steps=maximum_episode_steps,
        )
        self.rollout_state = self.collector.initial()
        self.iteration = 0
        self.environment_steps = 0
        self._loss_and_grad = nn.value_and_grad(
            self.model,
            self._loss,
        )

    def _loss(
        self,
        observations: mx.array,
        latents: mx.array,
        old_log_probabilities: mx.array,
        old_values: mx.array,
        advantages: mx.array,
        returns: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        log_probabilities, entropy, values = self.model.evaluate(
            observations,
            latents,
        )
        log_ratio = log_probabilities - old_log_probabilities
        ratio = mx.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = mx.clip(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio,
        ) * advantages
        policy_loss = -mx.mean(mx.minimum(unclipped, clipped))
        clipped_values = old_values + mx.clip(
            values - old_values,
            -self.config.clip_ratio,
            self.config.clip_ratio,
        )
        value_loss = 0.5 * mx.mean(
            mx.maximum(
                mx.square(values - returns),
                mx.square(clipped_values - returns),
            )
        )
        entropy_mean = mx.mean(entropy)
        total = (
            policy_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy_mean
        )
        return total, {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "approx_kl": mx.mean((ratio - 1.0) - log_ratio),
            "clip_fraction": mx.mean(
                (
                    mx.abs(ratio - 1.0)
                    > self.config.clip_ratio
                ).astype(mx.float32)
            ),
        }

    def _update(
        self,
        rollout: MLXRolloutBatch,
    ) -> dict[str, float]:
        batch = rollout.flattened()
        advantages = batch["advantages"]
        batch["advantages"] = (
            advantages - mx.mean(advantages)
        ) / (mx.std(advantages) + 1e-8)
        sample_count = int(batch["advantages"].shape[0])
        minibatch_size = min(
            self.config.minibatch_size,
            sample_count,
        )
        metric_sums: dict[str, float] = {}
        updates = 0
        early_stop = False
        start = time.perf_counter()
        for _ in range(self.config.update_epochs):
            permutation = mx.random.permutation(sample_count)
            for offset in range(
                0,
                sample_count,
                minibatch_size,
            ):
                indices = permutation[
                    offset : offset + minibatch_size
                ]
                (loss, metrics), gradients = self._loss_and_grad(
                    batch["observations"][indices],
                    batch["latents"][indices],
                    batch["old_log_probabilities"][indices],
                    batch["old_values"][indices],
                    batch["advantages"][indices],
                    batch["returns"][indices],
                )
                gradients, gradient_norm = optim.clip_grad_norm(
                    gradients,
                    self.config.max_gradient_norm,
                )
                self.optimizer.update(self.model, gradients)
                metrics["gradient_norm"] = gradient_norm
                mx.eval(
                    loss,
                    metrics,
                    self.model.parameters(),
                    self.optimizer.state,
                )
                numeric = _metric_values(metrics)
                for key, value in numeric.items():
                    metric_sums[key] = (
                        metric_sums.get(key, 0.0) + value
                    )
                updates += 1
                if (
                    self.config.target_kl is not None
                    and numeric["approx_kl"]
                    > self.config.target_kl
                ):
                    early_stop = True
                    break
            if early_stop:
                break
        result = {
            key: value / max(updates, 1)
            for key, value in metric_sums.items()
        }
        result["update_seconds"] = (
            time.perf_counter() - start
        )
        result["minibatch_updates"] = float(updates)
        result["kl_early_stop"] = float(early_stop)
        return result

    def train(
        self,
        *,
        resume: str | Path | None = None,
    ) -> Path:
        if resume is not None:
            self.load_checkpoint(resume)
        checkpoint: Path | None = None
        for iteration in range(
            self.iteration + 1,
            self.config.iterations + 1,
        ):
            self.iteration = iteration
            if self.config.anneal_learning_rate:
                remaining = (
                    1.0
                    - (iteration - 1.0)
                    / self.config.iterations
                )
                self.optimizer.learning_rate = (
                    self.config.learning_rate * remaining
                )

            rollout_start = time.perf_counter()
            self.rollout_state, rollout = (
                self.collector.collect(
                    self.rollout_state,
                    self.config.rollout_steps,
                )
            )
            # Declared rollout/logging boundary: all prior async chunks finish
            # here, never inside a physics step.
            mx.eval(
                rollout.rewards,
                rollout.physics_errors,
                self.rollout_state.world.q,
                self.rollout_state.world.v,
            )
            rollout_seconds = (
                time.perf_counter() - rollout_start
            )
            update_metrics = self._update(rollout)
            added_steps = (
                self.config.rollout_steps
                * self.config.environment_count
            )
            self.environment_steps += added_steps
            mean_reward = float(
                mx.mean(rollout.rewards).item()
            )
            physics_errors = int(
                mx.sum(
                    rollout.physics_errors.astype(mx.uint32)
                ).item()
            )
            report: dict[str, Any] = {
                "iteration": iteration,
                "environment_steps": self.environment_steps,
                "backend": "mlx_active_encoder",
                "task": self.task_name,
                "rollout_seconds": rollout_seconds,
                "rollout_env_steps_per_second": (
                    added_steps
                    / max(rollout_seconds, 1e-9)
                ),
                "mean_step_reward": mean_reward,
                "physics_errors": physics_errors,
                **update_metrics,
            }
            print(
                json.dumps(
                    report,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            if (
                self.config.checkpoint_interval
                and iteration
                % self.config.checkpoint_interval
                == 0
            ):
                checkpoint = self.save_checkpoint()
        if checkpoint is None or self.iteration % max(
            self.config.checkpoint_interval,
            1,
        ):
            checkpoint = self.save_checkpoint()
        return checkpoint

    def save_checkpoint(
        self,
        directory: str | Path | None = None,
    ) -> Path:
        root = Path(
            directory or self.config.checkpoint_directory
        ).expanduser()
        checkpoint = root / (
            f"checkpoint-{self.iteration:06d}"
        )
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(
            str(checkpoint / "model.safetensors")
        )
        mx.save_safetensors(
            str(checkpoint / "optimizer.safetensors"),
            dict(tree_flatten(self.optimizer.state)),
        )
        state = {
            "format_version": 2,
            "backend": "mlx_active_encoder",
            "task": self.task_name,
            "iteration": self.iteration,
            "environment_steps": self.environment_steps,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "physics_substeps": self.world.physics_substeps,
            "ppo_config": asdict(self.config),
        }
        tactile_sensor_count = int(
            getattr(self.world, "tactile_sensor_count", 0)
        )
        if tactile_sensor_count:
            metadata_text = str(
                self.world.tactile_observation_metadata_json
            )
            if not metadata_text:
                raise RuntimeError(
                    "authored tactile world did not publish its "
                    "observation contract"
                )
            metadata = json.loads(metadata_text)
            fingerprint = metadata.get("fingerprint")
            selection = tuple(
                getattr(
                    self,
                    "tactile_observation_selection",
                    (),
                )
            )
            if not fingerprint or not selection:
                raise RuntimeError(
                    "tactile checkpoint requires a fingerprinted "
                    "named observation selection"
                )
            state["tactile_observation_fingerprint"] = fingerprint
            state["tactile_observation_selection"] = list(selection)
            (checkpoint / "tactile_observation.json").write_text(
                metadata_text.rstrip() + "\n",
                encoding="utf-8",
            )
        (checkpoint / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def load_checkpoint(
        self,
        directory: str | Path,
    ) -> None:
        checkpoint = Path(directory).expanduser()
        state = json.loads(
            (checkpoint / "state.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            state.get("format_version") != 2
            or state.get("backend") != "mlx_active_encoder"
            or state.get("task") != self.task_name
        ):
            raise ValueError(
                "checkpoint task does not match this MLX "
                f"active-encoder trainer ({self.task_name})"
            )
        expected = (self.observation_size, self.action_size)
        actual = (
            state.get("observation_size"),
            state.get("action_size"),
        )
        if actual != expected:
            raise ValueError(
                f"checkpoint model shape {actual} "
                f"does not match {expected}"
            )
        saved_hidden = tuple(
            state["ppo_config"]["hidden_sizes"]
        )
        if saved_hidden != self.config.hidden_sizes:
            raise ValueError(
                f"checkpoint hidden sizes {saved_hidden} "
                f"do not match {self.config.hidden_sizes}"
            )
        tactile_sensor_count = int(
            getattr(self.world, "tactile_sensor_count", 0)
        )
        if tactile_sensor_count:
            metadata_text = str(
                self.world.tactile_observation_metadata_json
            )
            metadata = json.loads(metadata_text)
            expected_fingerprint = metadata.get("fingerprint")
            saved_fingerprint = state.get(
                "tactile_observation_fingerprint"
            )
            expected_selection = tuple(
                getattr(
                    self,
                    "tactile_observation_selection",
                    (),
                )
            )
            saved_selection = tuple(
                state.get("tactile_observation_selection", ())
            )
            checkpoint_metadata = json.loads(
                (
                    checkpoint / "tactile_observation.json"
                ).read_text(encoding="utf-8")
            )
            if (
                not expected_fingerprint
                or saved_fingerprint != expected_fingerprint
                or checkpoint_metadata.get("fingerprint")
                != expected_fingerprint
                or saved_selection != expected_selection
            ):
                raise ValueError(
                    "checkpoint tactile observation contract does "
                    "not match this authored world"
                )
        self.model.load_weights(
            str(checkpoint / "model.safetensors")
        )
        optimizer_flat = mx.load(
            str(checkpoint / "optimizer.safetensors")
        )
        self.optimizer.state = tree_unflatten(
            list(optimizer_flat.items())
        )
        self.iteration = int(state["iteration"])
        self.environment_steps = int(
            state["environment_steps"]
        )
        mx.eval(
            self.model.parameters(),
            self.optimizer.state,
        )


__all__ = [
    "MLXPPOTrainer",
    "MLXRolloutBatch",
    "MLXRolloutCollector",
    "MLXRolloutState",
]
