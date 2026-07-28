"""Continuous-control PPO implemented directly with MLX."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .env import FrankaEnv


def _activation() -> nn.Module:
    return nn.Tanh()


def _mlp(
    input_size: int, output_size: int, hidden_sizes: tuple[int, ...]
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_size
    for width in hidden_sizes:
        layers.extend([nn.Linear(previous, width), _activation()])
        previous = width
    layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Independent actor and critic MLPs with a squashed Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(width <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer widths")
        self.actor = _mlp(observation_size, action_size, hidden_sizes)
        self.critic = _mlp(observation_size, 1, hidden_sizes)
        self.log_std = mx.full((action_size,), initial_log_std, dtype=mx.float32)

    def __call__(self, observations: mx.array) -> tuple[mx.array, mx.array]:
        mean = self.actor(observations)
        value = self.critic(observations).squeeze(-1)
        return mean, value

    def sample(
        self, observations: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        mean, value = self(observations)
        log_std = mx.clip(self.log_std, -5.0, 2.0)
        latent = mean + mx.exp(log_std) * mx.random.normal(mean.shape)
        action = mx.tanh(latent)
        log_probability = self.log_probability(mean, log_std, latent, action)
        return action, latent, log_probability, value

    def deterministic(self, observations: mx.array) -> tuple[mx.array, mx.array]:
        mean, value = self(observations)
        return mx.tanh(mean), value

    @staticmethod
    def log_probability(
        mean: mx.array,
        log_std: mx.array,
        latent: mx.array,
        action: mx.array,
    ) -> mx.array:
        standardized = (latent - mean) * mx.exp(-log_std)
        gaussian = (
            -0.5 * mx.square(standardized)
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_jacobian = mx.log(mx.maximum(1.0 - mx.square(action), 1e-6))
        return mx.sum(gaussian - log_jacobian, axis=-1)

    def evaluate(
        self, observations: mx.array, latents: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        mean, value = self(observations)
        log_std = mx.clip(self.log_std, -5.0, 2.0)
        action = mx.tanh(latents)
        log_probability = self.log_probability(mean, log_std, latents, action)
        base_entropy = mx.sum(
            log_std + 0.5 * (1.0 + math.log(2.0 * math.pi)), axis=-1
        )
        return log_probability, base_entropy, value


@dataclass(slots=True)
class PPOConfig:
    environment_count: int = 1024
    rollout_steps: int = 64
    iterations: int = 1000
    update_epochs: int = 4
    minibatch_size: int = 8192
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_gradient_norm: float = 1.0
    target_kl: float | None = 0.02
    initial_log_std: float = -0.5
    anneal_learning_rate: bool = True
    seed: int = 1
    checkpoint_interval: int = 10
    checkpoint_directory: str = "runs/franka"

    def validate(self) -> None:
        integer_fields = {
            "environment_count": self.environment_count,
            "rollout_steps": self.rollout_steps,
            "iterations": self.iterations,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be positive")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval cannot be negative")


class RolloutBuffer:
    """Fixed-size host buffer for one synchronous vector rollout."""

    def __init__(
        self, steps: int, environments: int, observation_size: int, action_size: int
    ) -> None:
        self.observations = np.empty(
            (steps, environments, observation_size), dtype=np.float32
        )
        self.latents = np.empty((steps, environments, action_size), dtype=np.float32)
        self.log_probabilities = np.empty(
            (steps, environments), dtype=np.float32
        )
        self.rewards = np.empty((steps, environments), dtype=np.float32)
        self.dones = np.empty((steps, environments), dtype=np.bool_)
        self.values = np.empty((steps, environments), dtype=np.float32)
        self.advantages = np.empty((steps, environments), dtype=np.float32)
        self.returns = np.empty((steps, environments), dtype=np.float32)

    def finish(
        self,
        final_value: np.ndarray,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        advantage = np.zeros_like(final_value, dtype=np.float32)
        next_value = final_value.astype(np.float32, copy=False)
        for step in range(self.rewards.shape[0] - 1, -1, -1):
            continuing = 1.0 - self.dones[step].astype(np.float32)
            delta = (
                self.rewards[step]
                + gamma * next_value * continuing
                - self.values[step]
            )
            advantage = delta + gamma * gae_lambda * continuing * advantage
            self.advantages[step] = advantage
            next_value = self.values[step]
        self.returns[:] = self.advantages + self.values

    def flattened(self) -> dict[str, np.ndarray]:
        return {
            "observations": self.observations.reshape(
                -1, self.observations.shape[-1]
            ),
            "latents": self.latents.reshape(-1, self.latents.shape[-1]),
            "old_log_probabilities": self.log_probabilities.reshape(-1),
            "old_values": self.values.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
        }


def _arrays_to_float(metrics: dict[str, mx.array]) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in metrics.items()}


class PPOTrainer:
    """Synchronous on-policy trainer for one vectorized MetalRobo runtime."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        library_path: str | None = None,
        metallib_path: str | None = None,
    ) -> None:
        config.validate()
        self.config = config
        np.random.seed(config.seed)
        mx.random.seed(config.seed)
        self._rng = np.random.default_rng(config.seed)
        self.env = FrankaEnv(
            config.environment_count,
            seed=config.seed,
            library_path=library_path,
            metallib_path=metallib_path,
        )
        self.model = ActorCritic(
            self.env.runtime.observation_count,
            self.env.runtime.action_count,
            config.hidden_sizes,
            config.initial_log_std,
        )
        self.optimizer = optim.Adam(
            learning_rate=config.learning_rate, bias_correction=True
        )
        self.optimizer.init(self.model.trainable_parameters())
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.buffer = RolloutBuffer(
            config.rollout_steps,
            self.env.num_envs,
            self.env.runtime.observation_count,
            self.env.runtime.action_count,
        )
        self.iteration = 0
        self.environment_steps = 0
        self._completed_returns: list[float] = []
        self._completed_lengths: list[int] = []
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)

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
            observations, latents
        )
        log_ratio = log_probabilities - old_log_probabilities
        ratio = mx.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = mx.clip(
            ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
        ) * advantages
        policy_loss = -mx.mean(mx.minimum(unclipped, clipped))

        clipped_values = old_values + mx.clip(
            values - old_values,
            -self.config.clip_ratio,
            self.config.clip_ratio,
        )
        value_error = mx.square(values - returns)
        clipped_value_error = mx.square(clipped_values - returns)
        value_loss = 0.5 * mx.mean(mx.maximum(value_error, clipped_value_error))
        entropy_mean = mx.mean(entropy)
        total = (
            policy_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy_mean
        )
        metrics = {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "approx_kl": mx.mean((ratio - 1.0) - log_ratio),
            "clip_fraction": mx.mean(
                (mx.abs(ratio - 1.0) > self.config.clip_ratio).astype(mx.float32)
            ),
        }
        return total, metrics

    def _collect_rollout(self, observations: np.ndarray) -> tuple[np.ndarray, dict]:
        start = time.perf_counter()
        for step in range(self.config.rollout_steps):
            self.buffer.observations[step] = observations
            mlx_observations = mx.array(observations)
            actions, latents, log_probabilities, values = self.model.sample(
                mlx_observations
            )
            mx.eval(actions, latents, log_probabilities, values)
            numpy_actions = np.asarray(actions, dtype=np.float32)
            self.buffer.latents[step] = np.asarray(latents, dtype=np.float32)
            self.buffer.log_probabilities[step] = np.asarray(
                log_probabilities, dtype=np.float32
            )
            self.buffer.values[step] = np.asarray(values, dtype=np.float32)

            observations, rewards, terminated, truncated, info = self.env.step(
                numpy_actions
            )
            self.buffer.rewards[step] = rewards
            self.buffer.dones[step] = np.logical_or(terminated, truncated)
            final_info = info.get("final_info")
            if final_info is not None:
                self._completed_returns.extend(final_info["returns"].tolist())
                self._completed_lengths.extend(final_info["lengths"].tolist())

        _, final_value = self.model(mx.array(observations))
        mx.eval(final_value)
        self.buffer.finish(
            np.asarray(final_value, dtype=np.float32),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        elapsed = time.perf_counter() - start
        steps = self.config.rollout_steps * self.env.num_envs
        return observations, {
            "rollout_seconds": elapsed,
            "rollout_env_steps_per_second": steps / max(elapsed, 1e-9),
        }

    def _update(self) -> dict[str, float]:
        batch = {
            key: mx.array(value) for key, value in self.buffer.flattened().items()
        }
        advantages = batch["advantages"]
        batch["advantages"] = (advantages - mx.mean(advantages)) / (
            mx.std(advantages) + 1e-8
        )
        sample_count = int(batch["advantages"].shape[0])
        minibatch_size = min(self.config.minibatch_size, sample_count)
        metric_sums: dict[str, float] = {}
        updates = 0
        early_stop = False
        start = time.perf_counter()

        for _ in range(self.config.update_epochs):
            permutation = mx.random.permutation(sample_count)
            for offset in range(0, sample_count, minibatch_size):
                indices = permutation[offset : offset + minibatch_size]
                (loss, metrics), gradients = self._loss_and_grad(
                    batch["observations"][indices],
                    batch["latents"][indices],
                    batch["old_log_probabilities"][indices],
                    batch["old_values"][indices],
                    batch["advantages"][indices],
                    batch["returns"][indices],
                )
                gradients, gradient_norm = optim.clip_grad_norm(
                    gradients, self.config.max_gradient_norm
                )
                self.optimizer.update(self.model, gradients)
                metrics["gradient_norm"] = gradient_norm
                mx.eval(
                    loss,
                    metrics,
                    self.model.parameters(),
                    self.optimizer.state,
                )
                numeric = _arrays_to_float(metrics)
                for key, value in numeric.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                updates += 1
                if (
                    self.config.target_kl is not None
                    and numeric["approx_kl"] > self.config.target_kl
                ):
                    early_stop = True
                    break
            if early_stop:
                break

        elapsed = time.perf_counter() - start
        averaged = {
            key: value / max(updates, 1) for key, value in metric_sums.items()
        }
        averaged["update_seconds"] = elapsed
        averaged["minibatch_updates"] = float(updates)
        averaged["kl_early_stop"] = float(early_stop)
        return averaged

    def train(self, *, resume: str | Path | None = None) -> Path:
        observations, _ = self.env.reset(seed=self.config.seed)
        if resume is not None:
            self.load_checkpoint(resume)
        checkpoint: Path | None = None

        try:
            for iteration in range(self.iteration + 1, self.config.iterations + 1):
                self.iteration = iteration
                if self.config.anneal_learning_rate:
                    remaining = 1.0 - (iteration - 1.0) / self.config.iterations
                    self.optimizer.learning_rate = (
                        self.config.learning_rate * remaining
                    )

                observations, rollout_metrics = self._collect_rollout(observations)
                update_metrics = self._update()
                added_steps = self.config.rollout_steps * self.env.num_envs
                self.environment_steps += added_steps
                stats = self.env.runtime.stats
                recent_returns = self._completed_returns[-100:]
                recent_lengths = self._completed_lengths[-100:]
                report: dict[str, Any] = {
                    "iteration": iteration,
                    "environment_steps": self.environment_steps,
                    "device": self.env.runtime.device_name,
                    **rollout_metrics,
                    **update_metrics,
                    "gpu_ms": stats.last_gpu_milliseconds,
                    "episode_return_mean": (
                        float(np.mean(recent_returns)) if recent_returns else None
                    ),
                    "episode_length_mean": (
                        float(np.mean(recent_lengths)) if recent_lengths else None
                    ),
                }
                print(json.dumps(report, separators=(",", ":"), allow_nan=False))

                if (
                    self.config.checkpoint_interval
                    and iteration % self.config.checkpoint_interval == 0
                ):
                    checkpoint = self.save_checkpoint()
            if checkpoint is None or self.iteration % max(
                self.config.checkpoint_interval, 1
            ):
                checkpoint = self.save_checkpoint()
            return checkpoint
        finally:
            self.env.close()

    def save_checkpoint(self, directory: str | Path | None = None) -> Path:
        root = Path(directory or self.config.checkpoint_directory).expanduser()
        checkpoint = root / f"checkpoint-{self.iteration:06d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(checkpoint / "model.safetensors"))
        optimizer_arrays = dict(tree_flatten(self.optimizer.state))
        mx.save_safetensors(
            str(checkpoint / "optimizer.safetensors"), optimizer_arrays
        )
        state = {
            "format_version": 1,
            "iteration": self.iteration,
            "environment_steps": self.environment_steps,
            "observation_size": self.env.runtime.observation_count,
            "action_size": self.env.runtime.action_count,
            "native_version": self.env.runtime.version,
            "device": self.env.runtime.device_name,
            "ppo_config": asdict(self.config),
        }
        (checkpoint / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return checkpoint

    def load_checkpoint(self, directory: str | Path) -> None:
        checkpoint = Path(directory).expanduser()
        state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
        if state.get("format_version") != 1:
            raise ValueError(f"Unsupported checkpoint format in {checkpoint}")
        expected = (
            self.env.runtime.observation_count,
            self.env.runtime.action_count,
        )
        actual = (state.get("observation_size"), state.get("action_size"))
        if actual != expected:
            raise ValueError(
                f"Checkpoint model shape {actual} does not match runtime {expected}"
            )
        saved_hidden = tuple(state["ppo_config"]["hidden_sizes"])
        if saved_hidden != self.config.hidden_sizes:
            raise ValueError(
                f"Checkpoint hidden sizes {saved_hidden} do not match "
                f"{self.config.hidden_sizes}"
            )
        self.model.load_weights(str(checkpoint / "model.safetensors"))
        optimizer_flat = mx.load(str(checkpoint / "optimizer.safetensors"))
        self.optimizer.state = tree_unflatten(list(optimizer_flat.items()))
        self.iteration = int(state["iteration"])
        self.environment_steps = int(state["environment_steps"])
        mx.eval(self.model.parameters(), self.optimizer.state)


def load_policy(
    checkpoint: str | Path,
    observation_size: int,
    action_size: int,
) -> tuple[ActorCritic, dict[str, Any]]:
    """Load an inference-only actor-critic from a checkpoint directory."""

    directory = Path(checkpoint).expanduser()
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    actual = (state.get("observation_size"), state.get("action_size"))
    expected = (observation_size, action_size)
    if actual != expected:
        raise ValueError(f"Checkpoint model shape {actual} does not match {expected}")
    config = state["ppo_config"]
    model = ActorCritic(
        observation_size,
        action_size,
        tuple(config["hidden_sizes"]),
        float(config["initial_log_std"]),
    )
    model.load_weights(str(directory / "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())
    return model, state


def infer_actions(
    model: ActorCritic, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run a deterministic policy batch and return actions and values."""

    actions, values = model.deterministic(mx.array(observations))
    mx.eval(actions, values)
    return (
        np.asarray(actions, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
    )


def sample_actions(
    model: ActorCritic, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a stochastic policy batch and return actions and values."""

    actions, _, _, values = model.sample(mx.array(observations))
    mx.eval(actions, values)
    return (
        np.asarray(actions, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
    )
