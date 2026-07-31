"""MLX policy learning over rollout batches produced by the native runtime.

This module deliberately has no simulator, world-state, contact, reset, or
rollout-scheduling dependency. Swift/Metal owns collection. MLX receives one
compact batch at a declared learning boundary, performs PPO updates, and
publishes a PolicyPack through the canonical native artifact writer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .native import PolicyDenseLayerArtifact, write_policy_pack


@dataclass(frozen=True, slots=True)
class MLXPPOConfiguration:
    update_epochs: int = 4
    minibatch_size: int = 8192
    hidden_sizes: tuple[int, ...] = (512, 256, 128)
    learning_rate: float = 3.0e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    maximum_gradient_norm: float = 1.0
    target_kl: float | None = 0.02
    initial_log_standard_deviation: float = -0.5
    seed: int = 1

    def validate(self) -> None:
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("PPO epoch and minibatch counts must be positive")
        if not self.hidden_sizes or any(
            width <= 0 for width in self.hidden_sizes
        ):
            raise ValueError("PPO hidden widths must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("PPO learning rate must be positive")
        if self.clip_ratio <= 0.0:
            raise ValueError("PPO clip ratio must be positive")
        if self.value_coefficient < 0.0:
            raise ValueError("PPO value coefficient must be nonnegative")
        if self.entropy_coefficient < 0.0:
            raise ValueError("PPO entropy coefficient must be nonnegative")
        if self.maximum_gradient_norm <= 0.0:
            raise ValueError("PPO gradient norm limit must be positive")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("PPO target KL must be positive when enabled")


@dataclass(frozen=True, slots=True)
class MLXPolicyBatch:
    actor_observations: mx.array
    critic_observations: mx.array
    latents: mx.array
    old_log_probabilities: mx.array
    old_values: mx.array
    advantages: mx.array
    returns: mx.array

    @classmethod
    def from_numpy(
        cls,
        *,
        actor_observations: np.ndarray,
        critic_observations: np.ndarray,
        latents: np.ndarray,
        old_log_probabilities: np.ndarray,
        old_values: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> "MLXPolicyBatch":
        return cls(
            actor_observations=mx.array(
                actor_observations,
                dtype=mx.float32,
            ),
            critic_observations=mx.array(
                critic_observations,
                dtype=mx.float32,
            ),
            latents=mx.array(latents, dtype=mx.float32),
            old_log_probabilities=mx.array(
                old_log_probabilities,
                dtype=mx.float32,
            ),
            old_values=mx.array(old_values, dtype=mx.float32),
            advantages=mx.array(advantages, dtype=mx.float32),
            returns=mx.array(returns, dtype=mx.float32),
        )

    def validate(
        self,
        actor_observation_count: int,
        critic_observation_count: int,
        action_count: int,
    ) -> int:
        if self.actor_observations.ndim != 2 or int(
            self.actor_observations.shape[1]
        ) != actor_observation_count:
            raise ValueError("actor observation batch shape is invalid")
        sample_count = int(self.actor_observations.shape[0])
        expected = {
            "critic_observations": (
                self.critic_observations,
                (sample_count, critic_observation_count),
            ),
            "latents": (
                self.latents,
                (sample_count, action_count),
            ),
            "old_log_probabilities": (
                self.old_log_probabilities,
                (sample_count,),
            ),
            "old_values": (self.old_values, (sample_count,)),
            "advantages": (self.advantages, (sample_count,)),
            "returns": (self.returns, (sample_count,)),
        }
        for label, (value, shape) in expected.items():
            if tuple(int(dimension) for dimension in value.shape) != shape:
                raise ValueError(f"{label} batch shape is invalid")
        if sample_count == 0:
            raise ValueError("policy batch must contain samples")
        return sample_count


def _elu_mlp(
    input_count: int,
    output_count: int,
    hidden_sizes: tuple[int, ...],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_count
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, output_count))
    return nn.Sequential(*layers)


def _orthogonal_weight(
    shape: tuple[int, ...],
    gain: float,
    generator: np.random.Generator,
) -> mx.array:
    rows = shape[0]
    columns = int(np.prod(shape[1:]))
    sample = generator.standard_normal(
        (max(rows, columns), min(rows, columns)),
        dtype=np.float32,
    )
    q, r = np.linalg.qr(sample, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q *= signs[None, :]
    if rows < columns:
        q = q.T
    return mx.array(
        gain
        * q[:rows, :columns]
        .reshape(shape)
        .astype(np.float32, copy=False),
        dtype=mx.float32,
    )


def _initialize_mlp(
    network: nn.Sequential,
    *,
    output_gain: float,
    generator: np.random.Generator,
) -> None:
    linear = [
        layer
        for layer in network.layers
        if isinstance(layer, nn.Linear)
    ]
    for index, layer in enumerate(linear):
        layer.weight = _orthogonal_weight(
            tuple(int(value) for value in layer.weight.shape),
            (
                output_gain
                if index + 1 == len(linear)
                else math.sqrt(2.0)
            ),
            generator,
        )
        if getattr(layer, "bias", None) is not None:
            layer.bias = mx.zeros_like(layer.bias)


class MLXActorCritic(nn.Module):
    """Generic asymmetric actor/critic matching compiled TaskPack dimensions."""

    def __init__(
        self,
        actor_observation_count: int,
        critic_observation_count: int,
        action_count: int,
        configuration: MLXPPOConfiguration,
    ) -> None:
        super().__init__()
        if min(
            actor_observation_count,
            critic_observation_count,
            action_count,
        ) <= 0:
            raise ValueError("policy dimensions must be positive")
        self.actor = _elu_mlp(
            actor_observation_count,
            action_count,
            configuration.hidden_sizes,
        )
        self.critic = _elu_mlp(
            critic_observation_count,
            1,
            configuration.hidden_sizes,
        )
        generator = np.random.default_rng(configuration.seed)
        _initialize_mlp(
            self.actor,
            output_gain=0.01,
            generator=generator,
        )
        _initialize_mlp(
            self.critic,
            output_gain=1.0,
            generator=generator,
        )
        self.log_standard_deviation = mx.full(
            (action_count,),
            configuration.initial_log_standard_deviation,
            dtype=mx.float32,
        )

    def actor_mean(self, observations: mx.array) -> mx.array:
        return self.actor(observations)

    def value(self, observations: mx.array) -> mx.array:
        return self.critic(observations).squeeze(-1)

    @staticmethod
    def log_probability(
        mean: mx.array,
        log_standard_deviation: mx.array,
        latent: mx.array,
    ) -> mx.array:
        standardized = (
            latent - mean
        ) * mx.exp(-log_standard_deviation)
        gaussian = (
            -0.5 * mx.square(standardized)
            - log_standard_deviation
            - 0.5 * math.log(2.0 * math.pi)
        )
        action = mx.tanh(latent)
        jacobian = mx.log(
            mx.maximum(1.0 - mx.square(action), 1.0e-6)
        )
        return mx.sum(gaussian - jacobian, axis=-1)

    def evaluate(
        self,
        actor_observations: mx.array,
        critic_observations: mx.array,
        latents: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        mean = self.actor_mean(actor_observations)
        log_standard_deviation = mx.clip(
            self.log_standard_deviation,
            -5.0,
            2.0,
        )
        entropy = mx.sum(
            log_standard_deviation
            + 0.5 * (1.0 + math.log(2.0 * math.pi)),
            axis=-1,
        )
        return (
            self.log_probability(
                mean,
                log_standard_deviation,
                latents,
            ),
            entropy,
            self.value(critic_observations),
        )


class MLXPolicyLearner:
    """Owns only policy parameters, optimizer state, and PPO updates."""

    def __init__(
        self,
        actor_observation_count: int,
        critic_observation_count: int,
        action_count: int,
        configuration: MLXPPOConfiguration = MLXPPOConfiguration(),
    ) -> None:
        configuration.validate()
        self.actor_observation_count = actor_observation_count
        self.critic_observation_count = critic_observation_count
        self.action_count = action_count
        self.configuration = configuration
        mx.random.seed(configuration.seed)
        self.model = MLXActorCritic(
            actor_observation_count,
            critic_observation_count,
            action_count,
            configuration,
        )
        self.optimizer = optim.Adam(
            learning_rate=configuration.learning_rate,
            bias_correction=True,
        )
        self.optimizer.init(self.model.trainable_parameters())
        self._loss_and_gradient = nn.value_and_grad(
            self.model,
            self._loss,
        )
        self.revision = 1
        mx.eval(self.model.parameters(), self.optimizer.state)

    def _loss(
        self,
        actor_observations: mx.array,
        critic_observations: mx.array,
        latents: mx.array,
        old_log_probabilities: mx.array,
        old_values: mx.array,
        advantages: mx.array,
        returns: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        log_probabilities, entropy, values = self.model.evaluate(
            actor_observations,
            critic_observations,
            latents,
        )
        log_ratio = log_probabilities - old_log_probabilities
        ratio = mx.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = mx.clip(
            ratio,
            1.0 - self.configuration.clip_ratio,
            1.0 + self.configuration.clip_ratio,
        ) * advantages
        policy_loss = -mx.mean(mx.minimum(unclipped, clipped))
        clipped_values = old_values + mx.clip(
            values - old_values,
            -self.configuration.clip_ratio,
            self.configuration.clip_ratio,
        )
        value_loss = 0.5 * mx.mean(
            mx.maximum(
                mx.square(values - returns),
                mx.square(clipped_values - returns),
            )
        )
        entropy_mean = mx.mean(entropy)
        loss = (
            policy_loss
            + self.configuration.value_coefficient * value_loss
            - self.configuration.entropy_coefficient * entropy_mean
        )
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "approximate_kl": mx.mean(
                (ratio - 1.0) - log_ratio
            ),
            "clip_fraction": mx.mean(
                (
                    mx.abs(ratio - 1.0)
                    > self.configuration.clip_ratio
                ).astype(mx.float32)
            ),
        }

    def update(self, batch: MLXPolicyBatch) -> dict[str, float]:
        sample_count = batch.validate(
            self.actor_observation_count,
            self.critic_observation_count,
            self.action_count,
        )
        advantages = (
            batch.advantages - mx.mean(batch.advantages)
        ) / (mx.std(batch.advantages) + 1.0e-8)
        totals: dict[str, float] = {}
        update_count = 0
        start = time.perf_counter()
        stop = False
        for _ in range(self.configuration.update_epochs):
            permutation = mx.random.permutation(sample_count)
            for offset in range(
                0,
                sample_count,
                self.configuration.minibatch_size,
            ):
                indices = permutation[
                    offset : offset
                    + self.configuration.minibatch_size
                ]
                (loss, metrics), gradients = self._loss_and_gradient(
                    batch.actor_observations[indices],
                    batch.critic_observations[indices],
                    batch.latents[indices],
                    batch.old_log_probabilities[indices],
                    batch.old_values[indices],
                    advantages[indices],
                    batch.returns[indices],
                )
                gradients, gradient_norm = optim.clip_grad_norm(
                    gradients,
                    self.configuration.maximum_gradient_norm,
                )
                self.optimizer.update(self.model, gradients)
                metrics["gradient_norm"] = gradient_norm
                mx.eval(
                    loss,
                    metrics,
                    self.model.parameters(),
                    self.optimizer.state,
                )
                numeric = {
                    key: float(value.item())
                    for key, value in metrics.items()
                }
                for key, value in numeric.items():
                    totals[key] = totals.get(key, 0.0) + value
                update_count += 1
                if (
                    self.configuration.target_kl is not None
                    and numeric["approximate_kl"]
                    > self.configuration.target_kl
                ):
                    stop = True
                    break
            if stop:
                break
        self.revision += 1
        result = {
            key: value / max(update_count, 1)
            for key, value in totals.items()
        }
        result["update_seconds"] = time.perf_counter() - start
        result["minibatch_updates"] = float(update_count)
        result["policy_revision"] = float(self.revision)
        return result

    def write_policy_pack(
        self,
        output: str | Path,
        *,
        policy_id: str,
        observation_mean: np.ndarray | tuple[float, ...] = (),
        observation_inverse_standard_deviation:
            np.ndarray | tuple[float, ...] = (),
        action_bias: np.ndarray | tuple[float, ...] = (),
        action_scale: np.ndarray | tuple[float, ...] = (),
        observation_clip: float = 100.0,
        action_clip: float = 1.0,
        library_path: str | Path | None = None,
    ) -> Path:
        layers = []
        linear = [
            layer
            for layer in self.model.actor.layers
            if isinstance(layer, nn.Linear)
        ]
        for index, layer in enumerate(linear):
            mx.eval(layer.weight, layer.bias)
            layers.append(
                PolicyDenseLayerArtifact(
                    weights=np.asarray(
                        layer.weight,
                        dtype=np.float32,
                    ),
                    bias=np.asarray(
                        layer.bias,
                        dtype=np.float32,
                    ),
                    activation=(
                        2 if index + 1 == len(linear) else 3
                    ),
                )
            )
        return write_policy_pack(
            output,
            policy_id=policy_id,
            revision=self.revision,
            layers=layers,
            observation_mean=observation_mean,
            observation_inverse_standard_deviation=(
                observation_inverse_standard_deviation
            ),
            action_bias=action_bias,
            action_scale=action_scale,
            observation_clip=observation_clip,
            action_clip=action_clip,
            library_path=library_path,
        )


__all__ = [
    "MLXActorCritic",
    "MLXPPOConfiguration",
    "MLXPolicyBatch",
    "MLXPolicyLearner",
]
