"""MLX policy learning over rollout batches produced by the native runtime.

This module deliberately has no simulator, world-state, contact, reset, or
rollout-scheduling dependency. Swift/Metal owns collection. MLX receives one
compact batch at a declared learning boundary, performs PPO updates, and
publishes a PolicyPack through the canonical native artifact writer.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .native import (
    PolicyDenseLayerArtifact,
    learning_pack_content_hash,
    write_policy_pack,
)

_LEARNING_PACK_HEADER = struct.Struct("<8sIIQQ")
_POLICY_KIND = 2
_POLICY_FORMAT_VERSION = 2
_POLICY_ROLLOUT_KIND = 3
_POLICY_ROLLOUT_FORMAT_VERSION = 2
_TRANSITION_DTYPE = np.dtype(
    [
        ("reward", "<f4"),
        ("tracking_score", "<f4"),
        ("root_height", "<f4"),
        ("tilt", "<f4"),
        ("done", "<u4"),
        ("timeout", "<u4"),
        ("physics_error", "<u4"),
        ("termination_reason", "<u4"),
        ("task_reward", "<f4"),
        ("base_reward", "<f4"),
        ("joint_velocity_reward", "<f4"),
        ("joint_acceleration_reward", "<f4"),
        ("control_reward", "<f4"),
        ("posture_reward", "<f4"),
        ("energy_reward", "<f4"),
        ("contact_reward", "<f4"),
        ("policy_revision", "<u8"),
        ("reserved", "<u8"),
    ],
    align=False,
)


@dataclass(frozen=True, slots=True)
class NativePolicyRollout:
    """Validated, step-major rollout artifact published by Swift/native Metal."""

    id: str
    task_fingerprint: int
    policy_fingerprint: int
    policy_revision: int
    environment_count: int
    control_step_count: int
    actor_observation_count: int
    critic_observation_count: int
    action_count: int
    actor_observations: np.ndarray
    critic_observations: np.ndarray
    latents: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    bootstrap_values: np.ndarray
    transitions: np.ndarray

    @property
    def sample_count(self) -> int:
        return self.environment_count * self.control_step_count

    def policy_batch(
        self,
        *,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> "MLXPolicyBatch":
        """Compute terminal-safe GAE and publish only learner tensors to MLX."""

        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        steps = self.control_step_count
        environments = self.environment_count
        rewards = self.transitions["reward"].reshape(
            steps,
            environments,
        )
        done = self.transitions["done"].reshape(
            steps,
            environments,
        ).astype(np.float32)
        values = self.old_values.reshape(steps, environments)
        # A native timeout is an episode boundary. Its post-transition
        # observation is intentionally not replaced with the next episode's
        # reset observation, so do not invent a bootstrap from V(s_t).
        # Treat it as terminal until the rollout ABI publishes an explicit
        # terminal-observation value.
        advantages = np.zeros_like(values, dtype=np.float32)
        running = np.zeros((environments,), dtype=np.float32)
        for step in range(steps - 1, -1, -1):
            next_values = (
                self.bootstrap_values
                if step + 1 == steps
                else values[step + 1]
            )
            continuing = 1.0 - done[step]
            delta = (
                rewards[step]
                + discount * continuing * next_values
                - values[step]
            )
            running = (
                delta
                + discount
                * gae_lambda
                * continuing
                * running
            )
            advantages[step] = running
        returns = advantages + values
        return MLXPolicyBatch.from_numpy(
            actor_observations=self.actor_observations.reshape(
                self.sample_count,
                self.actor_observation_count,
            ),
            critic_observations=self.critic_observations.reshape(
                self.sample_count,
                self.critic_observation_count,
            ),
            latents=self.latents.reshape(
                self.sample_count,
                self.action_count,
            ),
            old_log_probabilities=self.old_log_probabilities,
            old_values=self.old_values,
            advantages=advantages.reshape(self.sample_count),
            returns=returns.reshape(self.sample_count),
        )


@dataclass(frozen=True, slots=True)
class NativePolicyDenseLayer:
    input_count: int
    output_count: int
    activation: int
    weights: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True, slots=True)
class NativePolicyPack:
    """Validated canonical actor/critic artifact consumed by Metal."""

    id: str
    revision: int
    observation_mean: np.ndarray
    observation_inverse_standard_deviation: np.ndarray
    layers: tuple[NativePolicyDenseLayer, ...]
    critic_observation_mean: np.ndarray
    critic_observation_inverse_standard_deviation: np.ndarray
    critic_layers: tuple[NativePolicyDenseLayer, ...]
    action_log_standard_deviation: np.ndarray
    action_bias: np.ndarray
    action_scale: np.ndarray
    observation_clip: float
    action_clip: float
    content_hash: int

    @property
    def actor_observation_count(self) -> int:
        return self.layers[0].input_count

    @property
    def critic_observation_count(self) -> int:
        return (
            self.critic_layers[0].input_count
            if self.critic_layers
            else 0
        )

    @property
    def action_count(self) -> int:
        return self.layers[-1].output_count

    @staticmethod
    def _effective(
        values: np.ndarray,
        count: int,
        default: float,
    ) -> np.ndarray:
        return (
            values
            if values.size
            else np.full((count,), default, dtype=np.float32)
        )

    @property
    def effective_observation_mean(self) -> np.ndarray:
        return self._effective(
            self.observation_mean,
            self.actor_observation_count,
            0.0,
        )

    @property
    def effective_observation_inverse_standard_deviation(
        self,
    ) -> np.ndarray:
        return self._effective(
            self.observation_inverse_standard_deviation,
            self.actor_observation_count,
            1.0,
        )

    @property
    def effective_critic_observation_mean(self) -> np.ndarray:
        return self._effective(
            self.critic_observation_mean,
            self.critic_observation_count,
            0.0,
        )

    @property
    def effective_critic_observation_inverse_standard_deviation(
        self,
    ) -> np.ndarray:
        return self._effective(
            self.critic_observation_inverse_standard_deviation,
            self.critic_observation_count,
            1.0,
        )

    @property
    def effective_action_bias(self) -> np.ndarray:
        return self._effective(
            self.action_bias,
            self.action_count,
            0.0,
        )

    @property
    def effective_action_scale(self) -> np.ndarray:
        return self._effective(
            self.action_scale,
            self.action_count,
            1.0,
        )

    def actor_mean(self, observations: np.ndarray) -> np.ndarray:
        value = np.asarray(observations, dtype=np.float32)
        if value.shape[-1] != self.actor_observation_count:
            raise ValueError(
                "actor observation width does not match PolicyPack"
            )
        value = np.clip(
            (
                value - self.effective_observation_mean
            )
            * self
                .effective_observation_inverse_standard_deviation,
            -self.observation_clip,
            self.observation_clip,
        )
        for layer in self.layers:
            value = value @ layer.weights.T + layer.bias
            if layer.activation == 1:
                value = np.maximum(value, 0.0)
            elif layer.activation == 2:
                value = np.tanh(value)
            elif layer.activation == 3:
                value = np.where(
                    value >= 0.0,
                    value,
                    np.expm1(np.minimum(value, 0.0)),
                )
            elif layer.activation == 4:
                value = value / (
                    1.0
                    + np.exp(
                        np.clip(-value, -80.0, 80.0)
                    )
                )
            elif layer.activation != 0:
                raise ValueError(
                    "PolicyPack contains an unsupported activation"
                )
        return value.astype(np.float32, copy=False)

    def actions(self, observations: np.ndarray) -> np.ndarray:
        squashed = np.tanh(self.actor_mean(observations))
        return np.clip(
            self.effective_action_scale * squashed
            + self.effective_action_bias,
            -self.action_clip,
            self.action_clip,
        ).astype(np.float32, copy=False)


class _PackReader:
    def __init__(self, payload: memoryview) -> None:
        self.payload = payload
        self.cursor = 0

    def _take(self, byte_count: int) -> memoryview:
        if byte_count < 0 or byte_count > len(self.payload) - self.cursor:
            raise ValueError("Learning-pack payload is truncated")
        result = self.payload[self.cursor : self.cursor + byte_count]
        self.cursor += byte_count
        return result

    def peek_unsigned64(self) -> int:
        if len(self.payload) - self.cursor < 8:
            raise ValueError("Learning-pack payload is truncated")
        return struct.unpack_from("<Q", self.payload, self.cursor)[0]

    def unsigned64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def unsigned32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def float32(self) -> float:
        return struct.unpack("<f", self._take(4))[0]

    def string(self) -> str:
        count = self.unsigned64()
        if count > (1 << 32) - 1:
            raise ValueError("Learning-pack string exceeds its ABI limit")
        try:
            return bytes(self._take(count)).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Learning-pack identity is not UTF-8") from error

    def vector(
        self,
        dtype: np.dtype,
        expected_count: int | None = None,
        *,
        maximum_count: int = (1 << 32) - 1,
    ) -> np.ndarray:
        count = self.unsigned64()
        if count > maximum_count or (
            expected_count is not None
            and count != expected_count
        ):
            raise ValueError(
                "Learning-pack vector count disagrees with its dimensions"
            )
        byte_count = count * dtype.itemsize
        return np.frombuffer(
            self._take(byte_count),
            dtype=dtype,
            count=count,
        )


def _read_dense_layers(
    reader: _PackReader,
    label: str,
) -> tuple[NativePolicyDenseLayer, ...]:
    layer_count = reader.unsigned64()
    if layer_count == 0 or layer_count > 32:
        raise ValueError(f"{label} layer count is invalid")
    layers: list[NativePolicyDenseLayer] = []
    expected_input = 0
    for index in range(layer_count):
        input_count = reader.unsigned32()
        output_count = reader.unsigned32()
        activation = reader.unsigned32()
        if (
            input_count == 0
            or output_count == 0
            or activation > 4
            or (index != 0 and input_count != expected_input)
        ):
            raise ValueError(f"{label} topology is invalid")
        weights = reader.vector(
            np.dtype("<f4"),
            input_count * output_count,
        ).reshape(output_count, input_count)
        bias = reader.vector(
            np.dtype("<f4"),
            output_count,
        )
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError(f"{label} contains non-finite parameters")
        layers.append(
            NativePolicyDenseLayer(
                input_count=input_count,
                output_count=output_count,
                activation=activation,
                weights=weights,
                bias=bias,
            )
        )
        expected_input = output_count
    if layers[-1].activation != 0:
        raise ValueError(f"{label} final layer must use identity")
    return tuple(layers)


def read_policy_pack(
    path: str | Path,
    *,
    library_path: str | Path | None = None,
) -> NativePolicyPack:
    """Read and validate the canonical native PolicyPack artifact."""

    source = Path(path)
    data = source.read_bytes()
    if len(data) < _LEARNING_PACK_HEADER.size:
        raise ValueError("PolicyPack file is truncated")
    (
        magic,
        format_version,
        kind,
        payload_bytes,
        expected_hash,
    ) = _LEARNING_PACK_HEADER.unpack_from(data)
    if magic != b"MRLEARN\0":
        raise ValueError("PolicyPack magic is invalid")
    if format_version != _POLICY_FORMAT_VERSION:
        raise ValueError("PolicyPack format version is unsupported")
    if kind != _POLICY_KIND:
        raise ValueError("Learning artifact is not a PolicyPack")
    if payload_bytes != len(data) - _LEARNING_PACK_HEADER.size:
        raise ValueError("PolicyPack payload length is invalid")
    payload = memoryview(data)[_LEARNING_PACK_HEADER.size :]
    if (
        learning_pack_content_hash(
            np.frombuffer(payload, dtype=np.uint8),
            path=library_path,
        )
        != expected_hash
    ):
        raise ValueError("PolicyPack content fingerprint is invalid")
    reader = _PackReader(payload)
    identifier = reader.string()
    revision = reader.unsigned64()
    observation_mean = reader.vector(np.dtype("<f4"))
    observation_inverse_standard_deviation = reader.vector(
        np.dtype("<f4")
    )
    layers = _read_dense_layers(reader, "actor")
    critic_observation_mean = reader.vector(np.dtype("<f4"))
    critic_observation_inverse_standard_deviation = reader.vector(
        np.dtype("<f4")
    )
    if reader.peek_unsigned64() == 0:
        reader.unsigned64()
        critic_layers: tuple[NativePolicyDenseLayer, ...] = ()
    else:
        critic_layers = _read_dense_layers(reader, "critic")
    action_log_standard_deviation = reader.vector(
        np.dtype("<f4")
    )
    action_bias = reader.vector(np.dtype("<f4"))
    action_scale = reader.vector(np.dtype("<f4"))
    observation_clip = reader.float32()
    action_clip = reader.float32()
    if reader.cursor != len(payload):
        raise ValueError("PolicyPack contains trailing payload bytes")
    actor_count = layers[0].input_count
    action_count = layers[-1].output_count
    critic_count = (
        critic_layers[0].input_count if critic_layers else 0
    )
    vector_contracts = (
        (observation_mean, actor_count, False),
        (
            observation_inverse_standard_deviation,
            actor_count,
            True,
        ),
        (critic_observation_mean, critic_count, False),
        (
            critic_observation_inverse_standard_deviation,
            critic_count,
            True,
        ),
        (
            action_log_standard_deviation,
            action_count,
            False,
        ),
        (action_bias, action_count, False),
        (action_scale, action_count, False),
    )
    if (
        not identifier
        or revision == 0
        or not math.isfinite(observation_clip)
        or observation_clip <= 0.0
        or not math.isfinite(action_clip)
        or action_clip <= 0.0
        or (critic_layers and critic_layers[-1].output_count != 1)
        or (action_log_standard_deviation.size and not critic_layers)
    ):
        raise ValueError("PolicyPack contract is invalid")
    for values, count, positive in vector_contracts:
        if values.size not in (0, count) or not np.isfinite(values).all():
            raise ValueError("PolicyPack vector shape or values are invalid")
        if positive and values.size and np.any(values <= 0.0):
            raise ValueError(
                "PolicyPack inverse standard deviations must be positive"
            )
    if action_log_standard_deviation.size and np.any(
        (action_log_standard_deviation < -5.0)
        | (action_log_standard_deviation > 2.0)
    ):
        raise ValueError(
            "PolicyPack log standard deviations exceed [-5, 2]"
        )
    return NativePolicyPack(
        id=identifier,
        revision=revision,
        observation_mean=observation_mean,
        observation_inverse_standard_deviation=(
            observation_inverse_standard_deviation
        ),
        layers=layers,
        critic_observation_mean=critic_observation_mean,
        critic_observation_inverse_standard_deviation=(
            critic_observation_inverse_standard_deviation
        ),
        critic_layers=critic_layers,
        action_log_standard_deviation=(
            action_log_standard_deviation
        ),
        action_bias=action_bias,
        action_scale=action_scale,
        observation_clip=observation_clip,
        action_clip=action_clip,
        content_hash=expected_hash,
    )


def read_policy_rollout_pack(
    path: str | Path,
    *,
    library_path: str | Path | None = None,
) -> NativePolicyRollout:
    """Read and validate the canonical native rollout handoff artifact."""

    source = Path(path)
    if source.stat().st_size < _LEARNING_PACK_HEADER.size:
        raise ValueError("PolicyRolloutPack file is truncated")
    data = np.memmap(source, mode="r", dtype=np.uint8)
    (
        magic,
        format_version,
        kind,
        payload_bytes,
        expected_hash,
    ) = _LEARNING_PACK_HEADER.unpack_from(data)
    if magic != b"MRLEARN\0":
        raise ValueError("PolicyRolloutPack magic is invalid")
    if format_version != _POLICY_ROLLOUT_FORMAT_VERSION:
        raise ValueError(
            "PolicyRolloutPack format version is unsupported"
        )
    if kind != _POLICY_ROLLOUT_KIND:
        raise ValueError("Learning artifact is not a PolicyRolloutPack")
    if payload_bytes != data.size - _LEARNING_PACK_HEADER.size:
        raise ValueError("PolicyRolloutPack payload length is invalid")
    payload_bytes_view = data[_LEARNING_PACK_HEADER.size :]
    if (
        learning_pack_content_hash(
            payload_bytes_view,
            path=library_path,
        )
        != expected_hash
    ):
        raise ValueError("PolicyRolloutPack content fingerprint is invalid")
    payload = memoryview(payload_bytes_view)

    reader = _PackReader(payload)
    identifier = reader.string()
    task_fingerprint = reader.unsigned64()
    policy_fingerprint = reader.unsigned64()
    policy_revision = reader.unsigned64()
    environment_count = reader.unsigned32()
    control_step_count = reader.unsigned32()
    actor_observation_count = reader.unsigned32()
    critic_observation_count = reader.unsigned32()
    action_count = reader.unsigned32()
    dimensions = (
        environment_count,
        control_step_count,
        actor_observation_count,
        critic_observation_count,
        action_count,
    )
    if (
        not identifier
        or task_fingerprint == 0
        or policy_fingerprint == 0
        or policy_revision == 0
        or any(value == 0 for value in dimensions)
    ):
        raise ValueError(
            "PolicyRolloutPack identity, fingerprints, or dimensions are invalid"
        )
    sample_count = environment_count * control_step_count
    actor = reader.vector(
        np.dtype("<f4"),
        sample_count * actor_observation_count,
    )
    critic = reader.vector(
        np.dtype("<f4"),
        sample_count * critic_observation_count,
    )
    latents = reader.vector(
        np.dtype("<f4"),
        sample_count * action_count,
    )
    log_probabilities = reader.vector(
        np.dtype("<f4"),
        sample_count,
    )
    values = reader.vector(np.dtype("<f4"), sample_count)
    bootstrap_values = reader.vector(
        np.dtype("<f4"),
        environment_count,
    )
    transitions = reader.vector(
        _TRANSITION_DTYPE,
        sample_count,
    )
    if reader.cursor != len(payload):
        raise ValueError("PolicyRolloutPack contains trailing payload bytes")
    float_tables = (
        actor,
        critic,
        latents,
        log_probabilities,
        values,
        bootstrap_values,
        transitions["reward"],
        transitions["tracking_score"],
        transitions["root_height"],
        transitions["tilt"],
        transitions["task_reward"],
        transitions["base_reward"],
        transitions["joint_velocity_reward"],
        transitions["joint_acceleration_reward"],
        transitions["control_reward"],
        transitions["posture_reward"],
        transitions["energy_reward"],
        transitions["contact_reward"],
    )
    if not all(np.isfinite(table).all() for table in float_tables):
        raise ValueError("PolicyRolloutPack contains non-finite values")
    if (
        np.any(transitions["done"] > 1)
        or np.any(transitions["timeout"] > 1)
        or np.any(transitions["physics_error"] > 1)
        or np.any(
            transitions["policy_revision"] != policy_revision
        )
    ):
        raise ValueError(
            "PolicyRolloutPack transition metadata is inconsistent"
        )
    return NativePolicyRollout(
        id=identifier,
        task_fingerprint=task_fingerprint,
        policy_fingerprint=policy_fingerprint,
        policy_revision=policy_revision,
        environment_count=environment_count,
        control_step_count=control_step_count,
        actor_observation_count=actor_observation_count,
        critic_observation_count=critic_observation_count,
        action_count=action_count,
        actor_observations=actor,
        critic_observations=critic,
        latents=latents,
        old_log_probabilities=log_probabilities,
        old_values=values,
        bootstrap_values=bootstrap_values,
        transitions=transitions,
    )


@dataclass(frozen=True, slots=True)
class MLXPPOConfiguration:
    update_epochs: int = 5
    minibatch_size: int = 8192
    hidden_sizes: tuple[int, ...] = (512, 256, 128)
    critic_hidden_sizes: tuple[int, ...] | None = None
    learning_rate: float = 1.0e-3
    clip_ratio: float = 0.2
    value_coefficient: float = 1.0
    entropy_coefficient: float = 0.01
    maximum_gradient_norm: float = 1.0
    target_kl: float | None = 0.01
    adaptive_learning_rate: bool = True
    minimum_learning_rate: float = 1.0e-5
    maximum_learning_rate: float = 1.0e-2
    discount: float = 0.99
    gae_lambda: float = 0.95
    # Match the pinned Unitree/RSL-RL exploration contract: initial standard
    # deviation is one in policy coordinates. MetalRobo scores the exact
    # bounded tanh-transformed distribution instead of clipping an unbounded
    # sample after the fact.
    initial_log_standard_deviation: float = 0.0
    observation_clip: float = 100.0
    seed: int = 1

    def validate(self) -> None:
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("PPO epoch and minibatch counts must be positive")
        if self.seed < 0:
            raise ValueError("PPO seed must be nonnegative")
        if any(width <= 0 for width in self.hidden_sizes):
            raise ValueError("PPO hidden widths must be positive")
        if self.critic_hidden_sizes is not None and (
            any(
                width <= 0
                for width in self.critic_hidden_sizes
            )
        ):
            raise ValueError(
                "PPO critic hidden widths must be positive"
            )
        finite_scalars = {
            "learning rate": self.learning_rate,
            "minimum learning rate": self.minimum_learning_rate,
            "maximum learning rate": self.maximum_learning_rate,
            "clip ratio": self.clip_ratio,
            "value coefficient": self.value_coefficient,
            "entropy coefficient": self.entropy_coefficient,
            "gradient norm limit": self.maximum_gradient_norm,
            "discount": self.discount,
            "GAE lambda": self.gae_lambda,
        }
        if not all(
            math.isfinite(value)
            for value in finite_scalars.values()
        ):
            raise ValueError("PPO scalar configuration must be finite")
        if self.learning_rate <= 0.0:
            raise ValueError("PPO learning rate must be positive")
        if (
            self.minimum_learning_rate <= 0.0
            or self.maximum_learning_rate
                < self.minimum_learning_rate
            or not (
                self.minimum_learning_rate
                <= self.learning_rate
                <= self.maximum_learning_rate
            )
        ):
            raise ValueError(
                "PPO learning-rate bounds are invalid"
            )
        if self.clip_ratio <= 0.0:
            raise ValueError("PPO clip ratio must be positive")
        if self.value_coefficient < 0.0:
            raise ValueError("PPO value coefficient must be nonnegative")
        if self.entropy_coefficient < 0.0:
            raise ValueError("PPO entropy coefficient must be nonnegative")
        if self.maximum_gradient_norm <= 0.0:
            raise ValueError("PPO gradient norm limit must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("PPO discount must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("PPO GAE lambda must be in [0, 1]")
        if not math.isfinite(
            self.observation_clip
        ) or self.observation_clip <= 0.0:
            raise ValueError(
                "PPO observation clip must be finite and positive"
            )
        if self.target_kl is not None and (
            not math.isfinite(self.target_kl)
            or self.target_kl <= 0.0
        ):
            raise ValueError(
                "PPO target KL must be finite and positive when enabled"
            )
        if (
            not math.isfinite(
                self.initial_log_standard_deviation
            )
            or self.initial_log_standard_deviation < -5.0
            or self.initial_log_standard_deviation > 2.0
        ):
            raise ValueError(
                "initial log standard deviation must be within [-5, 2]"
            )


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
        tables = {
            "actor_observations": self.actor_observations,
            **{
                label: value
                for label, (value, _) in expected.items()
            },
        }
        finite = [
            (label, mx.all(mx.isfinite(value)))
            for label, value in tables.items()
        ]
        mx.eval(*[value for _, value in finite])
        invalid = [
            label
            for label, value in finite
            if not bool(value.item())
        ]
        if invalid:
            raise ValueError(
                "policy batch contains non-finite values in "
                + ", ".join(invalid)
            )
        return sample_count


class _StableELU(nn.Module):
    """ELU whose inactive exponential branch cannot overflow."""

    def __call__(self, value: mx.array) -> mx.array:
        negative = mx.exp(mx.minimum(value, 0.0)) - 1.0
        return mx.where(value >= 0.0, value, negative)


def _elu_mlp(
    input_count: int,
    output_count: int,
    hidden_sizes: tuple[int, ...],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_count
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), _StableELU()))
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
            (
                configuration.critic_hidden_sizes
                if configuration.critic_hidden_sizes is not None
                else configuration.hidden_sizes
            ),
        )
        self.actor_observation_mean = mx.zeros(
            (actor_observation_count,),
            dtype=mx.float32,
        )
        self.actor_observation_inverse_standard_deviation = (
            mx.ones(
                (actor_observation_count,),
                dtype=mx.float32,
            )
        )
        self.critic_observation_mean = mx.zeros(
            (critic_observation_count,),
            dtype=mx.float32,
        )
        self.critic_observation_inverse_standard_deviation = (
            mx.ones(
                (critic_observation_count,),
                dtype=mx.float32,
            )
        )
        self.observation_clip = configuration.observation_clip
        self.freeze(
            recurse=False,
            keys=[
                "actor_observation_mean",
                "actor_observation_inverse_standard_deviation",
                "critic_observation_mean",
                "critic_observation_inverse_standard_deviation",
            ],
            strict=True,
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
        normalized = (
            observations - self.actor_observation_mean
        ) * self.actor_observation_inverse_standard_deviation
        return self.actor(
            mx.clip(
                normalized,
                -self.observation_clip,
                self.observation_clip,
            )
        )

    def value(self, observations: mx.array) -> mx.array:
        normalized = (
            observations - self.critic_observation_mean
        ) * self.critic_observation_inverse_standard_deviation
        return self.critic(
            mx.clip(
                normalized,
                -self.observation_clip,
                self.observation_clip,
            )
        ).squeeze(-1)

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
        magnitude = mx.abs(latent)
        log_jacobian = 2.0 * (
            math.log(2.0)
            - magnitude
            - mx.log1p(mx.exp(-2.0 * magnitude))
        )
        return mx.sum(gaussian - log_jacobian, axis=-1)

    @staticmethod
    def entropy(
        mean: mx.array,
        log_standard_deviation: mx.array,
    ) -> mx.array:
        """Five-point Gauss-Hermite entropy of the squashed distribution."""

        nodes = mx.array(
            (
                -2.0201828704560856,
                -0.9585724646138185,
                0.0,
                0.9585724646138185,
                2.0201828704560856,
            ),
            dtype=mx.float32,
        )
        weights = mx.array(
            (
                0.011257411327720689,
                0.22207592200561266,
                0.5333333333333333,
                0.22207592200561266,
                0.011257411327720689,
            ),
            dtype=mx.float32,
        )
        latent = (
            mean[..., None]
            + math.sqrt(2.0)
            * mx.exp(log_standard_deviation)[None, :, None]
            * nodes[None, None, :]
        )
        magnitude = mx.abs(latent)
        log_jacobian = 2.0 * (
            math.log(2.0)
            - magnitude
            - mx.log1p(mx.exp(-2.0 * magnitude))
        )
        transformed = (
            log_standard_deviation
            + 0.5 * (1.0 + math.log(2.0 * math.pi))
            + mx.sum(
                weights[None, None, :] * log_jacobian,
                axis=-1,
            )
        )
        return mx.sum(transformed, axis=-1)

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
        return (
            self.log_probability(
                mean,
                log_standard_deviation,
                latents,
            ),
            self.entropy(mean, log_standard_deviation),
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
        self._training_state: list[Any] = []
        self._compiled_training_step: Any = None
        self.refresh_compiled_training_state()
        self.policy_id: str | None = None
        self.action_bias = np.zeros(
            (action_count,),
            dtype=np.float32,
        )
        self.action_scale = np.ones(
            (action_count,),
            dtype=np.float32,
        )
        self.action_clip = 1.0
        self.revision = 1
        mx.eval(self.model.parameters(), self.optimizer.state)

    def refresh_compiled_training_state(self) -> None:
        """Rebind the compiled update after replacing model/optimizer state."""

        self._training_state = [
            self.model.state,
            self.optimizer.state,
        ]
        self._compiled_training_step = mx.compile(
            self._training_step,
            inputs=self._training_state,
            outputs=self._training_state,
        )

    @staticmethod
    def _copied_state_tree(state: Any) -> Any:
        """Copy tree containers while retaining immutable evaluated arrays."""

        return tree_unflatten(list(tree_flatten(state)))

    @staticmethod
    def _permutation_seed(
        base_seed: int,
        revision: int,
        epoch: int,
    ) -> int:
        """Counter-derived SplitMix64 seed for restart-stable shuffling."""

        mask = (1 << 64) - 1
        value = (
            int(base_seed)
            ^ (int(revision) * 0xD2B74407B1CE6E93)
            ^ (int(epoch) * 0xCA5A826395121157)
        ) & mask
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    def _require_finite_training_state(self) -> None:
        entries = [
            (f"model.{name}", value)
            for name, value in tree_flatten(self.model.state)
        ]
        entries.extend(
            (f"optimizer.{name}", value)
            for name, value in tree_flatten(self.optimizer.state)
        )
        checks = [
            (name, mx.all(mx.isfinite(value)))
            for name, value in entries
        ]
        mx.eval(*[value for _, value in checks])
        invalid = [
            name
            for name, value in checks
            if not bool(value.item())
        ]
        if invalid:
            raise FloatingPointError(
                "PPO update produced non-finite training state in "
                + ", ".join(invalid)
            )

    def restart_exploration(
        self,
        log_standard_deviation: float,
    ) -> None:
        """Start a fresh PPO optimizer from the current policy weights.

        This is the explicit curriculum boundary for changing the behavior
        distribution. Actor, critic, normalization, and action contracts are
        retained; exploration and every Adam moment are replaced atomically,
        and the behavior revision advances before another rollout is legal.
        """

        value = float(log_standard_deviation)
        if not np.isfinite(value) or value < -5.0 or value > 2.0:
            raise ValueError(
                "log_standard_deviation must be finite and in [-5, 2]"
            )
        if self.revision >= np.iinfo(np.uint64).max:
            raise OverflowError("policy revision cannot advance")
        self.model.log_standard_deviation = mx.full(
            (self.action_count,),
            value,
            dtype=mx.float32,
        )
        self.optimizer = optim.Adam(
            learning_rate=self.configuration.learning_rate,
            bias_correction=True,
        )
        self.optimizer.init(self.model.trainable_parameters())
        self.revision += 1
        self.refresh_compiled_training_state()
        mx.eval(
            self.model.parameters(),
            self.optimizer.state,
        )

    @classmethod
    def from_policy_pack(
        cls,
        path: str | Path,
        configuration: MLXPPOConfiguration = MLXPPOConfiguration(),
        *,
        library_path: str | Path | None = None,
    ) -> "MLXPolicyLearner":
        """Restore trainable PPO state from the canonical native artifact."""

        pack = read_policy_pack(
            path,
            library_path=library_path,
        )
        if not pack.critic_layers:
            raise ValueError(
                "PPO resume requires a PolicyPack critic network"
            )
        if pack.action_log_standard_deviation.size != pack.action_count:
            raise ValueError(
                "PPO resume requires a stochastic PolicyPack"
            )
        for label, layers in (
            ("actor", pack.layers),
            ("critic", pack.critic_layers),
        ):
            expected_activations = (
                [3] * (len(layers) - 1) + [0]
            )
            if [
                layer.activation for layer in layers
            ] != expected_activations:
                raise ValueError(
                    f"PPO resume supports ELU {label} hidden layers "
                    "and an identity output layer"
                )
        if pack.critic_layers[-1].output_count != 1:
            raise ValueError(
                "PPO resume requires a scalar critic output"
            )
        restored_configuration = replace(
            configuration,
            hidden_sizes=tuple(
                layer.output_count for layer in pack.layers[:-1]
            ),
            critic_hidden_sizes=tuple(
                layer.output_count
                for layer in pack.critic_layers[:-1]
            ),
            observation_clip=pack.observation_clip,
        )
        learner = cls(
            pack.actor_observation_count,
            pack.critic_observation_count,
            pack.action_count,
            restored_configuration,
        )

        def restore_layers(
            network: nn.Sequential,
            source: tuple[NativePolicyDenseLayer, ...],
        ) -> None:
            destination = [
                layer
                for layer in network.layers
                if isinstance(layer, nn.Linear)
            ]
            if len(destination) != len(source):
                raise RuntimeError(
                    "MLX network construction disagrees with PolicyPack"
                )
            for target, artifact in zip(destination, source, strict=True):
                target.weight = mx.array(
                    artifact.weights,
                    dtype=mx.float32,
                )
                target.bias = mx.array(
                    artifact.bias,
                    dtype=mx.float32,
                )

        restore_layers(learner.model.actor, pack.layers)
        restore_layers(learner.model.critic, pack.critic_layers)
        learner.model.log_standard_deviation = mx.array(
            pack.action_log_standard_deviation,
            dtype=mx.float32,
        )
        learner.set_observation_normalization(
            actor_mean=pack.effective_observation_mean,
            actor_inverse_standard_deviation=(
                pack
                    .effective_observation_inverse_standard_deviation
            ),
            critic_mean=pack.effective_critic_observation_mean,
            critic_inverse_standard_deviation=(
                pack
                    .effective_critic_observation_inverse_standard_deviation
            ),
            observation_clip=pack.observation_clip,
        )
        learner.set_action_contract(
            action_bias=pack.effective_action_bias,
            action_scale=pack.effective_action_scale,
            action_clip=pack.action_clip,
        )
        learner.policy_id = pack.id
        learner.revision = pack.revision
        learner.optimizer = optim.Adam(
            learning_rate=restored_configuration.learning_rate,
            bias_correction=True,
        )
        learner.optimizer.init(learner.model.trainable_parameters())
        learner.refresh_compiled_training_state()
        mx.eval(
            learner.model.parameters(),
            learner.optimizer.state,
        )
        return learner

    @staticmethod
    def _normalization_array(
        values: np.ndarray | tuple[float, ...],
        *,
        count: int,
        default: float,
        label: str,
        positive: bool,
    ) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.size == 0:
            result = np.full((count,), default, dtype=np.float32)
        result = np.ascontiguousarray(result.reshape(-1))
        if result.shape != (count,) or not np.isfinite(result).all():
            raise ValueError(f"{label} shape or values are invalid")
        if positive and np.any(result <= 0.0):
            raise ValueError(f"{label} values must be positive")
        return result

    def set_observation_normalization(
        self,
        *,
        actor_mean: np.ndarray | tuple[float, ...] | None = None,
        actor_inverse_standard_deviation:
            np.ndarray | tuple[float, ...] | None = None,
        critic_mean: np.ndarray | tuple[float, ...] | None = None,
        critic_inverse_standard_deviation:
            np.ndarray | tuple[float, ...] | None = None,
        observation_clip: float | None = None,
    ) -> None:
        """Install the frozen observation contract used by MLX and Metal."""

        actor_mean_array = self._normalization_array(
            (
                np.asarray(self.model.actor_observation_mean)
                if actor_mean is None
                else actor_mean
            ),
            count=self.actor_observation_count,
            default=0.0,
            label="actor observation mean",
            positive=False,
        )
        actor_inverse_array = self._normalization_array(
            (
                np.asarray(
                    self.model
                        .actor_observation_inverse_standard_deviation
                )
                if actor_inverse_standard_deviation is None
                else actor_inverse_standard_deviation
            ),
            count=self.actor_observation_count,
            default=1.0,
            label="actor observation inverse standard deviation",
            positive=True,
        )
        critic_mean_array = self._normalization_array(
            (
                np.asarray(self.model.critic_observation_mean)
                if critic_mean is None
                else critic_mean
            ),
            count=self.critic_observation_count,
            default=0.0,
            label="critic observation mean",
            positive=False,
        )
        critic_inverse_array = self._normalization_array(
            (
                np.asarray(
                    self.model
                        .critic_observation_inverse_standard_deviation
                )
                if critic_inverse_standard_deviation is None
                else critic_inverse_standard_deviation
            ),
            count=self.critic_observation_count,
            default=1.0,
            label="critic observation inverse standard deviation",
            positive=True,
        )
        if observation_clip is not None:
            if (
                not math.isfinite(observation_clip)
                or observation_clip <= 0.0
            ):
                raise ValueError(
                    "observation clip must be finite and positive"
                )
            self.model.observation_clip = float(observation_clip)
        self.model.actor_observation_mean = mx.array(
            actor_mean_array
        )
        self.model.actor_observation_inverse_standard_deviation = (
            mx.array(actor_inverse_array)
        )
        self.model.critic_observation_mean = mx.array(
            critic_mean_array
        )
        self.model.critic_observation_inverse_standard_deviation = (
            mx.array(critic_inverse_array)
        )
        mx.eval(
            self.model.actor_observation_mean,
            self.model.actor_observation_inverse_standard_deviation,
            self.model.critic_observation_mean,
            self.model
                .critic_observation_inverse_standard_deviation,
        )

    def set_action_contract(
        self,
        *,
        action_bias: np.ndarray | tuple[float, ...] | None = None,
        action_scale: np.ndarray | tuple[float, ...] | None = None,
        action_clip: float | None = None,
    ) -> None:
        """Install the action transform shared by training and deployment."""

        if action_bias is not None:
            self.action_bias = self._normalization_array(
                action_bias,
                count=self.action_count,
                default=0.0,
                label="action bias",
                positive=False,
            )
        if action_scale is not None:
            self.action_scale = self._normalization_array(
                action_scale,
                count=self.action_count,
                default=1.0,
                label="action scale",
                positive=False,
            )
        if action_clip is not None:
            if not math.isfinite(action_clip) or action_clip <= 0.0:
                raise ValueError(
                    "action clip must be finite and positive"
                )
            self.action_clip = float(action_clip)

    def _loss(
        self,
        actor_observations: mx.array,
        critic_observations: mx.array,
        latents: mx.array,
        old_log_probabilities: mx.array,
        old_means: mx.array,
        old_log_standard_deviation: mx.array,
        old_values: mx.array,
        advantages: mx.array,
        returns: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        log_probabilities, entropy, values = self.model.evaluate(
            actor_observations,
            critic_observations,
            latents,
        )
        current_means = self.model.actor_mean(
            actor_observations
        )
        current_log_standard_deviation = mx.clip(
            self.model.log_standard_deviation,
            -5.0,
            2.0,
        )
        inverse_current_variance = mx.exp(
            -2.0 * current_log_standard_deviation
        )
        kl_divergence = mx.mean(
            mx.sum(
                current_log_standard_deviation
                - old_log_standard_deviation
                + 0.5
                * (
                    mx.exp(
                        2.0
                        * (
                            old_log_standard_deviation
                            - current_log_standard_deviation
                        )
                    )
                    + mx.square(old_means - current_means)
                    * inverse_current_variance
                    - 1.0
                ),
                axis=-1,
            )
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
        value_loss = mx.mean(
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
            "kl_divergence": kl_divergence,
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

    def _training_step(
        self,
        actor_observations: mx.array,
        critic_observations: mx.array,
        latents: mx.array,
        old_log_probabilities: mx.array,
        old_means: mx.array,
        old_log_standard_deviation: mx.array,
        old_values: mx.array,
        advantages: mx.array,
        returns: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        (loss, metrics), gradients = self._loss_and_gradient(
            actor_observations,
            critic_observations,
            latents,
            old_log_probabilities,
            old_means,
            old_log_standard_deviation,
            old_values,
            advantages,
            returns,
        )
        _, actor_gradient_norm = optim.clip_grad_norm(
            {
                "actor": gradients["actor"],
                "log_standard_deviation":
                    gradients["log_standard_deviation"],
            },
            self.configuration.maximum_gradient_norm,
        )
        _, critic_gradient_norm = optim.clip_grad_norm(
            {"critic": gradients["critic"]},
            self.configuration.maximum_gradient_norm,
        )
        gradients, gradient_norm = optim.clip_grad_norm(
            gradients,
            self.configuration.maximum_gradient_norm,
        )
        self.optimizer.update(self.model, gradients)
        metrics["actor_gradient_norm"] = actor_gradient_norm
        metrics["critic_gradient_norm"] = critic_gradient_norm
        metrics["gradient_norm"] = gradient_norm
        return loss, metrics

    def update(self, batch: MLXPolicyBatch) -> dict[str, float]:
        model_before = self._copied_state_tree(self.model.state)
        optimizer_before = self._copied_state_tree(
            self.optimizer.state
        )
        revision_before = self.revision
        try:
            result = self._update_impl(batch)
            self._require_finite_training_state()
            return result
        except Exception:
            self.model.update(model_before, strict=True)
            self.optimizer.state = optimizer_before
            self.revision = revision_before
            self.refresh_compiled_training_state()
            mx.eval(self.model.state, self.optimizer.state)
            raise

    def _update_impl(
        self,
        batch: MLXPolicyBatch,
    ) -> dict[str, float]:
        sample_count = batch.validate(
            self.actor_observation_count,
            self.critic_observation_count,
            self.action_count,
        )
        advantages = (
            batch.advantages - mx.mean(batch.advantages)
        ) / (mx.std(batch.advantages) + 1.0e-8)
        old_means = self.model.actor_mean(
            batch.actor_observations
        )
        old_log_standard_deviation = mx.clip(
            self.model.log_standard_deviation,
            -5.0,
            2.0,
        )
        mx.eval(old_means, old_log_standard_deviation)
        totals: dict[str, float] = {}
        update_count = 0
        start = time.perf_counter()
        for epoch in range(self.configuration.update_epochs):
            permutation = mx.random.permutation(
                sample_count,
                key=mx.random.key(
                    self._permutation_seed(
                        self.configuration.seed,
                        self.revision,
                        epoch,
                    )
                ),
            )
            for offset in range(
                0,
                sample_count,
                self.configuration.minibatch_size,
            ):
                indices = permutation[
                    offset : offset
                    + self.configuration.minibatch_size
                ]
                loss, metrics = self._compiled_training_step(
                    batch.actor_observations[indices],
                    batch.critic_observations[indices],
                    batch.latents[indices],
                    batch.old_log_probabilities[indices],
                    old_means[indices],
                    old_log_standard_deviation,
                    batch.old_values[indices],
                    advantages[indices],
                    batch.returns[indices],
                )
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
                if not all(
                    math.isfinite(value)
                    for value in numeric.values()
                ):
                    raise FloatingPointError(
                        "PPO minibatch produced non-finite metrics"
                    )
                for key, value in numeric.items():
                    totals[key] = totals.get(key, 0.0) + value
                update_count += 1
                if (
                    self.configuration.adaptive_learning_rate
                    and
                    self.configuration.target_kl is not None
                ):
                    current_learning_rate = float(
                        self.optimizer.learning_rate.item()
                    )
                    if (
                        numeric["kl_divergence"]
                        > 2.0 * self.configuration.target_kl
                    ):
                        current_learning_rate = max(
                            self.configuration
                                .minimum_learning_rate,
                            current_learning_rate / 1.5,
                        )
                    elif (
                        0.0 < numeric["kl_divergence"]
                        < 0.5 * self.configuration.target_kl
                    ):
                        current_learning_rate = min(
                            self.configuration
                                .maximum_learning_rate,
                            current_learning_rate * 1.5,
                        )
                    self.optimizer.learning_rate = (
                        current_learning_rate
                    )
                totals["learning_rate"] = (
                    totals.get("learning_rate", 0.0)
                    + float(self.optimizer.learning_rate.item())
                )
        self.revision += 1
        result = {
            key: value / max(update_count, 1)
            for key, value in totals.items()
        }
        result["update_seconds"] = time.perf_counter() - start
        result["minibatch_updates"] = float(update_count)
        result["policy_revision"] = float(self.revision)
        log_standard_deviation = np.asarray(
            mx.clip(
                self.model.log_standard_deviation,
                -5.0,
                2.0,
            ),
            dtype=np.float32,
        )
        result["mean_action_standard_deviation"] = float(
            np.mean(np.exp(log_standard_deviation))
        )
        return result

    def write_policy_pack(
        self,
        output: str | Path,
        *,
        policy_id: str | None = None,
        observation_mean:
            np.ndarray | tuple[float, ...] | None = None,
        observation_inverse_standard_deviation:
            np.ndarray | tuple[float, ...] | None = None,
        critic_observation_mean:
            np.ndarray | tuple[float, ...] | None = None,
        critic_observation_inverse_standard_deviation:
            np.ndarray | tuple[float, ...] | None = None,
        action_bias:
            np.ndarray | tuple[float, ...] | None = None,
        action_scale:
            np.ndarray | tuple[float, ...] | None = None,
        observation_clip: float | None = None,
        action_clip: float | None = None,
        stochastic: bool = True,
        library_path: str | Path | None = None,
    ) -> Path:
        resolved_policy_id = policy_id or self.policy_id
        if not resolved_policy_id:
            raise ValueError(
                "policy_id is required for a new PolicyPack"
            )
        if any(
            value is not None
            for value in (
                observation_mean,
                observation_inverse_standard_deviation,
                critic_observation_mean,
                critic_observation_inverse_standard_deviation,
                observation_clip,
            )
        ):
            self.set_observation_normalization(
                actor_mean=observation_mean,
                actor_inverse_standard_deviation=(
                    observation_inverse_standard_deviation
                ),
                critic_mean=critic_observation_mean,
                critic_inverse_standard_deviation=(
                    critic_observation_inverse_standard_deviation
                ),
                observation_clip=observation_clip,
            )
        if any(
            value is not None
            for value in (
                action_bias,
                action_scale,
                action_clip,
            )
        ):
            self.set_action_contract(
                action_bias=action_bias,
                action_scale=action_scale,
                action_clip=action_clip,
            )
        self.policy_id = resolved_policy_id
        actor_layers = []
        actor_linear = [
            layer
            for layer in self.model.actor.layers
            if isinstance(layer, nn.Linear)
        ]
        for index, layer in enumerate(actor_linear):
            mx.eval(layer.weight, layer.bias)
            actor_layers.append(
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
                        0
                        if index + 1 == len(actor_linear)
                        else 3
                    ),
                )
            )
        critic_layers = []
        critic_linear = [
            layer
            for layer in self.model.critic.layers
            if isinstance(layer, nn.Linear)
        ]
        for index, layer in enumerate(critic_linear):
            mx.eval(layer.weight, layer.bias)
            critic_layers.append(
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
                        0
                        if index + 1 == len(critic_linear)
                        else 3
                    ),
                )
            )
        log_standard_deviation = mx.clip(
            self.model.log_standard_deviation,
            -5.0,
            2.0,
        )
        mx.eval(log_standard_deviation)
        mx.eval(
            self.model.actor_observation_mean,
            self.model.actor_observation_inverse_standard_deviation,
            self.model.critic_observation_mean,
            self.model
                .critic_observation_inverse_standard_deviation,
        )
        return write_policy_pack(
            output,
            policy_id=resolved_policy_id,
            revision=self.revision,
            layers=actor_layers,
            critic_layers=critic_layers,
            observation_mean=np.asarray(
                self.model.actor_observation_mean,
                dtype=np.float32,
            ),
            observation_inverse_standard_deviation=(
                np.asarray(
                    self.model
                        .actor_observation_inverse_standard_deviation,
                    dtype=np.float32,
                )
            ),
            critic_observation_mean=np.asarray(
                self.model.critic_observation_mean,
                dtype=np.float32,
            ),
            critic_observation_inverse_standard_deviation=(
                np.asarray(
                    self.model
                        .critic_observation_inverse_standard_deviation,
                    dtype=np.float32,
                )
            ),
            action_log_standard_deviation=(
                np.asarray(
                    log_standard_deviation,
                    dtype=np.float32,
                )
                if stochastic
                else ()
            ),
            action_bias=self.action_bias,
            action_scale=self.action_scale,
            observation_clip=self.model.observation_clip,
            action_clip=self.action_clip,
            library_path=library_path,
        )


__all__ = [
    "MLXActorCritic",
    "MLXPPOConfiguration",
    "MLXPolicyBatch",
    "MLXPolicyLearner",
    "NativePolicyDenseLayer",
    "NativePolicyPack",
    "NativePolicyRollout",
    "read_policy_pack",
    "read_policy_rollout_pack",
]
