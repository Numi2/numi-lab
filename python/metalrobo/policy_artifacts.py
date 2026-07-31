"""Read-only PolicyPack and PolicyRolloutPack artifact contracts.

Production learning lives in Swift/MLX.  This offline module deliberately has
no optimizer, simulator, rollout scheduler, or Metal command-queue ownership.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native import (
    learning_pack_content_hash,
)

_LEARNING_PACK_HEADER = struct.Struct("<8sIIQQ")
_POLICY_KIND = 2
_POLICY_FORMAT_VERSION = 3
_POLICY_ROLLOUT_KIND = 3
_POLICY_ROLLOUT_FORMAT_VERSION = 3
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
        ("timeout_bootstrap_value", "<f4"),
        ("episode_tracking_score", "<f4"),
        ("curriculum_level", "<u4"),
        ("terrain_level", "<u4"),
        ("reserved_0", "<u4"),
        ("reserved_1", "<u4"),
    ],
    align=False,
)
__all__ = [
    "NativePolicyDenseLayer",
    "NativePolicyPack",
    "NativePolicyRollout",
    "read_policy_pack",
    "read_policy_rollout_pack",
]
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
    format_version: int

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
        mean = self.actor_mean(observations)
        # V2 is readable only for truthful historical deployment evidence.
        # Its persisted action contract was tanh(mean); V3 is raw Gaussian.
        policy_output = (
            np.tanh(mean) if self.format_version == 2 else mean
        )
        return np.clip(
            self.effective_action_scale * policy_output
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
    if format_version not in (2, _POLICY_FORMAT_VERSION):
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
        format_version=format_version,
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
        transitions["timeout_bootstrap_value"],
        transitions["episode_tracking_score"],
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


__all__ = [
    "NativePolicyDenseLayer",
    "NativePolicyPack",
    "NativePolicyRollout",
    "read_policy_pack",
    "read_policy_rollout_pack",
]
