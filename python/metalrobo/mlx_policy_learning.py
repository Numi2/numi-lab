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
from dataclasses import dataclass, field, replace
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
_POLICY_FORMAT_VERSION = 4
_POLICY_ROLLOUT_KIND = 3
_POLICY_ROLLOUT_FORMAT_VERSION = 7
_MOTION_KIND = 4
_MOTION_FORMAT_VERSION = 1
_TRANSITION_DTYPE = np.dtype(
    [
        ("reward", "<f4"),
        ("timeout_bootstrap_value", "<f4"),
        ("reserved_float0", "<f4"),
        ("reserved_float1", "<f4"),
        ("done", "<u4"),
        ("timeout", "<u4"),
        ("physics_error", "<u4"),
        ("termination_reason", "<u4"),
        ("difficulty_band", "<u4"),
        ("terrain_level", "<u4"),
        ("impact_sequence_index", "<u4"),
        ("impact_event_flags", "<u4"),
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
    motion_feature_count: int
    actor_observations: np.ndarray
    critic_observations: np.ndarray
    motion_features: np.ndarray
    latents: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    bootstrap_values: np.ndarray
    transitions: np.ndarray
    outcomes: dict[str, np.ndarray] = field(default_factory=dict)
    outcome_units: dict[str, str] = field(default_factory=dict)
    outcome_directions: dict[str, int] = field(default_factory=dict)
    teacher_actions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    @property
    def sample_count(self) -> int:
        return self.environment_count * self.control_step_count

    def outcome(
        self,
        identifier: str,
        *,
        default: float = 0.0,
    ) -> np.ndarray:
        values = self.outcomes.get(identifier)
        if values is not None:
            return values
        return np.full(self.sample_count, default, dtype=np.float32)

    def policy_batch(
        self,
        *,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        rewards: np.ndarray | None = None,
        navigation_teacher_minimum_waypoint: int = 5,
        navigation_return_weighted_self_imitation: bool = False,
    ) -> "MLXPolicyBatch":
        """Compute terminal-safe GAE and publish only learner tensors to MLX."""

        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 1 <= navigation_teacher_minimum_waypoint <= 5:
            raise ValueError(
                "navigation teacher minimum waypoint must be in [1, 5]"
            )
        if (
            navigation_return_weighted_self_imitation
            and not self.teacher_actions.size
        ):
            raise ValueError(
                "navigation return-weighted self-imitation requires the "
                "interaction teacher stream"
            )
        steps = self.control_step_count
        environments = self.environment_count
        if rewards is None:
            rewards = self.transitions["reward"]
        rewards = np.asarray(
            rewards,
            dtype=np.float32,
        ).reshape(steps, environments)
        done = self.transitions["done"].reshape(
            steps,
            environments,
        ).astype(np.float32)
        timeout = self.transitions["timeout"].reshape(
            steps,
            environments,
        ).astype(np.float32)
        physics_error = self.transitions["physics_error"].reshape(
            steps,
            environments,
        ).astype(np.float32)
        timeout_values = self.transitions[
            "timeout_bootstrap_value"
        ].reshape(steps, environments)
        values = self.old_values.reshape(steps, environments)
        advantages = np.zeros_like(values, dtype=np.float32)
        running = np.zeros((environments,), dtype=np.float32)
        for step in range(steps - 1, -1, -1):
            next_values = (
                self.bootstrap_values
                if step + 1 == steps
                else values[step + 1]
            )
            continuing = 1.0 - done[step]
            timeout_bootstrap = (
                timeout[step]
                * (1.0 - physics_error[step])
                * timeout_values[step]
            )
            delta = (
                rewards[step]
                + discount * timeout_bootstrap
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
        teacher_actions = self.teacher_actions.reshape(
            self.sample_count,
            self.action_count,
        ) if self.teacher_actions.size else np.zeros(
            (self.sample_count, self.action_count),
            dtype=np.float32,
        )
        if navigation_return_weighted_self_imitation:
            # Actor means and sampled latents share raw Gaussian policy space.
            # Reinforce the sampled action only when the native physical
            # episode passes the navigation outcome gate below; failed and
            # partial routes retain zero teacher weight.
            teacher_actions = self.latents.reshape(
                self.sample_count,
                self.action_count,
            )
        teacher_weights = np.zeros(
            (steps, environments),
            dtype=np.float32,
        )
        policy_weights = np.ones(
            (steps, environments),
            dtype=np.float32,
        )
        if self.teacher_actions.size:
            sequence = self.transitions[
                "impact_sequence_index"
            ].reshape(steps, environments)
            flags = self.transitions[
                "impact_event_flags"
            ].reshape(steps, environments)
            failed = np.logical_or(
                self.transitions["physics_error"].reshape(
                    steps,
                    environments,
                ).astype(bool),
                self.transitions["done"].reshape(
                    steps,
                    environments,
                ).astype(bool),
            )
            root_height = self.outcome("root_height").reshape(
                steps,
                environments,
            )
            tilt = self.outcome("tilt").reshape(
                steps,
                environments,
            )
            for environment in range(environments):
                start: int | None = None
                active_sequence = 0
                environment_flags = flags[:, environment].astype(
                    np.uint32
                )
                get_up = bool(
                    root_height[0, environment] < 0.5
                    or np.any(
                        environment_flags
                        & np.uint32((1 << 31) | (1 << 30))
                    )
                )
                for step in range(steps):
                    current = int(sequence[step, environment])
                    residual_control = bool(
                        int(environment_flags[step]) & (1 << 23)
                    )
                    interaction_teacher = bool(
                        int(environment_flags[step]) & (1 << 22)
                    )
                    policy_weights[step, environment] = float(
                        residual_control or
                        (
                            not interaction_teacher
                            and current == 0
                            and not get_up
                        )
                    )
                    if residual_control:
                        # ARDY already supplies the absolute motion. The
                        # sampled actor action is an executed residual, so it
                        # has a valid PPO ratio and must not be trained toward
                        # the absolute teacher pose a second time.
                        start = None
                        continue
                    if current != active_sequence:
                        active_sequence = current
                        start = step if current != 0 else None
                    if failed[step, environment]:
                        start = None
                    if (
                        start is not None
                        and int(flags[step, environment]) & 4
                        and not np.any(
                            failed[start : step + 1, environment]
                        )
                    ):
                        prefix = slice(start, step + 1)
                        prefix_stability = np.clip(
                            1.0
                            - tilt[prefix, environment]
                            / np.float32(np.pi),
                            0.0,
                            1.0,
                        )
                        teacher_weights[prefix, environment] = np.maximum(
                            teacher_weights[prefix, environment],
                            0.5 + 0.5 * prefix_stability,
                        )
                        start = None
                    if not get_up or failed[
                        step,
                        environment,
                    ]:
                        if interaction_teacher and not failed[
                            step,
                            environment,
                        ]:
                            # Generated motion is intent, while Metal supplies
                            # the physical evidence. Retain stable partial
                            # progress continuously instead of requiring a
                            # binary task-completion gate.
                            stability = np.clip(
                                1.0
                                - tilt[step, environment]
                                / np.float32(np.pi),
                                0.0,
                                1.0,
                            )
                            height_quality = np.clip(
                                root_height[step, environment]
                                / np.float32(0.78),
                                0.0,
                                1.0,
                            )
                            teacher_weights[step, environment] = max(
                                teacher_weights[step, environment],
                                np.float32(
                                    0.15
                                    + 0.45 * stability
                                    + 0.40 * height_quality
                                ),
                            )
                        continue
                    previous = max(step - 1, 0)
                    height_progress = max(
                        float(root_height[step, environment])
                        - float(root_height[previous, environment]),
                        0.0,
                    )
                    tilt_recovery = max(
                        float(tilt[previous, environment])
                        - float(tilt[step, environment]),
                        0.0,
                    )
                    stability = float(
                        np.clip(
                            1.0
                            - tilt[step, environment]
                            / np.float32(np.pi),
                            0.0,
                            1.0,
                        )
                    )
                    outcome = int(flags[step, environment])
                    standing = float((outcome & (1 << 31)) != 0)
                    restored = float((outcome & (1 << 30)) != 0)
                    teacher_weights[step, environment] = np.float32(
                        np.clip(
                            0.10
                            + 0.35 * stability
                            + 2.0 * height_progress
                            + 0.5 * tilt_recovery
                            + 0.20 * standing
                            + 0.35 * restored,
                            0.0,
                            1.0,
                        )
                    )
            navigation_waypoints = self.outcome(
                "navigation_waypoints_reached"
            ).reshape(steps, environments)
            navigation_completion = self.outcome(
                "navigation_completion"
            ).reshape(steps, environments)
            interaction_teacher = (
                flags.astype(np.uint32) & np.uint32(1 << 22)
            ) != 0
            navigation_teacher = np.logical_and(
                interaction_teacher,
                np.logical_or(
                    navigation_waypoints > 0.0,
                    navigation_completion > 0.0,
                ),
            )
            # A navigation teacher is useful only when its accepted physical
            # episode actually reaches the terminal waypoint. Do not distill
            # stable-looking partial routes or failure prefixes. Back-label
            # the realized successful episode, including its completion step,
            # while keeping teacher-executed actions out of the PPO ratio.
            for environment in range(environments):
                if not np.any(navigation_teacher[:, environment]):
                    continue
                teacher_weights[
                    navigation_teacher[:, environment], environment
                ] = 0.0
                episode_start = 0
                for step in range(steps):
                    episode_end = bool(done[step, environment]) or (
                        step + 1 == steps
                    )
                    if not episode_end:
                        continue
                    upper = step + 1
                    segment = slice(episode_start, upper)
                    eligible = navigation_teacher[segment, environment]
                    realized = bool(
                        np.any(
                            navigation_waypoints[segment, environment] >=
                                float(navigation_teacher_minimum_waypoint)
                        ) or (
                            navigation_teacher_minimum_waypoint == 5 and
                            np.any(
                                navigation_completion[segment, environment]
                                > 0.0
                            )
                        )
                    )
                    if realized and np.any(eligible):
                        stability = np.clip(
                            1.0
                            - tilt[segment, environment]
                            / np.float32(np.pi),
                            0.0,
                            1.0,
                        )
                        qualified = np.where(
                            eligible,
                            0.5 + 0.5 * stability,
                            0.0,
                        ).astype(np.float32)
                        teacher_weights[segment, environment] = np.maximum(
                            teacher_weights[segment, environment],
                            qualified,
                        )
                    episode_start = upper
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
            teacher_actions=teacher_actions,
            teacher_weights=teacher_weights.reshape(
                self.sample_count
            ),
            policy_weights=policy_weights.reshape(
                self.sample_count
            ),
        )


@dataclass(frozen=True, slots=True)
class NativeMotionClip:
    id: str
    frames_per_second: float
    features: np.ndarray


@dataclass(frozen=True, slots=True)
class NativeMotionPack:
    id: str
    source_repository: str
    source_revision: str
    license: str
    anchor_body: str
    tracked_bodies: tuple[str, ...]
    feature_count: int
    clips: tuple[NativeMotionClip, ...]
    content_hash: int


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
    contract_version: int
    world_fingerprint: int
    task_fingerprint: int
    observation_fingerprint: int
    action_fingerprint: int
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
    if format_version not in (2, 3, _POLICY_FORMAT_VERSION):
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
    if format_version >= 4:
        contract_version = reader.unsigned64()
        world_fingerprint = reader.unsigned64()
        task_fingerprint = reader.unsigned64()
        observation_fingerprint = reader.unsigned64()
        action_fingerprint = reader.unsigned64()
        if (
            contract_version != 1
            or min(
                world_fingerprint,
                task_fingerprint,
                observation_fingerprint,
                action_fingerprint,
            ) <= 0
        ):
            raise ValueError("PolicyPack semantic contract is incomplete")
    else:
        contract_version = 0
        world_fingerprint = 0
        task_fingerprint = 0
        observation_fingerprint = 0
        action_fingerprint = 0
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
        contract_version=contract_version,
        world_fingerprint=world_fingerprint,
        task_fingerprint=task_fingerprint,
        observation_fingerprint=observation_fingerprint,
        action_fingerprint=action_fingerprint,
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
    motion_feature_count = reader.unsigned32()
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
    motion_features = reader.vector(
        np.dtype("<f4"),
        sample_count * motion_feature_count,
    )
    teacher_actions = reader.vector(np.dtype("<f4"))
    if teacher_actions.size not in (0, sample_count * action_count):
        raise ValueError(
            "PolicyRolloutPack teacher action shape is invalid"
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
    outcome_count = reader.unsigned64()
    if outcome_count > 4096:
        raise ValueError("PolicyRolloutPack outcome schema is invalid")
    outcome_schema: list[tuple[str, str, int]] = []
    for _ in range(outcome_count):
        outcome_id = reader.string()
        outcome_unit = reader.string()
        outcome_direction = reader.unsigned32()
        if (
            not outcome_id
            or outcome_direction > 2
            or any(existing[0] == outcome_id for existing in outcome_schema)
        ):
            raise ValueError("PolicyRolloutPack outcome descriptor is invalid")
        outcome_schema.append(
            (outcome_id, outcome_unit, outcome_direction)
        )
    outcome_values = reader.vector(
        np.dtype("<f4"),
        sample_count * outcome_count,
    ).reshape(sample_count, outcome_count)
    transitions = reader.vector(
        _TRANSITION_DTYPE,
        sample_count,
    )
    if reader.cursor != len(payload):
        raise ValueError("PolicyRolloutPack contains trailing payload bytes")
    float_tables = (
        actor,
        critic,
        motion_features,
        teacher_actions,
        latents,
        log_probabilities,
        values,
        bootstrap_values,
        transitions["reward"],
        outcome_values,
        transitions["timeout_bootstrap_value"],
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
        motion_feature_count=motion_feature_count,
        actor_observations=actor,
        critic_observations=critic,
        motion_features=motion_features,
        teacher_actions=teacher_actions,
        latents=latents,
        old_log_probabilities=log_probabilities,
        old_values=values,
        bootstrap_values=bootstrap_values,
        transitions=transitions,
        outcomes={
            identifier: outcome_values[:, index]
            for index, (identifier, _, _) in enumerate(outcome_schema)
        },
        outcome_units={
            identifier: unit for identifier, unit, _ in outcome_schema
        },
        outcome_directions={
            identifier: direction
            for identifier, _, direction in outcome_schema
        },
    )


def read_motion_pack(
    path: str | Path,
    *,
    library_path: str | Path | None = None,
) -> NativeMotionPack:
    """Read the canonical provenance-carrying expert MotionPack."""

    source = Path(path)
    data = np.memmap(source, mode="r", dtype=np.uint8)
    if data.size < _LEARNING_PACK_HEADER.size:
        raise ValueError("MotionPack file is truncated")
    magic, version, kind, payload_bytes, expected_hash = (
        _LEARNING_PACK_HEADER.unpack_from(data)
    )
    if (
        magic != b"MRLEARN\0"
        or version != _MOTION_FORMAT_VERSION
        or kind != _MOTION_KIND
    ):
        raise ValueError("MotionPack header, kind, or version is invalid")
    if payload_bytes != data.size - _LEARNING_PACK_HEADER.size:
        raise ValueError("MotionPack payload length is invalid")
    payload_view = data[_LEARNING_PACK_HEADER.size :]
    if (
        learning_pack_content_hash(
            payload_view,
            path=library_path,
        )
        != expected_hash
    ):
        raise ValueError("MotionPack content fingerprint is invalid")
    reader = _PackReader(memoryview(payload_view))
    identifier = reader.string()
    repository = reader.string()
    revision = reader.string()
    license_name = reader.string()
    anchor = reader.string()
    tracked_count = reader.unsigned64()
    tracked = tuple(
        reader.string() for _ in range(tracked_count)
    )
    feature_count = reader.unsigned32()
    clip_count = reader.unsigned64()
    if (
        not identifier
        or not repository
        or not revision
        or not license_name
        or not anchor
        or not tracked
        or feature_count != 9 * len(tracked)
        or clip_count == 0
    ):
        raise ValueError("MotionPack semantic contract is invalid")
    clips: list[NativeMotionClip] = []
    for _ in range(clip_count):
        clip_id = reader.string()
        fps = reader.float32()
        features = reader.vector(np.dtype("<f4"))
        if (
            not clip_id
            or not math.isfinite(fps)
            or fps <= 0.0
            or features.size < 2 * feature_count
            or features.size % feature_count != 0
            or not np.isfinite(features).all()
        ):
            raise ValueError("MotionPack clip is invalid")
        clips.append(
            NativeMotionClip(
                id=clip_id,
                frames_per_second=fps,
                features=features.reshape(
                    -1,
                    feature_count,
                ),
            )
        )
    if reader.cursor != len(reader.payload):
        raise ValueError(
            "MotionPack contains trailing payload bytes"
        )
    return NativeMotionPack(
        id=identifier,
        source_repository=repository,
        source_revision=revision,
        license=license_name,
        anchor_body=anchor,
        tracked_bodies=tracked,
        feature_count=feature_count,
        clips=tuple(clips),
        content_hash=expected_hash,
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
    entropy_coefficient: float = 0.001
    imagination_distillation_coefficient: float = 1.0
    maximum_gradient_norm: float = 1.0
    target_kl: float | None = 0.01
    adaptive_learning_rate: bool = True
    minimum_learning_rate: float = 1.0e-5
    maximum_learning_rate: float = 1.0e-2
    discount: float = 0.99
    gae_lambda: float = 0.95
    # Native stand-first production begins at 0.2 standard deviation in policy
    # coordinates. Unit-standard-deviation parity remains available through
    # the explicit CLI/configuration field.
    initial_log_standard_deviation: float = -1.6094379124341003
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
            "imagination distillation coefficient": (
                self.imagination_distillation_coefficient
            ),
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
        if self.imagination_distillation_coefficient < 0.0:
            raise ValueError(
                "imagination distillation coefficient must be nonnegative"
            )
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
    # Keep rollout-sized tables as NumPy views over the memory-mapped native
    # artifact. Only the active minibatch is transferred to MLX. Materializing
    # the full visual observation tensor on the unified heap needlessly doubles
    # multi-gigabyte rollouts before the first optimizer step.
    actor_observations: np.ndarray
    critic_observations: np.ndarray
    latents: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    teacher_actions: np.ndarray | None = None
    teacher_weights: np.ndarray | None = None
    policy_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Preserve the public direct-construction path while normalizing legacy
        # MLX-array callers to the bounded-memory NumPy representation.
        for name in (
            "actor_observations",
            "critic_observations",
            "latents",
            "old_log_probabilities",
            "old_values",
            "advantages",
            "returns",
        ):
            object.__setattr__(
                self,
                name,
                np.asarray(getattr(self, name), dtype=np.float32),
            )
        sample_count = int(self.actor_observations.shape[0])
        action_count = int(self.latents.shape[1])
        object.__setattr__(
            self,
            "teacher_actions",
            np.zeros(
                (sample_count, action_count),
                dtype=np.float32,
            ) if self.teacher_actions is None else np.asarray(
                self.teacher_actions,
                dtype=np.float32,
            ),
        )
        object.__setattr__(
            self,
            "teacher_weights",
            np.zeros(sample_count, dtype=np.float32)
            if self.teacher_weights is None else np.asarray(
                self.teacher_weights,
                dtype=np.float32,
            ),
        )
        object.__setattr__(
            self,
            "policy_weights",
            np.ones(sample_count, dtype=np.float32)
            if self.policy_weights is None else np.asarray(
                self.policy_weights,
                dtype=np.float32,
            ),
        )

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
        teacher_actions: np.ndarray | None = None,
        teacher_weights: np.ndarray | None = None,
        policy_weights: np.ndarray | None = None,
    ) -> "MLXPolicyBatch":
        return cls(
            actor_observations=np.asarray(
                actor_observations,
                dtype=np.float32,
            ),
            critic_observations=np.asarray(
                critic_observations,
                dtype=np.float32,
            ),
            latents=np.asarray(latents, dtype=np.float32),
            old_log_probabilities=np.asarray(
                old_log_probabilities,
                dtype=np.float32,
            ),
            old_values=np.asarray(old_values, dtype=np.float32),
            advantages=np.asarray(advantages, dtype=np.float32),
            returns=np.asarray(returns, dtype=np.float32),
            teacher_actions=teacher_actions,
            teacher_weights=teacher_weights,
            policy_weights=policy_weights,
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
            "teacher_actions": (
                self.teacher_actions,
                (sample_count, action_count),
            ),
            "teacher_weights": (
                self.teacher_weights,
                (sample_count,),
            ),
            "policy_weights": (
                self.policy_weights,
                (sample_count,),
            ),
        }
        for label, (value, shape) in expected.items():
            if tuple(int(dimension) for dimension in value.shape) != shape:
                raise ValueError(f"{label} batch shape is invalid")
        if sample_count == 0:
            raise ValueError("policy batch must contain samples")
        if np.any(self.teacher_weights < 0.0) or np.any(
            self.teacher_weights > 1.0
        ):
            raise ValueError("teacher weights must be in [0, 1]")
        if np.any(self.policy_weights < 0.0) or np.any(
            self.policy_weights > 1.0
        ):
            raise ValueError("policy weights must be in [0, 1]")
        tables = {
            "actor_observations": self.actor_observations,
            **{
                label: value
                for label, (value, _) in expected.items()
            },
        }
        # Bound validation scratch memory as rollout width grows. The canonical
        # reader already validates its mapped tables, while this also covers
        # derived advantage and return arrays and direct callers.
        rows_per_chunk = 16_384
        invalid = []
        for label, value in tables.items():
            flat = value.reshape(-1)
            if any(
                not np.isfinite(
                    flat[offset : offset + rows_per_chunk]
                ).all()
                for offset in range(0, flat.size, rows_per_chunk)
            ):
                invalid.append(label)
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


def _initialize_mlp(
    network: nn.Sequential,
    *,
    generator: np.random.Generator,
) -> None:
    """Match the default linear-layer scale used by pinned RSL-RL."""

    for layer in network.layers:
        if not isinstance(layer, nn.Linear):
            continue
        shape = tuple(int(value) for value in layer.weight.shape)
        bound = 1.0 / math.sqrt(float(shape[1]))
        layer.weight = mx.array(
            generator.uniform(-bound, bound, shape).astype(np.float32),
            dtype=mx.float32,
        )
        if getattr(layer, "bias", None) is not None:
            layer.bias = mx.array(
                generator.uniform(
                    -bound,
                    bound,
                    tuple(int(value) for value in layer.bias.shape),
                ).astype(np.float32),
                dtype=mx.float32,
            )


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
            generator=generator,
        )
        _initialize_mlp(
            self.critic,
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
        return mx.sum(gaussian, axis=-1)

    @staticmethod
    def entropy(
        mean: mx.array,
        log_standard_deviation: mx.array,
    ) -> mx.array:
        """Exact entropy of a diagonal Gaussian."""

        entropy = mx.sum(
            log_standard_deviation
            + 0.5 * (1.0 + math.log(2.0 * math.pi))
        )
        return mx.broadcast_to(entropy, mean.shape[:-1])

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


@dataclass(frozen=True, slots=True)
class MLXMotionPriorConfiguration:
    hidden_sizes: tuple[int, ...] = (512, 256)
    learning_rate: float = 1.0e-3
    minibatch_size: int = 4096
    update_epochs: int = 2
    reward_coefficient: float = 0.3
    activation_mode: str = "impact"
    maximum_gradient_norm: float = 1.0
    seed: int = 1

    def validate(self) -> None:
        if (
            not self.hidden_sizes
            or any(width <= 0 for width in self.hidden_sizes)
            or self.learning_rate <= 0.0
            or self.minibatch_size <= 0
            or self.update_epochs <= 0
            or self.reward_coefficient < 0.0
            or self.activation_mode not in {"impact", "always"}
            or self.maximum_gradient_norm <= 0.0
            or self.seed < 0
        ):
            raise ValueError(
                "motion-prior configuration is invalid"
            )


class _MotionDiscriminator(nn.Module):
    def __init__(
        self,
        input_count: int,
        hidden_sizes: tuple[int, ...],
        mean: np.ndarray,
        inverse_standard_deviation: np.ndarray,
        seed: int,
    ) -> None:
        super().__init__()
        self.network = _elu_mlp(
            input_count,
            1,
            hidden_sizes,
        )
        self.mean = mx.array(mean, dtype=mx.float32)
        self.inverse_standard_deviation = mx.array(
            inverse_standard_deviation,
            dtype=mx.float32,
        )
        self.freeze(
            recurse=False,
            keys=["mean", "inverse_standard_deviation"],
            strict=True,
        )
        _initialize_mlp(
            self.network,
            generator=np.random.default_rng(seed),
        )

    def __call__(self, values: mx.array) -> mx.array:
        normalized = (
            values - self.mean
        ) * self.inverse_standard_deviation
        return self.network(
            mx.clip(normalized, -10.0, 10.0)
        ).squeeze(-1)


class MLXMotionPrior:
    """Apple-GPU discriminator over consecutive native tracked-link frames."""

    def __init__(
        self,
        motion_pack: NativeMotionPack,
        configuration: MLXMotionPriorConfiguration | None = None,
    ) -> None:
        if configuration is None:
            configuration = MLXMotionPriorConfiguration()
        configuration.validate()
        self.motion_pack = motion_pack
        self.configuration = configuration
        expert = np.concatenate(
            [
                np.concatenate(
                    (
                        clip.features[:-1],
                        clip.features[1:],
                    ),
                    axis=1,
                )
                for clip in motion_pack.clips
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        mean = expert.mean(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        standard_deviation = expert.std(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        inverse = 1.0 / np.maximum(
            standard_deviation,
            1.0e-4,
        )
        self.expert_transitions = mx.array(
            expert,
            dtype=mx.float32,
        )
        self.model = _MotionDiscriminator(
            2 * motion_pack.feature_count,
            configuration.hidden_sizes,
            mean,
            inverse,
            configuration.seed,
        )
        self.optimizer = optim.Adam(
            learning_rate=configuration.learning_rate,
            bias_correction=True,
        )
        self.optimizer.init(
            self.model.trainable_parameters()
        )
        self._loss_and_gradient = nn.value_and_grad(
            self.model,
            self._loss,
        )
        mx.eval(
            self.expert_transitions,
            self.model.parameters(),
            self.optimizer.state,
        )

    def _loss(
        self,
        expert: mx.array,
        policy: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        expert_score = self.model(expert)
        policy_score = self.model(policy)
        expert_loss = mx.mean(
            mx.square(expert_score - 1.0)
        )
        policy_loss = mx.mean(
            mx.square(policy_score + 1.0)
        )
        loss = 0.5 * (expert_loss + policy_loss)
        return loss, {
            "motion_discriminator_loss": loss,
            "motion_expert_score": mx.mean(expert_score),
            "motion_policy_score": mx.mean(policy_score),
        }

    def update_and_rewards(
        self,
        rollout: NativePolicyRollout,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if (
            rollout.motion_feature_count
            != self.motion_pack.feature_count
        ):
            raise ValueError(
                "rollout and MotionPack feature contracts differ"
            )
        steps = rollout.control_step_count
        environments = rollout.environment_count
        frames = rollout.motion_features.reshape(
            steps,
            environments,
            rollout.motion_feature_count,
        )
        if steps < 2:
            raise ValueError(
                "motion prior requires at least two rollout steps"
            )
        continuing = rollout.transitions["done"].reshape(
            steps,
            environments,
        )[:-1] == 0
        policy_pairs = np.concatenate(
            (frames[:-1], frames[1:]),
            axis=2,
        )[continuing].astype(np.float32, copy=False)
        if policy_pairs.shape[0] == 0:
            return (
                np.zeros(
                    rollout.sample_count,
                    dtype=np.float32,
                ),
                {},
            )
        policy = mx.array(policy_pairs, dtype=mx.float32)
        generator = np.random.default_rng(
            self.configuration.seed + rollout.policy_revision
        )
        last_metrics: dict[str, mx.array] = {}
        batch_size = min(
            self.configuration.minibatch_size,
            policy_pairs.shape[0],
            int(self.expert_transitions.shape[0]),
        )
        for _ in range(self.configuration.update_epochs):
            policy_indices = generator.integers(
                0,
                policy_pairs.shape[0],
                size=batch_size,
            )
            expert_indices = generator.integers(
                0,
                int(self.expert_transitions.shape[0]),
                size=batch_size,
            )
            (loss, last_metrics), gradients = (
                self._loss_and_gradient(
                    self.expert_transitions[
                        mx.array(expert_indices)
                    ],
                    policy[mx.array(policy_indices)],
                )
            )
            gradients, _ = optim.clip_grad_norm(
                gradients,
                max_norm=(
                    self.configuration.maximum_gradient_norm
                ),
            )
            self.optimizer.update(self.model, gradients)
            mx.eval(
                loss,
                self.model.parameters(),
                self.optimizer.state,
            )
        scores = self.model(policy)
        style = mx.clip(
            1.0 - 0.25 * mx.square(scores - 1.0),
            0.0,
            1.0,
        )
        mx.eval(style, *last_metrics.values())
        rewards = np.zeros(
            (steps, environments),
            dtype=np.float32,
        )
        active = (
            np.ones((steps - 1, environments), dtype=bool)
            if self.configuration.activation_mode == "always"
            else rollout.transitions[
                "impact_sequence_index"
            ].reshape(steps, environments)[1:] != 0
        )
        rewards[1:][continuing] = np.asarray(
            style,
            dtype=np.float32,
        )
        rewards[1:] *= active
        metrics = {
            key: float(value.item())
            for key, value in last_metrics.items()
        }
        metrics["motion_reward_mean"] = float(
            rewards.mean()
        )
        return rewards.reshape(-1), metrics

    def blend_rewards(
        self,
        rollout: NativePolicyRollout,
    ) -> tuple[np.ndarray, dict[str, float]]:
        motion_rewards, metrics = self.update_and_rewards(
            rollout
        )
        task_rewards = np.asarray(
            rollout.transitions["reward"],
            dtype=np.float32,
        )
        return (
            task_rewards
            + self.configuration.reward_coefficient
            * motion_rewards,
            metrics,
        )


def _actor_observation_extension_only_gradients(
    gradients: Any,
    offset: int,
    count: int | None = None,
    train_output_layer: bool = False,
    output_action_indices: tuple[int, ...] | None = None,
) -> Any:
    """Keep critic gradients and a protected actor route adapter.

    The default preserves the historical extension-only behavior.  The
    optional output-layer path also trains the final control projection while
    freezing every inherited hidden layer and the exploration parameter.
    """

    flattened = list(tree_flatten(gradients))
    actor_weight_layers = [
        (int(name.split(".")[2]), name.rsplit(".", 1)[0])
        for name, gradient in flattened
        if name.startswith("actor.layers.")
        and name.endswith(".weight")
        and gradient.ndim == 2
    ]
    train_any_output = train_output_layer or output_action_indices is not None
    output_layer = (
        max(actor_weight_layers)[1]
        if train_any_output and actor_weight_layers
        else None
    )
    masked_gradients: list[tuple[str, mx.array]] = []
    found_first_layer = False
    found_output_layer = False
    for name, gradient in flattened:
        if name == "actor.layers.0.weight":
            extension_count = (
                gradient.shape[1] - offset if count is None else count
            )
            if (
                offset < 0
                or extension_count <= 0
                or offset + extension_count > gradient.shape[1]
            ):
                raise ValueError(
                    "actor observation extension range is invalid"
                )
            extension_mask = mx.concatenate(
                (
                    mx.zeros(
                        (gradient.shape[0], offset),
                        dtype=gradient.dtype,
                    ),
                    mx.ones(
                        (gradient.shape[0], extension_count),
                        dtype=gradient.dtype,
                    ),
                    mx.zeros(
                        (
                            gradient.shape[0],
                            gradient.shape[1] - offset - extension_count,
                        ),
                        dtype=gradient.dtype,
                    ),
                ),
                axis=1,
            )
            gradient = gradient * extension_mask
            found_first_layer = True
        elif output_layer is not None and name.startswith(output_layer + "."):
            if output_action_indices is not None:
                if (
                    not output_action_indices
                    or min(output_action_indices) < 0
                    or max(output_action_indices) >= gradient.shape[0]
                    or len(set(output_action_indices)) !=
                        len(output_action_indices)
                ):
                    raise ValueError(
                        "actor output action range is invalid"
                    )
                row_mask = mx.array(
                    [
                        1.0 if index in output_action_indices else 0.0
                        for index in range(gradient.shape[0])
                    ],
                    dtype=gradient.dtype,
                ).reshape(
                    (gradient.shape[0],) +
                    (1,) * (gradient.ndim - 1)
                )
                gradient = gradient * row_mask
            found_output_layer = True
        elif name.startswith("actor.") or name == "log_standard_deviation":
            gradient = mx.zeros_like(gradient)
        masked_gradients.append((name, gradient))
    if not found_first_layer:
        raise ValueError("actor gradient tree has no first dense layer")
    if train_any_output and not found_output_layer:
        raise ValueError("actor gradient tree has no dense output layer")
    return tree_unflatten(masked_gradients)


def _actor_residual_output_only_gradients(
    gradients: Any,
    residual_width: int,
    output_action_indices: tuple[int, ...] | None = None,
    hidden_observation_range: tuple[int, int] | None = None,
) -> Any:
    """Train an isolated residual tail and optional hidden feature path.

    The residual feature path is fixed at initialization, every inherited
    actor parameter and the exploration head remain frozen, and the critic
    remains fully trainable.  This makes the initial deployment exactly equal
    to its source actor while giving route features their own control readout.
    """

    flattened = list(tree_flatten(gradients))
    actor_weight_layers = [
        (int(name.split(".")[2]), name.rsplit(".", 1)[0])
        for name, gradient in flattened
        if name.startswith("actor.layers.")
        and name.endswith(".weight")
        and gradient.ndim == 2
    ]
    if not actor_weight_layers:
        raise ValueError("actor gradient tree has no dense output layer")
    output_layer_index, output_layer = max(actor_weight_layers)
    masked_gradients: list[tuple[str, mx.array]] = []
    found_output_weight = False
    for name, gradient in flattened:
        if name == output_layer + ".weight":
            if residual_width <= 0 or residual_width >= gradient.shape[1]:
                raise ValueError("actor residual width is invalid")
            column_mask = mx.concatenate(
                (
                    mx.zeros(
                        (gradient.shape[0], gradient.shape[1] - residual_width),
                        dtype=gradient.dtype,
                    ),
                    mx.ones(
                        (gradient.shape[0], residual_width),
                        dtype=gradient.dtype,
                    ),
                ),
                axis=1,
            )
            gradient = gradient * column_mask
            if output_action_indices is not None:
                if (
                    not output_action_indices
                    or min(output_action_indices) < 0
                    or max(output_action_indices) >= gradient.shape[0]
                    or len(set(output_action_indices)) !=
                        len(output_action_indices)
                ):
                    raise ValueError(
                        "actor residual output actions are invalid"
                    )
                row_mask = mx.array(
                    [
                        1.0 if index in output_action_indices else 0.0
                        for index in range(gradient.shape[0])
                    ],
                    dtype=gradient.dtype,
                ).reshape((gradient.shape[0], 1))
                gradient = gradient * row_mask
            found_output_weight = True
        elif (
            hidden_observation_range is not None
            and name.startswith("actor.layers.")
            and name.endswith(".weight")
            and gradient.ndim == 2
        ):
            layer_index = int(name.split(".")[2])
            if layer_index >= output_layer_index:
                raise ValueError("actor residual hidden-layer order is invalid")
            if residual_width <= 0 or residual_width >= gradient.shape[0]:
                raise ValueError("actor residual hidden width is invalid")
            row_mask = mx.concatenate(
                (
                    mx.zeros(
                        (gradient.shape[0] - residual_width, 1),
                        dtype=gradient.dtype,
                    ),
                    mx.ones(
                        (residual_width, 1),
                        dtype=gradient.dtype,
                    ),
                ),
                axis=0,
            )
            if layer_index == min(actor_weight_layers)[0]:
                observation_offset, observation_count = (
                    hidden_observation_range
                )
                if (
                    observation_offset < 0
                    or observation_count <= 0
                    or observation_offset + observation_count >
                        gradient.shape[1]
                ):
                    raise ValueError(
                        "actor residual hidden observation range is invalid"
                    )
                column_mask = mx.concatenate(
                    (
                        mx.zeros(
                            (1, observation_offset),
                            dtype=gradient.dtype,
                        ),
                        mx.ones(
                            (1, observation_count),
                            dtype=gradient.dtype,
                        ),
                        mx.zeros(
                            (
                                1,
                                gradient.shape[1] - observation_offset -
                                    observation_count,
                            ),
                            dtype=gradient.dtype,
                        ),
                    ),
                    axis=1,
                )
            else:
                if residual_width >= gradient.shape[1]:
                    raise ValueError(
                        "actor residual hidden input width is invalid"
                    )
                column_mask = mx.concatenate(
                    (
                        mx.zeros(
                            (1, gradient.shape[1] - residual_width),
                            dtype=gradient.dtype,
                        ),
                        mx.ones(
                            (1, residual_width),
                            dtype=gradient.dtype,
                        ),
                    ),
                    axis=1,
                )
            gradient = gradient * row_mask * column_mask
        elif name.startswith("actor.") or name == "log_standard_deviation":
            gradient = mx.zeros_like(gradient)
        masked_gradients.append((name, gradient))
    if not found_output_weight:
        raise ValueError("actor gradient tree has no residual output weight")
    return tree_unflatten(masked_gradients)


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
        self._actor_observation_extension_only_range: (
            tuple[
                int,
                int | None,
                bool,
                tuple[int, ...] | None,
            ] | None
        ) = None
        self._actor_residual_output_only: (
            tuple[
                int,
                tuple[int, ...] | None,
                tuple[int, int] | None,
            ] | None
        ) = None
        self.refresh_compiled_training_state()
        self.policy_id: str | None = None
        self.contract_version = 0
        self.world_fingerprint = 0
        self.task_fingerprint = 0
        self.observation_fingerprint = 0
        self.action_fingerprint = 0
        self.action_bias = np.zeros(
            (action_count,),
            dtype=np.float32,
        )
        self.action_scale = np.ones(
            (action_count,),
            dtype=np.float32,
        )
        self.action_clip = float(np.finfo(np.float32).max)
        self.revision = 1
        mx.eval(self.model.parameters(), self.optimizer.state)

    def bind_contract(
        self,
        *,
        world_fingerprint: int,
        task_fingerprint: int,
        observation_fingerprint: int,
        action_fingerprint: int,
    ) -> None:
        values = (
            int(world_fingerprint),
            int(task_fingerprint),
            int(observation_fingerprint),
            int(action_fingerprint),
        )
        if min(values) <= 0 or max(values) > np.iinfo(np.uint64).max:
            raise ValueError("policy semantic fingerprints must be nonzero uint64 values")
        self.contract_version = 1
        (
            self.world_fingerprint,
            self.task_fingerprint,
            self.observation_fingerprint,
            self.action_fingerprint,
        ) = values

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

    def train_actor_observation_extension_only(
        self,
        offset: int,
        count: int | None = None,
        *,
        train_output_layer: bool = False,
        output_action_indices: tuple[int, ...] | None = None,
    ) -> None:
        """Freeze the inherited actor and train only new input columns.

        The critic remains fully trainable. This is intended for a
        zero-connected observation extension whose inherited actor already
        owns a qualified control behavior.
        """

        if (
            offset < 0
            or offset >= self.actor_observation_count
            or (
                count is not None
                and (
                    count <= 0
                    or offset + count > self.actor_observation_count
                )
            )
        ):
            raise ValueError(
                "actor observation extension range is invalid"
            )
        self._actor_observation_extension_only_range = (
            offset,
            count,
            train_output_layer,
            output_action_indices,
        )
        self.refresh_compiled_training_state()

    def train_actor_residual_output_only(
        self,
        residual_width: int,
        *,
        output_action_indices: tuple[int, ...] | None = None,
        hidden_observation_range: tuple[int, int] | None = None,
    ) -> None:
        """Freeze the carrier and train an isolated residual subnetwork."""

        linear_layers = [
            layer
            for layer in self.model.actor.layers
            if isinstance(layer, nn.Linear)
        ]
        if not linear_layers:
            raise RuntimeError("actor contains no linear output layer")
        if (
            residual_width <= 0
            or residual_width >= linear_layers[-1].weight.shape[1]
        ):
            raise ValueError("actor residual width is invalid")
        if hidden_observation_range is not None:
            offset, count = hidden_observation_range
            if (
                offset < 0
                or count <= 0
                or offset + count > self.actor_observation_count
            ):
                raise ValueError(
                    "actor residual hidden observation range is invalid"
                )
        self._actor_residual_output_only = (
            residual_width,
            output_action_indices,
            hidden_observation_range,
        )
        self.refresh_compiled_training_state()

    def zero_actor_output(self) -> None:
        """Start a stabilizer from the mechanism-default action exactly."""

        linear_layers = [
            layer
            for layer in self.model.actor.layers
            if isinstance(layer, nn.Linear)
        ]
        if not linear_layers:
            raise RuntimeError("actor contains no linear output layer")
        output = linear_layers[-1]
        output.weight = mx.zeros_like(output.weight)
        output.bias = mx.zeros_like(output.bias)
        self.optimizer = optim.Adam(
            learning_rate=self.configuration.learning_rate,
            bias_correction=True,
        )
        self.optimizer.init(self.model.trainable_parameters())
        self.refresh_compiled_training_state()
        mx.eval(self.model.parameters(), self.optimizer.state)

    def zero_actor_observation_range(
        self,
        offset: int,
        count: int,
    ) -> None:
        """Disconnect an unqualified observation range from the actor.

        This preserves every inherited bias, hidden layer, and output row.
        It is intended for a policy that was accidentally optimized while an
        authored sensor range was identically zero: enabling the live sensor
        can then begin from the exact qualified non-sensor controller before
        those first-layer columns are reopened deliberately.
        """

        if (
            offset < 0
            or count <= 0
            or offset + count > self.actor_observation_count
        ):
            raise ValueError("actor observation zero range is invalid")
        first = next(
            (
                layer
                for layer in self.model.actor.layers
                if isinstance(layer, nn.Linear)
            ),
            None,
        )
        if first is None:
            raise RuntimeError("actor contains no linear input layer")
        disconnected = first.weight[:, offset:offset + count]
        zero_input = (
            -self.model.actor_observation_mean[offset:offset + count]
            * self.model.actor_observation_inverse_standard_deviation[
                offset:offset + count
            ]
        )
        if first.bias is None:
            raise RuntimeError("actor input layer has no bias to preserve")
        # PolicyPack normalization happens before the first dense layer.
        # Fold the exact raw-zero contribution into the bias before clearing
        # these columns, so the transformed actor is numerically equivalent
        # on the previously observed all-zero sensor contract.
        first.bias = first.bias + disconnected @ zero_input
        first.weight = mx.concatenate(
            (
                first.weight[:, :offset],
                mx.zeros_like(first.weight[:, offset:offset + count]),
                first.weight[:, offset + count:],
            ),
            axis=1,
        )
        self.refresh_compiled_training_state()
        mx.eval(self.model.parameters())

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

    @classmethod
    def from_policy_pack(
        cls,
        path: str | Path,
        configuration: MLXPPOConfiguration = MLXPPOConfiguration(),
        *,
        actor_observation_count: int | None = None,
        actor_observation_extension_mean: float = 0.0,
        actor_observation_extension_inverse_standard_deviation: float = 1.0,
        actor_observation_extension_offset: int | None = None,
        library_path: str | Path | None = None,
    ) -> "MLXPolicyLearner":
        """Restore trainable PPO state from the canonical native artifact."""

        pack = read_policy_pack(
            path,
            library_path=library_path,
        )
        if pack.format_version == 2:
            raise ValueError(
                "PPO resume requires a v3+ raw-Gaussian "
                "PolicyPack; v2 is evaluation-only because its squashed "
                "behavior distribution cannot be migrated exactly"
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
        target_actor_observations = (
            pack.actor_observation_count
            if actor_observation_count is None
            else actor_observation_count
        )
        if target_actor_observations < pack.actor_observation_count:
            raise ValueError(
                "PPO resume cannot discard PolicyPack observations"
            )
        if not math.isfinite(actor_observation_extension_mean):
            raise ValueError(
                "actor observation extension mean must be finite"
            )
        if (
            not math.isfinite(
                actor_observation_extension_inverse_standard_deviation
            ) or
            actor_observation_extension_inverse_standard_deviation <= 0.0
        ):
            raise ValueError(
                "actor observation extension inverse standard deviation "
                "must be finite and positive"
            )
        extension_count = (
            target_actor_observations - pack.actor_observation_count
        )
        extension_offset = (
            pack.actor_observation_count
            if actor_observation_extension_offset is None
            else actor_observation_extension_offset
        )
        if not 0 <= extension_offset <= pack.actor_observation_count:
            raise ValueError(
                "actor observation extension offset is outside the source contract"
            )
        learner = cls(
            target_actor_observations,
            pack.critic_observation_count,
            pack.action_count,
            restored_configuration,
        )

        def restore_layers(
            network: nn.Sequential,
            source: tuple[NativePolicyDenseLayer, ...],
            *,
            extend_first_input: bool = False,
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
            for layer_index, (target, artifact) in enumerate(
                zip(destination, source, strict=True)
            ):
                weights = np.asarray(
                    artifact.weights,
                    dtype=np.float32,
                )
                if (extend_first_input and layer_index == 0 and
                        extension_count > 0):
                    weights = np.concatenate(
                        (
                            weights[:, :extension_offset],
                            np.zeros(
                                (weights.shape[0], extension_count),
                                dtype=np.float32,
                            ),
                            weights[:, extension_offset:],
                        ),
                        axis=1,
                    )
                target.weight = mx.array(weights, dtype=mx.float32)
                target.bias = mx.array(
                    artifact.bias,
                    dtype=mx.float32,
                )

        restore_layers(
            learner.model.actor,
            pack.layers,
            extend_first_input=True,
        )
        restore_layers(learner.model.critic, pack.critic_layers)
        learner.model.log_standard_deviation = mx.array(
            pack.action_log_standard_deviation,
            dtype=mx.float32,
        )
        actor_mean = np.concatenate(
            (
                pack.effective_observation_mean[:extension_offset],
                np.full(
                    extension_count,
                    actor_observation_extension_mean,
                    dtype=np.float32,
                ),
                pack.effective_observation_mean[extension_offset:],
            ),
        )
        actor_inverse_standard_deviation = np.concatenate(
            (
                pack.effective_observation_inverse_standard_deviation[
                    :extension_offset
                ],
                np.full(
                    extension_count,
                    actor_observation_extension_inverse_standard_deviation,
                    dtype=np.float32,
                ),
                pack.effective_observation_inverse_standard_deviation[
                    extension_offset:
                ],
            ),
        )
        learner.set_observation_normalization(
            actor_mean=actor_mean,
            actor_inverse_standard_deviation=(
                actor_inverse_standard_deviation
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
        learner.contract_version = pack.contract_version
        learner.world_fingerprint = pack.world_fingerprint
        learner.task_fingerprint = pack.task_fingerprint
        learner.observation_fingerprint = pack.observation_fingerprint
        learner.action_fingerprint = pack.action_fingerprint
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

    @classmethod
    def from_actor_policy_pack(
        cls,
        path: str | Path,
        critic_observation_count: int,
        configuration: MLXPPOConfiguration = MLXPPOConfiguration(),
        *,
        preserve_critic: bool = True,
        actor_observation_count: int | None = None,
        actor_observation_extension_mean: float = 0.0,
        actor_observation_extension_inverse_standard_deviation: float = 1.0,
        actor_observation_extension_offset: int | None = None,
        actor_route_residual_observation_offset: int | None = None,
        actor_route_residual_observation_count: int | None = None,
        library_path: str | Path | None = None,
    ) -> "MLXPolicyLearner":
        """Start PPO from an existing actor contract.

        By default, a stochastic PolicyPack preserves its critic and
        exploration head; a deterministic deployment pack creates fresh
        versions of those two training-only components. ``preserve_critic``
        can be disabled when migrating an actor into a task with a different
        privileged critic contract, so the critic is freshly initialized at
        the requested width while the actor remains unchanged. A wider actor
        observation contract is initialized with zero-connected new inputs,
        preserving the actor's exact output until learning uses them.  An
        optional paired-sign route residual widens each hidden layer with a
        fixed feature path and a zero output projection, also preserving the
        actor exactly while isolating later route learning from the carrier.
        Optimizer restoration remains an explicit learner-checkpoint
        operation.
        """

        pack = read_policy_pack(path, library_path=library_path)
        if (
            preserve_critic and
            pack.critic_layers and
            pack.action_log_standard_deviation.size == pack.action_count and
            actor_route_residual_observation_offset is None and
            actor_route_residual_observation_count is None
        ):
            return cls.from_policy_pack(
                path,
                configuration,
                actor_observation_count=actor_observation_count,
                actor_observation_extension_mean=(
                    actor_observation_extension_mean
                ),
                actor_observation_extension_inverse_standard_deviation=(
                    actor_observation_extension_inverse_standard_deviation
                ),
                actor_observation_extension_offset=(
                    actor_observation_extension_offset
                ),
                library_path=library_path,
            )
        expected_activations = [3] * (len(pack.layers) - 1) + [0]
        if [layer.activation for layer in pack.layers] != expected_activations:
            raise ValueError(
                "actor initialization supports ELU hidden layers and an "
                "identity output layer"
            )
        residual_arguments = (
            actor_route_residual_observation_offset,
            actor_route_residual_observation_count,
        )
        if (residual_arguments[0] is None) != (residual_arguments[1] is None):
            raise ValueError(
                "actor route residual requires both observation offset and count"
            )
        residual_observation_count = residual_arguments[1] or 0
        residual_width = 2 * residual_observation_count
        source_hidden_sizes = tuple(
            layer.output_count for layer in pack.layers[:-1]
        )
        if (
            preserve_critic
            and pack.critic_layers
            and pack.critic_observation_count != critic_observation_count
        ):
            raise ValueError(
                "actor initialization source critic disagrees with the "
                "requested observation contract; request a fresh critic"
            )
        preserve_source_critic = bool(
            preserve_critic
            and pack.critic_layers
            and pack.critic_observation_count == critic_observation_count
        )
        restored_configuration = replace(
            configuration,
            hidden_sizes=tuple(
                size + residual_width for size in source_hidden_sizes
            ),
            critic_hidden_sizes=(
                tuple(
                    layer.output_count
                    for layer in pack.critic_layers[:-1]
                )
                if preserve_source_critic
                else configuration.critic_hidden_sizes
            ),
            observation_clip=pack.observation_clip,
        )
        target_actor_observations = (
            pack.actor_observation_count
            if actor_observation_count is None
            else actor_observation_count
        )
        if target_actor_observations < pack.actor_observation_count:
            raise ValueError(
                "actor initialization cannot discard PolicyPack observations"
            )
        if residual_width > 0 and (
            actor_route_residual_observation_offset is None
            or actor_route_residual_observation_offset < 0
            or actor_route_residual_observation_offset +
                residual_observation_count > target_actor_observations
        ):
            raise ValueError(
                "actor route residual observation range is invalid"
            )
        if not math.isfinite(actor_observation_extension_mean):
            raise ValueError(
                "actor observation extension mean must be finite"
            )
        if (
            not math.isfinite(
                actor_observation_extension_inverse_standard_deviation
            ) or
            actor_observation_extension_inverse_standard_deviation <= 0.0
        ):
            raise ValueError(
                "actor observation extension inverse standard deviation "
                "must be finite and positive"
            )
        extension_count = (
            target_actor_observations - pack.actor_observation_count
        )
        extension_offset = (
            pack.actor_observation_count
            if actor_observation_extension_offset is None
            else actor_observation_extension_offset
        )
        if (
            extension_offset < 0
            or extension_offset > pack.actor_observation_count
        ):
            raise ValueError(
                "actor observation extension offset is outside the source contract"
            )
        learner = cls(
            target_actor_observations,
            critic_observation_count,
            pack.action_count,
            restored_configuration,
        )
        destination = [
            layer
            for layer in learner.model.actor.layers
            if isinstance(layer, nn.Linear)
        ]
        if len(destination) != len(pack.layers):
            raise RuntimeError(
                "MLX actor construction disagrees with PolicyPack"
            )
        for layer_index, (target, artifact) in enumerate(
            zip(destination, pack.layers, strict=True)
        ):
            source_weights = np.asarray(
                artifact.weights,
                dtype=np.float32,
            )
            if layer_index == 0 and extension_count > 0:
                source_weights = np.concatenate(
                    (
                        source_weights[:, :extension_offset],
                        np.zeros(
                            (source_weights.shape[0], extension_count),
                            dtype=np.float32,
                        ),
                        source_weights[:, extension_offset:],
                    ),
                    axis=1,
                )
            if residual_width == 0:
                target.weight = mx.array(
                    source_weights,
                    dtype=mx.float32,
                )
                target.bias = mx.array(
                    artifact.bias,
                    dtype=mx.float32,
                )
                continue
            weights = np.zeros(target.weight.shape, dtype=np.float32)
            bias = np.zeros(target.bias.shape, dtype=np.float32)
            source_output_count = artifact.output_count
            source_input_count = source_weights.shape[1]
            weights[
                :source_output_count,
                :source_input_count,
            ] = source_weights
            bias[:source_output_count] = np.asarray(
                artifact.bias,
                dtype=np.float32,
            )
            if layer_index == 0:
                residual_offset = int(
                    actor_route_residual_observation_offset
                )
                for component in range(residual_observation_count):
                    weights[
                        source_output_count + component,
                        residual_offset + component,
                    ] = 1.0
                    weights[
                        source_output_count +
                            residual_observation_count + component,
                        residual_offset + component,
                    ] = -1.0
            elif layer_index < len(pack.layers) - 1:
                previous_source_output_count = (
                    pack.layers[layer_index - 1].output_count
                )
                for component in range(residual_width):
                    weights[
                        source_output_count + component,
                        previous_source_output_count + component,
                    ] = 1.0
            # The final residual columns intentionally remain zero.  Loading
            # this pack is therefore bit-exact with the inherited actor.
            target.weight = mx.array(weights, dtype=mx.float32)
            target.bias = mx.array(bias, dtype=mx.float32)
        if preserve_source_critic:
            critic_destination = [
                layer
                for layer in learner.model.critic.layers
                if isinstance(layer, nn.Linear)
            ]
            if len(critic_destination) != len(pack.critic_layers):
                raise RuntimeError(
                    "MLX critic construction disagrees with PolicyPack"
                )
            for target, artifact in zip(
                critic_destination,
                pack.critic_layers,
                strict=True,
            ):
                target.weight = mx.array(
                    artifact.weights,
                    dtype=mx.float32,
                )
                target.bias = mx.array(
                    artifact.bias,
                    dtype=mx.float32,
                )
        if pack.action_log_standard_deviation.size == pack.action_count:
            learner.model.log_standard_deviation = mx.array(
                pack.action_log_standard_deviation,
                dtype=mx.float32,
            )
        actor_mean = np.concatenate(
            (
                pack.effective_observation_mean[:extension_offset],
                np.full(
                    extension_count,
                    actor_observation_extension_mean,
                    dtype=np.float32,
                ),
                pack.effective_observation_mean[extension_offset:],
            ),
        )
        actor_inverse_standard_deviation = np.concatenate(
            (
                pack.effective_observation_inverse_standard_deviation[
                    :extension_offset
                ],
                np.full(
                    extension_count,
                    actor_observation_extension_inverse_standard_deviation,
                    dtype=np.float32,
                ),
                pack.effective_observation_inverse_standard_deviation[
                    extension_offset:
                ],
            ),
        )
        learner.set_observation_normalization(
            actor_mean=actor_mean,
            actor_inverse_standard_deviation=(
                actor_inverse_standard_deviation
            ),
            observation_clip=pack.observation_clip,
        )
        learner.set_action_contract(
            action_bias=pack.effective_action_bias,
            action_scale=pack.effective_action_scale,
            action_clip=pack.action_clip,
        )
        learner.policy_id = pack.id
        learner.revision = 1
        learner.optimizer = optim.Adam(
            learning_rate=restored_configuration.learning_rate,
            bias_correction=True,
        )
        learner.optimizer.init(learner.model.trainable_parameters())
        learner.refresh_compiled_training_state()
        mx.eval(learner.model.parameters(), learner.optimizer.state)
        return learner

    @classmethod
    def from_projected_actor_policy_pack(
        cls,
        path: str | Path,
        critic_observation_count: int,
        source_actor_indices: tuple[int | None, ...],
        configuration: MLXPPOConfiguration = MLXPPOConfiguration(),
        *,
        action_multiplier:
            np.ndarray | tuple[float, ...] | None = None,
        source_observation_defaults:
            np.ndarray | tuple[float, ...] | None = None,
        library_path: str | Path | None = None,
    ) -> "MLXPolicyLearner":
        """Project an actor onto a semantically selected input contract.

        Each target observation names one source observation index. ``None``
        creates a zero-connected target input with identity normalization.
        Source observations not selected are evaluated at their normalization
        mean unless explicit raw defaults are supplied. Their constant
        normalized contribution is folded into the first-layer bias, so a
        no-threat camera image can be removed without changing policy output.
        An optional per-action multiplier adapts normalized outputs when the
        destination TaskPack uses different joint scales.
        """

        pack = read_policy_pack(path, library_path=library_path)
        expected_activations = [3] * (len(pack.layers) - 1) + [0]
        if [layer.activation for layer in pack.layers] != expected_activations:
            raise ValueError(
                "actor projection supports ELU hidden layers and an "
                "identity output layer"
            )
        if not source_actor_indices:
            raise ValueError("actor projection requires target observations")
        selected = [
            index
            for index in source_actor_indices
            if index is not None
        ]
        if any(
            index < 0 or index >= pack.actor_observation_count
            for index in selected
        ):
            raise ValueError(
                "actor projection source index is outside the PolicyPack contract"
            )
        if len(set(selected)) != len(selected):
            raise ValueError("actor projection source indices must be unique")
        if source_observation_defaults is None:
            source_defaults = pack.effective_observation_mean.copy()
        else:
            source_defaults = np.asarray(
                source_observation_defaults,
                dtype=np.float32,
            )
        if (
            source_defaults.shape != (pack.actor_observation_count,)
            or not np.all(np.isfinite(source_defaults))
        ):
            raise ValueError(
                "actor projection source defaults must be finite and match the source observation count"
            )
        if action_multiplier is None:
            multiplier = np.ones(pack.action_count, dtype=np.float32)
        else:
            multiplier = np.asarray(
                action_multiplier,
                dtype=np.float32,
            )
        if (
            multiplier.shape != (pack.action_count,)
            or not np.all(np.isfinite(multiplier))
            or np.any(multiplier <= 0.0)
        ):
            raise ValueError(
                "actor projection action multiplier must be finite, positive, and match the action count"
            )

        restored_configuration = replace(
            configuration,
            hidden_sizes=tuple(
                layer.output_count for layer in pack.layers[:-1]
            ),
            observation_clip=pack.observation_clip,
        )
        learner = cls(
            len(source_actor_indices),
            critic_observation_count,
            pack.action_count,
            restored_configuration,
        )
        destination = [
            layer
            for layer in learner.model.actor.layers
            if isinstance(layer, nn.Linear)
        ]
        if len(destination) != len(pack.layers):
            raise RuntimeError(
                "MLX actor construction disagrees with PolicyPack"
            )
        for layer_index, (target, artifact) in enumerate(
            zip(destination, pack.layers, strict=True)
        ):
            weights = np.asarray(artifact.weights, dtype=np.float32)
            bias = np.asarray(artifact.bias, dtype=np.float32)
            if layer_index == 0:
                source_weights = weights
                projected = np.zeros(
                    (weights.shape[0], len(source_actor_indices)),
                    dtype=np.float32,
                )
                for target_index, source_index in enumerate(
                    source_actor_indices
                ):
                    if source_index is not None:
                        projected[:, target_index] = weights[:, source_index]
                weights = projected
                omitted = np.ones(
                    pack.actor_observation_count,
                    dtype=bool,
                )
                omitted[selected] = False
                normalized_defaults = np.clip(
                    (
                        source_defaults.astype(np.float64)
                        - pack.effective_observation_mean.astype(np.float64)
                    ) *
                    pack.effective_observation_inverse_standard_deviation.astype(
                        np.float64
                    ),
                    -pack.observation_clip,
                    pack.observation_clip,
                )
                bias = bias.astype(np.float64) + np.sum(
                    source_weights[:, omitted].astype(np.float64)
                    * normalized_defaults[omitted][None, :],
                    axis=1,
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(bias)) or np.any(
                    np.abs(bias) > np.finfo(np.float32).max
                ):
                    raise ValueError(
                        "actor projection defaults produce a non-finite first-layer bias"
                    )
                bias = bias.astype(np.float32)
            target.weight = mx.array(weights, dtype=mx.float32)
            target.bias = mx.array(bias, dtype=mx.float32)

        actor_mean = np.zeros(len(source_actor_indices), dtype=np.float32)
        actor_inverse_standard_deviation = np.ones(
            len(source_actor_indices),
            dtype=np.float32,
        )
        for target_index, source_index in enumerate(source_actor_indices):
            if source_index is None:
                continue
            actor_mean[target_index] = (
                pack.effective_observation_mean[source_index]
            )
            actor_inverse_standard_deviation[target_index] = (
                pack.effective_observation_inverse_standard_deviation[
                    source_index
                ]
            )
        learner.set_observation_normalization(
            actor_mean=actor_mean,
            actor_inverse_standard_deviation=(
                actor_inverse_standard_deviation
            ),
            observation_clip=pack.observation_clip,
        )
        learner.set_action_contract(
            action_bias=pack.effective_action_bias * multiplier,
            action_scale=pack.effective_action_scale * multiplier,
            action_clip=pack.action_clip,
        )
        learner.policy_id = pack.id
        learner.revision = 1
        learner.optimizer = optim.Adam(
            learning_rate=restored_configuration.learning_rate,
            bias_correction=True,
        )
        learner.optimizer.init(learner.model.trainable_parameters())
        learner.refresh_compiled_training_state()
        mx.eval(learner.model.parameters(), learner.optimizer.state)
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
        teacher_actions: mx.array,
        teacher_weights: mx.array,
        policy_weights: mx.array,
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
        ppo_weight = mx.maximum(mx.sum(policy_weights), 1.0)
        kl_per_sample = mx.sum(
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
        kl_divergence = mx.sum(
            kl_per_sample * policy_weights
        ) / ppo_weight
        log_ratio = log_probabilities - old_log_probabilities
        ratio = mx.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = mx.clip(
            ratio,
            1.0 - self.configuration.clip_ratio,
            1.0 + self.configuration.clip_ratio,
        ) * advantages
        policy_loss = -mx.sum(
            mx.minimum(unclipped, clipped) * policy_weights
        ) / ppo_weight
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
        entropy_mean = mx.sum(
            entropy * policy_weights
        ) / ppo_weight
        teacher_error = mx.abs(current_means - teacher_actions)
        teacher_huber = mx.where(
            teacher_error <= 1.0,
            0.5 * mx.square(teacher_error),
            teacher_error - 0.5,
        )
        teacher_weight = mx.maximum(mx.sum(teacher_weights), 1.0)
        imagination_loss = mx.sum(
            mx.mean(teacher_huber, axis=-1) * teacher_weights
        ) / teacher_weight
        loss = (
            policy_loss
            + self.configuration.value_coefficient * value_loss
            - self.configuration.entropy_coefficient * entropy_mean
            + self.configuration.imagination_distillation_coefficient
            * imagination_loss
        )
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "imagination_loss": imagination_loss,
            "imagination_fraction": mx.mean(teacher_weights),
            "kl_divergence": kl_divergence,
            "approximate_kl": mx.sum(
                ((ratio - 1.0) - log_ratio) * policy_weights
            ) / ppo_weight,
            "clip_fraction": mx.sum(
                (
                    mx.abs(ratio - 1.0)
                    > self.configuration.clip_ratio
                ).astype(mx.float32) * policy_weights
            ) / ppo_weight,
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
        teacher_actions: mx.array,
        teacher_weights: mx.array,
        policy_weights: mx.array,
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
            teacher_actions,
            teacher_weights,
            policy_weights,
        )
        if self._actor_observation_extension_only_range is not None:
            offset, count, train_output_layer, output_action_indices = (
                self._actor_observation_extension_only_range
            )
            gradients = _actor_observation_extension_only_gradients(
                gradients,
                offset,
                count,
                train_output_layer,
                output_action_indices,
            )
        elif self._actor_residual_output_only is not None:
            (
                residual_width,
                output_action_indices,
                hidden_observation_range,
            ) = (
                self._actor_residual_output_only
            )
            gradients = _actor_residual_output_only_gradients(
                gradients,
                residual_width,
                output_action_indices,
                hidden_observation_range,
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
            batch.advantages - np.mean(batch.advantages)
        ) / (np.std(batch.advantages) + 1.0e-8)
        # Old-policy means must remain fixed across every minibatch. Compute
        # them once in bounded chunks instead of promoting the complete visual
        # rollout to a second device-resident tensor.
        old_mean_chunks: list[np.ndarray] = []
        for offset in range(
            0,
            sample_count,
            self.configuration.minibatch_size,
        ):
            actor_chunk = mx.array(
                batch.actor_observations[
                    offset : offset + self.configuration.minibatch_size
                ],
                dtype=mx.float32,
            )
            means = self.model.actor_mean(actor_chunk)
            mx.eval(means)
            old_mean_chunks.append(
                np.asarray(means, dtype=np.float32)
            )
        old_means = np.concatenate(old_mean_chunks, axis=0)
        old_log_standard_deviation = mx.clip(
            self.model.log_standard_deviation,
            -5.0,
            2.0,
        )
        mx.eval(old_log_standard_deviation)
        totals: dict[str, float] = {}
        update_count = 0
        start = time.perf_counter()
        for epoch in range(self.configuration.update_epochs):
            permutation = np.random.default_rng(
                self._permutation_seed(
                    self.configuration.seed,
                    self.revision,
                    epoch,
                )
            ).permutation(
                sample_count
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
                    mx.array(
                        batch.actor_observations[indices],
                        dtype=mx.float32,
                    ),
                    mx.array(
                        batch.critic_observations[indices],
                        dtype=mx.float32,
                    ),
                    mx.array(batch.latents[indices], dtype=mx.float32),
                    mx.array(
                        batch.old_log_probabilities[indices],
                        dtype=mx.float32,
                    ),
                    mx.array(old_means[indices], dtype=mx.float32),
                    old_log_standard_deviation,
                    mx.array(
                        batch.old_values[indices],
                        dtype=mx.float32,
                    ),
                    mx.array(advantages[indices], dtype=mx.float32),
                    mx.array(batch.returns[indices], dtype=mx.float32),
                    mx.array(
                        batch.teacher_actions[indices],
                        dtype=mx.float32,
                    ),
                    mx.array(
                        batch.teacher_weights[indices],
                        dtype=mx.float32,
                    ),
                    mx.array(
                        batch.policy_weights[indices],
                        dtype=mx.float32,
                    ),
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
        if stochastic:
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
            contract_version=self.contract_version,
            world_fingerprint=self.world_fingerprint,
            task_fingerprint=self.task_fingerprint,
            observation_fingerprint=self.observation_fingerprint,
            action_fingerprint=self.action_fingerprint,
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
            critic_observation_mean=(
                np.asarray(
                    self.model.critic_observation_mean,
                    dtype=np.float32,
                )
                if stochastic
                else ()
            ),
            critic_observation_inverse_standard_deviation=(
                np.asarray(
                    self.model
                        .critic_observation_inverse_standard_deviation,
                    dtype=np.float32,
                )
                if stochastic
                else ()
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
    "MLXMotionPrior",
    "MLXMotionPriorConfiguration",
    "MLXPPOConfiguration",
    "MLXPolicyBatch",
    "MLXPolicyLearner",
    "NativeMotionClip",
    "NativeMotionPack",
    "NativePolicyDenseLayer",
    "NativePolicyPack",
    "NativePolicyRollout",
    "read_motion_pack",
    "read_policy_pack",
    "read_policy_rollout_pack",
]
