"""Apple-native G1 locomotion rollouts and asymmetric PPO.

The deployable actor sees only the Unitree-compatible proprioceptive contract.
Physics truth, compact contact reductions, terrain samples, and randomized
parameters are carried in a separate critic observation. Dense tactile and
visual products are deliberately absent from the online training graph.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .mlx_world import (
    ActuatorState,
    RodState,
    SceneBodyState,
    SolverCache,
    TactileState,
    WorldState,
    compile_world,
    initial_state,
    step,
)
from .ppo import PPOConfig


G1_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_BODY_ORDER = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

G1_ACTOR_FRAME_SIZE = 96
G1_ACTOR_HISTORY = 5
G1_ACTOR_OBSERVATION_SIZE = (
    G1_ACTOR_FRAME_SIZE * G1_ACTOR_HISTORY
)
G1_TERRAIN_GRID_X = 17
G1_TERRAIN_GRID_Y = 11
G1_TERRAIN_SAMPLE_COUNT = (
    G1_TERRAIN_GRID_X * G1_TERRAIN_GRID_Y
)
G1_COMPACT_CONTACT_SIZE = 16
G1_DOMAIN_PARAMETER_SIZE = 9
G1_CRITIC_OBSERVATION_SIZE = (
    G1_ACTOR_OBSERVATION_SIZE
    + 4
    + G1_COMPACT_CONTACT_SIZE
    + G1_TERRAIN_SAMPLE_COUNT
    + G1_DOMAIN_PARAMETER_SIZE
)


class G1Termination:
    continuing = 0
    height = 1
    tilt = 2
    body_contact = 3
    timeout = 4
    physics_error = 5


@dataclass(frozen=True, slots=True)
class G1LocomotionTaskSpec:
    """One current locomotion contract; values are policy-facing ABI."""

    control_timestep: float = 0.02
    physics_substeps: int = 4
    maximum_episode_steps: int = 1_000
    actor_frame_size: int = G1_ACTOR_FRAME_SIZE
    actor_history: int = G1_ACTOR_HISTORY
    action_scale_radians: float = 0.25
    base_height_target: float = 0.78
    gait_period_seconds: float = 0.8
    foot_clearance_target: float = 0.10
    standing_interval_fraction: float = 0.15
    command_interval_min_seconds: float = 5.0
    command_interval_max_seconds: float = 10.0
    cohort_size: int = 256
    rollout_steps: int = 24
    rollout_chunk_size: int = 8
    source_repository: str = (
        "https://github.com/unitreerobotics/unitree_ros"
    )
    source_revision: str = (
        "aa0f5c68b5aba347bad409e71b6430407da758d7"
    )
    source_model: str = (
        "robots/g1_description/g1_29dof_rev_1_0.urdf"
    )
    deployment_repository: str = (
        "https://github.com/unitreerobotics/unitree_rl_lab"
    )
    deployment_revision: str = (
        "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
    )

    def validate(self) -> None:
        if (
            self.control_timestep != 0.02
            or self.physics_substeps != 4
            or self.actor_frame_size != G1_ACTOR_FRAME_SIZE
            or self.actor_history != G1_ACTOR_HISTORY
            or self.rollout_steps % self.rollout_chunk_size
        ):
            raise ValueError(
                "G1 locomotion requires 200 Hz physics, 50 Hz policy, "
                "five 96-value frames, and integral eight-step chunks"
            )
        if self.maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be positive")

    @property
    def physics_timestep(self) -> float:
        return self.control_timestep / self.physics_substeps


class G1LocomotionState(NamedTuple):
    """Explicit device state carried across bounded rollout chunks."""

    world: WorldState
    actor_history: mx.array
    clean_history: mx.array
    critic_observation: mx.array
    previous_action: mx.array
    previous_joint_velocity: mx.array
    commands: mx.array
    command_steps_remaining: mx.array
    episode_steps: mx.array
    gait_phase: mx.array
    foot_air_time: mx.array
    action_delay_history: mx.array
    actuator_delay: mx.array
    observation_delay: mx.array
    gyro_bias: mx.array
    encoder_bias: mx.array
    body_parameters: mx.array
    controller_parameters: mx.array
    curriculum_level: mx.array
    terrain_level: mx.array
    push_steps_remaining: mx.array
    episode_return: mx.array
    episode_tracking: mx.array


class G1LocomotionStepOutput(NamedTuple):
    """One policy/physics/reward transition and its PPO evidence."""

    state: G1LocomotionState
    actor_observation: mx.array
    critic_observation: mx.array
    latent: mx.array
    log_probability: mx.array
    value: mx.array
    reward: mx.array
    done: mx.array
    timeout: mx.array
    physics_error: mx.array
    physics_status: mx.array
    termination_reason: mx.array
    timeout_bootstrap_value: mx.array
    metrics: mx.array


class _G1CompiledWorldState(NamedTuple):
    """Non-empty physics leaves carried through the compiled G1 graph."""

    q: mx.array
    v: mx.array
    scene_bodies: SceneBodyState
    manifold_headers: mx.array
    manifold_points: mx.array
    manifold_counts: mx.array
    pair_cache: mx.array
    actuators: ActuatorState


class _G1CompiledLocomotionState(NamedTuple):
    world: _G1CompiledWorldState
    actor_history: mx.array
    clean_history: mx.array
    critic_observation: mx.array
    previous_action: mx.array
    previous_joint_velocity: mx.array
    commands: mx.array
    command_steps_remaining: mx.array
    episode_steps: mx.array
    gait_phase: mx.array
    foot_air_time: mx.array
    action_delay_history: mx.array
    actuator_delay: mx.array
    observation_delay: mx.array
    gyro_bias: mx.array
    encoder_bias: mx.array
    body_parameters: mx.array
    controller_parameters: mx.array
    curriculum_level: mx.array
    terrain_level: mx.array
    push_steps_remaining: mx.array
    episode_return: mx.array
    episode_tracking: mx.array


class _G1CompiledStepOutput(NamedTuple):
    state: _G1CompiledLocomotionState
    actor_observation: mx.array
    critic_observation: mx.array
    latent: mx.array
    log_probability: mx.array
    value: mx.array
    reward: mx.array
    done: mx.array
    timeout: mx.array
    physics_error: mx.array
    physics_status: mx.array
    termination_reason: mx.array
    timeout_bootstrap_value: mx.array
    metrics: mx.array


def _compact_g1_state(
    state: G1LocomotionState,
) -> _G1CompiledLocomotionState:
    world = state.world
    return _G1CompiledLocomotionState(
        _G1CompiledWorldState(
            q=world.q,
            v=world.v,
            scene_bodies=world.scene_bodies,
            manifold_headers=world.solver_cache.manifold_headers,
            manifold_points=world.solver_cache.manifold_points,
            manifold_counts=world.solver_cache.manifold_counts,
            pair_cache=world.solver_cache.pair_cache,
            actuators=world.actuators,
        ),
        *state[1:],
    )


def _expand_g1_state(
    state: _G1CompiledLocomotionState,
    empty_world: WorldState,
) -> G1LocomotionState:
    compact = state.world
    world = WorldState(
        q=compact.q,
        v=compact.v,
        scene_bodies=compact.scene_bodies,
        rods=empty_world.rods,
        solver_cache=SolverCache(
            manifold_headers=compact.manifold_headers,
            manifold_points=compact.manifold_points,
            manifold_counts=compact.manifold_counts,
            pair_cache=compact.pair_cache,
            rod_witnesses=empty_world.solver_cache.rod_witnesses,
        ),
        tactile=empty_world.tactile,
        actuators=compact.actuators,
    )
    return G1LocomotionState(world, *state[1:])


def _nonempty_abi_leaf(value: mx.array) -> mx.array:
    if value.size != 0:
        return value
    return mx.zeros(
        tuple(max(int(dimension), 1) for dimension in value.shape),
        dtype=value.dtype,
    )


def _g1_compiled_world_template(world: WorldState) -> WorldState:
    """Provide one-element storage for unused generic-world ABI channels."""

    return world._replace(
        rods=RodState(*(_nonempty_abi_leaf(value) for value in world.rods)),
        solver_cache=world.solver_cache._replace(
            rod_witnesses=_nonempty_abi_leaf(
                world.solver_cache.rod_witnesses
            )
        ),
        tactile=TactileState(
            *(
                _nonempty_abi_leaf(value)
                for value in world.tactile
            )
        ),
    )


class G1LocomotionRolloutBatch(NamedTuple):
    """Device-backed asymmetric PPO rollout."""

    actor_observations: mx.array
    critic_observations: mx.array
    latents: mx.array
    old_log_probabilities: mx.array
    old_values: mx.array
    advantages: mx.array
    returns: mx.array
    rewards: mx.array
    dones: mx.array
    timeouts: mx.array
    physics_errors: mx.array
    physics_statuses: mx.array
    termination_reasons: mx.array
    metrics: mx.array

    def flattened(self) -> dict[str, mx.array]:
        return {
            "actor_observations": self.actor_observations.reshape(
                (-1, G1_ACTOR_OBSERVATION_SIZE)
            ),
            "critic_observations": self.critic_observations.reshape(
                (-1, G1_CRITIC_OBSERVATION_SIZE)
            ),
            "latents": self.latents.reshape((-1, 29)),
            "old_log_probabilities":
                self.old_log_probabilities.reshape((-1,)),
            "old_values": self.old_values.reshape((-1,)),
            "advantages": self.advantages.reshape((-1,)),
            "returns": self.returns.reshape((-1,)),
        }


class _G1RandomInputs(NamedTuple):
    policy_noise: mx.array
    command: mx.array
    reset_pose: mx.array
    reset_velocity: mx.array
    domain: mx.array
    push: mx.array


class _G1RolloutRecord(NamedTuple):
    """PPO fields retained from a transition, without its world state."""

    actor_observation: mx.array
    critic_observation: mx.array
    latent: mx.array
    log_probability: mx.array
    value: mx.array
    reward: mx.array
    done: mx.array
    timeout: mx.array
    physics_error: mx.array
    physics_status: mx.array
    termination_reason: mx.array
    timeout_bootstrap_value: mx.array
    metrics: mx.array


def _elu_mlp(
    input_size: int,
    output_size: int,
    hidden_sizes: tuple[int, ...],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_size
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


def _orthogonal_weight(
    shape: tuple[int, ...],
    gain: float,
    rng: np.random.Generator,
) -> mx.array:
    rows = int(shape[0])
    columns = int(np.prod(shape[1:]))
    tall_rows = max(rows, columns)
    tall_columns = min(rows, columns)
    sample = rng.standard_normal(
        (tall_rows, tall_columns),
        dtype=np.float32,
    )
    q, r = np.linalg.qr(sample, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q *= signs[None, :]
    if rows < columns:
        q = q.T
    weight = (
        q[:rows, :columns]
        .reshape(shape)
        .astype(np.float32, copy=False)
    )
    return mx.array(gain * weight, dtype=mx.float32)


def _initialize_policy_mlp(
    network: nn.Sequential,
    *,
    output_gain: float,
    rng: np.random.Generator,
) -> None:
    linear_layers = [
        layer
        for layer in network.layers
        if isinstance(layer, nn.Linear)
    ]
    for index, layer in enumerate(linear_layers):
        gain = (
            output_gain
            if index + 1 == len(linear_layers)
            else math.sqrt(2.0)
        )
        layer.weight = _orthogonal_weight(
            tuple(int(value) for value in layer.weight.shape),
            gain,
            rng,
        )
        if getattr(layer, "bias", None) is not None:
            layer.bias = mx.zeros_like(layer.bias)


class G1ActorCritic(nn.Module):
    """Separate deployable actor and privileged critic networks."""

    def __init__(
        self,
        actor_observation_size: int = G1_ACTOR_OBSERVATION_SIZE,
        critic_observation_size: int = G1_CRITIC_OBSERVATION_SIZE,
        action_size: int = 29,
        hidden_sizes: tuple[int, ...] = (512, 256, 128),
        initial_log_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(width <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must be positive")
        self.actor = _elu_mlp(
            actor_observation_size,
            action_size,
            hidden_sizes,
        )
        self.critic = _elu_mlp(
            critic_observation_size,
            1,
            hidden_sizes,
        )
        initialization_rng = np.random.default_rng(seed)
        _initialize_policy_mlp(
            self.actor,
            output_gain=0.01,
            rng=initialization_rng,
        )
        _initialize_policy_mlp(
            self.critic,
            output_gain=1.0,
            rng=initialization_rng,
        )
        self.log_std = mx.full(
            (action_size,),
            initial_log_std,
            dtype=mx.float32,
        )

    def actor_mean(self, observation: mx.array) -> mx.array:
        return self.actor(observation)

    def value(self, observation: mx.array) -> mx.array:
        return self.critic(observation).squeeze(-1)

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
        jacobian = mx.log(
            mx.maximum(1.0 - mx.square(action), 1.0e-6)
        )
        return mx.sum(gaussian - jacobian, axis=-1)

    def sample(
        self,
        actor_observation: mx.array,
        critic_observation: mx.array,
        noise: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        mean = self.actor_mean(actor_observation)
        log_std = mx.clip(self.log_std, -5.0, 2.0)
        latent = mean + mx.exp(log_std) * noise
        action = mx.tanh(latent)
        return (
            action,
            latent,
            self.log_probability(
                mean,
                log_std,
                latent,
                action,
            ),
            self.value(critic_observation),
        )

    def deterministic(self, observation: mx.array) -> mx.array:
        return mx.tanh(self.actor_mean(observation))

    def evaluate(
        self,
        actor_observation: mx.array,
        critic_observation: mx.array,
        latent: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        mean = self.actor_mean(actor_observation)
        log_std = mx.clip(self.log_std, -5.0, 2.0)
        action = mx.tanh(latent)
        log_probability = self.log_probability(
            mean,
            log_std,
            latent,
            action,
        )
        entropy = mx.sum(
            log_std
            + 0.5 * (1.0 + math.log(2.0 * math.pi)),
            axis=-1,
        )
        return (
            log_probability,
            entropy,
            self.value(critic_observation),
        )


@dataclass(frozen=True, slots=True)
class G1PolicyPack:
    """Artifact-bound deployment contract for one trained actor."""

    format: str
    task: str
    actor_observation_size: int
    critic_observation_size: int
    action_size: int
    joint_order: tuple[str, ...]
    default_pose: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    position_limits: tuple[tuple[float, float], ...]
    effort_limits: tuple[float, ...]
    velocity_limits: tuple[float, ...]
    action_scale_radians: float
    command_limits: tuple[tuple[float, float], ...]
    drive_prediction_seconds: float
    source_revision: str
    source_model: str
    world_fingerprint: str
    weights_sha256: str
    renderer_free_training: bool
    evaluation_summary: dict[str, float]

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()


def _cross(a: mx.array, b: mx.array) -> mx.array:
    return mx.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def _rotate_inverse(
    quaternion_xyzw: mx.array,
    vector: mx.array,
) -> mx.array:
    q = quaternion_xyzw[..., :3]
    w = quaternion_xyzw[..., 3:4]
    tangent = 2.0 * _cross(q, vector)
    return vector - w * tangent + _cross(q, tangent)


def _rotate(
    quaternion_xyzw: mx.array,
    vector: mx.array,
) -> mx.array:
    q = quaternion_xyzw[..., :3]
    w = quaternion_xyzw[..., 3:4]
    tangent = 2.0 * _cross(q, vector)
    return vector + w * tangent + _cross(q, tangent)


def _select(
    mask: mx.array,
    replacement: mx.array,
    current: mx.array,
) -> mx.array:
    shape = (int(mask.shape[0]),) + (1,) * (current.ndim - 1)
    return mx.where(mask.reshape(shape), replacement, current)


def _select_nonempty(
    mask: mx.array,
    replacement: mx.array,
    current: mx.array,
) -> mx.array:
    # Empty rod/tactile leaves are ABI placeholders in the generic world
    # state. Keeping the existing leaf avoids asking mx.compile to launch a
    # zero-grid elementwise kernel in the G1-specialized transition.
    if current.size == 0:
        return current
    return _select(mask, replacement, current)


def _select_world(
    mask: mx.array,
    replacement: WorldState,
    current: WorldState,
) -> WorldState:
    return WorldState(
        q=_select(mask, replacement.q, current.q),
        v=_select(mask, replacement.v, current.v),
        scene_bodies=SceneBodyState(
            *(
                _select(mask, new, old)
                for new, old in zip(
                    replacement.scene_bodies,
                    current.scene_bodies,
                    strict=True,
                )
            )
        ),
        rods=RodState(
            *(
                _select_nonempty(mask, new, old)
                for new, old in zip(
                    replacement.rods,
                    current.rods,
                    strict=True,
                )
            )
        ),
        solver_cache=SolverCache(
            *(
                _select_nonempty(mask, new, old)
                for new, old in zip(
                    replacement.solver_cache,
                    current.solver_cache,
                    strict=True,
                )
            )
        ),
        tactile=TactileState(
            *(
                _select_nonempty(mask, new, old)
                for new, old in zip(
                    replacement.tactile,
                    current.tactile,
                    strict=True,
                )
            )
        ),
        actuators=ActuatorState(
            *(
                _select(mask, new, old)
                for new, old in zip(
                    replacement.actuators,
                    current.actuators,
                    strict=True,
                )
            )
        ),
    )


def _terrain_height(x: mx.array, y: mx.array) -> mx.array:
    slope = 0.08 * (x + 2.0)
    stairs = 0.16 + 0.03 * mx.floor(x / 0.25)
    rough = (
        0.37
        + 0.035
        * (mx.sin(1.7 * x) - math.sin(3.4))
        * mx.sin(1.3 * y)
        + 0.015
        * (mx.sin(3.1 * x) - math.sin(6.2))
        * mx.sin(2.3 * y)
    )
    return mx.where(
        x < -2.0,
        mx.zeros_like(x),
        mx.where(x < 0.0, slope, mx.where(x < 2.0, stairs, rough)),
    )


class MLXG1RolloutCollector:
    """One fixed-shape G1 execution cohort."""

    def __init__(
        self,
        world: Any,
        model: G1ActorCritic,
        environment_count: int,
        *,
        task: G1LocomotionTaskSpec,
        gamma: float,
        gae_lambda: float,
        terrain: bool,
        diagnostic_allow_pgs: bool = False,
    ) -> None:
        task.validate()
        if environment_count <= 0:
            raise ValueError("environment_count must be positive")
        if (
            world.nq != 36
            or world.nv != 35
            or not world.floating_root
            or world.actuation_mode != "implicit_position"
            or (
                world.solver_mode != "throughput_tgs"
                and not (
                    diagnostic_allow_pgs
                    and world.solver_mode == "throughput_pgs"
                )
            )
            or not world.publishes_body_states
        ):
            raise ValueError(
                "G1 locomotion requires the 36/35 floating-base "
                "implicit-position TGS world with compact body publication"
            )
        if (
            int(world.tactile_sensor_count) != 0
            or int(world.tactile_sample_count) != 0
        ):
            raise ValueError(
                "dense tactile sensing is not part of G1 locomotion"
            )
        self.world = world
        self.model = model
        self.environment_count = environment_count
        self.task = task
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.terrain = bool(terrain)
        self.default_world = initial_state(world, environment_count)
        self._compiled_world_template = _g1_compiled_world_template(
            self.default_world
        )
        self.default_q = mx.array(world.default_q, dtype=mx.float32)
        self.default_v = mx.array(world.default_v, dtype=mx.float32)
        self.default_joint_pose = self.default_q[7:]
        self.joint_lower = mx.array(
            world.joint_lower_limits,
            dtype=mx.float32,
        )
        self.joint_upper = mx.array(
            world.joint_upper_limits,
            dtype=mx.float32,
        )
        self.joint_velocity_limit = mx.array(
            world.joint_velocity_limits,
            dtype=mx.float32,
        )
        self.stiffness = mx.array(
            world.drive_stiffness,
            dtype=mx.float32,
        )
        self.damping = mx.array(
            world.drive_damping,
            dtype=mx.float32,
        )
        self.effort_limit = mx.array(
            world.effort_limits[6:],
            dtype=mx.float32,
        )
        self.shape_body_indices = mx.array(
            world.shape_body_indices,
            dtype=mx.uint32,
        )
        shape_bodies = np.asarray(
            world.shape_body_indices,
            dtype=np.uint32,
        )
        shape_types = np.asarray(
            world.shape_types,
            dtype=np.uint32,
        )
        shape_local_positions = np.asarray(
            world.shape_local_positions,
            dtype=np.float32,
        ).reshape((-1, 3))
        shape_dimensions = np.asarray(
            world.shape_dimensions,
            dtype=np.float32,
        ).reshape((-1, 3))
        foot_spheres = []
        for body in (6, 12):
            indices = np.flatnonzero(
                (shape_bodies == body) & (shape_types == 0)
            )
            if indices.size != 4:
                raise ValueError(
                    "G1 locomotion requires four official sole spheres "
                    f"on body {body}, found {indices.size}"
                )
            foot_spheres.append(indices)
        foot_sphere_indices = np.stack(foot_spheres)
        self._foot_sphere_local = mx.array(
            shape_local_positions[foot_sphere_indices],
            dtype=mx.float32,
        )
        self._foot_sphere_radius = mx.array(
            shape_dimensions[foot_sphere_indices, 0],
            dtype=mx.float32,
        )
        self.body_masses = mx.array(
            world.body_masses,
            dtype=mx.float32,
        )
        self._torso_body_index = 15
        self._foot_body_indices = mx.array(
            [6, 12],
            dtype=mx.uint32,
        )
        self._hip_posture_indices = mx.array(
            [1, 2, 7, 8],
            dtype=mx.uint32,
        )
        self.robot_body_count = len(G1_BODY_ORDER)
        self._terrain_offsets = self._make_terrain_offsets()
        self._compiled_policy = mx.compile(
            self.model.sample,
            inputs=self.model.state,
        )
        self._compiled_value = mx.compile(
            self.model.value,
            inputs=self.model.state,
        )
        self._compiled_transition = mx.compile(
            self._compact_transition_impl,
            inputs=self.model.state,
        )

    @staticmethod
    def _make_terrain_offsets() -> mx.array:
        values = []
        for ix in range(G1_TERRAIN_GRID_X):
            x = -0.8 + 0.1 * ix
            for iy in range(G1_TERRAIN_GRID_Y):
                y = -0.5 + 0.1 * iy
                values.append((x, y))
        return mx.array(values, dtype=mx.float32)

    def _commands(
        self,
        random: mx.array,
        curriculum: mx.array,
    ) -> mx.array:
        progress = mx.clip(
            (curriculum.astype(mx.float32) + 1.0) / 10.0,
            0.1,
            1.0,
        )
        lower = mx.stack(
            (-0.5 * progress, -0.3 * progress, -0.2 * progress),
            axis=-1,
        )
        upper = mx.stack(
            (1.0 * progress, 0.3 * progress, 0.2 * progress),
            axis=-1,
        )
        command = lower + (upper - lower) * random[:, :3]
        standing = (
            random[:, 3] < self.task.standing_interval_fraction
        )
        return mx.where(
            standing[:, None],
            mx.zeros_like(command),
            command,
        )

    def _command_steps(self, random: mx.array) -> mx.array:
        seconds = (
            self.task.command_interval_min_seconds
            + (
                self.task.command_interval_max_seconds
                - self.task.command_interval_min_seconds
            )
            * random
        )
        return mx.maximum(
            mx.array(1, dtype=mx.int32),
            mx.floor(
                seconds / self.task.control_timestep
            ).astype(mx.int32),
        )

    def _frame(
        self,
        q: mx.array,
        v: mx.array,
        command: mx.array,
        previous_action: mx.array,
    ) -> mx.array:
        orientation = q[:, 3:7]
        angular_velocity = _rotate_inverse(
            orientation,
            v[:, 3:6],
        )
        gravity = _rotate_inverse(
            orientation,
            mx.broadcast_to(
                mx.array(
                    [0.0, 0.0, -1.0],
                    dtype=mx.float32,
                ),
                (self.environment_count, 3),
            ),
        )
        gravity = gravity / mx.maximum(
            mx.sqrt(mx.sum(mx.square(gravity), axis=-1))[:, None],
            1.0e-6,
        )
        return mx.concatenate(
            (
                0.2 * angular_velocity,
                gravity,
                command,
                q[:, 7:] - self.default_joint_pose,
                0.05 * v[:, 6:],
                previous_action,
            ),
            axis=-1,
        )

    @staticmethod
    def _corrupt_frame(
        clean: mx.array,
        noise: mx.array,
        gyro_bias: mx.array,
        encoder_bias: mx.array,
    ) -> mx.array:
        gyro = (
            clean[:, :3]
            + 0.04 * noise[:, :3]
            + 0.2 * gyro_bias
        )
        gravity = clean[:, 3:6] + 0.05 * noise[:, 3:6]
        gravity = gravity / mx.maximum(
            mx.sqrt(mx.sum(mx.square(gravity), axis=-1))[:, None],
            1.0e-6,
        )
        joint_position = (
            clean[:, 9:38]
            + 0.01 * noise[:, 6:35]
            + encoder_bias
        )
        joint_velocity = (
            clean[:, 38:67] + 0.075 * noise[:, 35:64]
        )
        return mx.concatenate(
            (
                gyro,
                gravity,
                clean[:, 6:9],
                joint_position,
                joint_velocity,
                clean[:, 67:96],
            ),
            axis=-1,
        )

    def _terrain_profile_translation(
        self,
        level: mx.array,
    ) -> mx.array:
        center = mx.where(
            level < 3,
            -3.0,
            mx.where(level < 5, -1.0, mx.where(level < 7, 3.0, 1.0)),
        )
        center_height = mx.where(
            level < 3,
            0.0,
            mx.where(level < 5, 0.08, mx.where(level < 7, 0.37, 0.28)),
        )
        return mx.stack(
            (
                -center,
                mx.zeros_like(center),
                -center_height,
                mx.ones_like(center),
            ),
            axis=-1,
        )

    def _profiled_world(
        self,
        world: WorldState,
        level: mx.array,
    ) -> WorldState:
        if not self.terrain:
            return world
        scene = world.scene_bodies
        translation = self._terrain_profile_translation(level)
        position = mx.concatenate(
            (
                translation[:, None, :],
                scene.position[:, 1:, :],
            ),
            axis=1,
        )
        return world._replace(
            scene_bodies=scene._replace(position=position)
        )

    def _reset_world(
        self,
        candidate: WorldState,
        reset_q: mx.array,
        reset_v: mx.array,
        terrain_level: mx.array,
    ) -> WorldState:
        """Reset from live-shaped arrays so compiled state has no host leaves."""

        scene = candidate.scene_bodies
        reset = WorldState(
            q=reset_q,
            v=reset_v,
            scene_bodies=SceneBodyState(
                position=scene.position,
                orientation=scene.orientation,
                linear_velocity=mx.zeros_like(scene.linear_velocity),
                angular_velocity=mx.zeros_like(scene.angular_velocity),
            ),
            rods=RodState(
                *(
                    mx.zeros_like(value)
                    for value in candidate.rods
                )
            ),
            solver_cache=SolverCache(
                *(
                    mx.zeros_like(value)
                    for value in candidate.solver_cache
                )
            ),
            tactile=TactileState(
                *(
                    mx.zeros_like(value)
                    for value in candidate.tactile
                )
            ),
            actuators=ActuatorState(
                effective_position_target=mx.broadcast_to(
                    mx.array(
                        self.world.default_actuator_targets,
                        dtype=mx.float32,
                    )[None, :],
                    candidate.actuators.effective_position_target.shape,
                ),
                profile_values=candidate.actuators.profile_values,
            ),
        )
        return self._profiled_world(reset, terrain_level)

    def _terrain_grid(
        self,
        q: mx.array,
        scene_position: mx.array,
    ) -> mx.array:
        if not self.terrain:
            return mx.zeros(
                (self.environment_count, G1_TERRAIN_SAMPLE_COUNT),
                dtype=mx.float32,
            )
        orientation = q[:, 3:7]
        local = mx.broadcast_to(
            mx.concatenate(
                (
                    self._terrain_offsets,
                    mx.zeros(
                        (G1_TERRAIN_SAMPLE_COUNT, 1),
                        dtype=mx.float32,
                    ),
                ),
                axis=-1,
            )[None, :, :],
            (
                self.environment_count,
                G1_TERRAIN_SAMPLE_COUNT,
                3,
            ),
        )
        quaternion = mx.broadcast_to(
            orientation[:, None, :],
            (
                self.environment_count,
                G1_TERRAIN_SAMPLE_COUNT,
                4,
            ),
        )
        world_offset = (
            local
            + 2.0
            * (
                orientation[:, None, 3:4]
                * _cross(quaternion[..., :3], local)
                + _cross(
                    quaternion[..., :3],
                    _cross(quaternion[..., :3], local),
                )
            )
        )
        world_x = q[:, 0:1] + world_offset[..., 0]
        world_y = q[:, 1:2] + world_offset[..., 1]
        translation = scene_position[:, :1, :]
        local_x = world_x - translation[..., 0]
        local_y = world_y - translation[..., 1]
        return (
            _terrain_height(local_x, local_y)
            + translation[..., 2]
            - q[:, 2:3]
        )

    def _contact_aggregates(
        self,
        physics: Any,
    ) -> tuple[mx.array, mx.array, mx.array]:
        ids = physics.contacts.stable_ids
        mask = physics.contacts.mask.astype(mx.float32)
        maximum_shape = int(self.shape_body_indices.shape[0]) - 1
        collider_a = mx.minimum(ids[..., 0], maximum_shape)
        collider_b = mx.minimum(ids[..., 1], maximum_shape)
        body_a = self.shape_body_indices[collider_a]
        body_b = self.shape_body_indices[collider_b]
        scene_a = body_a >= self.robot_body_count
        scene_b = body_b >= self.robot_body_count
        values = physics.contacts.values
        normal_impulse = mx.abs(values[..., 7]) * mask
        foot_masks = []
        for body in (6, 12):
            foot_masks.append(
                (
                    ((body_a == body) & scene_b)
                    | ((body_b == body) & scene_a)
                ).astype(mx.float32)
                * mask
            )
        foot_mask = mx.stack(foot_masks, axis=1)
        force = mx.sum(
            foot_mask * normal_impulse[:, None, :],
            axis=-1,
        ) / self.task.physics_timestep
        contact = force > 1.0

        body_words = physics.body_states.view(mx.float32)
        foot_body_position = body_words[
            :, self._foot_body_indices, :3
        ]
        foot_orientation = body_words[
            :, self._foot_body_indices, 4:8
        ]
        foot_linear_velocity = body_words[
            :, self._foot_body_indices, 8:11
        ]
        foot_angular_velocity = body_words[
            :, self._foot_body_indices, 12:15
        ]
        sphere_offset = _rotate(
            foot_orientation[:, :, None, :],
            self._foot_sphere_local[None, :, :, :],
        )
        sphere_position = (
            foot_body_position[:, :, None, :] + sphere_offset
        )
        sphere_velocity = (
            foot_linear_velocity[:, :, None, :]
            + _cross(
                foot_angular_velocity[:, :, None, :],
                sphere_offset,
            )
        )
        foot_position = mx.mean(sphere_position, axis=2)
        foot_velocity = mx.mean(sphere_velocity, axis=2)
        slip = (
            mx.sqrt(
                mx.sum(mx.square(foot_velocity[..., :2]), axis=-1)
            )
            * contact.astype(mx.float32)
        )
        terrain_translation = (
            physics.next_state.scene_bodies.position[:, 0, :]
        )
        terrain_height = (
            _terrain_height(
                sphere_position[..., 0]
                - terrain_translation[:, None, None, 0],
                sphere_position[..., 1]
                - terrain_translation[:, None, None, 1],
            )
            + terrain_translation[:, None, None, 2]
            if self.terrain
            else mx.zeros_like(sphere_position[..., 2])
        )
        clearance = mx.min(
            sphere_position[..., 2]
            - self._foot_sphere_radius[None, :, :]
            - terrain_height,
            axis=2,
        )

        points = values[..., :2]
        center_of_pressure = []
        for index in range(2):
            weights = foot_mask[:, index, :] * normal_impulse
            denominator = mx.maximum(
                mx.sum(weights, axis=-1, keepdims=True),
                1.0e-6,
            )
            center = mx.sum(
                weights[..., None] * points,
                axis=1,
            ) / denominator
            center_of_pressure.append(
                center - foot_position[:, index, :2]
            )
        cop = mx.concatenate(center_of_pressure, axis=-1)

        ground_robot_body = mx.where(
            scene_a,
            body_b,
            body_a,
        )
        robot_scene = (scene_a ^ scene_b) & (mask > 0.0)
        undesired = mx.stack(
            (
                mx.max(
                    (
                        robot_scene
                        & (
                            (ground_robot_body == 4)
                            | (ground_robot_body == 10)
                        )
                    ).astype(mx.float32),
                    axis=-1,
                ),
                mx.max(
                    (
                        robot_scene & (ground_robot_body == 0)
                    ).astype(mx.float32),
                    axis=-1,
                ),
                mx.max(
                    (
                        robot_scene & (ground_robot_body == 15)
                    ).astype(mx.float32),
                    axis=-1,
                ),
                mx.max(
                    (
                        robot_scene & (ground_robot_body >= 16)
                    ).astype(mx.float32),
                    axis=-1,
                ),
            ),
            axis=-1,
        )
        compact = mx.concatenate(
            (
                force,
                slip,
                mx.zeros_like(force),
                clearance,
                cop,
                undesired,
            ),
            axis=-1,
        )
        terminal_contact = (
            undesired[:, 1] > 0.0
        ) | (undesired[:, 2] > 0.0)
        contact_shape_count = int(
            self.shape_body_indices.shape[0]
        )
        contact_key = (
            collider_a * contact_shape_count + collider_b
        )
        no_contact_key = contact_shape_count * contact_shape_count
        first_contact_key = mx.min(
            mx.where(
                mask > 0.0,
                contact_key,
                mx.full_like(contact_key, no_contact_key),
            ),
            axis=-1,
        )
        has_contact = first_contact_key < no_contact_key
        first_contact = mx.stack(
            (
                mx.where(
                    has_contact,
                    (
                        first_contact_key //
                        contact_shape_count
                    ).astype(mx.float32),
                    mx.full(
                        has_contact.shape,
                        -1.0,
                        dtype=mx.float32,
                    ),
                ),
                mx.where(
                    has_contact,
                    (
                        first_contact_key %
                        contact_shape_count
                    ).astype(mx.float32),
                    mx.full(
                        has_contact.shape,
                        -1.0,
                        dtype=mx.float32,
                    ),
                ),
                mx.sum(mask, axis=-1),
            ),
            axis=-1,
        )
        return (
            compact,
            terminal_contact,
            first_contact,
        )

    def _critic(
        self,
        clean_history: mx.array,
        q: mx.array,
        v: mx.array,
        compact_contact: mx.array,
        body_parameters: mx.array,
        controller_parameters: mx.array,
        scene_position: mx.array,
    ) -> mx.array:
        true_base_velocity = _rotate_inverse(
            q[:, 3:7],
            v[:, :3],
        )
        link_height = (
            self.task.base_height_target
            + q[:, 2:3]
            - self.default_q[2]
        )
        physical = mx.stack(
            (
                mx.mean(body_parameters[..., 0], axis=1),
                body_parameters[:, self._torso_body_index, 0],
                mx.mean(body_parameters[..., 1], axis=1),
                mx.mean(body_parameters[..., 2], axis=1),
                mx.mean(body_parameters[..., 3], axis=1),
            ),
            axis=-1,
        )
        controller = controller_parameters[:, 0, :]
        return mx.concatenate(
            (
                clean_history.reshape(
                    (self.environment_count, -1)
                ),
                true_base_velocity,
                link_height,
                compact_contact,
                self._terrain_grid(q, scene_position),
                physical,
                controller,
            ),
            axis=-1,
        )

    def _domain_parameters(
        self,
        random: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        mass_scale = 0.9 + 0.2 * random[:, 0]
        friction = 0.3 + 0.7 * random[:, 1]
        inertia_scale = 0.9 + 0.2 * random[:, 2]
        damping_scale = 0.8 + 0.4 * random[:, 3]
        payload_mass = -1.0 + 4.0 * random[:, 9]
        mass_scales = mx.broadcast_to(
            mass_scale[:, None],
            (
                self.environment_count,
                int(self.world.model_body_count),
            ),
        )
        torso_selector = mx.arange(
            int(self.world.model_body_count),
            dtype=mx.uint32,
        ) == self._torso_body_index
        torso_scale = mx.maximum(
            0.5,
            mass_scale
            + payload_mass
            / self.body_masses[self._torso_body_index],
        )
        mass_scales = mx.where(
            torso_selector[None, :],
            torso_scale[:, None],
            mass_scales,
        )
        physical = mx.stack(
            (
                mass_scales,
                mx.broadcast_to(friction[:, None], mass_scales.shape),
                mx.broadcast_to(
                    inertia_scale[:, None],
                    mass_scales.shape,
                ),
                mx.broadcast_to(
                    damping_scale[:, None],
                    mass_scales.shape,
                ),
            ),
            axis=-1,
        )
        body_parameters = physical
        gain = 0.8 + 0.4 * random[:, 4]
        damping = 0.8 + 0.4 * random[:, 5]
        delay = mx.minimum(
            mx.floor(3.0 * random[:, 6]).astype(mx.int32),
            mx.array(2, dtype=mx.int32),
        )
        observation_delay = mx.minimum(
            mx.floor(2.0 * random[:, 7]).astype(mx.int32),
            mx.array(1, dtype=mx.int32),
        )
        strength = 0.8 + 0.4 * random[:, 8]
        controller = mx.stack(
            (
                gain,
                damping,
                delay.astype(mx.float32)
                * self.task.control_timestep,
                strength,
            ),
            axis=-1,
        )[:, None, :]
        return (
            body_parameters,
            controller,
            delay,
            observation_delay,
            random[:, 10:13],
        )

    def initial(self) -> G1LocomotionState:
        zero_action = mx.zeros(
            (self.environment_count, 29),
            dtype=mx.float32,
        )
        curriculum = mx.zeros(
            (self.environment_count,),
            dtype=mx.int32,
        )
        command_random = mx.random.uniform(
            shape=(self.environment_count, 5)
        )
        commands = self._commands(command_random, curriculum)
        initial_world = self._profiled_world(
            self.default_world,
            curriculum,
        )
        clean = self._frame(
            initial_world.q,
            initial_world.v,
            commands,
            zero_action,
        )
        history = mx.broadcast_to(
            clean[:, None, :],
            (
                self.environment_count,
                G1_ACTOR_HISTORY,
                G1_ACTOR_FRAME_SIZE,
            ),
        )
        domain = mx.random.uniform(
            shape=(self.environment_count, 15)
        )
        (
            body_parameters,
            controller_parameters,
            actuator_delay,
            observation_delay,
            sensor_bias,
        ) = self._domain_parameters(domain)
        contact = mx.zeros(
            (self.environment_count, G1_COMPACT_CONTACT_SIZE),
            dtype=mx.float32,
        )
        critic = self._critic(
            history,
            self.default_world.q,
            self.default_world.v,
            contact,
            body_parameters,
            controller_parameters,
            initial_world.scene_bodies.position,
        )
        return G1LocomotionState(
            world=initial_world,
            actor_history=history,
            clean_history=history,
            critic_observation=critic,
            previous_action=zero_action,
            previous_joint_velocity=mx.zeros_like(zero_action),
            commands=commands,
            command_steps_remaining=self._command_steps(
                command_random[:, 4]
            ),
            episode_steps=mx.zeros(
                (self.environment_count,),
                dtype=mx.int32,
            ),
            gait_phase=mx.zeros(
                (self.environment_count,),
                dtype=mx.float32,
            ),
            foot_air_time=mx.zeros(
                (self.environment_count, 2),
                dtype=mx.float32,
            ),
            action_delay_history=mx.zeros(
                (self.environment_count, 3, 29),
                dtype=mx.float32,
            ),
            actuator_delay=actuator_delay,
            observation_delay=observation_delay,
            gyro_bias=0.02 * (2.0 * sensor_bias[:, :3] - 1.0),
            encoder_bias=mx.zeros(
                (self.environment_count, 29),
                dtype=mx.float32,
            ),
            body_parameters=body_parameters,
            controller_parameters=controller_parameters,
            curriculum_level=curriculum,
            terrain_level=curriculum,
            push_steps_remaining=mx.full(
                (self.environment_count,),
                150,
                dtype=mx.int32,
            ),
            episode_return=mx.zeros(
                (self.environment_count,),
                dtype=mx.float32,
            ),
            episode_tracking=mx.zeros(
                (self.environment_count,),
                dtype=mx.float32,
            ),
        )

    def evaluation_initial(self) -> G1LocomotionState:
        """Create held-out full-range commands, terrain, and one push."""

        state = self.initial()
        curriculum = mx.full(
            (self.environment_count,),
            10,
            dtype=mx.int32,
        )
        environment = mx.arange(
            self.environment_count,
            dtype=mx.int32,
        )
        split = self.environment_count // 2
        terrain_level = (
            mx.where(
                environment < split,
                mx.zeros_like(environment),
                3 + environment % 8,
            )
            if self.terrain
            else mx.zeros_like(environment)
        )
        world = self._profiled_world(
            self.default_world,
            terrain_level,
        )
        command_random = mx.random.uniform(
            shape=(self.environment_count, 5)
        )
        commands = self._commands(command_random, curriculum)
        zero_action = mx.zeros(
            (self.environment_count, 29),
            dtype=mx.float32,
        )
        clean = self._frame(
            world.q,
            world.v,
            commands,
            zero_action,
        )
        actor = self._corrupt_frame(
            clean,
            mx.zeros(
                (self.environment_count, 64),
                dtype=mx.float32,
            ),
            state.gyro_bias,
            state.encoder_bias,
        )
        actor_history = mx.broadcast_to(
            actor[:, None, :],
            (
                self.environment_count,
                G1_ACTOR_HISTORY,
                G1_ACTOR_FRAME_SIZE,
            ),
        )
        clean_history = mx.broadcast_to(
            clean[:, None, :],
            actor_history.shape,
        )
        critic = self._critic(
            clean_history,
            world.q,
            world.v,
            mx.zeros(
                (
                    self.environment_count,
                    G1_COMPACT_CONTACT_SIZE,
                ),
                dtype=mx.float32,
            ),
            state.body_parameters,
            state.controller_parameters,
            world.scene_bodies.position,
        )
        return state._replace(
            world=world,
            actor_history=actor_history,
            clean_history=clean_history,
            critic_observation=critic,
            commands=commands,
            command_steps_remaining=self._command_steps(
                command_random[:, 4]
            ),
            curriculum_level=curriculum,
            terrain_level=terrain_level,
            push_steps_remaining=mx.zeros(
                (self.environment_count,),
                dtype=mx.int32,
            ),
        )

    def _random_inputs(self) -> _G1RandomInputs:
        return _G1RandomInputs(
            policy_noise=mx.random.uniform(
                low=-1.0,
                high=1.0,
                shape=(self.environment_count, 64),
            ),
            command=mx.random.uniform(
                shape=(self.environment_count, 5)
            ),
            reset_pose=mx.random.uniform(
                low=-1.0,
                high=1.0,
                shape=(self.environment_count, 32),
            ),
            reset_velocity=mx.random.uniform(
                low=-1.0,
                high=1.0,
                shape=(self.environment_count, 35),
            ),
            domain=mx.random.uniform(
                shape=(self.environment_count, 15)
            ),
            push=mx.random.uniform(
                low=-1.0,
                high=1.0,
                shape=(self.environment_count, 3),
            ),
        )

    def _transition_impl(
        self,
        state: G1LocomotionState,
        random: _G1RandomInputs,
        policy_noise: mx.array,
    ) -> G1LocomotionStepOutput:
        actor_observation = state.actor_history.reshape(
            (self.environment_count, -1)
        )
        critic_observation = state.critic_observation
        (
            normalized_action,
            latent,
            log_probability,
            value,
        ) = self._compiled_policy(
            actor_observation,
            critic_observation,
            policy_noise,
        )

        delay_history = mx.concatenate(
            (
                state.action_delay_history[:, 1:, :],
                normalized_action[:, None, :],
            ),
            axis=1,
        )
        selected_index = (
            2 - state.actuator_delay
        )[:, None, None]
        delayed_action = mx.take_along_axis(
            delay_history,
            mx.broadcast_to(
                selected_index,
                (self.environment_count, 1, 29),
            ),
            axis=1,
        ).squeeze(1)
        joint_target = (
            self.default_joint_pose
            + self.task.action_scale_radians * delayed_action
        )
        joint_target = mx.clip(
            joint_target,
            self.joint_lower,
            self.joint_upper,
        )
        target = mx.concatenate(
            (
                mx.zeros(
                    (self.environment_count, 6),
                    dtype=mx.float32,
                ),
                joint_target,
            ),
            axis=-1,
        )

        push_due = state.push_steps_remaining <= 0
        push_scale = (
            0.5
            * mx.clip(
                state.curriculum_level.astype(mx.float32) / 10.0,
                0.0,
                1.0,
            )
        )
        pushed_velocity = mx.concatenate(
            (
                state.world.v[:, :2]
                + mx.where(
                    push_due[:, None],
                    push_scale[:, None] * random.push[:, :2],
                    mx.zeros(
                        (self.environment_count, 2),
                        dtype=mx.float32,
                    ),
                ),
                state.world.v[:, 2:],
            ),
            axis=-1,
        )
        pushed_world = state.world._replace(v=pushed_velocity)
        physics = step(
            self.world,
            pushed_world,
            target,
            body_parameters=state.body_parameters,
            controller_parameters=state.controller_parameters,
            nonempty_unused_outputs=True,
        )
        candidate = physics.next_state
        compact, terminal_contact, contact_metrics = (
            self._contact_aggregates(physics)
        )
        foot_contact = compact[:, :2] > 1.0
        air_time = mx.where(
            foot_contact,
            mx.zeros_like(state.foot_air_time),
            state.foot_air_time + self.task.control_timestep,
        )
        compact = mx.concatenate(
            (compact[:, :4], air_time, compact[:, 6:]),
            axis=-1,
        )

        orientation = candidate.q[:, 3:7]
        base_linear = _rotate_inverse(
            orientation,
            candidate.v[:, :3],
        )
        base_angular = _rotate_inverse(
            orientation,
            candidate.v[:, 3:6],
        )
        gravity = _rotate_inverse(
            orientation,
            mx.broadcast_to(
                mx.array([0.0, 0.0, -1.0], dtype=mx.float32),
                (self.environment_count, 3),
            ),
        )
        tracking_error = mx.sum(
            mx.square(
                base_linear[:, :2] - state.commands[:, :2]
            ),
            axis=-1,
        )
        yaw_error = mx.square(
            base_angular[:, 2] - state.commands[:, 2]
        )
        tracking = mx.exp(-tracking_error / 0.25)
        yaw_tracking = mx.exp(-yaw_error / 0.25)
        link_height = (
            self.task.base_height_target
            + candidate.q[:, 2]
            - self.default_q[2]
        )
        tilt = mx.arctan2(
            mx.sqrt(mx.sum(mx.square(gravity[:, :2]), axis=-1)),
            mx.maximum(-gravity[:, 2], 1.0e-6),
        )
        joint_velocity = candidate.v[:, 6:]
        joint_acceleration = (
            joint_velocity - state.previous_joint_velocity
        ) / self.task.control_timestep
        action_rate = normalized_action - state.previous_action
        joint_position = candidate.q[:, 7:]
        lower_violation = mx.maximum(
            self.joint_lower - joint_position,
            0.0,
        )
        upper_violation = mx.maximum(
            joint_position - self.joint_upper,
            0.0,
        )
        controller = state.controller_parameters[:, 0, :]
        torque = mx.clip(
            controller[:, 0:1] * self.stiffness
            * (
                joint_target
                - joint_position
                - self.task.physics_timestep * joint_velocity
            )
            - controller[:, 1:2]
            * self.damping
            * joint_velocity,
            -self.effort_limit,
            self.effort_limit,
        ) * controller[:, 3:4]
        energy = mx.sum(
            mx.abs(torque * joint_velocity),
            axis=-1,
        )

        phase = (
            state.gait_phase
            + 2.0
            * math.pi
            * self.task.control_timestep
            / self.task.gait_period_seconds
        ) % (2.0 * math.pi)
        desired_contact = mx.stack(
            (mx.sin(phase) >= 0.0, mx.sin(phase) < 0.0),
            axis=-1,
        )
        moving = (
            mx.sqrt(
                mx.sum(
                    mx.square(state.commands[:, :2]),
                    axis=-1,
                )
                + mx.square(state.commands[:, 2])
            )
            > 0.05
        )
        gait_reward = (
            mx.mean(
                (
                    desired_contact == foot_contact
                ).astype(mx.float32),
                axis=-1,
            )
            * moving.astype(mx.float32)
        )
        swing = (~desired_contact).astype(mx.float32)
        clearance_error = mx.square(
            compact[:, 6:8] - self.task.foot_clearance_target
        )
        clearance_reward = (
            mx.sum(
                swing * mx.exp(-clearance_error / 0.0025),
                axis=-1,
            )
            / mx.maximum(mx.sum(swing, axis=-1), 1.0)
            * moving.astype(mx.float32)
        )
        posture_waist = mx.mean(
            mx.square(
                joint_position[:, 12:15]
                - self.default_joint_pose[12:15]
            ),
            axis=-1,
        )
        posture_hips = mx.mean(
            mx.square(
                joint_position[:, self._hip_posture_indices]
                - self.default_joint_pose[
                    self._hip_posture_indices
                ]
            ),
            axis=-1,
        )
        posture_arms = mx.mean(
            mx.square(
                joint_position[:, 15:]
                - self.default_joint_pose[15:]
            ),
            axis=-1,
        )
        reward = (
            tracking
            + 0.5 * yaw_tracking
            + 0.15
            - 2.0 * mx.square(base_linear[:, 2])
            - 0.05 * mx.sum(mx.square(base_angular[:, :2]), axis=-1)
            - 5.0 * mx.square(tilt)
            - 10.0 * mx.square(
                link_height - self.task.base_height_target
            )
            - 0.001 * mx.sum(mx.square(joint_velocity), axis=-1)
            - 2.5e-7
            * mx.sum(mx.square(joint_acceleration), axis=-1)
            - 0.05 * mx.sum(mx.square(action_rate), axis=-1)
            - 5.0
            * mx.sum(
                mx.square(lower_violation)
                + mx.square(upper_violation),
                axis=-1,
            )
            - 2.0e-5 * energy
            - 0.2 * posture_waist
            - 0.1 * posture_hips
            - 0.05 * posture_arms
            + 0.5 * gait_reward
            + clearance_reward
            - 0.2 * mx.mean(compact[:, 2:4], axis=-1)
            - mx.max(compact[:, 12:16], axis=-1)
        )

        episode_steps = state.episode_steps + 1
        timeout = (
            episode_steps >= self.task.maximum_episode_steps
        )
        low = link_height < 0.2
        tipped = tilt > 0.8
        physics_error = physics.physics_error
        done = (
            timeout
            | low
            | tipped
            | terminal_contact
            | physics_error
        )
        reason = mx.zeros(
            episode_steps.shape,
            dtype=mx.uint32,
        )
        reason = mx.where(
            low,
            mx.array(G1Termination.height, dtype=mx.uint32),
            reason,
        )
        reason = mx.where(
            tipped,
            mx.array(G1Termination.tilt, dtype=mx.uint32),
            reason,
        )
        reason = mx.where(
            terminal_contact,
            mx.array(
                G1Termination.body_contact,
                dtype=mx.uint32,
            ),
            reason,
        )
        reason = mx.where(
            timeout,
            mx.array(G1Termination.timeout, dtype=mx.uint32),
            reason,
        )
        reason = mx.where(
            physics_error,
            mx.array(
                G1Termination.physics_error,
                dtype=mx.uint32,
            ),
            reason,
        )

        command_due = state.command_steps_remaining <= 1
        new_commands = self._commands(
            random.command,
            state.curriculum_level,
        )
        next_commands = _select(
            command_due | done,
            new_commands,
            state.commands,
        )
        next_command_steps = mx.where(
            command_due | done,
            self._command_steps(random.command[:, 4]),
            state.command_steps_remaining - 1,
        )
        candidate_clean = self._frame(
            candidate.q,
            candidate.v,
            next_commands,
            normalized_action,
        )
        candidate_actor = self._corrupt_frame(
            candidate_clean,
            random.policy_noise,
            state.gyro_bias,
            state.encoder_bias,
        )
        candidate_actor = _select(
            state.observation_delay > 0,
            state.actor_history[:, -1, :],
            candidate_actor,
        )
        candidate_actor_history = mx.concatenate(
            (
                state.actor_history[:, 1:, :],
                candidate_actor[:, None, :],
            ),
            axis=1,
        )
        candidate_clean_history = mx.concatenate(
            (
                state.clean_history[:, 1:, :],
                candidate_clean[:, None, :],
            ),
            axis=1,
        )
        candidate_critic = self._critic(
            candidate_clean_history,
            candidate.q,
            candidate.v,
            compact,
            state.body_parameters,
            state.controller_parameters,
            candidate.scene_bodies.position,
        )
        timeout_bootstrap = mx.where(
            timeout & (~physics_error),
            self._compiled_value(candidate_critic),
            mx.zeros_like(value),
        )

        episode_return = state.episode_return + reward
        episode_tracking = (
            state.episode_tracking
            + 0.5 * (tracking + yaw_tracking)
        )
        tracking_score = episode_tracking / mx.maximum(
            episode_steps.astype(mx.float32),
            1.0,
        )
        successful = timeout & (tracking_score >= 0.8)
        failed = done & (~successful)
        next_curriculum = mx.clip(
            state.curriculum_level
            + successful.astype(mx.int32)
            - failed.astype(mx.int32),
            0,
            10,
        )
        next_terrain_level = mx.clip(
            state.terrain_level
            + (
                successful
                & (state.curriculum_level >= 3)
            ).astype(mx.int32)
            - failed.astype(mx.int32),
            0,
            10,
        )

        default_q = mx.broadcast_to(
            mx.array(
                self.world.default_q,
                dtype=mx.float32,
            )[None, :],
            candidate.q.shape,
        )
        half_yaw = 0.04 * random.reset_pose[:, 2:3]
        reset_q = mx.concatenate(
            (
                default_q[:, :2]
                + 0.015 * random.reset_pose[:, :2],
                default_q[:, 2:3],
                mx.zeros(
                    (self.environment_count, 2),
                    dtype=mx.float32,
                ),
                mx.sin(half_yaw),
                mx.cos(half_yaw),
                default_q[:, 7:]
                + 0.025 * random.reset_pose[:, 3:32],
            ),
            axis=-1,
        )
        reset_v = 0.05 * random.reset_velocity
        reset_world = self._reset_world(
            candidate,
            reset_q,
            reset_v,
            next_terrain_level,
        )
        next_world = _select_world(done, reset_world, candidate)

        (
            sampled_body,
            sampled_controller,
            sampled_actuator_delay,
            sampled_observation_delay,
            sensor_bias,
        ) = self._domain_parameters(random.domain)
        next_body = _select(
            done,
            sampled_body,
            state.body_parameters,
        )
        next_controller = _select(
            done,
            sampled_controller,
            state.controller_parameters,
        )
        next_actuator_delay = mx.where(
            done,
            sampled_actuator_delay,
            state.actuator_delay,
        )
        next_observation_delay = mx.where(
            done,
            sampled_observation_delay,
            state.observation_delay,
        )
        next_gyro_bias = _select(
            done,
            0.02 * (2.0 * sensor_bias[:, :3] - 1.0),
            state.gyro_bias,
        )
        sampled_encoder_bias = (
            0.005
            * random.policy_noise[:, :29]
        )
        next_encoder_bias = _select(
            done,
            sampled_encoder_bias,
            state.encoder_bias,
        )
        zero_action = mx.zeros_like(normalized_action)
        reset_clean = self._frame(
            reset_q,
            reset_v,
            next_commands,
            zero_action,
        )
        reset_actor = self._corrupt_frame(
            reset_clean,
            random.policy_noise,
            next_gyro_bias,
            next_encoder_bias,
        )
        reset_actor_history = mx.broadcast_to(
            reset_actor[:, None, :],
            candidate_actor_history.shape,
        )
        reset_clean_history = mx.broadcast_to(
            reset_clean[:, None, :],
            candidate_clean_history.shape,
        )
        reset_compact = mx.zeros_like(compact)
        reset_critic = self._critic(
            reset_clean_history,
            reset_q,
            reset_v,
            reset_compact,
            next_body,
            next_controller,
            reset_world.scene_bodies.position,
        )

        next_actor_history = _select(
            done,
            reset_actor_history,
            candidate_actor_history,
        )
        next_clean_history = _select(
            done,
            reset_clean_history,
            candidate_clean_history,
        )
        next_critic = _select(
            done,
            reset_critic,
            candidate_critic,
        )
        next_air_time = _select(
            done,
            mx.zeros_like(air_time),
            air_time,
        )
        next_delay_history = _select(
            done,
            mx.zeros_like(delay_history),
            delay_history,
        )
        next_push_steps = mx.where(
            done | push_due,
            (
                100
                + mx.floor(
                    150.0
                    * (0.5 + 0.5 * random.push[:, 2])
                ).astype(mx.int32)
            ),
            state.push_steps_remaining - 1,
        )
        next_state = G1LocomotionState(
            world=next_world,
            actor_history=next_actor_history,
            clean_history=next_clean_history,
            critic_observation=next_critic,
            previous_action=_select(
                done,
                zero_action,
                normalized_action,
            ),
            previous_joint_velocity=_select(
                done,
                mx.zeros_like(joint_velocity),
                joint_velocity,
            ),
            commands=next_commands,
            command_steps_remaining=next_command_steps,
            episode_steps=mx.where(
                done,
                mx.zeros_like(episode_steps),
                episode_steps,
            ),
            gait_phase=mx.where(
                done,
                mx.zeros_like(phase),
                phase,
            ),
            foot_air_time=next_air_time,
            action_delay_history=next_delay_history,
            actuator_delay=next_actuator_delay,
            observation_delay=next_observation_delay,
            gyro_bias=next_gyro_bias,
            encoder_bias=next_encoder_bias,
            body_parameters=next_body,
            controller_parameters=next_controller,
            curriculum_level=next_curriculum,
            terrain_level=next_terrain_level,
            push_steps_remaining=next_push_steps,
            episode_return=mx.where(
                done,
                mx.zeros_like(episode_return),
                episode_return,
            ),
            episode_tracking=mx.where(
                done,
                mx.zeros_like(episode_tracking),
                episode_tracking,
            ),
        )
        metrics = mx.stack(
            (
                tracking,
                yaw_tracking,
                link_height,
                tilt,
                mx.max(compact[:, 12:16], axis=-1),
                episode_return,
                tracking_score,
                state.curriculum_level.astype(mx.float32),
                tracking_error,
                yaw_error,
            ),
            axis=-1,
        )
        metrics = mx.concatenate((metrics, contact_metrics), axis=-1)
        return G1LocomotionStepOutput(
            state=next_state,
            actor_observation=actor_observation,
            critic_observation=critic_observation,
            latent=latent,
            log_probability=log_probability,
            value=value,
            reward=reward,
            done=done,
            timeout=timeout,
            physics_error=physics_error,
            physics_status=physics.status,
            termination_reason=reason,
            timeout_bootstrap_value=timeout_bootstrap,
            metrics=metrics,
        )

    def _compact_transition_impl(
        self,
        state: _G1CompiledLocomotionState,
        random: _G1RandomInputs,
        policy_noise: mx.array,
    ) -> _G1CompiledStepOutput:
        output = self._transition_impl(
            _expand_g1_state(state, self._compiled_world_template),
            random,
            policy_noise,
        )
        return _G1CompiledStepOutput(
            _compact_g1_state(output.state),
            *output[1:],
        )

    def _transition(
        self,
        state: G1LocomotionState,
        random: _G1RandomInputs,
        policy_noise: mx.array,
    ) -> G1LocomotionStepOutput:
        """Execute the stable G1 transition through one cached MLX graph."""

        output = self._compiled_transition(
            _compact_g1_state(state),
            random,
            policy_noise,
        )
        return G1LocomotionStepOutput(
            _expand_g1_state(output.state, self.default_world),
            *output[1:],
        )

    def warmup(self) -> None:
        """Materialize one transition before constructing lazy rollouts."""

        for _, value in tree_flatten(self.model.parameters()):
            mx.eval(value)
        random = self._random_inputs()
        policy_noise = mx.random.normal(
            (self.environment_count, 29)
        )
        # MLX specializes RandomBits pipelines by output layout. Materialize
        # each layout independently so Apple's Metal compiler file cache is
        # not entered repeatedly by one large first-use lazy graph.
        for value in (*random, policy_noise):
            mx.eval(value)
        warm_state = self.initial()
        for _, value in tree_flatten(_compact_g1_state(warm_state)):
            mx.eval(value)
        output = self._transition(
            warm_state,
            random,
            policy_noise,
        )
        mx.eval(
            _compact_g1_state(output.state),
            output.reward,
            output.physics_error,
            output.physics_status,
        )

    def collect(
        self,
        state: G1LocomotionState,
        rollout_steps: int,
        *,
        deterministic: bool = False,
    ) -> tuple[G1LocomotionState, G1LocomotionRolloutBatch]:
        if (
            rollout_steps <= 0
            or rollout_steps % self.task.rollout_chunk_size
        ):
            raise ValueError(
                "rollout_steps must be a positive multiple of eight"
        )
        current = state
        records: list[_G1RolloutRecord] = []
        for index in range(rollout_steps):
            output = self._transition(
                current,
                self._random_inputs(),
                (
                    mx.zeros(
                        (self.environment_count, 29),
                        dtype=mx.float32,
                    )
                    if deterministic
                    else mx.random.normal(
                        (self.environment_count, 29)
                    )
                ),
            )
            current = output.state
            records.append(
                _G1RolloutRecord(
                    actor_observation=output.actor_observation,
                    critic_observation=output.critic_observation,
                    latent=output.latent,
                    log_probability=output.log_probability,
                    value=output.value,
                    reward=output.reward,
                    done=output.done,
                    timeout=output.timeout,
                    physics_error=output.physics_error,
                    physics_status=output.physics_status,
                    termination_reason=output.termination_reason,
                    timeout_bootstrap_value=(
                        output.timeout_bootstrap_value
                    ),
                    metrics=output.metrics,
                )
            )
            if (index + 1) % self.task.rollout_chunk_size == 0:
                chunk = records[
                    index + 1 - self.task.rollout_chunk_size :
                    index + 1
                ]
                # The state alone does not depend on every rollout field.
                # Materialize the complete record payload so no physics,
                # status, or policy branch crosses the bounded chunk.
                mx.eval(
                    current,
                    chunk,
                )

        final_value = self._compiled_value(
            current.critic_observation
        )
        advantages: list[mx.array] = [
            mx.zeros_like(final_value)
            for _ in range(rollout_steps)
        ]
        advantage = mx.zeros_like(final_value)
        next_value = final_value
        for index in range(rollout_steps - 1, -1, -1):
            record = records[index]
            continuing = (~record.done).astype(mx.float32)
            bootstrap = (
                continuing * next_value
                + record.timeout.astype(mx.float32)
                * record.timeout_bootstrap_value
            )
            delta = (
                record.reward
                + self.gamma * bootstrap
                - record.value
            )
            advantage = (
                delta
                + self.gamma
                * self.gae_lambda
                * continuing
                * advantage
            )
            advantages[index] = advantage
            next_value = record.value

        values = mx.stack([record.value for record in records])
        batch = G1LocomotionRolloutBatch(
            actor_observations=mx.stack(
                [record.actor_observation for record in records]
            ),
            critic_observations=mx.stack(
                [record.critic_observation for record in records]
            ),
            latents=mx.stack(
                [record.latent for record in records]
            ),
            old_log_probabilities=mx.stack(
                [record.log_probability for record in records]
            ),
            old_values=values,
            advantages=mx.stack(advantages),
            returns=mx.stack(advantages) + values,
            rewards=mx.stack(
                [record.reward for record in records]
            ),
            dones=mx.stack([record.done for record in records]),
            timeouts=mx.stack(
                [record.timeout for record in records]
            ),
            physics_errors=mx.stack(
                [record.physics_error for record in records]
            ),
            physics_statuses=mx.stack(
                [record.physics_status for record in records]
            ),
            termination_reasons=mx.stack(
                [record.termination_reason for record in records]
            ),
            metrics=mx.stack(
                [record.metrics for record in records]
            ),
        )
        mx.eval(
            batch.actor_observations,
            batch.critic_observations,
            batch.returns,
            batch.physics_errors,
            current.world.q,
            current.world.v,
        )
        return current, batch


def _concatenate_rollouts(
    batches: list[G1LocomotionRolloutBatch],
) -> G1LocomotionRolloutBatch:
    if not batches:
        raise ValueError("at least one rollout cohort is required")
    if len(batches) == 1:
        return batches[0]
    return G1LocomotionRolloutBatch(
        *(
            mx.concatenate(values, axis=1)
            for values in zip(*batches, strict=True)
        )
    )


class MLXG1PPOTrainer:
    """Serial-cohort G1 PPO with one active Metal execution cohort."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        metallib_path: str | None = None,
        maximum_episode_steps: int = 1_000,
        physics_substeps: int = 4,
        velocity_iterations: int = 4,
        final_velocity_iterations: int = 2,
        terrain: bool = True,
        cohort_size: int = 256,
    ) -> None:
        config.validate()
        self.task = G1LocomotionTaskSpec(
            physics_substeps=physics_substeps,
            maximum_episode_steps=maximum_episode_steps,
            cohort_size=min(cohort_size, config.environment_count),
            rollout_steps=config.rollout_steps,
        )
        self.task.validate()
        if (
            config.environment_count > self.task.cohort_size
            and config.environment_count % self.task.cohort_size
        ):
            raise ValueError(
                "logical G1 environments must divide into fixed "
                "execution cohorts"
            )
        self.config = config
        self.task_name = "g1-locomotion"
        self.observation_size = G1_ACTOR_OBSERVATION_SIZE
        self.critic_observation_size = G1_CRITIC_OBSERVATION_SIZE
        self.action_size = 29
        self.terrain = terrain
        mx.random.seed(config.seed)

        execution_count = min(
            config.environment_count,
            self.task.cohort_size,
        )
        self.world = compile_world(
            "g1",
            scene="terrain" if terrain else "ground",
            environment_capacity=execution_count,
            control_timestep=self.task.control_timestep,
            physics_substeps=self.task.physics_substeps,
            actuation_mode="implicit_position",
            solver_mode="throughput_tgs",
            velocity_iterations=velocity_iterations,
            final_velocity_iterations=final_velocity_iterations,
            ccd_mode="disabled",
            publish_body_states=True,
            metallib_path=metallib_path or "",
        )
        self.model = G1ActorCritic(
            hidden_sizes=config.hidden_sizes,
            initial_log_std=config.initial_log_std,
            seed=config.seed,
        )
        self.optimizer = optim.Adam(
            learning_rate=config.learning_rate,
            bias_correction=True,
        )
        self.optimizer.init(self.model.trainable_parameters())
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.collector = MLXG1RolloutCollector(
            self.world,
            self.model,
            execution_count,
            task=self.task,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            terrain=terrain,
        )
        # Compile and execute one bounded graph before a rollout cohort builds
        # eight lazy transitions. This keeps first-use Metal compilation out
        # of the first production command-buffer burst.
        self.collector.warmup()
        cohort_count = (
            config.environment_count // execution_count
        )
        self.rollout_states = [
            self.collector.initial()
            for _ in range(cohort_count)
        ]
        self.iteration = 0
        self.environment_steps = 0
        self._loss_and_grad = nn.value_and_grad(
            self.model,
            self._loss,
        )
        self._joint_mirror_indices = mx.array(
            (
                6, 7, 8, 9, 10, 11,
                0, 1, 2, 3, 4, 5,
                12, 13, 14,
                22, 23, 24, 25, 26, 27, 28,
                15, 16, 17, 18, 19, 20, 21,
            ),
            dtype=mx.uint32,
        )
        self._joint_mirror_sign = mx.array(
            (
                1, -1, -1, 1, 1, -1,
                1, -1, -1, 1, 1, -1,
                -1, -1, 1,
                1, -1, -1, 1, -1, 1, -1,
                1, -1, -1, 1, -1, 1, -1,
            ),
            dtype=mx.float32,
        )
        self._left_right = mx.array(
            [1, 0],
            dtype=mx.uint32,
        )

    def _mirror_frame(self, frame: mx.array) -> mx.array:
        angular = frame[:, :3] * mx.array(
            [-1.0, 1.0, -1.0],
            dtype=mx.float32,
        )
        gravity = frame[:, 3:6] * mx.array(
            [1.0, -1.0, 1.0],
            dtype=mx.float32,
        )
        command = frame[:, 6:9] * mx.array(
            [1.0, -1.0, -1.0],
            dtype=mx.float32,
        )
        components = [angular, gravity, command]
        for start in (9, 38, 67):
            components.append(
                frame[:, start : start + 29][
                    :, self._joint_mirror_indices
                ]
                * self._joint_mirror_sign
            )
        return mx.concatenate(components, axis=-1)

    def _mirror_actor(self, observation: mx.array) -> mx.array:
        history = observation.reshape(
            (-1, G1_ACTOR_HISTORY, G1_ACTOR_FRAME_SIZE)
        )
        frames = [
            self._mirror_frame(history[:, index, :])
            for index in range(G1_ACTOR_HISTORY)
        ]
        return mx.stack(frames, axis=1).reshape(
            (-1, G1_ACTOR_OBSERVATION_SIZE)
        )

    def _mirror_critic(self, observation: mx.array) -> mx.array:
        actor = self._mirror_actor(
            observation[:, :G1_ACTOR_OBSERVATION_SIZE]
        )
        cursor = G1_ACTOR_OBSERVATION_SIZE
        velocity_height = observation[:, cursor : cursor + 4]
        velocity_height = mx.concatenate(
            (
                velocity_height[:, :1],
                -velocity_height[:, 1:2],
                velocity_height[:, 2:],
            ),
            axis=-1,
        )
        cursor += 4
        contact = observation[
            :, cursor : cursor + G1_COMPACT_CONTACT_SIZE
        ]
        mirrored_contact = mx.concatenate(
            (
                contact[:, :2][:, self._left_right],
                contact[:, 2:4][:, self._left_right],
                contact[:, 4:6][:, self._left_right],
                contact[:, 6:8][:, self._left_right],
                contact[:, 10:12]
                * mx.array([1.0, -1.0], dtype=mx.float32),
                contact[:, 8:10]
                * mx.array([1.0, -1.0], dtype=mx.float32),
                contact[:, 12:],
            ),
            axis=-1,
        )
        cursor += G1_COMPACT_CONTACT_SIZE
        terrain = observation[
            :, cursor : cursor + G1_TERRAIN_SAMPLE_COUNT
        ].reshape((-1, G1_TERRAIN_GRID_X, G1_TERRAIN_GRID_Y))
        terrain = terrain[:, :, ::-1].reshape(
            (-1, G1_TERRAIN_SAMPLE_COUNT)
        )
        cursor += G1_TERRAIN_SAMPLE_COUNT
        return mx.concatenate(
            (
                actor,
                velocity_height,
                mirrored_contact,
                terrain,
                observation[:, cursor:],
            ),
            axis=-1,
        )

    def _loss(
        self,
        actor_observation: mx.array,
        critic_observation: mx.array,
        latent: mx.array,
        old_log_probability: mx.array,
        old_value: mx.array,
        advantage: mx.array,
        returns: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        log_probability, entropy, value = self.model.evaluate(
            actor_observation,
            critic_observation,
            latent,
        )
        log_ratio = log_probability - old_log_probability
        ratio = mx.exp(log_ratio)
        unclipped = ratio * advantage
        clipped = mx.clip(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio,
        ) * advantage
        policy_loss = -mx.mean(mx.minimum(unclipped, clipped))
        clipped_value = old_value + mx.clip(
            value - old_value,
            -self.config.clip_ratio,
            self.config.clip_ratio,
        )
        value_loss = 0.5 * mx.mean(
            mx.maximum(
                mx.square(value - returns),
                mx.square(clipped_value - returns),
            )
        )
        entropy_mean = mx.mean(entropy)
        loss = (
            policy_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy_mean
        )
        return loss, {
            "loss": loss,
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
        rollout: G1LocomotionRolloutBatch,
    ) -> dict[str, float]:
        batch = rollout.flattened()
        advantage = batch["advantages"]
        batch["advantages"] = (
            advantage - mx.mean(advantage)
        ) / (mx.std(advantage) + 1.0e-8)
        mirrored_actor = self._mirror_actor(
            batch["actor_observations"]
        )
        mirrored_critic = self._mirror_critic(
            batch["critic_observations"]
        )
        mirrored_latent = (
            batch["latents"][:, self._joint_mirror_indices]
            * self._joint_mirror_sign
        )
        (
            mirrored_old_log_probability,
            _,
            mirrored_old_value,
        ) = self.model.evaluate(
            mirrored_actor,
            mirrored_critic,
            mirrored_latent,
        )
        # Materialize behavior-policy probabilities before the first update.
        # Reusing the original sample's probability for mirrored observations
        # would produce a biased PPO ratio once the actor is not symmetric.
        mx.eval(
            mirrored_actor,
            mirrored_critic,
            mirrored_latent,
            mirrored_old_log_probability,
            mirrored_old_value,
        )
        sample_count = int(advantage.shape[0])
        minibatch_size = math.ceil(sample_count / 4)
        totals: dict[str, float] = {}
        updates = 0
        start = time.perf_counter()
        for _ in range(self.config.update_epochs):
            permutation = mx.random.permutation(sample_count)
            for offset in range(0, sample_count, minibatch_size):
                index = permutation[
                    offset : offset + minibatch_size
                ]
                split = int(index.shape[0]) // 2
                direct_index = index[:split]
                mirror_index = index[split:]
                actor = mx.concatenate(
                    (
                        batch["actor_observations"][direct_index],
                        mirrored_actor[mirror_index],
                    ),
                    axis=0,
                )
                critic = mx.concatenate(
                    (
                        batch["critic_observations"][direct_index],
                        mirrored_critic[mirror_index],
                    ),
                    axis=0,
                )
                latent = mx.concatenate(
                    (
                        batch["latents"][direct_index],
                        mirrored_latent[mirror_index],
                    ),
                    axis=0,
                )
                old_log_probability = mx.concatenate(
                    (
                        batch["old_log_probabilities"][direct_index],
                        mirrored_old_log_probability[mirror_index],
                    ),
                    axis=0,
                )
                old_value = mx.concatenate(
                    (
                        batch["old_values"][direct_index],
                        mirrored_old_value[mirror_index],
                    ),
                    axis=0,
                )
                minibatch_advantage = mx.concatenate(
                    (
                        batch["advantages"][direct_index],
                        batch["advantages"][mirror_index],
                    ),
                    axis=0,
                )
                minibatch_returns = mx.concatenate(
                    (
                        batch["returns"][direct_index],
                        batch["returns"][mirror_index],
                    ),
                    axis=0,
                )
                (loss, metrics), gradients = self._loss_and_grad(
                    actor,
                    critic,
                    latent,
                    old_log_probability,
                    old_value,
                    minibatch_advantage,
                    minibatch_returns,
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
                numeric = {
                    key: float(value.item())
                    for key, value in metrics.items()
                }
                for key, value in numeric.items():
                    totals[key] = totals.get(key, 0.0) + value
                updates += 1
        if self.config.target_kl is not None:
            learning_rate = float(self.optimizer.learning_rate)
            mean_kl = totals.get("approx_kl", 0.0) / max(
                updates,
                1,
            )
            if mean_kl > 2.0 * self.config.target_kl:
                learning_rate = max(1.0e-5, learning_rate / 1.5)
            elif mean_kl < 0.5 * self.config.target_kl:
                learning_rate = min(
                    self.config.learning_rate,
                    learning_rate * 1.5,
                )
            self.optimizer.learning_rate = learning_rate
        result = {
            key: value / max(updates, 1)
            for key, value in totals.items()
        }
        result["update_seconds"] = time.perf_counter() - start
        result["minibatch_updates"] = float(updates)
        result["learning_rate"] = float(
            self.optimizer.learning_rate
        )
        return result

    def _collect(self) -> G1LocomotionRolloutBatch:
        batches = []
        next_states = []
        for state in self.rollout_states:
            next_state, batch = self.collector.collect(
                state,
                self.config.rollout_steps,
            )
            next_states.append(next_state)
            batches.append(batch)
        self.rollout_states = next_states
        rollout = _concatenate_rollouts(batches)
        mx.eval(
            rollout.rewards,
            rollout.physics_errors,
            rollout.metrics,
        )
        return rollout

    def evaluate(self, steps: int = 1_000) -> dict[str, float]:
        """Stream one deterministic held-out episode per environment."""

        if steps <= 0 or steps % self.task.rollout_chunk_size:
            raise ValueError(
                "evaluation steps must be a positive multiple of eight"
            )
        state = self.collector.evaluation_initial()
        environment_count = self.collector.environment_count
        alive = np.ones(environment_count, dtype=bool)
        first_done_step = np.full(
            environment_count,
            steps,
            dtype=np.int32,
        )
        first_reason = np.zeros(environment_count, dtype=np.uint32)
        planar_squared_error = 0.0
        yaw_squared_error = 0.0
        reward_sum = 0.0
        active_samples = 0
        physics_errors = 0
        for offset in range(0, steps, self.task.rollout_chunk_size):
            state, batch = self.collector.collect(
                state,
                self.task.rollout_chunk_size,
                deterministic=True,
            )
            done = np.asarray(batch.dones, dtype=bool)
            reason = np.asarray(
                batch.termination_reasons,
                dtype=np.uint32,
            )
            errors = np.asarray(
                batch.physics_errors,
                dtype=bool,
            )
            metrics = np.asarray(batch.metrics, dtype=np.float32)
            rewards = np.asarray(batch.rewards, dtype=np.float32)
            physics_errors += int(errors.sum())
            for local_step in range(self.task.rollout_chunk_size):
                active = alive.copy()
                active_count = int(active.sum())
                if active_count:
                    planar_squared_error += float(
                        metrics[local_step, active, 8].sum()
                    )
                    yaw_squared_error += float(
                        metrics[local_step, active, 9].sum()
                    )
                    reward_sum += float(
                        rewards[local_step, active].sum()
                    )
                    active_samples += active_count
                finished = active & done[local_step]
                first_done_step[finished] = (
                    offset + local_step + 1
                )
                first_reason[finished] = reason[
                    local_step,
                    finished,
                ]
                alive[finished] = False

        survived = (
            alive
            | (first_reason == G1Termination.timeout)
        )
        split = environment_count // 2
        flat = np.arange(environment_count) < split
        mixed = ~flat

        def fraction(mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            return float(np.mean(survived[mask]))

        denominator = max(active_samples, 1)
        return {
            "held_out_environments": float(environment_count),
            "held_out_steps": float(steps),
            "flat_full_episode_survival": fraction(flat),
            "mixed_full_episode_survival": fraction(mixed),
            "push_recovery_fraction": fraction(flat),
            "mean_survival_fraction": float(
                np.mean(first_done_step / float(steps))
            ),
            "planar_velocity_rmse": math.sqrt(
                planar_squared_error / denominator
            ),
            "yaw_velocity_rmse": math.sqrt(
                yaw_squared_error / denominator
            ),
            "mean_reward": reward_sum / denominator,
            "physics_errors": float(physics_errors),
        }

    def policy_pack(
        self,
        weights_sha256: str,
        evaluation_summary: dict[str, float] | None = None,
    ) -> G1PolicyPack:
        if len(weights_sha256) != 64:
            raise ValueError("policy weights require a SHA-256 identity")
        return G1PolicyPack(
            format="metalrobo.g1-policy-pack",
            task=self.task_name,
            actor_observation_size=G1_ACTOR_OBSERVATION_SIZE,
            critic_observation_size=G1_CRITIC_OBSERVATION_SIZE,
            action_size=29,
            joint_order=G1_JOINT_ORDER,
            default_pose=tuple(
                float(value) for value in self.world.default_q[7:]
            ),
            stiffness=tuple(
                float(value) for value in self.world.drive_stiffness
            ),
            damping=tuple(
                float(value) for value in self.world.drive_damping
            ),
            position_limits=tuple(
                (float(lower), float(upper))
                for lower, upper in zip(
                    self.world.joint_lower_limits,
                    self.world.joint_upper_limits,
                    strict=True,
                )
            ),
            effort_limits=tuple(
                float(value)
                for value in self.world.effort_limits[6:]
            ),
            velocity_limits=tuple(
                float(value)
                for value in self.world.joint_velocity_limits
            ),
            action_scale_radians=self.task.action_scale_radians,
            command_limits=(
                (-0.5, 1.0),
                (-0.3, 0.3),
                (-0.2, 0.2),
            ),
            drive_prediction_seconds=self.task.physics_timestep,
            source_revision=self.task.source_revision,
            source_model=self.task.source_model,
            world_fingerprint=(
                f"{int(self.world.world_fingerprint):016x}"
            ),
            weights_sha256=weights_sha256,
            renderer_free_training=True,
            evaluation_summary=evaluation_summary or {},
        )

    def save_checkpoint(
        self,
        directory: str | Path | None = None,
        *,
        evaluation_summary: dict[str, float] | None = None,
    ) -> Path:
        root = Path(
            directory or self.config.checkpoint_directory
        ).expanduser()
        checkpoint = root / f"checkpoint-{self.iteration:06d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        weight_path = checkpoint / "model.safetensors"
        self.model.save_weights(str(weight_path))
        with weight_path.open("rb") as weight_file:
            weights_sha256 = hashlib.file_digest(
                weight_file,
                "sha256",
            ).hexdigest()
        mx.save_safetensors(
            str(checkpoint / "optimizer.safetensors"),
            dict(tree_flatten(self.optimizer.state)),
        )
        policy = self.policy_pack(
            weights_sha256,
            evaluation_summary,
        )
        policy_record = {
            **asdict(policy),
            "fingerprint": policy.fingerprint,
            "world_pack_hash": int(self.world.authored_pack_hash),
            "iteration": self.iteration,
            "environment_steps": self.environment_steps,
            "ppo": asdict(self.config),
        }
        (checkpoint / "policy.json").write_text(
            json.dumps(
                policy_record,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def load_checkpoint(self, directory: str | Path) -> None:
        checkpoint = Path(directory).expanduser()
        record = json.loads(
            (checkpoint / "policy.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            record.get("format") != "metalrobo.g1-policy-pack"
            or record.get("task") != self.task_name
            or tuple(record.get("joint_order", ())) != G1_JOINT_ORDER
            or record.get("actor_observation_size")
            != G1_ACTOR_OBSERVATION_SIZE
        ):
            raise ValueError(
                "checkpoint does not match the G1 locomotion ABI"
            )
        self.model.load_weights(
            str(checkpoint / "model.safetensors")
        )
        self.iteration = int(record["iteration"])
        self.environment_steps = int(record["environment_steps"])
        mx.eval(self.model.parameters())

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
            rollout_start = time.perf_counter()
            rollout = self._collect()
            rollout_seconds = time.perf_counter() - rollout_start
            update = self._update(rollout)
            added_steps = (
                self.config.environment_count
                * self.config.rollout_steps
            )
            self.environment_steps += added_steps
            report: dict[str, Any] = {
                "iteration": iteration,
                "environment_steps": self.environment_steps,
                "backend": "mlx_active_encoder",
                "task": self.task_name,
                "logical_environments":
                    self.config.environment_count,
                "execution_cohort": self.collector.environment_count,
                "rollout_seconds": rollout_seconds,
                "rollout_env_steps_per_second": (
                    added_steps / max(rollout_seconds, 1.0e-9)
                ),
                "mean_step_reward": float(
                    mx.mean(rollout.rewards).item()
                ),
                "physics_errors": int(
                    mx.sum(
                        rollout.physics_errors.astype(mx.uint32)
                    ).item()
                ),
                "mean_planar_tracking": float(
                    mx.mean(rollout.metrics[..., 0]).item()
                ),
                "mean_yaw_tracking": float(
                    mx.mean(rollout.metrics[..., 1]).item()
                ),
                "mlx_active_memory_bytes": int(
                    mx.get_active_memory()
                ),
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                **update,
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
                and iteration % self.config.checkpoint_interval == 0
            ):
                checkpoint = self.save_checkpoint()
        if checkpoint is None or self.iteration % max(
            self.config.checkpoint_interval,
            1,
        ):
            checkpoint = self.save_checkpoint()
        evaluation = self.evaluate(self.task.maximum_episode_steps)
        checkpoint = self.save_checkpoint(
            evaluation_summary=evaluation,
        )
        print(
            json.dumps(
                {
                    "iteration": self.iteration,
                    "task": self.task_name,
                    "evaluation": evaluation,
                },
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return checkpoint


__all__ = [
    "G1_ACTOR_FRAME_SIZE",
    "G1_ACTOR_HISTORY",
    "G1_ACTOR_OBSERVATION_SIZE",
    "G1_BODY_ORDER",
    "G1_CRITIC_OBSERVATION_SIZE",
    "G1_JOINT_ORDER",
    "G1ActorCritic",
    "G1LocomotionRolloutBatch",
    "G1LocomotionState",
    "G1LocomotionStepOutput",
    "G1LocomotionTaskSpec",
    "G1PolicyPack",
    "G1Termination",
    "MLXG1PPOTrainer",
    "MLXG1RolloutCollector",
]
