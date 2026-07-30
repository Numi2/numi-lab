"""Contact-capable G1 PPO rollouts on MLX's active Metal encoder.

The policy, implicit joint targets, rigid contact step, reward, termination,
transactional reset, rollout storage, and PPO update remain MLX array
operations. Host synchronization is reserved for the existing logging and
optimizer boundaries in :class:`MLXPPOTrainer`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .mlx_ppo import (
    MLXPPOTrainer,
    MLXRolloutBatch,
    MLXRolloutState,
)
from .mlx_world import (
    ActuatorState,
    RodState,
    SceneBodyState,
    SolverCache,
    TactileState,
    WorldState,
    compile_world_pack,
    initial_state,
    step,
)
from .ppo import ActorCritic, PPOConfig
from .tactile_latent import (
    TactileAtlasRange,
    canonical_metric_tactile_policy_observation,
)


_DUAL_32_TACTILE_ATLASES = (
    TactileAtlasRange(offset=0, width=32, height=32),
    TactileAtlasRange(offset=32 * 32, width=32, height=32),
)


def _select_environment(
    mask: mx.array,
    reset_value: mx.array,
    current_value: mx.array,
) -> mx.array:
    shape = (int(mask.shape[0]),) + (1,) * (
        current_value.ndim - 1
    )
    return mx.where(
        mask.reshape(shape),
        reset_value,
        current_value,
    )


def _tactile_policy_observation(tactile) -> mx.array:
    """Flatten the shared metric stem with explicit source availability."""

    return canonical_metric_tactile_policy_observation(
        tactile,
        _DUAL_32_TACTILE_ATLASES,
    )


class MLXG1RolloutCollector:
    """Pure-MLX G1 standing/locomotion rollout over the contact world."""

    def __init__(
        self,
        world,
        model: ActorCritic,
        environment_count: int,
        *,
        gamma: float,
        gae_lambda: float,
        chunk_size: int = 16,
        maximum_episode_steps: int = 1_200,
        action_scale: float = 0.35,
        command_tracking: bool = False,
        command_scales: tuple[float, float, float] = (
            0.5,
            0.3,
            0.5,
        ),
        reset_root_xy_range: float = 0.015,
        reset_root_yaw_range: float = 0.08,
        reset_joint_range: float = 0.025,
        reset_velocity_range: float = 0.05,
    ) -> None:
        if environment_count <= 0:
            raise ValueError("environment_count must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be positive")
        if (
            world.solver_mode == "free_motion_aba"
            or world.actuation_mode != "implicit_position"
            or not world.floating_root
        ):
            raise ValueError(
                "G1 rollouts require a floating-root contact world "
                "with implicit_position actuation"
            )
        if world.nv < 7 or world.nq != world.nv + 1:
            raise ValueError("G1 state must use quaternion floating-root ABI")
        if (
            int(world.tactile_sensor_count) != 2
            or int(world.tactile_sample_count) != 2 * 32 * 32
        ):
            raise ValueError(
                "G1 tactile rollouts require the authored dual-sole "
                "32x32 observation contract"
            )
        self.world = world
        self.model = model
        self.environment_count = environment_count
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.chunk_size = chunk_size
        self.maximum_episode_steps = maximum_episode_steps
        self.action_scale = float(action_scale)
        self.command_tracking = bool(command_tracking)
        self.command_scales = mx.array(
            command_scales,
            dtype=mx.float32,
        )
        for name, value in (
            ("reset_root_xy_range", reset_root_xy_range),
            ("reset_root_yaw_range", reset_root_yaw_range),
            ("reset_joint_range", reset_joint_range),
            ("reset_velocity_range", reset_velocity_range),
        ):
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        self.reset_root_xy_range = float(reset_root_xy_range)
        self.reset_root_yaw_range = float(reset_root_yaw_range)
        self.reset_joint_range = float(reset_joint_range)
        self.reset_velocity_range = float(reset_velocity_range)
        self.action_size = int(world.nv) - 6
        self.default_state = initial_state(
            world,
            environment_count,
        )
        self.default_q = mx.array(
            world.default_q,
            dtype=mx.float32,
        )
        self.default_joint_targets = self.default_q[7:]
        self._compiled_step = mx.compile(
            self._policy_physics_reward,
            inputs=self.model.state,
        )

    def _randomized_reset_qv(
        self,
    ) -> tuple[mx.array, mx.array]:
        root_xy = mx.random.uniform(
            low=-self.reset_root_xy_range,
            high=self.reset_root_xy_range,
            shape=(self.environment_count, 2),
        )
        root_yaw = mx.random.uniform(
            low=-self.reset_root_yaw_range,
            high=self.reset_root_yaw_range,
            shape=(self.environment_count, 1),
        )
        half_yaw = 0.5 * root_yaw
        joint_offset = mx.random.uniform(
            low=-self.reset_joint_range,
            high=self.reset_joint_range,
            shape=(self.environment_count, self.world.nq - 7),
        )
        default_q = self.default_state.q
        reset_q = mx.concatenate(
            (
                default_q[:, :2] + root_xy,
                default_q[:, 2:3],
                mx.zeros(
                    (self.environment_count, 2),
                    dtype=mx.float32,
                ),
                mx.sin(half_yaw),
                mx.cos(half_yaw),
                default_q[:, 7:] + joint_offset,
            ),
            axis=-1,
        )
        reset_v = self.default_state.v + mx.random.uniform(
            low=-self.reset_velocity_range,
            high=self.reset_velocity_range,
            shape=(self.environment_count, self.world.nv),
        )
        return reset_q, reset_v

    def _observations(
        self,
        state: WorldState,
        commands: mx.array,
        tactile_policy: mx.array,
    ) -> mx.array:
        components = [state.q, state.v, tactile_policy]
        if self.command_tracking:
            components.append(commands)
        return mx.concatenate(components, axis=-1)

    def initial(self) -> MLXRolloutState:
        commands = (
            mx.random.uniform(
                low=-1.0,
                high=1.0,
                shape=(self.environment_count, 3),
            )
            * self.command_scales
            if self.command_tracking
            else mx.zeros(
                (self.environment_count, 3),
                dtype=mx.float32,
            )
        )
        return MLXRolloutState(
            world=self.default_state,
            observations=self._observations(
                self.default_state,
                commands,
                mx.zeros(
                    (self.environment_count, 132),
                    dtype=mx.float32,
                ),
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
        scene_position: mx.array,
        scene_orientation: mx.array,
        scene_linear_velocity: mx.array,
        scene_angular_velocity: mx.array,
        rod_position: mx.array,
        rod_velocity: mx.array,
        rod_twist: mx.array,
        rod_twist_rate: mx.array,
        manifold_headers: mx.array,
        manifold_points: mx.array,
        manifold_counts: mx.array,
        pair_cache: mx.array,
        rod_witnesses: mx.array,
        tactile_previous_depth: mx.array,
        tactile_previous_validity: mx.array,
        tactile_previous_object: mx.array,
        tactile_previous_motion: mx.array,
        tactile_target_anchor: mx.array,
        tactile_frame_index: mx.array,
        tactile_time: mx.array,
        actuator_effective_position_target: mx.array,
        actuator_profile_values: mx.array,
        observations: mx.array,
        episode_steps: mx.array,
        noise: mx.array,
        reset_commands: mx.array,
        default_q: mx.array,
        default_joint_targets: mx.array,
        reset_q: mx.array,
        reset_v: mx.array,
        reset_scene_position: mx.array,
        reset_scene_orientation: mx.array,
        reset_scene_linear_velocity: mx.array,
        reset_scene_angular_velocity: mx.array,
        reset_manifold_headers: mx.array,
        reset_manifold_points: mx.array,
        reset_manifold_counts: mx.array,
        reset_pair_cache: mx.array,
        reset_rod_witnesses: mx.array,
        reset_actuator_effective_position_target: mx.array,
        reset_actuator_profile_values: mx.array,
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
        joint_targets = (
            default_joint_targets
            + self.action_scale * normalized_action
        )
        actions = mx.concatenate(
            (
                mx.zeros(
                    (self.environment_count, 6),
                    dtype=mx.float32,
                ),
                joint_targets,
            ),
            axis=-1,
        )
        current = WorldState(
            q=q,
            v=v,
            scene_bodies=SceneBodyState(
                scene_position,
                scene_orientation,
                scene_linear_velocity,
                scene_angular_velocity,
            ),
            rods=RodState(
                rod_position,
                rod_velocity,
                rod_twist,
                rod_twist_rate,
            ),
            solver_cache=SolverCache(
                manifold_headers,
                manifold_points,
                manifold_counts,
                pair_cache,
                rod_witnesses,
            ),
            tactile=TactileState(
                tactile_previous_depth,
                tactile_previous_validity,
                tactile_previous_object,
                tactile_previous_motion,
                tactile_target_anchor,
                tactile_frame_index,
                tactile_time,
            ),
            actuators=ActuatorState(
                actuator_effective_position_target,
                actuator_profile_values,
            ),
        )
        physics = step(self.world, current, actions)
        candidate = physics.next_state

        # Quaternion x/y encode root tilt. This is the world-z projection of
        # the pelvis local up direction and remains sign-invariant.
        upright = (
            1.0
            - 2.0
            * (
                mx.square(candidate.q[:, 3])
                + mx.square(candidate.q[:, 4])
            )
        )
        height_error = candidate.q[:, 2] - default_q[2]
        planar_velocity = mx.sum(
            mx.square(candidate.v[:, :2]),
            axis=-1,
        )
        vertical_velocity = mx.square(candidate.v[:, 2])
        angular_velocity = mx.mean(
            mx.square(candidate.v[:, 3:6]),
            axis=-1,
        )
        joint_error = mx.mean(
            mx.square(
                candidate.q[:, 7:]
                - default_joint_targets
            ),
            axis=-1,
        )
        action_cost = mx.mean(
            mx.square(normalized_action),
            axis=-1,
        )
        normal_load = physics.sensors[:, 6]
        support = mx.clip(
            normal_load / (33.34114202 * 9.81),
            0.0,
            1.0,
        )
        tactile_summary = physics.tactile.summary
        tactile_force = mx.sqrt(
            mx.sum(
                mx.square(
                    tactile_summary.
                        net_force_and_contact_area[..., :3]
                ),
                axis=-1,
            )
        )
        tactile_support = mx.clip(
            mx.sum(tactile_force, axis=-1) /
                (33.34114202 * 9.81),
            0.0,
            1.0,
        )
        center_of_pressure = (
            tactile_summary.
                center_of_pressure_local_and_force_weight[..., :2]
        )
        center_of_pressure_cost = mx.mean(
            mx.sum(mx.square(center_of_pressure), axis=-1),
            axis=-1,
        )
        tactile_motion_cost = mx.mean(
            mx.square(
                tactile_summary.
                    tangential_motion_and_friction[..., 0]
            ),
            axis=-1,
        )
        commands = (
            observations[:, -3:]
            if self.command_tracking
            else mx.zeros(
                (self.environment_count, 3),
                dtype=mx.float32,
            )
        )
        command_error = (
            mx.square(candidate.v[:, 0] - commands[:, 0])
            + mx.square(candidate.v[:, 1] - commands[:, 1])
            + 0.25
            * mx.square(candidate.v[:, 5] - commands[:, 2])
        )
        tracking_reward = (
            mx.exp(-2.0 * command_error)
            if self.command_tracking
            else mx.zeros_like(command_error)
        )
        reward = (
            1.5 * mx.clip(upright, 0.0, 1.0)
            + mx.exp(-20.0 * mx.square(height_error))
            + 0.2 * support
            + 0.2 * tactile_support
            + tracking_reward
            - 0.25 * planar_velocity
            - 0.05 * vertical_velocity
            - 0.02 * angular_velocity
            - 0.5 * joint_error
            - 0.002 * action_cost
            - 2.0 * center_of_pressure_cost
            - 0.01 * tactile_motion_cost
        )

        next_episode_steps = episode_steps + mx.array(
            1,
            dtype=mx.uint32,
        )
        horizon = (
            next_episode_steps >= self.maximum_episode_steps
        )
        fallen = (
            (candidate.q[:, 2] < 0.35)
            | (upright < 0.0)
        )
        done = physics.physics_error | horizon | fallen

        next_q = _select_environment(
            done,
            reset_q,
            candidate.q,
        )
        next_v = _select_environment(
            done,
            reset_v,
            candidate.v,
        )
        reset_scene = (
            reset_scene_position,
            reset_scene_orientation,
            reset_scene_linear_velocity,
            reset_scene_angular_velocity,
        )
        next_scene = [
            _select_environment(done, reset, value)
            for reset, value in zip(
                reset_scene,
                candidate.scene_bodies,
                strict=True,
            )
        ]
        reset_cache = (
            reset_manifold_headers,
            reset_manifold_points,
            reset_manifold_counts,
            reset_pair_cache,
            reset_rod_witnesses,
        )
        next_cache = [
            _select_environment(done, reset, value)
            for reset, value in zip(
                reset_cache,
                candidate.solver_cache,
                strict=True,
            )
        ]
        next_episode_steps = mx.where(
            done,
            mx.zeros_like(next_episode_steps),
            next_episode_steps,
        )
        next_commands = _select_environment(
            done,
            reset_commands,
            commands,
        )
        next_tactile = TactileState(
            *(
                _select_environment(done, reset, value)
                for reset, value in zip(
                    (
                        mx.zeros_like(
                            candidate.tactile.previous_depth_m
                        ),
                        mx.zeros_like(
                            candidate.tactile.previous_validity
                        ),
                        mx.full_like(
                            candidate.tactile.
                                previous_object_shape_ids,
                            0xFFFFFFFF,
                        ),
                        mx.zeros_like(
                            candidate.tactile.
                                previous_tangential_motion
                        ),
                        mx.zeros_like(
                            candidate.tactile.target_local_anchor
                        ),
                        mx.zeros_like(
                            candidate.tactile.frame_index
                        ),
                        mx.zeros_like(
                            candidate.tactile.time_seconds
                        ),
                    ),
                    candidate.tactile,
                    strict=True,
                )
            )
        )
        next_actuators = ActuatorState(
            *(
                _select_environment(done, reset, value)
                for reset, value in zip(
                    (
                        reset_actuator_effective_position_target,
                        reset_actuator_profile_values,
                    ),
                    candidate.actuators,
                    strict=True,
                )
            )
        )
        tactile_policy = _select_environment(
            done,
            mx.zeros(
                (self.environment_count, 132),
                dtype=mx.float32,
            ),
            _tactile_policy_observation(physics.tactile),
        )
        next_observations = self._observations(
            WorldState(
                q=next_q,
                v=next_v,
                scene_bodies=SceneBodyState(*next_scene),
                rods=current.rods,
                solver_cache=SolverCache(*next_cache),
                tactile=next_tactile,
                actuators=next_actuators,
            ),
            next_commands,
            tactile_policy,
        )
        return (
            next_q,
            next_v,
            *next_scene,
            *next_cache,
            next_observations,
            next_episode_steps,
            latent,
            log_probability,
            value,
            reward,
            done,
            physics.physics_error,
            *next_tactile,
            *next_actuators,
        )

    def collect(
        self,
        state: MLXRolloutState,
        rollout_steps: int,
    ) -> tuple[MLXRolloutState, MLXRolloutBatch]:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        current = state.world
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
                (self.environment_count, self.action_size)
            )
            reset_commands = (
                mx.random.uniform(
                    low=-1.0,
                    high=1.0,
                    shape=(self.environment_count, 3),
                )
                * self.command_scales
            )
            reset_q, reset_v = self._randomized_reset_qv()
            result = self._compiled_step(
                current.q,
                current.v,
                *current.scene_bodies,
                *current.rods,
                *current.solver_cache,
                *current.tactile,
                *current.actuators,
                observations,
                episode_steps,
                noise,
                reset_commands,
                self.default_q,
                self.default_joint_targets,
                reset_q,
                reset_v,
                *self.default_state.scene_bodies,
                *self.default_state.solver_cache,
                *self.default_state.actuators,
            )
            current = WorldState(
                q=result[0],
                v=result[1],
                scene_bodies=SceneBodyState(*result[2:6]),
                rods=current.rods,
                solver_cache=SolverCache(*result[6:11]),
                tactile=TactileState(*result[19:26]),
                actuators=ActuatorState(*result[26:28]),
            )
            observations = result[11]
            episode_steps = result[12]
            latent = result[13]
            log_probability = result[14]
            value = result[15]
            reward = result[16]
            done = result[17]
            physics_error = result[18]
            stored_observations.append(policy_observations)
            stored_latents.append(latent)
            stored_log_probabilities.append(log_probability)
            stored_values.append(value)
            stored_rewards.append(reward)
            stored_dones.append(done)
            stored_errors.append(physics_error)
            if (index + 1) % self.chunk_size == 0:
                mx.async_eval(
                    current.q,
                    current.v,
                    current.solver_cache.manifold_counts,
                    observations,
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
            world=current,
            observations=observations,
            episode_steps=episode_steps,
        )
        mx.async_eval(
            batch.observations,
            batch.returns,
            batch.physics_errors,
            next_state.world.q,
            next_state.world.solver_cache.manifold_counts,
        )
        return next_state, batch


class MLXG1PPOTrainer(MLXPPOTrainer):
    """PPO trainer for the contact-capable G1 standing task."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        metallib_path: str | None = None,
        rollout_chunk_size: int = 16,
        maximum_episode_steps: int = 1_200,
        physics_substeps: int = 4,
        velocity_iterations: int = 2,
        final_velocity_iterations: int = 1,
        scene: str = "ground",
        command_tracking: bool = False,
        world_pack_path: str,
    ) -> None:
        config.validate()
        if scene not in {"ground", "terrain"}:
            raise ValueError(
                "G1 PPO scene must be 'ground' or 'terrain'"
            )
        self.config = config
        mx.random.seed(config.seed)
        if not world_pack_path:
            raise ValueError(
                "G1 tactile training requires an explicit authored "
                "world pack"
            )
        self.world = compile_world_pack(
            world_pack_path,
            environment_capacity=config.environment_count,
            actuation_mode="implicit_position",
            solver_mode="throughput_tgs",
            ccd_mode="speculative",
            physics_substeps=physics_substeps,
            velocity_iterations=velocity_iterations,
            final_velocity_iterations=final_velocity_iterations,
            metallib_path=metallib_path or "",
        )
        self.task_name = (
            f"g1_{'flat_ground' if scene == 'ground' else 'rough_mesh'}_"
            f"{'command_tracking' if command_tracking else 'standing'}_"
            "tactile"
        )
        self.observation_size = (
            self.world.nq
            + self.world.nv
            + 132
            + (3 if command_tracking else 0)
        )
        self.tactile_observation_selection = (
            "canonical_metric_stem_64_per_sensor",
            "presence",
            "confidence",
        )
        self.action_size = self.world.nv - 6
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
        self.collector = MLXG1RolloutCollector(
            self.world,
            self.model,
            config.environment_count,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            chunk_size=rollout_chunk_size,
            maximum_episode_steps=maximum_episode_steps,
            command_tracking=command_tracking,
        )
        self.rollout_state = self.collector.initial()
        self.iteration = 0
        self.environment_steps = 0
        self._loss_and_grad = nn.value_and_grad(
            self.model,
            self._loss,
        )


__all__ = [
    "MLXG1PPOTrainer",
    "MLXG1RolloutCollector",
]
