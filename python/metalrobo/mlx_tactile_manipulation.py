"""Franka tactile grasp stabilization on an explicit authored world pack."""

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
    return canonical_metric_tactile_policy_observation(
        tactile,
        _DUAL_32_TACTILE_ATLASES,
    )


class MLXFrankaTactileRolloutCollector:
    """Compiled Franka grasp rollout with a real dynamic grasp object."""

    def __init__(
        self,
        world,
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
        if chunk_size <= 0 or maximum_episode_steps <= 0:
            raise ValueError(
                "chunk_size and maximum_episode_steps must be positive"
            )
        if (
            world.solver_mode == "free_motion_aba"
            or world.actuation_mode != "implicit_position"
            or world.floating_root
            or world.nq != 9
            or world.nv != 9
            or world.scene_body_count < 1
        ):
            raise ValueError(
                "Franka tactile grasp requires the authored fixed-base "
                "nine-coordinate contact world"
            )
        if (
            int(world.tactile_sensor_count) != 2
            or int(world.tactile_sample_count) != 2 * 32 * 32
        ):
            raise ValueError(
                "Franka tactile grasp requires the authored dual-finger "
                "32x32 observation contract"
            )
        self.world = world
        self.model = model
        self.environment_count = environment_count
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.chunk_size = chunk_size
        self.maximum_episode_steps = maximum_episode_steps
        self.action_size = int(world.nv)
        base_state = initial_state(world, environment_count)
        initial_disturbance = mx.broadcast_to(
            mx.array(
                (0.0, 0.08, 0.0, 0.0),
                dtype=mx.float32,
            ).reshape((1, 1, 4)),
            (
                environment_count,
                int(world.scene_body_count),
                4,
            ),
        )
        body_mask = (
            mx.arange(int(world.scene_body_count))[None, :, None]
            == 0
        )
        self.default_state = base_state._replace(
            scene_bodies=base_state.scene_bodies._replace(
                linear_velocity=(
                    base_state.scene_bodies.linear_velocity
                    + mx.where(
                        body_mask,
                        initial_disturbance,
                        mx.zeros_like(initial_disturbance),
                    )
                )
            )
        )
        self.default_q = mx.array(
            world.default_q,
            dtype=mx.float32,
        )
        self.default_object_position = (
            base_state.scene_bodies.position[:, 0, :3]
        )
        self.action_scales = mx.array(
            (
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                0.002,
                0.002,
            ),
            dtype=mx.float32,
        )
        self._compiled_step = mx.compile(
            self._policy_physics_reward,
            inputs=self.model.state,
        )

    @staticmethod
    def _observations(
        state: WorldState,
        contact_counts: mx.array,
        tactile_policy: mx.array,
    ) -> mx.array:
        scene = state.scene_bodies
        return mx.concatenate(
            (
                state.q,
                state.v,
                scene.position[:, 0, :3],
                scene.orientation[:, 0, :4],
                scene.linear_velocity[:, 0, :3],
                scene.angular_velocity[:, 0, :3],
                contact_counts[:, None].astype(mx.float32) * 0.125,
                tactile_policy,
            ),
            axis=-1,
        )

    def initial(self) -> MLXRolloutState:
        return MLXRolloutState(
            world=self.default_state,
            observations=self._observations(
                self.default_state,
                mx.zeros(
                    (self.environment_count,),
                    dtype=mx.uint32,
                ),
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

    def _reset_scene_linear_velocity(self) -> mx.array:
        disturbance = mx.concatenate(
            (
                mx.zeros(
                    (self.environment_count, 1),
                    dtype=mx.float32,
                ),
                mx.random.uniform(
                    low=-0.12,
                    high=0.12,
                    shape=(self.environment_count, 1),
                ),
                mx.zeros(
                    (self.environment_count, 2),
                    dtype=mx.float32,
                ),
            ),
            axis=-1,
        )
        body_mask = (
            mx.arange(int(self.world.scene_body_count))[None, :, None]
            == 0
        )
        return (
            self.default_state.scene_bodies.linear_velocity
            + mx.where(
                body_mask,
                disturbance[:, None, :],
                mx.zeros(
                    (
                        self.environment_count,
                        int(self.world.scene_body_count),
                        4,
                    ),
                    dtype=mx.float32,
                ),
            )
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
        default_q: mx.array,
        action_scales: mx.array,
        default_object_position: mx.array,
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
        actions = default_q + action_scales * normalized_action
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

        object_position = candidate.scene_bodies.position[:, 0, :3]
        object_velocity = candidate.scene_bodies.linear_velocity[:, 0, :3]
        displacement = object_position - default_object_position
        displacement_cost = mx.sum(mx.square(displacement), axis=-1)
        velocity_cost = mx.sum(mx.square(object_velocity), axis=-1)
        joint_cost = mx.mean(
            mx.square(candidate.q - default_q),
            axis=-1,
        )
        action_cost = mx.mean(
            mx.square(normalized_action),
            axis=-1,
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
        bilateral_support = mx.clip(
            mx.min(tactile_force, axis=-1) / 1.0,
            0.0,
            1.0,
        )
        force_balance = mx.square(
            tactile_force[:, 0] - tactile_force[:, 1]
        )
        tactile_depth = mx.mean(
            tactile_summary.
                centroid_local_and_mean_depth[..., 3],
            axis=-1,
        )
        tangent_speed = mx.mean(
            tactile_summary.
                tangential_motion_and_friction[..., 0],
            axis=-1,
        )
        reward = (
            2.0 * mx.exp(-5_000.0 * displacement_cost)
            + 0.75 * bilateral_support
            + 0.25 * mx.clip(
                tactile_depth / 1.0e-4,
                0.0,
                1.0,
            )
            - 0.05 * velocity_cost
            - 0.01 * force_balance
            - 0.25 * joint_cost
            - 0.002 * action_cost
            - 0.02 * mx.square(tangent_speed)
        )

        next_episode_steps = episode_steps + mx.array(
            1,
            dtype=mx.uint32,
        )
        done = (
            physics.physics_error
            | (next_episode_steps >= self.maximum_episode_steps)
            | (mx.sqrt(displacement_cost) > 0.08)
        )
        next_q = _select_environment(
            done,
            mx.broadcast_to(
                default_q,
                candidate.q.shape,
            ),
            candidate.q,
        )
        next_v = _select_environment(
            done,
            mx.zeros_like(candidate.v),
            candidate.v,
        )
        reset_scene = (
            reset_scene_position,
            reset_scene_orientation,
            reset_scene_linear_velocity,
            reset_scene_angular_velocity,
        )
        next_scene = [
            _select_environment(done, reset, current_value)
            for reset, current_value in zip(
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
            _select_environment(done, reset, current_value)
            for reset, current_value in zip(
                reset_cache,
                candidate.solver_cache,
                strict=True,
            )
        ]
        next_tactile = TactileState(
            *(
                _select_environment(done, reset, current_value)
                for reset, current_value in zip(
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
                _select_environment(done, reset, current_value)
                for reset, current_value in zip(
                    (
                        reset_actuator_effective_position_target,
                        reset_actuator_profile_values,
                    ),
                    candidate.actuators,
                    strict=True,
                )
            )
        )
        next_episode_steps = mx.where(
            done,
            mx.zeros_like(next_episode_steps),
            next_episode_steps,
        )
        next_contact_counts = mx.where(
            done,
            mx.zeros_like(physics.contacts.counts),
            physics.contacts.counts,
        )
        tactile_policy = _select_environment(
            done,
            mx.zeros(
                (self.environment_count, 132),
                dtype=mx.float32,
            ),
            _tactile_policy_observation(physics.tactile),
        )
        next_state = WorldState(
            q=next_q,
            v=next_v,
            scene_bodies=SceneBodyState(*next_scene),
            rods=current.rods,
            solver_cache=SolverCache(*next_cache),
            tactile=next_tactile,
            actuators=next_actuators,
        )
        return (
            next_q,
            next_v,
            *next_scene,
            *next_cache,
            self._observations(
                next_state,
                next_contact_counts,
                tactile_policy,
            ),
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
                mx.random.normal(
                    (self.environment_count, self.action_size)
                ),
                self.default_q,
                self.action_scales,
                self.default_object_position,
                self.default_state.scene_bodies.position,
                self.default_state.scene_bodies.orientation,
                self._reset_scene_linear_velocity(),
                self.default_state.scene_bodies.angular_velocity,
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
            stored_observations.append(policy_observations)
            stored_latents.append(result[13])
            stored_log_probabilities.append(result[14])
            stored_values.append(result[15])
            stored_rewards.append(result[16])
            stored_dones.append(result[17])
            stored_errors.append(result[18])
            if (index + 1) % self.chunk_size == 0:
                mx.async_eval(
                    current.q,
                    current.scene_bodies.position,
                    current.tactile.frame_index,
                    observations,
                    result[16],
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
            next_state.world.scene_bodies.position,
        )
        return next_state, batch


class MLXFrankaTactilePPOTrainer(MLXPPOTrainer):
    """PPO trainer for authored Franka tactile grasp stabilization."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        world_pack_path: str,
        metallib_path: str | None = None,
        rollout_chunk_size: int = 16,
        maximum_episode_steps: int = 256,
        physics_substeps: int = 8,
        velocity_iterations: int = 4,
        final_velocity_iterations: int = 2,
    ) -> None:
        config.validate()
        self.config = config
        mx.random.seed(config.seed)
        if not world_pack_path:
            raise ValueError(
                "Franka tactile training requires an explicit authored "
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
        self.task_name = "franka_tactile_grasp_stabilization"
        self.observation_size = 164
        self.action_size = int(self.world.nv)
        self.tactile_observation_selection = (
            "canonical_metric_stem_64_per_sensor",
            "presence",
            "confidence",
        )
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
        self.collector = MLXFrankaTactileRolloutCollector(
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


__all__ = [
    "MLXFrankaTactilePPOTrainer",
    "MLXFrankaTactileRolloutCollector",
]
