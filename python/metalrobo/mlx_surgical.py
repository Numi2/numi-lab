"""Pure-MLX surgical control and learning for the PSM contact world."""

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
    SceneBodyState,
    SolverCache,
    WorldState,
    compile_world,
    initial_state,
    step,
)
from .ppo import ActorCritic, PPOConfig


def psm_physical_position_targets(
    logical_targets: mx.array,
) -> mx.array:
    """Expand six arm targets plus jaw aperture into eight PSM coordinates.

    ``logical_targets[..., :6]`` map directly to the arm. The final
    non-negative aperture is split symmetrically across the independent jaws.
    The physics model intentionally keeps both jaw coordinates explicit until
    tendon/transmission constraints are executable.
    """

    if logical_targets.ndim < 1 or logical_targets.shape[-1] != 7:
        raise ValueError(
            "PSM logical targets must have a final dimension of 7"
        )
    aperture = mx.maximum(
        logical_targets[..., 6:7],
        mx.array(0.0, dtype=logical_targets.dtype),
    )
    return mx.concatenate(
        (
            logical_targets[..., :6],
            -0.5 * aperture,
            0.5 * aperture,
        ),
        axis=-1,
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


class MLXPSMNeedleRolloutCollector:
    """Physics-owned PSM needle hold/lift rollout.

    The needle remains a dynamic rigid body. Success is rewarded from its
    measured pose and published contacts; no weld, teleport, or native grasp
    state is introduced.
    """

    def __init__(
        self,
        world,
        model: ActorCritic,
        environment_count: int,
        *,
        gamma: float,
        gae_lambda: float,
        chunk_size: int = 16,
        maximum_episode_steps: int = 400,
    ) -> None:
        if environment_count <= 0:
            raise ValueError("environment_count must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if maximum_episode_steps <= 0:
            raise ValueError(
                "maximum_episode_steps must be positive"
            )
        if (
            world.solver_mode == "free_motion_aba"
            or world.actuation_mode != "implicit_position"
            or world.floating_root
            or world.nq != 8
            or world.nv != 8
            or world.scene_body_count != 1
        ):
            raise ValueError(
                "PSM needle rollouts require the fixed-base "
                "eight-coordinate needle contact world"
            )
        self.world = world
        self.model = model
        self.environment_count = environment_count
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.chunk_size = chunk_size
        self.maximum_episode_steps = maximum_episode_steps
        self.action_size = 7
        self.default_state = initial_state(
            world,
            environment_count,
        )
        self.default_q = mx.array(
            world.default_q,
            dtype=mx.float32,
        )
        self.default_logical_targets = mx.concatenate(
            (
                self.default_q[:6],
                self.default_q[7:8] - self.default_q[6:7],
            )
        )
        self.target_scales = mx.array(
            (0.05, 0.05, 0.015, 0.10, 0.10, 0.10, 0.18),
            dtype=mx.float32,
        )
        self.default_needle_position = (
            self.default_state.scene_bodies.position[:, 0, :3]
        )
        self._compiled_step = mx.compile(
            self._policy_physics_reward,
            inputs=self.model.state,
        )

    @staticmethod
    def _observations(
        state: WorldState,
        contact_counts: mx.array,
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
                contact_counts[:, None].astype(mx.float32) * 0.25,
            ),
            axis=-1,
        )

    def initial(self) -> MLXRolloutState:
        contact_counts = mx.zeros(
            (self.environment_count,),
            dtype=mx.uint32,
        )
        return MLXRolloutState(
            world=self.default_state,
            observations=self._observations(
                self.default_state,
                contact_counts,
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
        manifold_headers: mx.array,
        manifold_points: mx.array,
        manifold_counts: mx.array,
        pair_cache: mx.array,
        observations: mx.array,
        episode_steps: mx.array,
        noise: mx.array,
        default_logical_targets: mx.array,
        target_scales: mx.array,
        default_needle_position: mx.array,
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
        logical_targets = (
            default_logical_targets
            + target_scales * normalized_action
        )
        actions = psm_physical_position_targets(logical_targets)
        current = WorldState(
            q=q,
            v=v,
            scene_bodies=SceneBodyState(
                scene_position,
                scene_orientation,
                scene_linear_velocity,
                scene_angular_velocity,
            ),
            rods=self.default_state.rods,
            solver_cache=SolverCache(
                manifold_headers,
                manifold_points,
                manifold_counts,
                pair_cache,
            ),
        )
        physics = step(self.world, current, actions)
        candidate = physics.next_state
        needle_position = candidate.scene_bodies.position[:, 0, :3]
        needle_velocity = candidate.scene_bodies.linear_velocity[:, 0, :3]
        displacement = needle_position - default_needle_position
        lift_error = displacement[:, 2] - 0.008
        lateral_error = mx.sum(
            mx.square(displacement[:, :2]),
            axis=-1,
        )
        contact_counts = physics.contacts.counts
        contact_support = mx.clip(
            contact_counts.astype(mx.float32) / 2.0,
            0.0,
            1.0,
        )
        aperture = candidate.q[:, 7] - candidate.q[:, 6]
        action_cost = mx.mean(
            mx.square(normalized_action),
            axis=-1,
        )
        reward = (
            2.0 * mx.exp(-20_000.0 * mx.square(lift_error))
            + contact_support
            + 0.25 * mx.clip(displacement[:, 2] / 0.008, 0.0, 1.0)
            - 2_000.0 * lateral_error
            - 0.01 * mx.sum(mx.square(needle_velocity), axis=-1)
            - 0.02 * mx.square(aperture)
            - 0.002 * action_cost
        )

        next_episode_steps = episode_steps + mx.array(
            1,
            dtype=mx.uint32,
        )
        horizon = (
            next_episode_steps >= self.maximum_episode_steps
        )
        lost = (
            mx.sum(mx.square(displacement), axis=-1)
            > 0.0064
        )
        done = physics.physics_error | horizon | lost
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
        next_scene = [
            _select_environment(done, reset, current_value)
            for reset, current_value in zip(
                (
                    reset_scene_position,
                    reset_scene_orientation,
                    reset_scene_linear_velocity,
                    reset_scene_angular_velocity,
                ),
                candidate.scene_bodies,
                strict=True,
            )
        ]
        next_cache = [
            _select_environment(done, reset, current_value)
            for reset, current_value in zip(
                (
                    reset_manifold_headers,
                    reset_manifold_points,
                    reset_manifold_counts,
                    reset_pair_cache,
                ),
                candidate.solver_cache,
                strict=True,
            )
        ]
        next_episode_steps = mx.where(
            done,
            mx.zeros_like(next_episode_steps),
            next_episode_steps,
        )
        next_contact_counts = mx.where(
            done,
            mx.zeros_like(contact_counts),
            contact_counts,
        )
        next_state = WorldState(
            q=next_q,
            v=next_v,
            scene_bodies=SceneBodyState(*next_scene),
            rods=self.default_state.rods,
            solver_cache=SolverCache(*next_cache),
        )
        return (
            next_q,
            next_v,
            *next_scene,
            *next_cache,
            self._observations(next_state, next_contact_counts),
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
            result = self._compiled_step(
                current.q,
                current.v,
                *current.scene_bodies,
                *current.solver_cache,
                observations,
                episode_steps,
                noise,
                self.default_logical_targets,
                self.target_scales,
                self.default_needle_position,
                self.default_state.q,
                self.default_state.v,
                *self.default_state.scene_bodies,
                *self.default_state.solver_cache,
            )
            current = WorldState(
                q=result[0],
                v=result[1],
                scene_bodies=SceneBodyState(*result[2:6]),
                rods=self.default_state.rods,
                solver_cache=SolverCache(*result[6:10]),
            )
            observations = result[10]
            episode_steps = result[11]
            stored_observations.append(policy_observations)
            stored_latents.append(result[12])
            stored_log_probabilities.append(result[13])
            stored_values.append(result[14])
            stored_rewards.append(result[15])
            stored_dones.append(result[16])
            stored_errors.append(result[17])
            if (index + 1) % self.chunk_size == 0:
                mx.async_eval(
                    current.q,
                    current.scene_bodies.position,
                    current.solver_cache.manifold_counts,
                    observations,
                    result[15],
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


class MLXPSMNeedlePPOTrainer(MLXPPOTrainer):
    """PPO trainer for physics-owned PSM needle hold and lift."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        metallib_path: str | None = None,
        rollout_chunk_size: int = 16,
        maximum_episode_steps: int = 400,
        physics_substeps: int = 4,
        velocity_iterations: int = 2,
        final_velocity_iterations: int = 1,
    ) -> None:
        config.validate()
        self.config = config
        mx.random.seed(config.seed)
        self.world = compile_world(
            "psm",
            scene="needle",
            environment_capacity=config.environment_count,
            actuation_mode="implicit_position",
            solver_mode="throughput_tgs",
            ccd_mode="hybrid",
            physics_substeps=physics_substeps,
            velocity_iterations=velocity_iterations,
            final_velocity_iterations=final_velocity_iterations,
            metallib_path=metallib_path or "",
        )
        self.task_name = "psm_needle_hold_lift_v1"
        self.observation_size = 30
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
        self.collector = MLXPSMNeedleRolloutCollector(
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
    "MLXPSMNeedlePPOTrainer",
    "MLXPSMNeedleRolloutCollector",
    "psm_physical_position_targets",
]
