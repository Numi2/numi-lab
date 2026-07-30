#!/usr/bin/env python3
"""Focused mechanics/observation/action check for G1 locomotion."""

from __future__ import annotations

import argparse
import json

import mlx.core as mx
import numpy as np

from metalrobo.mlx_locomotion import (
    G1_ACTOR_OBSERVATION_SIZE,
    G1_CRITIC_OBSERVATION_SIZE,
    G1ActorCritic,
    G1LocomotionTaskSpec,
    MLXG1RolloutCollector,
)
from metalrobo.mlx_world import compile_world


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="metalrobo_g1_locomotion_check"
    )
    parser.add_argument("--metallib", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def mark(stage: str) -> None:
        if args.verbose:
            print(stage, flush=True)

    mark("compile world")
    task = G1LocomotionTaskSpec(
        cohort_size=2,
        rollout_steps=8,
    )
    world = compile_world(
        "g1",
        scene="terrain",
        environment_capacity=2,
        control_timestep=0.02,
        physics_substeps=4,
        actuation_mode="implicit_position",
        solver_mode="throughput_tgs",
        velocity_iterations=4,
        final_velocity_iterations=2,
        ccd_mode="disabled",
        publish_body_states=True,
        metallib_path=args.metallib,
    )
    if (
        int(world.collider_count) != 61
        or int(world.tactile_sample_count) != 0
        or len(world.shape_body_indices) != 61
        or len(world.body_masses) != int(world.model_body_count)
    ):
        raise RuntimeError(
            "G1 locomotion did not compile 60 robot colliders plus terrain"
        )
    mark("compile locomotion transition")
    model = G1ActorCritic(hidden_sizes=(32, 16))
    collector = MLXG1RolloutCollector(
        world,
        model,
        2,
        task=task,
        gamma=0.99,
        gae_lambda=0.95,
        terrain=True,
    )
    state = collector.initial()
    if (
        state.actor_history.shape != (2, 5, 96)
        or state.critic_observation.shape
        != (2, G1_CRITIC_OBSERVATION_SIZE)
        or state.actor_history.reshape((2, -1)).shape[-1]
        != G1_ACTOR_OBSERVATION_SIZE
    ):
        raise RuntimeError("G1 observation contract has changed")

    policy_noise = mx.zeros((2, 29), dtype=mx.float32)
    mx.random.seed(17)
    mark("first deterministic transition")
    first = collector._transition(
        state,
        collector._random_inputs(),
        policy_noise,
    )
    mx.eval(
        first.state.world.q,
        first.reward,
        first.physics_error,
    )
    mx.random.seed(17)
    mark("second deterministic transition")
    second = collector._transition(
        state,
        collector._random_inputs(),
        policy_noise,
    )
    mx.eval(
        second.state.world.q,
        second.reward,
        second.physics_error,
    )
    if (
        not np.array_equal(
            np.asarray(first.state.world.q),
            np.asarray(second.state.world.q),
        )
        or not np.array_equal(
            np.asarray(first.reward),
            np.asarray(second.reward),
        )
    ):
        raise RuntimeError("G1 fixed-input transition is not deterministic")

    mark("eight-step rollout")
    next_state, rollout = collector.collect(state, 8)
    mx.eval(
        rollout.rewards,
        rollout.physics_errors,
        next_state.world.q,
    )
    errors = int(
        mx.sum(rollout.physics_errors.astype(mx.uint32)).item()
    )
    if errors:
        status = np.asarray(rollout.physics_statuses)
        failed = status[
            np.asarray(rollout.physics_errors, dtype=bool)
        ]
        raise RuntimeError(
            "G1 production transition reported "
            f"{errors} physics errors; first status="
            f"{failed[0].tolist() if len(failed) else 'missing'}"
        )
    print(
        json.dumps(
            {
                "check": "metalrobo_g1_locomotion_check",
                "physics_hz": 200,
                "policy_hz": 50,
                "official_collision_elements": 36,
                "robot_colliders": 60,
                "actor_observation": 480,
                "critic_observation": G1_CRITIC_OBSERVATION_SIZE,
                "tactile_samples": 0,
                "steps": 8,
                "deterministic": True,
                "physics_errors": errors,
                "mlx_active_memory_bytes": int(mx.get_active_memory()),
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
