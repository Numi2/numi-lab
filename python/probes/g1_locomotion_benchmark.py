#!/usr/bin/env python3
"""Throughput and retained-memory benchmark for one G1 cohort."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from metalrobo.mlx_locomotion import (
    G1ActorCritic,
    G1LocomotionTaskSpec,
    MLXG1RolloutCollector,
)
from metalrobo.mlx_world import compile_world


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="metalrobo_g1_locomotion_benchmark"
    )
    parser.add_argument("--metallib", required=True)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--scene",
        choices=("ground", "terrain"),
        default="terrain",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stepwise-warmup", action="store_true")
    args = parser.parse_args()
    if (
        args.envs <= 0
        or args.steps <= 0
        or args.steps % 8
        or args.repeats <= 0
    ):
        raise ValueError(
            "envs/repeats must be positive and steps a multiple of 8"
        )
    mx.random.seed(args.seed)

    def mark(stage: str) -> None:
        if args.verbose:
            print(stage, flush=True)

    task = G1LocomotionTaskSpec(
        cohort_size=args.envs,
        rollout_steps=args.steps,
    )
    mark("compile world")
    world = compile_world(
        "g1",
        scene=args.scene,
        environment_capacity=args.envs,
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
    mark("initialize policy and collector")
    model = G1ActorCritic()
    collector = MLXG1RolloutCollector(
        world,
        model,
        args.envs,
        task=task,
        gamma=0.99,
        gae_lambda=0.95,
        terrain=args.scene == "terrain",
    )
    state = collector.initial()
    mark("warm rollout")
    if args.stepwise_warmup:
        for step in range(8):
            mark(f"warm transition {step + 1}")
            output = collector._transition(
                state,
                collector._random_inputs(),
                mx.random.normal((args.envs, 29)),
            )
            mx.eval(
                output.state,
                output.reward,
                output.physics_error,
                output.physics_status,
            )
            if args.verbose:
                errors = int(
                    mx.sum(
                        output.physics_error.astype(mx.uint32)
                    ).item()
                )
                statuses = np.asarray(output.physics_status)
                print(
                    json.dumps(
                        {
                            "warm_transition": step + 1,
                            "physics_errors": errors,
                            "first_status": statuses[0].tolist(),
                            "contact_pairs": np.asarray(
                                output.metrics[:, -3:],
                                dtype=np.float32,
                            ).tolist(),
                        }
                    ),
                    flush=True,
                )
            state = output.state
    else:
        state, _ = collector.collect(state, 8)
        mx.eval(state)
    mark("measured rollouts")
    mx.reset_peak_memory()
    retained = []
    start = time.perf_counter()
    for repeat in range(args.repeats):
        mark(f"repeat {repeat + 1}")
        state, rollout = collector.collect(state, args.steps)
        mx.eval(
            state.world.q,
            rollout.rewards,
            rollout.physics_errors,
        )
        retained.append(int(mx.get_active_memory()))
    elapsed = time.perf_counter() - start
    environment_steps = (
        args.envs * args.steps * args.repeats
    )
    print(
        json.dumps(
            {
                "benchmark": "g1_locomotion_cohort",
                "environments": args.envs,
                "steps_per_repeat": args.steps,
                "repeats": args.repeats,
                "environment_control_steps_per_second":
                    environment_steps / elapsed,
                "elapsed_seconds": elapsed,
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                "mlx_active_memory_bytes": retained[-1],
                "retained_growth_bytes": retained[-1] - retained[0],
                "physics_errors": int(
                    mx.sum(
                        rollout.physics_errors.astype(mx.uint32)
                    ).item()
                ),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
