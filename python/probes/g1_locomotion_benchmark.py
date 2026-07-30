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
    parser.add_argument(
        "--solver-mode",
        choices=("throughput_pgs", "throughput_tgs"),
        default="throughput_tgs",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stepwise-warmup", action="store_true")
    parser.add_argument("--clear-contact-cache", action="store_true")
    parser.add_argument("--uncompiled-transition", action="store_true")
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
        solver_mode=args.solver_mode,
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
        diagnostic_allow_pgs=args.solver_mode == "throughput_pgs",
    )
    collector.warmup()
    if args.uncompiled_transition:
        collector._compiled_transition = collector._compact_transition_impl
    state = collector.initial()
    mark("warm rollout")
    if args.stepwise_warmup:
        for step in range(args.steps):
            mark(f"warm transition {step + 1}")
            transition_start = time.perf_counter()
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
                q = np.asarray(output.state.world.q, dtype=np.float32)
                v = np.asarray(output.state.world.v, dtype=np.float32)
                manifold_counts = np.asarray(
                    output.state.world.solver_cache.manifold_counts,
                    dtype=np.uint32,
                )
                print(
                    json.dumps(
                        {
                            "warm_transition": step + 1,
                            "elapsed_seconds": (
                                time.perf_counter() - transition_start
                            ),
                            "physics_errors": errors,
                            "done": np.asarray(
                                output.done,
                                dtype=bool,
                            ).tolist(),
                            "reward": np.asarray(
                                output.reward,
                                dtype=np.float32,
                            ).tolist(),
                            "root_height": q[:, 2].tolist(),
                            "root_linear_velocity": v[:, :3].tolist(),
                            "manifold_counts": manifold_counts.tolist(),
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
            if args.clear_contact_cache:
                state = state._replace(
                    world=state.world._replace(
                        solver_cache=collector.default_world.solver_cache
                    )
                )
    mark("measured rollouts")
    mx.reset_peak_memory()
    retained = []
    physics_errors = 0
    terminations = 0
    timeouts = 0
    start = time.perf_counter()
    for repeat in range(args.repeats):
        mark(f"repeat {repeat + 1}")
        state, rollout = collector.collect(state, args.steps)
        mx.eval(
            state.world.q,
            rollout.rewards,
            rollout.physics_errors,
            rollout.dones,
            rollout.timeouts,
        )
        physics_errors += int(
            mx.sum(
                rollout.physics_errors.astype(mx.uint32)
            ).item()
        )
        if args.verbose:
            error_mask = np.asarray(
                rollout.physics_errors,
                dtype=bool,
            )
            if np.any(error_mask):
                failing = np.argwhere(error_mask)
                statuses = np.asarray(
                    rollout.physics_statuses,
                    dtype=np.uint32,
                )
                metrics = np.asarray(
                    rollout.metrics,
                    dtype=np.float32,
                )
                print(
                    json.dumps(
                        {
                            "failing_rollout_indices":
                                failing.tolist(),
                            "failing_statuses": [
                                statuses[tuple(index)].tolist()
                                for index in failing
                            ],
                            "failing_first_contacts": [
                                metrics[tuple(index)][-3:].tolist()
                                for index in failing
                            ],
                        }
                    ),
                    flush=True,
                )
        terminations += int(
            mx.sum(rollout.dones.astype(mx.uint32)).item()
        )
        timeouts += int(
            mx.sum(rollout.timeouts.astype(mx.uint32)).item()
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
                "solver_mode": args.solver_mode,
                "environments": args.envs,
                "steps_per_repeat": args.steps,
                "repeats": args.repeats,
                "environment_control_steps_per_second":
                    environment_steps / elapsed,
                "elapsed_seconds": elapsed,
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                "mlx_active_memory_bytes": retained[-1],
                "retained_growth_bytes": retained[-1] - retained[0],
                "physics_errors": physics_errors,
                "terminations": terminations,
                "timeouts": timeouts,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
