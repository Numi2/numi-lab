#!/usr/bin/env python3
"""Exercise exact-candidate physical replay through real Metal physics."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from metalrobo.mlx_replay import (
    MLXPhysicalReplayEvaluator,
    PhysicalReplayTrace,
)
from metalrobo.mlx_r2s2r import SMCConfig, fit_alignment_smc
from metalrobo.mlx_world import (
    ControllerDelayState,
    compile_world,
    sampled_state_from_world_family,
    step_sampled_world_family,
)
from metalrobo.worlds import FrankaPickPlaceWorldFamily
from metalrobo.r2s2r import R2S2RCoordinator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--solver-mode",
        choices=("throughput_tgs", "quality_newton"),
        default="throughput_tgs",
    )
    parser.add_argument("--library")
    parser.add_argument("--metallib")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.envs < 2 or args.steps <= 0:
        raise ValueError("probe requires at least two worlds and one step")
    with FrankaPickPlaceWorldFamily(
        capacity=args.envs,
        library_path=args.library,
        metallib_path=args.metallib,
    ) as family:
        feature_count = len(family.scenario_schema.features)
        true_quantiles = np.full(
            (1, feature_count),
            0.5,
            dtype=np.float32,
        )
        feature_index = {
            feature.id: index
            for index, feature in enumerate(
                family.scenario_schema.features
            )
        }
        true_quantiles[0, feature_index["robot_gain"]] = 0.82
        true_quantiles[0, feature_index["robot_damping"]] = 0.27
        true_quantiles[0, feature_index["robot_latency"]] = 0.63
        family.configure_sampling(
            alignment_fingerprint=1,
            particle_quantiles=true_quantiles,
            particle_weights=np.ones((1,), dtype=np.float32),
            alignment_jitter=0.0,
        )
        family.sample(1, seed=args.seed, mode="replay")
        world = compile_world(
            "franka",
            scene="pick_place",
            environment_capacity=args.envs,
            solver_mode=args.solver_mode,
            actuation_mode="implicit_position",
            physics_substeps=4,
            metallib_path=args.metallib or "",
        )
        state = sampled_state_from_world_family(world, family)
        default_q = mx.array(world.default_q, dtype=mx.float32)
        commands = []
        for step_index in range(args.steps):
            phase = float(step_index + 1) / float(args.steps)
            offset = np.zeros((world.nv,), dtype=np.float32)
            offset[0] = 0.18 * np.sin(phase * np.pi)
            offset[1] = -0.11 * np.sin(phase * np.pi * 0.75)
            commands.append(
                np.asarray(default_q) + offset
            )
        command_array = np.asarray(commands, dtype=np.float32)
        maximum_delay_steps = 2
        delay = ControllerDelayState(
            mx.broadcast_to(
                mx.array(command_array[0])[None, None, :],
                (1, maximum_delay_steps + 1, world.nv),
            )
        )

        robot_q = [np.asarray(state.world.q)[0].copy()]
        robot_v = [np.asarray(state.world.v)[0].copy()]
        objects = [
            np.asarray(state.world.scene_bodies.position)[0, :1, :3].copy()
        ]
        contacts = []
        physics_status_codes = []
        for command in command_array:
            stepped = step_sampled_world_family(
                world,
                state,
                mx.array(command)[None, :],
                delay,
                control_period_seconds=world.control_timestep,
            )
            state = stepped.next_state
            delay = stepped.delay_state
            mx.eval(
                state.world.q,
                state.world.v,
                state.world.scene_bodies.position,
                stepped.physics.contacts.mask,
                stepped.physics.status,
            )
            robot_q.append(np.asarray(state.world.q)[0].copy())
            robot_v.append(np.asarray(state.world.v)[0].copy())
            objects.append(
                np.asarray(
                    state.world.scene_bodies.position
                )[0, :1, :3].copy()
            )
            contacts.append(
                bool(np.any(np.asarray(stepped.physics.contacts.mask)[0]))
            )
            physics_status_codes.append(
                int(np.asarray(stepped.physics.status)[0, 0])
            )

        with tempfile.TemporaryDirectory(
            prefix="metalrobo-replay-probe-"
        ) as directory:
            trace_path = Path(directory) / "physical_episode.npz"
            np.savez_compressed(
                trace_path,
                commands=command_array,
                robot_q=np.asarray(robot_q, dtype=np.float32),
                robot_v=np.asarray(robot_v, dtype=np.float32),
                object_positions=np.asarray(
                    objects,
                    dtype=np.float32,
                ),
                scene_body_indices=np.asarray([0], dtype=np.uint32),
                contact_active=np.asarray(
                    contacts,
                    dtype=np.float32,
                ),
                control_period_seconds=np.asarray(
                    world.control_timestep,
                    dtype=np.float32,
                ),
            )
            trace = PhysicalReplayTrace.from_npz(trace_path)
            evaluator = MLXPhysicalReplayEvaluator(
                world,
                family,
                trace,
                seed=args.seed,
            )
            rng = np.random.default_rng(args.seed)
            candidates = rng.uniform(
                0.05,
                0.95,
                size=(args.envs, feature_count),
            ).astype(np.float32)
            candidates[0] = true_quantiles[0]
            evaluation = evaluator(mx.array(candidates))
            mx.eval(*evaluation)
            residual_values = np.asarray(evaluation.residuals)
            valid_values = np.asarray(evaluation.valid)
            if any(code != 0 for code in physics_status_codes):
                if bool(np.any(valid_values)):
                    raise RuntimeError(
                        "physics-failed replay candidates were marked valid"
                    )
                replay_statuses = {
                    int(value)
                    for value in np.asarray(
                        evaluation.physics_status
                    )
                }
                if not replay_statuses.issubset(
                    set(physics_status_codes)
                ):
                    raise RuntimeError(
                        "replay validity reported inconsistent physics status"
                    )
                try:
                    fit_alignment_smc(
                        mx.array(candidates),
                        evaluator.residual_count,
                        evaluator,
                        config=SMCConfig(
                            rounds=1,
                            minimum_effective_sample_fraction=0.01,
                            jitter_scale=0.0,
                            seed=args.seed,
                        ),
                    )
                except RuntimeError as error:
                    if "no physically valid candidates" not in str(error):
                        raise
                else:
                    raise RuntimeError(
                        "physics-invalid replay published an alignment"
                    )
                print(
                    f'device="{family.device_name}" '
                    f"worlds={args.envs} steps={args.steps} "
                    f"physics_statuses={sorted(set(physics_status_codes))} "
                    "invalid_replay_rejected=yes "
                    "alignment_published=no"
                )
                return 0
            population = fit_alignment_smc(
                mx.array(candidates),
                evaluator.residual_count,
                evaluator,
                config=SMCConfig(
                    rounds=2,
                    minimum_effective_sample_fraction=0.01,
                    jitter_scale=0.0,
                    seed=args.seed,
                ),
            )
            mx.eval(*population)
            coordinator = R2S2RCoordinator(
                Path(directory) / "r2s2r"
            )
            trace_reference = coordinator.ingest_replay_trace(
                trace_path
            )
            artifact = coordinator.align(
                family.scenario_schema,
                {
                    "schema_version": 2,
                    "scenario_schema": family.scenario_schema.id,
                    "episode_twin_hash": "probe-twin",
                    "trace_artifact_hash": (
                        trace_reference.content_hash
                    ),
                    "command_semantics": "joint_position_target",
                    "control_period_seconds": world.control_timestep,
                    "residual_names": list(
                        evaluator.residual_names
                    ),
                    "replay_backend": (
                        "mlx-metal-exact-candidate-v1"
                    ),
                },
                evaluator,
                initial_quantiles=mx.array(candidates),
                residual_count=evaluator.residual_count,
                config=SMCConfig(
                    rounds=1,
                    minimum_effective_sample_fraction=0.01,
                    jitter_scale=0.0,
                    seed=args.seed,
                ),
                world_hash="probe-world",
                engine_hash="probe-engine",
            )
            loaded_artifact, loaded_arrays = (
                coordinator.load_alignment(artifact.content_hash)
            )

        true_loss = float(
            np.sum(residual_values[0] * residual_values[0])
        )
        physics_index = evaluator.residual_names.index(
            "physics_failure_rate"
        )
        true_state_loss = float(
            np.sum(
                np.delete(
                    residual_values[0],
                    physics_index,
                )
                ** 2
            )
        )
        other_losses = np.sum(
            residual_values[1:] * residual_values[1:],
            axis=-1,
        )
        if (
            not np.all(np.isfinite(residual_values))
            or true_state_loss > 1.0e-8
            or float(np.median(other_losses)) <= true_loss + 1.0e-5
            or float(np.asarray(population.weights)[0])
            <= 1.0 / float(args.envs)
            or loaded_artifact.particle_count != args.envs
            or loaded_arrays["quantiles"].shape
            != (args.envs, feature_count)
        ):
            raise RuntimeError(
                "physical replay did not identify the generating twin: "
                f"true_loss={true_loss:.8g}, "
                f"median_other_loss={float(np.median(other_losses)):.8g}, "
                f"true_residuals={residual_values[0].tolist()}"
            )
        print(
            f'device="{family.device_name}" '
            f"worlds={args.envs} steps={args.steps} "
            f"residuals={evaluator.residual_count} "
            f"true_loss={true_loss:.8g} "
            f"true_state_loss={true_state_loss:.8g} "
            f"median_other_loss={float(np.median(other_losses)):.8g} "
            f"top_weight={float(np.asarray(population.weights)[0]):.8g} "
            f"physics_statuses={sorted(set(physics_status_codes))} "
            "exact_candidate=yes gpu_replay=yes artifact_graph=yes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
