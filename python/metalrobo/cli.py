"""Command-line entry points for training and throughput measurement."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .env import FrankaEnv
from .mlx_locomotion import MLXG1PPOTrainer
from .mlx_ppo import MLXPPOTrainer
from .mlx_surgical import MLXPSMNeedlePPOTrainer
from .mlx_world import compile_world, initial_state, step
from .ppo import (
    PPOConfig,
    PPOTrainer,
    infer_actions,
    load_policy,
    sample_actions,
)


def _runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--envs", type=int, default=1024, help="parallel environments")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--library", help="path to libmetalrobo.dylib")
    parser.add_argument("--metallib", help="path to MetalRobo.metallib")


def _train_parser(parser: argparse.ArgumentParser) -> None:
    _runtime_arguments(parser)
    parser.add_argument(
        "--backend",
        choices=("mlx", "ctypes-debug"),
        default="mlx",
        help=(
            "MLX active-encoder physics (default) or the legacy "
            "NumPy/ctypes validation adapter"
        ),
    )
    parser.add_argument(
        "--task",
        choices=(
            "franka-stabilization",
            "g1-standing",
            "g1-command",
            "g1-terrain",
            "psm-needle",
        ),
        default="franka-stabilization",
        help="device-resident MLX task to train",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="early-stop an update above this approximate KL; <=0 disables",
    )
    parser.add_argument("--initial-log-std", type=float, default=-0.5)
    parser.add_argument("--no-anneal-lr", action="store_true")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--rollout-chunk-size",
        type=int,
        default=16,
        help="bounded lazy MLX steps before mx.async_eval",
    )
    parser.add_argument(
        "--maximum-episode-steps",
        type=int,
        help="task horizon; defaults to 256 for Franka and 1200 for G1",
    )
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=4,
    )


def _benchmark_parser(parser: argparse.ArgumentParser) -> None:
    _runtime_arguments(parser)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--action",
        choices=("zero", "random"),
        default="zero",
        help="native benchmark action source when no checkpoint is supplied",
    )
    parser.add_argument("--checkpoint", type=Path)


def _rollout_parser(parser: argparse.ArgumentParser) -> None:
    _runtime_arguments(parser)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample the policy instead of using its deterministic mean",
    )


def _worker_tuning_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--metallib")
    parser.add_argument(
        "--model",
        choices=("franka", "g1", "psm"),
        default="franka",
    )
    parser.add_argument(
        "--scene",
        choices=("cube", "ground", "terrain", "needle"),
        default="cube",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--physics-substeps", type=int, default=4)
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=(32, 64, 96, 128),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metalrobo",
        description="Metal-native Franka simulation and MLX PPO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _train_parser(subparsers.add_parser("train", help="train an MLX PPO policy"))
    _benchmark_parser(
        subparsers.add_parser("benchmark", help="benchmark batched rollouts")
    )
    _rollout_parser(
        subparsers.add_parser("rollout", help="evaluate a saved policy")
    )
    _worker_tuning_parser(
        subparsers.add_parser(
            "tune-workers",
            help="measure fixed MLX Wave32 worker-grid choices",
        )
    )
    return parser


def run_train(args: argparse.Namespace) -> int:
    if (
        args.backend != "mlx"
        and args.task != "franka-stabilization"
    ):
        raise ValueError(
            "G1 and PSM tasks require --backend mlx; the ctypes "
            "adapter is Franka-only"
        )
    g1_task = args.task in {
        "g1-standing",
        "g1-command",
        "g1-terrain",
    }
    psm_task = args.task == "psm-needle"
    maximum_episode_steps = (
        args.maximum_episode_steps
        if args.maximum_episode_steps is not None
        else (1_200 if g1_task else 400 if psm_task else 256)
    )
    checkpoint_directory = args.checkpoint_dir or (
        "runs/g1-terrain"
        if args.task == "g1-terrain"
        else "runs/g1-standing"
        if args.task == "g1-standing"
        else "runs/g1-command"
        if args.task == "g1-command"
        else "runs/psm-needle"
        if psm_task
        else "runs/franka"
    )
    config = PPOConfig(
        environment_count=args.envs,
        rollout_steps=args.rollout_steps,
        iterations=args.iterations,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_sizes=tuple(args.hidden_sizes),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_gradient_norm=args.max_gradient_norm,
        target_kl=args.target_kl if args.target_kl > 0 else None,
        initial_log_std=args.initial_log_std,
        anneal_learning_rate=not args.no_anneal_lr,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_directory=checkpoint_directory,
    )
    if args.backend == "mlx":
        if args.library:
            raise ValueError(
                "--library applies only to --backend ctypes-debug; "
                "the MLX primitive links the compiled engine directly"
            )
        if g1_task:
            trainer = MLXG1PPOTrainer(
                config,
                metallib_path=args.metallib,
                rollout_chunk_size=args.rollout_chunk_size,
                maximum_episode_steps=maximum_episode_steps,
                physics_substeps=args.physics_substeps,
                scene=(
                    "terrain"
                    if args.task == "g1-terrain"
                    else "ground"
                ),
                command_tracking=(
                    args.task in {"g1-command", "g1-terrain"}
                ),
            )
        elif psm_task:
            trainer = MLXPSMNeedlePPOTrainer(
                config,
                metallib_path=args.metallib,
                rollout_chunk_size=args.rollout_chunk_size,
                maximum_episode_steps=maximum_episode_steps,
                physics_substeps=args.physics_substeps,
            )
        else:
            trainer = MLXPPOTrainer(
                config,
                metallib_path=args.metallib,
                rollout_chunk_size=args.rollout_chunk_size,
                maximum_episode_steps=maximum_episode_steps,
                physics_substeps=args.physics_substeps,
            )
    else:
        trainer = PPOTrainer(
            config,
            library_path=args.library,
            metallib_path=args.metallib,
        )
    checkpoint = trainer.train(resume=args.resume)
    print(json.dumps({"checkpoint": str(checkpoint.resolve())}))
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    if args.steps <= 0 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup-steps cannot be negative")
    rng = np.random.default_rng(args.seed)
    with FrankaEnv(
        args.envs,
        seed=args.seed,
        library_path=args.library,
        metallib_path=args.metallib,
    ) as env:
        observations, _ = env.reset(seed=args.seed)
        model = None
        if args.checkpoint:
            model, _ = load_policy(
                args.checkpoint,
                env.runtime.observation_count,
                env.runtime.action_count,
            )

        zeros = np.zeros(env.action_shape, dtype=np.float32)

        def next_actions() -> np.ndarray:
            if model is not None:
                actions, _ = infer_actions(model, observations)
                return actions
            if args.action == "random":
                return rng.uniform(-1.0, 1.0, env.action_shape).astype(np.float32)
            return zeros

        for _ in range(args.warmup_steps):
            observations, _, _, _, _ = env.step(next_actions())

        before = env.runtime.stats
        start = time.perf_counter()
        for _ in range(args.steps):
            observations, _, _, _, _ = env.step(next_actions())
        elapsed = time.perf_counter() - start
        after = env.runtime.stats
        environment_steps = args.steps * env.num_envs
        gpu_ms = after.total_gpu_milliseconds - before.total_gpu_milliseconds
        report = {
            "device": env.runtime.device_name,
            "native_version": env.runtime.version,
            "environments": env.num_envs,
            "control_steps": args.steps,
            "environment_steps": environment_steps,
            "wall_seconds": elapsed,
            "control_steps_per_second": args.steps / max(elapsed, 1e-9),
            "environment_steps_per_second": environment_steps
            / max(elapsed, 1e-9),
            "gpu_milliseconds": gpu_ms,
            "policy": str(args.checkpoint) if args.checkpoint else None,
        }
        print(json.dumps(report, indent=2))
    return 0


def run_rollout(args: argparse.Namespace) -> int:
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    with FrankaEnv(
        args.envs,
        seed=args.seed,
        library_path=args.library,
        metallib_path=args.metallib,
    ) as env:
        observations, _ = env.reset(seed=args.seed)
        model, state = load_policy(
            args.checkpoint,
            env.runtime.observation_count,
            env.runtime.action_count,
        )
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        reward_sum = 0.0
        start = time.perf_counter()
        for _ in range(args.steps):
            if args.stochastic:
                actions, _ = sample_actions(model, observations)
            else:
                actions, _ = infer_actions(model, observations)
            observations, rewards, _, _, info = env.step(actions)
            reward_sum += float(np.sum(rewards, dtype=np.float64))
            final_info = info.get("final_info")
            if final_info:
                completed_returns.extend(final_info["returns"].tolist())
                completed_lengths.extend(final_info["lengths"].tolist())
        elapsed = time.perf_counter() - start
        environment_steps = args.steps * env.num_envs
        report = {
            "device": env.runtime.device_name,
            "checkpoint_iteration": state["iteration"],
            "environments": env.num_envs,
            "environment_steps": environment_steps,
            "environment_steps_per_second": environment_steps
            / max(elapsed, 1e-9),
            "mean_step_reward": reward_sum / environment_steps,
            "completed_episodes": len(completed_returns),
            "episode_return_mean": (
                float(np.mean(completed_returns)) if completed_returns else None
            ),
            "episode_length_mean": (
                float(np.mean(completed_lengths)) if completed_lengths else None
            ),
        }
        print(json.dumps(report, indent=2))
    return 0


def run_worker_tuning(args: argparse.Namespace) -> int:
    if (
        args.envs <= 0
        or args.steps <= 0
        or args.warmup_steps < 0
        or args.physics_substeps <= 0
    ):
        raise ValueError(
            "envs, steps, and physics-substeps must be positive; "
            "warmup-steps cannot be negative"
        )
    allowed = {32, 64, 96, 128}
    candidates = tuple(dict.fromkeys(args.candidates))
    if not candidates or any(value not in allowed for value in candidates):
        raise ValueError(
            "worker candidates must be selected from 32, 64, 96, 128"
        )
    valid_scene = {
        "franka": {"cube", "ground", "terrain"},
        "g1": {"ground", "terrain"},
        "psm": {"needle"},
    }
    if args.scene not in valid_scene[args.model]:
        raise ValueError(
            f"scene {args.scene!r} is not valid for model "
            f"{args.model!r}"
        )

    import mlx.core as mx

    device = mx.device_info()
    reports: list[dict[str, float | int]] = []
    for worker_groups in candidates:
        implicit = args.model in {"g1", "psm"}
        world = compile_world(
            args.model,
            scene=args.scene,
            environment_capacity=args.envs,
            actuation_mode=(
                "implicit_position" if implicit else "effort"
            ),
            solver_mode="throughput_tgs",
            ccd_mode=(
                "hybrid" if args.model == "psm" else "speculative"
            ),
            physics_substeps=args.physics_substeps,
            wave_worker_groups=worker_groups,
            metallib_path=args.metallib or "",
        )
        state = initial_state(world, args.envs)
        if args.model == "g1":
            actions = mx.concatenate(
                (
                    mx.zeros((args.envs, 6), dtype=mx.float32),
                    mx.broadcast_to(
                        mx.array(
                            world.default_q[7:],
                            dtype=mx.float32,
                        ),
                        (args.envs, world.nv - 6),
                    ),
                ),
                axis=-1,
            )
        elif args.model == "psm":
            actions = mx.broadcast_to(
                mx.array(world.default_q, dtype=mx.float32),
                (args.envs, world.nv),
            )
        else:
            actions = mx.zeros(
                (args.envs, world.nv),
                dtype=mx.float32,
            )

        def advance(current, current_actions):
            output = step(world, current, current_actions)
            return output.next_state, output.physics_error

        compiled_advance = mx.compile(advance)
        physics_error = mx.zeros((args.envs,), dtype=mx.bool_)
        for _ in range(args.warmup_steps):
            state, step_error = compiled_advance(state, actions)
            physics_error = physics_error | step_error
        mx.eval(state, physics_error)

        physics_error = mx.zeros((args.envs,), dtype=mx.bool_)
        started = time.perf_counter()
        for _ in range(args.steps):
            state, step_error = compiled_advance(state, actions)
            physics_error = physics_error | step_error
            mx.async_eval(
                state.q,
                state.solver_cache.manifold_counts,
            )
        mx.eval(state, physics_error)
        elapsed = time.perf_counter() - started
        error_count = int(
            mx.sum(physics_error.astype(mx.uint32)).item()
        )
        if error_count:
            raise RuntimeError(
                f"{worker_groups} worker groups produced "
                f"{error_count} failed environments"
            )
        environment_steps = args.envs * args.steps
        reports.append(
            {
                "worker_groups": worker_groups,
                "wall_seconds": elapsed,
                "environment_steps_per_second":
                    environment_steps / max(elapsed, 1e-9),
            }
        )
        del compiled_advance, state, actions, world
        gc.collect()

    best = max(
        reports,
        key=lambda report: report[
            "environment_steps_per_second"
        ],
    )
    print(
        json.dumps(
            {
                "benchmark": "mlx_wave32_worker_grid",
                "device": device.get("device_name", "unknown"),
                "architecture": device.get(
                    "architecture",
                    "unknown",
                ),
                "model": args.model,
                "scene": args.scene,
                "environments": args.envs,
                "steps": args.steps,
                "physics_substeps": args.physics_substeps,
                "results": reports,
                "recommended_worker_groups":
                    best["worker_groups"],
                "scope": (
                    "explicit wall-throughput benchmark; "
                    "never runs inside lazy rollout execution"
                ),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return run_train(args)
        if args.command == "benchmark":
            return run_benchmark(args)
        if args.command == "rollout":
            return run_rollout(args)
        if args.command == "tune-workers":
            return run_worker_tuning(args)
        raise AssertionError(f"unhandled command: {args.command}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"metalrobo: {error}", file=sys.stderr)
        return 2


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metalrobo-train")
    _train_parser(parser)
    args = parser.parse_args(argv)
    try:
        return run_train(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"metalrobo-train: {error}", file=sys.stderr)
        return 2


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metalrobo-benchmark")
    _benchmark_parser(parser)
    args = parser.parse_args(argv)
    try:
        return run_benchmark(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"metalrobo-benchmark: {error}", file=sys.stderr)
        return 2


def rollout_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metalrobo-rollout")
    _rollout_parser(parser)
    args = parser.parse_args(argv)
    try:
        return run_rollout(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"metalrobo-rollout: {error}", file=sys.stderr)
        return 2
