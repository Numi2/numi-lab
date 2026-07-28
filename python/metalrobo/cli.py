"""Command-line entry points for training and throughput measurement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .env import FrankaEnv
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
    parser.add_argument("--checkpoint-dir", default="runs/franka")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", type=Path)


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
    return parser


def run_train(args: argparse.Namespace) -> int:
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
        checkpoint_directory=args.checkpoint_dir,
    )
    trainer = PPOTrainer(
        config, library_path=args.library, metallib_path=args.metallib
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return run_train(args)
        if args.command == "benchmark":
            return run_benchmark(args)
        if args.command == "rollout":
            return run_rollout(args)
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
