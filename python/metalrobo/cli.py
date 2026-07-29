"""Command-line entry points for training and throughput measurement."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .env import FrankaEnv
from .mlx_locomotion import MLXG1PPOTrainer
from .mlx_ppo import MLXPPOTrainer
from .mlx_family_ppo import MLXWorldFamilyPPOTrainer
from .mlx_replay import (
    MLXPhysicalReplayEvaluator,
    PhysicalReplayTrace,
    ReplayResidualScales,
)
from .mlx_surgical import MLXPSMNeedlePPOTrainer
from .mlx_world import compile_world, initial_state, step
from .ppo import (
    PPOConfig,
    PPOTrainer,
    infer_actions,
    load_policy,
    sample_actions,
)
from .r2s2r import (
    PolicyDescriptor,
    R2S2RCoordinator,
    make_affine_replay_evaluator,
)
from .mlx_r2s2r import SMCConfig, deterministic_candidate_scenarios
from .worlds import FrankaPickPlaceWorldFamily


def _runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--envs", type=int, default=1024, help="parallel environments")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--library", help="path to libmetalrobo.dylib")
    parser.add_argument("--metallib", help="path to MetalRobo.metallib")


def _checkpoint_content_hash(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    paths = (
        [checkpoint]
        if checkpoint.is_file()
        else sorted(
            path
            for path in checkpoint.rglob("*")
            if path.is_file()
        )
    )
    for path in paths:
        relative = (
            path.name
            if checkpoint.is_file()
            else path.relative_to(checkpoint).as_posix()
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as checkpoint_file:
            for chunk in iter(
                lambda: checkpoint_file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
    return digest.hexdigest()


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
            "franka-family-pick-place",
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
    parser.add_argument(
        "--r2s2r-root",
        help="record the resulting policy and exact train invocation",
    )
    parser.add_argument(
        "--policy-id",
        help="stable policy identity for R2S2R provenance",
    )
    parser.add_argument(
        "--embodiment-id",
        default="franka",
        help="embodiment identity recorded with the trained policy",
    )
    parser.add_argument(
        "--alignment-hash",
        help="WorldAlignmentPopulation artifact for family training",
    )
    parser.add_argument(
        "--feedback-hash",
        help="optional WorldFeedbackProgram artifact for curriculum training",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("coverage", "curriculum"),
        default="curriculum",
        help="world-family sampling mode for aligned training",
    )
    parser.add_argument(
        "--family-metallib",
        help="native world-family MetalRobo.metallib override",
    )


def _r2s2r_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".metalrobo-r2s2r"),
        help="R2S2R SQLite/WAL and content-addressed artifact root",
    )
    parser.add_argument("--library", help="path to libmetalrobo.dylib")
    parser.add_argument("--metallib", help="path to MetalRobo.metallib")


def _align_parser(parser: argparse.ArgumentParser) -> None:
    _r2s2r_runtime_arguments(parser)
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        required=True,
        help=(
            "physical replay manifest containing a time-aligned trace_npz "
            "or legacy replay_residual_modes"
        ),
    )
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--physics-substeps", type=int, default=4)


def _record_sim_parser(parser: argparse.ArgumentParser) -> None:
    _r2s2r_runtime_arguments(parser)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="chunk-drained simulation outcome manifest",
    )


def _ingest_real_parser(parser: argparse.ArgumentParser) -> None:
    _r2s2r_runtime_arguments(parser)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="versioned hardware outcome manifest",
    )


def _fit_feedback_parser(parser: argparse.ArgumentParser) -> None:
    _r2s2r_runtime_arguments(parser)
    parser.add_argument("--policy-fingerprint", required=True)
    parser.add_argument("--task-fingerprint", required=True)
    parser.add_argument("--embodiment-fingerprint", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--candidates", type=int, default=65536)
    parser.add_argument("--regions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)


def _evaluate_parser(parser: argparse.ArgumentParser) -> None:
    _r2s2r_runtime_arguments(parser)
    parser.add_argument("--task-fingerprint", required=True)
    parser.add_argument(
        "--policy-fingerprints",
        nargs="+",
        required=True,
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
    _align_parser(
        subparsers.add_parser(
            "align",
            help="fit and publish a replay-aligned world population",
        )
    )
    _record_sim_parser(
        subparsers.add_parser(
            "record-sim",
            help="persist a chunk-drained simulation outcome batch",
        )
    )
    _ingest_real_parser(
        subparsers.add_parser(
            "ingest-real",
            help="ingest a versioned hardware outcome manifest",
        )
    )
    _fit_feedback_parser(
        subparsers.add_parser(
            "fit-feedback",
            help="learn and compile policy-specific failure regions",
        )
    )
    _evaluate_parser(
        subparsers.add_parser(
            "evaluate",
            help="compare policies on identical coverage scenario keys",
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
    family_task = args.task == "franka-family-pick-place"
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
        else "runs/franka-family"
        if family_task
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
        if args.library and not family_task:
            raise ValueError(
                "--library applies only to --backend ctypes-debug; "
                "the MLX primitive links the compiled engine directly"
            )
        if family_task:
            if not args.r2s2r_root or not args.alignment_hash:
                raise ValueError(
                    "franka-family-pick-place requires --r2s2r-root "
                    "and --alignment-hash"
                )
            family = FrankaPickPlaceWorldFamily(
                capacity=args.envs,
                library_path=args.library,
                metallib_path=args.family_metallib,
            )
            coordinator = R2S2RCoordinator(args.r2s2r_root)
            coordinator.configure_world_family(
                family,
                alignment_hash=args.alignment_hash,
                feedback_hash=args.feedback_hash,
            )
            trainer = MLXWorldFamilyPPOTrainer(
                config,
                family,
                metallib_path=args.metallib,
                rollout_chunk_size=args.rollout_chunk_size,
                maximum_episode_steps=maximum_episode_steps,
                physics_substeps=args.physics_substeps,
                sampling_mode=args.sampling_mode,
            )
        elif g1_task:
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
    try:
        checkpoint = trainer.train(resume=args.resume)
    finally:
        close = getattr(trainer, "close", None)
        if close is not None:
            close()
    report: dict[str, object] = {
        "checkpoint": str(checkpoint.resolve()),
    }
    if args.r2s2r_root:
        descriptor = PolicyDescriptor(
            id=args.policy_id
            or f"{args.task}.iteration-{config.iterations}",
            content_hash=_checkpoint_content_hash(checkpoint),
            observation_schema=f"{args.task}.observation.v1",
            action_schema=f"{args.task}.action.v1",
            embodiment=args.embodiment_id,
        )
        coordinator = R2S2RCoordinator(args.r2s2r_root)
        coordinator.register_policy(descriptor)
        iteration = coordinator.record_iteration(
            "train",
            alignment_hash=args.alignment_hash or "",
            sampling_hash=args.feedback_hash or "",
            policy_hash=descriptor.content_hash,
            provenance={
                "policy_fingerprint": (
                    f"{descriptor.fingerprint:016x}"
                ),
                "task": args.task,
                "backend": args.backend,
                "environment_count": args.envs,
                "rollout_steps": args.rollout_steps,
                "iterations": args.iterations,
                "checkpoint": str(checkpoint.resolve()),
                "sampling_mode": (
                    args.sampling_mode if family_task else "fixed"
                ),
            },
        )
        report.update(
            policy_fingerprint=f"{descriptor.fingerprint:016x}",
            r2s2r_iteration=iteration,
        )
    print(json.dumps(report))
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

        def advance(
            current,
            current_actions,
            compiled_world=world,
        ):
            output = step(
                compiled_world,
                current,
                current_actions,
            )
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


def _cli_fingerprint(value: str) -> int:
    text = value.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if (
        not text
        or len(text) > 16
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(
            f"fingerprint must be up to 16 hexadecimal digits: {value!r}"
        )
    return int(text, 16)


def _world_family(args: argparse.Namespace) -> FrankaPickPlaceWorldFamily:
    return FrankaPickPlaceWorldFamily(
        capacity=max(1, int(getattr(args, "particles", 1))),
        library_path=args.library,
        metallib_path=args.metallib,
    )


def run_align(args: argparse.Namespace) -> int:
    if not 1 <= args.particles <= 4096:
        raise ValueError("--particles must be between 1 and 4096")
    if args.physics_substeps <= 0:
        raise ValueError("--physics-substeps must be positive")
    manifest_path = args.replay_manifest.expanduser().resolve()
    replay = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )
    replay_schema_version = int(replay.get("schema_version", 1))
    if replay_schema_version not in {1, 2}:
        raise ValueError("unsupported replay-manifest schema version")
    if not replay.get("scenario_schema") or not replay.get(
        "episode_twin_hash"
    ):
        raise ValueError(
            "replay manifest requires scenario_schema and "
            "episode_twin_hash provenance"
        )
    if (
        replay_schema_version == 2
        and (
            "trace_npz" not in replay
            or "control_period_seconds" not in replay
            or "command_semantics" not in replay
        )
    ):
        raise ValueError(
            "version-2 replay manifest requires trace_npz and "
            "control_period_seconds with command semantics"
        )
    if (
        replay_schema_version == 1
        and (
            "replay_residual_modes" not in replay
            or not replay.get("command_stream_hash")
        )
    ):
        raise ValueError(
            "version-1 replay manifest requires "
            "command_stream_hash and replay_residual_modes"
        )
    with _world_family(args) as family:
        schema = family.scenario_schema
        expected_schema = replay.get("scenario_schema")
        if expected_schema is not None and expected_schema != schema.id:
            raise ValueError(
                "replay manifest and compiled world use different "
                "ScenarioSchema ids"
            )
        coordinator = R2S2RCoordinator(args.root)
        replay_artifact = dict(replay)
        trace_hash = ""
        engine_hash = str(replay.get("engine_hash", ""))
        residual_names: tuple[str, ...] = ()
        if "trace_npz" in replay:
            if replay.get(
                "command_semantics",
                "joint_position_target",
            ) != "joint_position_target":
                raise ValueError(
                    "the first executable replay path requires "
                    "joint_position_target commands"
                )
            trace_path = Path(str(replay["trace_npz"])).expanduser()
            if not trace_path.is_absolute():
                trace_path = manifest_path.parent / trace_path
            scale_values = replay.get("residual_scales", {})
            scales = ReplayResidualScales(
                robot_q=float(scale_values.get("robot_q", 0.05)),
                robot_v=float(scale_values.get("robot_v", 0.20)),
                object_position=float(
                    scale_values.get("object_position", 0.02)
                ),
                rod_position=float(
                    scale_values.get("rod_position", 0.01)
                ),
                contact_timing=float(
                    scale_values.get("contact_timing", 1.0)
                ),
                physics_failure=float(
                    scale_values.get("physics_failure", 1.0)
                ),
            )
            period = replay.get("control_period_seconds")
            trace = PhysicalReplayTrace.from_npz(
                trace_path,
                control_period_seconds=(
                    None if period is None else float(period)
                ),
                scales=scales,
            )
            trace_reference = coordinator.ingest_replay_trace(
                trace_path
            )
            trace_hash = trace_reference.content_hash
            replay_solver_mode = str(
                replay.get("solver_mode", "throughput_tgs")
            )
            if replay_solver_mode not in {
                "throughput_tgs",
                "quality_newton",
            }:
                raise ValueError(
                    "replay solver_mode must be throughput_tgs or "
                    "quality_newton"
                )
            world = compile_world(
                "franka",
                scene="pick_place",
                environment_capacity=args.particles,
                solver_mode=replay_solver_mode,
                actuation_mode="implicit_position",
                physics_substeps=args.physics_substeps,
                control_timestep=trace.control_period_seconds,
                metallib_path=args.metallib or "",
            )
            physical_evaluator = MLXPhysicalReplayEvaluator(
                world,
                family,
                trace,
                seed=args.seed,
            )
            extension_module = sys.modules.get(
                "metalrobo._mlx_ext"
            )
            extension_file = getattr(
                extension_module,
                "__file__",
                None,
            )
            extension_path = (
                None
                if extension_file is None
                else Path(str(extension_file)).resolve()
            )
            engine_components = {
                "library_sha256": _checkpoint_content_hash(
                    family.library_path
                ),
                "metallib_sha256": (
                    ""
                    if family.metallib_path is None
                    else _checkpoint_content_hash(
                        family.metallib_path
                    )
                ),
                "mlx_extension_sha256": (
                    ""
                    if extension_path is None
                    else _checkpoint_content_hash(extension_path)
                ),
                "solver_mode": replay_solver_mode,
                "physics_substeps": args.physics_substeps,
                "control_period_seconds": (
                    trace.control_period_seconds
                ),
            }
            engine_hash = hashlib.sha256(
                json.dumps(
                    engine_components,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            evaluator = physical_evaluator
            residual_count = physical_evaluator.residual_count
            residual_names = physical_evaluator.residual_names
            replay_artifact.pop("trace_npz", None)
            replay_artifact["trace_artifact_hash"] = trace_hash
            replay_artifact["trace_file_name"] = trace_path.name
            replay_artifact["residual_names"] = list(residual_names)
            replay_artifact["replay_backend"] = (
                "mlx-metal-exact-candidate-v1"
            )
            replay_artifact["solver_mode"] = replay_solver_mode
            replay_artifact["declared_engine_hash"] = str(
                replay.get("engine_hash", "")
            )
            replay_artifact["engine_hash"] = engine_hash
            replay_artifact["engine_components"] = engine_components
        else:
            evaluator, residual_count = make_affine_replay_evaluator(
                replay["replay_residual_modes"],
                feature_count=len(schema.features),
            )
            residual_names = tuple(
                f"provider_residual_{index}"
                for index in range(residual_count)
            )
            replay_artifact["residual_names"] = list(residual_names)
            replay_artifact["replay_backend"] = (
                "provider-affine-residual-v1"
            )
        artifact = coordinator.align(
            schema,
            replay_artifact,
            evaluator,
            initial_quantiles=deterministic_candidate_scenarios(
                len(schema.features),
                count=args.particles,
            ),
            residual_count=residual_count,
            config=SMCConfig(seed=args.seed),
            world_hash=str(
                replay.get("world_hash")
                or replay["episode_twin_hash"]
            ),
            engine_hash=engine_hash,
        )
    print(
        json.dumps(
            {
                "alignment_hash": artifact.content_hash,
                "alignment_fingerprint": (
                    f"{artifact.fingerprint:016x}"
                ),
                "scenario_schema_fingerprint": (
                    f"{artifact.schema_fingerprint:016x}"
                ),
                "particles": artifact.particle_count,
                "replay_artifact_hash": (
                    artifact.replay_artifact_hash
                ),
                "trace_artifact_hash": trace_hash or None,
                "engine_hash": engine_hash or None,
                "residual_names": residual_names,
            },
            indent=2,
        )
    )
    return 0


def run_record_sim(args: argparse.Namespace) -> int:
    with _world_family(args) as family:
        coordinator = R2S2RCoordinator(args.root)
        artifact = coordinator.ingest_simulation_manifest(
            family.scenario_schema,
            args.manifest,
        )
    print(
        json.dumps(
            {"simulation_outcome_artifact": artifact.content_hash},
            indent=2,
        )
    )
    return 0


def run_ingest_real(args: argparse.Namespace) -> int:
    with _world_family(args) as family:
        coordinator = R2S2RCoordinator(args.root)
        artifact = coordinator.ingest_hardware_manifest(
            family.scenario_schema,
            args.manifest,
        )
    print(
        json.dumps(
            {"hardware_outcome_artifact": artifact.content_hash},
            indent=2,
        )
    )
    return 0


def run_fit_feedback(args: argparse.Namespace) -> int:
    with _world_family(args) as family:
        coordinator = R2S2RCoordinator(args.root)
        artifact = coordinator.fit_feedback(
            family.scenario_schema,
            policy_fingerprint=_cli_fingerprint(
                args.policy_fingerprint
            ),
            task_fingerprint=_cli_fingerprint(
                args.task_fingerprint
            ),
            embodiment_fingerprint=_cli_fingerprint(
                args.embodiment_fingerprint
            ),
            steps=args.steps,
            candidate_count=args.candidates,
            maximum_regions=args.regions,
            seed=args.seed,
        )
    print(
        json.dumps(
            {
                "feedback_hash": artifact.content_hash,
                "feedback_fingerprint": (
                    f"{artifact.fingerprint:016x}"
                ),
                "model_hash": artifact.model_content_hash,
                "regions": artifact.region_count,
                "hardware_prediction": (
                    "available"
                    if artifact.hardware_available
                    else "unavailable_sim_only"
                ),
            },
            indent=2,
        )
    )
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    with _world_family(args) as family:
        coordinator = R2S2RCoordinator(args.root)
        report = coordinator.evaluate(
            family.scenario_schema,
            task_fingerprint=_cli_fingerprint(
                args.task_fingerprint
            ),
            policy_fingerprints=[
                _cli_fingerprint(value)
                for value in args.policy_fingerprints
            ],
        )
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
        if args.command == "tune-workers":
            return run_worker_tuning(args)
        if args.command == "align":
            return run_align(args)
        if args.command == "record-sim":
            return run_record_sim(args)
        if args.command == "ingest-real":
            return run_ingest_real(args)
        if args.command == "fit-feedback":
            return run_fit_feedback(args)
        if args.command == "evaluate":
            return run_evaluate(args)
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


if __name__ == "__main__":
    raise SystemExit(main())
