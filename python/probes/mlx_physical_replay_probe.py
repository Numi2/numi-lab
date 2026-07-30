#!/usr/bin/env python3
"""Exercise exact-candidate physical replay through real Metal physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import selectors
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from metalrobo._mlx_abi import (
    ENGINE_ABI_VERSION as engine_abi_version,
    RUNTIME_ABI_FINGERPRINT as runtime_abi_fingerprint,
)
from metalrobo.mlx_replay import (
    MLXPhysicalReplayEvaluator,
    PhysicalReplayTrace,
)
from metalrobo.mlx_r2s2r import SMCConfig, fit_alignment_smc
from metalrobo.mlx_world import (
    ControllerDelayState,
    compile_world,
    compile_world_pack,
    sampled_state_from_world_family,
    step_sampled_world_family,
)
from metalrobo.worlds import (
    FrankaPickPlaceWorldFamily,
    PackedWorldFamily,
)
from metalrobo.r2s2r import R2S2RCoordinator


_WORLD_PACK_FORMAT_VERSION = 5
_WORLD_COMPILER_ABI_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=1,
        help=(
            "physics substeps per control step; correctness smoke tests "
            "default to one to keep integrated-GPU submissions short"
        ),
    )
    parser.add_argument(
        "--command-buffer-step-limit",
        type=int,
        default=1,
        help=(
            "maximum physics steps encoded into one replay command "
            "buffer"
        ),
    )
    parser.add_argument(
        "--solver-mode",
        choices=("throughput_tgs", "quality_newton"),
        default="throughput_tgs",
    )
    parser.add_argument("--library")
    parser.add_argument("--metallib")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "validate native/extension ABI and the world-pack header "
            "without creating a Metal pipeline or evaluating physics"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress flushed replay stage timings on stderr",
    )
    parser.add_argument(
        "--watchdog-seconds",
        type=float,
        default=20.0,
        help=(
            "terminate a GPU worker after this many seconds without "
            "stage progress; zero runs in-process"
        ),
    )
    parser.add_argument(
        "--repeat-evaluations",
        type=int,
        default=0,
        help=(
            "optional identical replay repetitions for determinism "
            "diagnostics; disabled in the bounded smoke path"
        ),
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--world-pack",
        help=(
            "optional authored pack; tactile observations are replayed "
            "when the pack publishes them"
        ),
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    if (
        args.envs < 2
        or args.steps <= 0
        or args.physics_substeps <= 0
        or args.physics_substeps > 64
        or args.repeat_evaluations < 0
        or args.repeat_evaluations > 3
    ):
        raise ValueError("probe requires at least two worlds and one step")
    started = time.perf_counter()
    previous_stage = started

    def stage(name: str, **values: object) -> None:
        nonlocal previous_stage
        now = time.perf_counter()
        if not args.quiet or args.worker:
            print(
                json.dumps(
                    {
                        "stage": name,
                        "stage_seconds": now - previous_stage,
                        "elapsed_seconds": now - started,
                        **values,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        previous_stage = now

    stage(
        "abi_preflight",
        engine_abi=int(engine_abi_version),
        runtime_abi_fingerprint=int(runtime_abi_fingerprint),
    )

    def compile_requested_world():
        if args.world_pack:
            return compile_world_pack(
                args.world_pack,
                environment_capacity=args.envs,
                solver_mode=args.solver_mode,
                actuation_mode="implicit_position",
                physics_substeps=args.physics_substeps,
                metallib_path=args.metallib or "",
            )
        return compile_world(
            "franka",
            scene="pick_place",
            environment_capacity=args.envs,
            solver_mode=args.solver_mode,
            actuation_mode="implicit_position",
            physics_substeps=args.physics_substeps,
            metallib_path=args.metallib or "",
        )

    if args.preflight_only:
        pack_metadata: dict[str, object] = {}
        if args.world_pack:
            pack_path = Path(args.world_pack)
            header_layout = struct.Struct("<8sIIIIQQQQQ3Q")
            with pack_path.open("rb") as stream:
                header = stream.read(header_layout.size)
                if len(header) != header_layout.size:
                    raise ValueError(
                        "world pack is shorter than its 88-byte header"
                    )
                (
                    magic,
                    format_version,
                    endian_marker,
                    compiler_abi,
                    pack_engine_abi,
                    payload_bytes,
                    content_hash,
                    family_fingerprint,
                    template_fingerprint,
                    program_fingerprint,
                    *_,
                ) = header_layout.unpack(header)
                if (
                    magic != b"MRWPACK1"
                    or endian_marker != 0x01020304
                    or format_version != _WORLD_PACK_FORMAT_VERSION
                    or compiler_abi != _WORLD_COMPILER_ABI_VERSION
                    or pack_engine_abi != int(engine_abi_version)
                ):
                    raise ValueError(
                        "world pack magic, format, endian marker, or ABI "
                        "does not match this runtime"
                    )
                hash_value = 14695981039346656037
                measured_payload_bytes = 0
                while block := stream.read(1024 * 1024):
                    measured_payload_bytes += len(block)
                    for value in block:
                        hash_value ^= value
                        hash_value = (
                            hash_value * 1099511628211
                        ) & 0xFFFFFFFFFFFFFFFF
                if (
                    measured_payload_bytes != payload_bytes
                    or hash_value != content_hash
                ):
                    raise ValueError(
                        "world pack payload length or content hash is "
                        "invalid"
                    )
            pack_metadata = {
                "compiler_abi": compiler_abi,
                "content_hash": content_hash,
                "family_fingerprint": family_fingerprint,
                "format_version": format_version,
                "pack_engine_abi": pack_engine_abi,
                "payload_bytes": payload_bytes,
                "program_fingerprint": program_fingerprint,
                "template_fingerprint": template_fingerprint,
                "world_pack": str(pack_path.resolve()),
            }
        stage(
            "host_preflight_complete",
            world_pack=bool(args.world_pack),
        )
        print(
            json.dumps(
                {
                    "device": str(mx.default_device()),
                    "engine_abi": int(engine_abi_version),
                    "environment_capacity": args.envs,
                    "metal_pipeline_created": False,
                    "physics_evaluated": False,
                    "physics_substeps": args.physics_substeps,
                    "runtime_abi_fingerprint": int(
                        runtime_abi_fingerprint
                    ),
                    "status": "ok",
                    **pack_metadata,
                },
                sort_keys=True,
            )
        )
        return 0

    family_context = (
        PackedWorldFamily(
            args.world_pack,
            capacity=args.envs,
            library_path=args.library,
            metallib_path=args.metallib,
        )
        if args.world_pack
        else FrankaPickPlaceWorldFamily(
            capacity=args.envs,
            library_path=args.library,
            metallib_path=args.metallib,
        )
    )
    with family_context as family:
        stage("world_family_open", device=family.device_name)
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
        world = compile_requested_world()
        stage(
            "world_compiled",
            environments=args.envs,
            physics_substeps=args.physics_substeps,
            tactile_sensors=int(world.tactile_sensor_count),
            tactile_samples=int(world.tactile_sample_count),
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
        tactile_wrenches = []
        tactile_depths = []
        physics_status_codes = []
        for step_index, command in enumerate(command_array):
            stepped = step_sampled_world_family(
                world,
                state,
                mx.array(command)[None, :],
                delay,
                control_period_seconds=world.control_timestep,
            )
            state = stepped.next_state
            delay = stepped.delay_state
            stage(
                "capture_step_submitted",
                step=step_index + 1,
            )
            mx.eval(
                state.world.q,
                state.world.v,
                state.world.scene_bodies.position,
                stepped.physics.contacts.mask,
                stepped.physics.tactile.penetration_depth_m,
                stepped.physics.tactile.summary
                .net_force_and_contact_area,
                stepped.physics.tactile.summary
                .net_torque_and_maximum_depth,
                stepped.physics.status,
            )
            stage(
                "capture_step_materialized",
                step=step_index + 1,
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
            if int(world.tactile_sensor_count) > 0:
                tactile_wrenches.append(
                    np.concatenate(
                        (
                            np.asarray(
                                stepped.physics.tactile.summary
                                .net_force_and_contact_area
                            )[0, :, :3],
                            np.asarray(
                                stepped.physics.tactile.summary
                                .net_torque_and_maximum_depth
                            )[0, :, :3],
                        ),
                        axis=-1,
                    )
                )
                tactile_depths.append(
                    np.asarray(
                        stepped.physics.tactile.penetration_depth_m
                    )[0].copy()
                )
            physics_status_codes.append(
                int(np.asarray(stepped.physics.status)[0, 0])
            )

        with tempfile.TemporaryDirectory(
            prefix="metalrobo-replay-probe-"
        ) as directory:
            trace_path = Path(directory) / "physical_episode.npz"
            trace_arrays = {
                "commands": command_array,
                "robot_q": np.asarray(robot_q, dtype=np.float32),
                "robot_v": np.asarray(robot_v, dtype=np.float32),
                "object_positions": np.asarray(
                    objects,
                    dtype=np.float32,
                ),
                "scene_body_indices": np.asarray(
                    [0],
                    dtype=np.uint32,
                ),
                "contact_active": np.asarray(
                    contacts,
                    dtype=np.float32,
                ),
                "control_period_seconds": np.asarray(
                    world.control_timestep,
                    dtype=np.float32,
                ),
            }
            tactile_alignment = {}
            if tactile_wrenches:
                trace_arrays["tactile_wrench"] = np.asarray(
                    tactile_wrenches,
                    dtype=np.float32,
                )
                trace_arrays["tactile_depth"] = np.asarray(
                    tactile_depths,
                    dtype=np.float32,
                )
                tactile_metadata = json.loads(
                    world.tactile_observation_metadata_json
                )
                tactile_alignment = {
                    "stream_fingerprint": hashlib.sha256(
                        Path(args.world_pack).read_bytes()
                    ).hexdigest(),
                    "canonical_observation_fingerprint": (
                        tactile_metadata["fingerprint"]
                    ),
                    "sensor_ids": list(world.tactile_sensor_ids),
                    "wrench_verified": True,
                    "depth_verified": True,
                }
            np.savez_compressed(trace_path, **trace_arrays)
            trace = PhysicalReplayTrace.from_npz(
                trace_path,
                tactile_sensor_ids=tuple(
                    tactile_alignment.get("sensor_ids", ())
                ),
                tactile_stream_fingerprint=(
                    tactile_alignment.get("stream_fingerprint")
                ),
                canonical_tactile_fingerprint=(
                    tactile_alignment.get(
                        "canonical_observation_fingerprint"
                    )
                ),
                tactile_wrench_verified=bool(tactile_alignment),
                tactile_depth_verified=bool(tactile_alignment),
            )
            evaluator = MLXPhysicalReplayEvaluator(
                world,
                family,
                trace,
                seed=args.seed,
                command_buffer_step_limit=(
                    args.command_buffer_step_limit
                ),
            )
            stage(
                "replay_evaluator_compiled",
                residual_count=evaluator.residual_count,
            )
            rng = np.random.default_rng(args.seed)
            candidates = rng.uniform(
                0.05,
                0.95,
                size=(args.envs, feature_count),
            ).astype(np.float32)
            candidates[0] = true_quantiles[0]
            evaluation = evaluator(mx.array(candidates))
            stage("replay_evaluation_submitted")
            mx.eval(*evaluation)
            residual_values = np.asarray(evaluation.residuals)
            valid_values = np.asarray(evaluation.valid)
            replay_status_values = np.asarray(
                evaluation.physics_status
            )
            finite_candidates = np.all(
                np.isfinite(residual_values),
                axis=-1,
            )
            stage(
                "replay_evaluation_materialized",
                capture_statuses=sorted(set(physics_status_codes)),
                replay_statuses=sorted(
                    {
                        int(value)
                        for value in replay_status_values.reshape(-1)
                    }
                ),
                finite_candidates=int(
                    np.count_nonzero(finite_candidates)
                ),
                nonfinite_residuals=[
                    name
                    for index, name in enumerate(
                        evaluator.residual_names
                    )
                    if not np.all(
                        np.isfinite(residual_values[:, index])
                    )
                ],
                valid_candidates=int(np.count_nonzero(valid_values)),
            )
            for repeat_index in range(args.repeat_evaluations):
                repeated = evaluator(mx.array(candidates))
                mx.eval(*repeated)
                repeated_residuals = np.asarray(repeated.residuals)
                repeated_valid = np.asarray(repeated.valid)
                stage(
                    "replay_repeat_materialized",
                    finite_candidates=int(
                        np.count_nonzero(
                            np.all(
                                np.isfinite(repeated_residuals),
                                axis=-1,
                            )
                        )
                    ),
                    repeat=repeat_index + 1,
                    replay_statuses=sorted(
                        {
                            int(value)
                            for value in np.asarray(
                                repeated.physics_status
                            ).reshape(-1)
                        }
                    ),
                    valid_candidates=int(
                        np.count_nonzero(repeated_valid)
                    ),
                )
            if any(code != 0 for code in physics_status_codes):
                expected_tactile_residuals = {
                    "tactile_force_trajectory",
                    "tactile_torque_trajectory",
                    "tactile_depth_trajectory",
                }
                if trace.tactile_wrench is not None and (
                    not expected_tactile_residuals.issubset(
                        evaluator.residual_names
                    )
                    or not np.all(np.isfinite(residual_values))
                ):
                    raise RuntimeError(
                        "tactile replay residual graph was not compiled"
                    )
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
                    f"tactile_residuals={trace.tactile_wrench is not None} "
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
            stage("smc_submitted")
            mx.eval(*population)
            stage("smc_materialized")
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
                    **(
                        {"tactile_alignment": tactile_alignment}
                        if tactile_alignment
                        else {}
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
            f"tactile_residuals={trace.tactile_wrench is not None} "
            "exact_candidate=yes gpu_replay=yes artifact_graph=yes"
        )
    return 0


def _terminate_worker(worker: subprocess.Popen[str]) -> None:
    worker.terminate()
    try:
        worker.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()


def _supervise(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
    ]
    worker = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert worker.stdout is not None
    assert worker.stderr is not None
    streams = selectors.DefaultSelector()
    streams.register(worker.stdout, selectors.EVENT_READ, sys.stdout)
    streams.register(worker.stderr, selectors.EVENT_READ, sys.stderr)
    deadline = time.monotonic() + args.watchdog_seconds
    try:
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate_worker(worker)
                raise RuntimeError(
                    "Metal replay worker made no stage progress for "
                    f"{args.watchdog_seconds:g}s and was terminated "
                    "before the WindowServer watchdog window"
                )
            ready = streams.select(timeout=remaining)
            if not ready:
                continue
            for key, _ in ready:
                line = key.fileobj.readline()
                if not line:
                    streams.unregister(key.fileobj)
                    continue
                is_stage = False
                if key.data is sys.stderr:
                    try:
                        is_stage = "stage" in json.loads(line)
                    except (
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ):
                        pass
                if not (args.quiet and is_stage):
                    print(
                        line,
                        end="",
                        file=key.data,
                        flush=True,
                    )
                deadline = time.monotonic() + args.watchdog_seconds
        return worker.wait()
    finally:
        streams.close()
        if worker.poll() is None:
            _terminate_worker(worker)


def main() -> int:
    args = parse_args()
    if args.watchdog_seconds < 0.0:
        raise ValueError("watchdog_seconds must be non-negative")
    if not args.worker and args.watchdog_seconds > 0.0:
        return _supervise(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
