"""Run a sampled Franka world family through one MLX contact step."""

from __future__ import annotations

import argparse
import json

import mlx.core as mx

from metalrobo import (
    FrankaPickPlaceWorldFamily,
    compile_world,
    initial_state_from_world_family,
    step,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0x1234)
    arguments = parser.parse_args()
    if arguments.envs <= 0:
        raise ValueError("--envs must be positive")

    with FrankaPickPlaceWorldFamily(arguments.envs) as family:
        family.sample(arguments.envs, seed=arguments.seed)
        world = compile_world(
            "franka",
            scene="pick_place",
            environment_capacity=arguments.envs,
            solver_mode="throughput_tgs",
            actuation_mode="implicit_position",
            physics_substeps=2,
        )
        state = initial_state_from_world_family(world, family)
        actions = mx.broadcast_to(
            mx.array(world.default_q, dtype=mx.float32),
            (arguments.envs, world.nv),
        )
        output = step(world, state, actions)
        mx.eval(
            state.scene_bodies.position,
            output.next_state.q,
            output.physics_error,
        )
        object_x = state.scene_bodies.position[:, 0, 0]
        report = {
            "device": family.device_name,
            "environments": arguments.envs,
            "q_shape": tuple(state.q.shape),
            "scene_shape": tuple(state.scene_bodies.position.shape),
            "object_x_span": float(mx.max(object_x) - mx.min(object_x)),
            "physics_errors": int(
                mx.sum(output.physics_error.astype(mx.uint32))
            ),
            "gpu_resident_family_state": True,
        }
        if report["object_x_span"] <= 0.0:
            raise RuntimeError("sampled object poses collapsed")
        if report["physics_errors"] != 0:
            raise RuntimeError(
                f"{report['physics_errors']} physics environments failed"
            )
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
