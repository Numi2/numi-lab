"""Run a sampled Franka world family through one MLX contact step."""

from __future__ import annotations

import argparse
import json

import mlx.core as mx

from metalrobo import (
    ControllerDelayState,
    FrankaPickPlaceWorldFamily,
    compile_world,
    reset_sampled_world_family,
    sampled_state_from_world_family,
    step,
    step_sampled_world_family,
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
        sampled = sampled_state_from_world_family(world, family)
        state = sampled.world
        mx.eval(
            state.scene_bodies.position,
            state.scene_bodies.linear_velocity,
            sampled.scenarios.headers,
            sampled.scenarios.values,
            sampled.parameters.body_values,
            sampled.parameters.controller_values,
        )
        family.sample(arguments.envs, seed=arguments.seed + 1)
        replacement = sampled_state_from_world_family(world, family)
        reset_mask = mx.arange(arguments.envs) % 2 == 0
        selected = reset_sampled_world_family(
            sampled,
            replacement,
            reset_mask,
        )
        target = list(world.default_q)
        target[0] += 0.05
        actions = mx.broadcast_to(
            mx.array(target, dtype=mx.float32),
            (arguments.envs, world.nv),
        )
        delay = ControllerDelayState(
            history=mx.broadcast_to(
                mx.array(
                    world.default_q,
                    dtype=mx.float32,
                ).reshape((1, 1, world.nv)),
                (arguments.envs, 5, world.nv),
            )
        )
        sampled_output = step_sampled_world_family(
            world,
            selected,
            actions,
            delay,
            control_period_seconds=world.control_timestep,
        )
        output = sampled_output.physics
        unit_payload = mx.concatenate(
            (
                selected.parameters.controller_values[:, :, :3],
                mx.ones_like(
                    selected.parameters.controller_values[:, :, 3:4]
                ),
            ),
            axis=-1,
        )
        baseline_payload = step(
            world,
            selected.world,
            sampled_output.applied_actions,
            body_parameters=selected.parameters.body_values,
            controller_parameters=unit_payload,
        )
        mx.eval(
            selected.scenarios.headers,
            output.next_state.q,
            output.physics_error,
            sampled_output.applied_actions,
            baseline_payload.next_state.q,
        )
        object_x = state.scene_bodies.position[:, 0, 0]
        object_inverse_mass = (
            state.scene_bodies.linear_velocity[:, 0, 3]
        )
        body_mass_scale = sampled.parameters.body_values[:, 11, 0]
        controller_gain = (
            selected.parameters.controller_values[:, 0, 0]
        )
        joint_q = output.next_state.q[:, 0]
        applied_joint = sampled_output.applied_actions[:, 0]
        payload_effect = mx.max(
            mx.abs(
                output.next_state.q
                - baseline_payload.next_state.q
            )
        )
        report = {
            "device": family.device_name,
            "environments": arguments.envs,
            "q_shape": tuple(state.q.shape),
            "scene_shape": tuple(state.scene_bodies.position.shape),
            "object_x_span": float(mx.max(object_x) - mx.min(object_x)),
            "object_inverse_mass_span": float(
                mx.max(object_inverse_mass) -
                mx.min(object_inverse_mass)
            ),
            "body_mass_scale_span": float(
                mx.max(body_mass_scale) - mx.min(body_mass_scale)
            ),
            "controller_gain_span": float(
                mx.max(controller_gain) - mx.min(controller_gain)
            ),
            "controlled_joint_span": float(
                mx.max(joint_q) - mx.min(joint_q)
            ),
            "delayed_action_span": float(
                mx.max(applied_joint) - mx.min(applied_joint)
            ),
            "payload_compensation_effect": float(payload_effect),
            "scenario_width": int(sampled.scenarios.values.shape[1]),
            "physics_errors": int(
                mx.sum(output.physics_error.astype(mx.uint32))
            ),
            "gpu_resident_family_state": True,
        }
        if report["object_x_span"] <= 0.0:
            raise RuntimeError("sampled object poses collapsed")
        if report["object_inverse_mass_span"] <= 0.0:
            raise RuntimeError("sampled mass/inertia did not reach MLX")
        if report["body_mass_scale_span"] <= 0.0:
            raise RuntimeError("body parameter population collapsed")
        if report["controller_gain_span"] <= 0.0:
            raise RuntimeError("controller parameter population collapsed")
        if report["controlled_joint_span"] <= 0.0:
            raise RuntimeError(
                "sampled controller gains did not affect MLX stepping"
            )
        if report["delayed_action_span"] <= 0.0:
            raise RuntimeError(
                "sampled controller latency did not affect applied actions"
            )
        if report["payload_compensation_effect"] <= 0.0:
            raise RuntimeError(
                "payload compensation did not affect the MLX drive"
            )
        if report["physics_errors"] != 0:
            raise RuntimeError(
                f"{report['physics_errors']} physics environments failed"
            )
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
