#!/usr/bin/env python3
"""Exercise the compiled Franka action/task contracts without host stepping."""

from __future__ import annotations

import mlx.core as mx

from metalrobo.mlx_manipulation import (
    FrankaTaskPhase,
    adapt_cartesian_action,
    adapt_operational_space_action,
    franka_kinematics,
    initial_franka_pick_place_task,
    step_franka_pick_place_task,
)
from metalrobo.mlx_teacher import privileged_franka_teacher_action
from metalrobo.mlx_world import (
    ActuatorState,
    ContactEvidence,
    RodState,
    SampledWorldFamilyState,
    ScenarioState,
    SceneBodyState,
    SolverCache,
    TactileState,
    WorldPhysicalParameters,
    WorldState,
)


def _with_scene(
    sampled: SampledWorldFamilyState,
    *,
    object_position: mx.array,
    object_velocity: mx.array,
    q: mx.array | None = None,
) -> SampledWorldFamilyState:
    scene = sampled.world.scene_bodies
    positions = mx.concatenate(
        (
            object_position[:, None, :],
            scene.position[:, 1:, :],
        ),
        axis=1,
    )
    velocities = mx.concatenate(
        (
            object_velocity[:, None, :],
            scene.linear_velocity[:, 1:, :],
        ),
        axis=1,
    )
    return SampledWorldFamilyState(
        world=WorldState(
            q=sampled.world.q if q is None else q,
            v=sampled.world.v,
            scene_bodies=SceneBodyState(
                position=positions,
                orientation=scene.orientation,
                linear_velocity=velocities,
                angular_velocity=scene.angular_velocity,
            ),
            rods=sampled.world.rods,
            solver_cache=sampled.world.solver_cache,
            tactile=sampled.world.tactile,
            actuators=sampled.world.actuators,
        ),
        scenarios=sampled.scenarios,
        parameters=sampled.parameters,
    )


def _contacts(
    pairs: tuple[tuple[int, int], ...],
) -> ContactEvidence:
    slot_count = 3
    stable_ids = [
        list(pairs[index]) if index < len(pairs) else [0, 0]
        for index in range(slot_count)
    ]
    mask = [index < len(pairs) for index in range(slot_count)]
    return ContactEvidence(
        values=mx.zeros((1, slot_count, 8), dtype=mx.float32),
        stable_ids=mx.array([stable_ids], dtype=mx.uint32),
        counts=mx.array([len(pairs)], dtype=mx.uint32),
        mask=mx.array([mask], dtype=mx.bool_),
    )


def _phase(value: mx.array) -> int:
    mx.eval(value)
    return int(value.item())


def _sampled_state() -> SampledWorldFamilyState:
    q = mx.array(
        [[
            0.0,
            -0.7853982,
            0.0,
            -2.3561945,
            0.0,
            1.5707963,
            0.7853982,
            0.04,
            0.04,
        ]],
        dtype=mx.float32,
    )
    zero_scene = mx.zeros((1, 4, 4), dtype=mx.float32)
    target = mx.array(
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.45, 0.25, 0.0, 0.0],
            [0.39, 0.13, 0.20, 0.0],
        ]],
        dtype=mx.float32,
    )
    empty_float4 = mx.zeros((1, 0, 4), dtype=mx.float32)
    empty_float = mx.zeros((1, 0), dtype=mx.float32)
    empty_uint4 = mx.zeros((1, 0, 4), dtype=mx.uint32)
    solver_values = {
        "manifold_headers": empty_uint4,
        "manifold_points": mx.zeros(
            (1, 0, 0, 4),
            dtype=mx.uint32,
        ),
        "manifold_counts": mx.zeros((1,), dtype=mx.uint32),
        "pair_cache": empty_uint4,
    }
    if "rod_witnesses" in SolverCache._fields:
        solver_values["rod_witnesses"] = mx.zeros(
            (1, 0, 12),
            dtype=mx.uint32,
        )
    return SampledWorldFamilyState(
        world=WorldState(
            q=q,
            v=mx.zeros_like(q),
            scene_bodies=SceneBodyState(
                position=target,
                orientation=zero_scene,
                linear_velocity=zero_scene,
                angular_velocity=zero_scene,
            ),
            rods=RodState(
                position=empty_float4,
                velocity=empty_float4,
                twist=empty_float,
                twist_rate=empty_float,
            ),
            solver_cache=SolverCache(**solver_values),
            tactile=TactileState(
                previous_depth_m=empty_float,
                previous_validity=mx.zeros(
                    (1, 0),
                    dtype=mx.uint32,
                ),
                previous_object_shape_ids=mx.zeros(
                    (1, 0),
                    dtype=mx.uint32,
                ),
                previous_tangential_motion=empty_float4,
                target_local_anchor=empty_float4,
                frame_index=mx.zeros(
                    (1, 0),
                    dtype=mx.uint64,
                ),
                time_seconds=empty_float,
            ),
            actuators=ActuatorState(
                effective_position_target=q,
                profile_values=mx.broadcast_to(
                    mx.array(
                        [
                            0.0,
                            0.0,
                            float(mx.finfo(mx.float32).max),
                            1.0,
                            0.0,
                            0.0,
                            float(mx.finfo(mx.float32).max),
                        ],
                        dtype=mx.float32,
                    ).reshape((1, 1, 7)),
                    (1, int(q.shape[1]), 7),
                ),
            ),
        ),
        scenarios=ScenarioState(
            headers=mx.zeros((1, 3, 4), dtype=mx.uint32),
            values=empty_float4,
            identities=empty_uint4,
        ),
        parameters=WorldPhysicalParameters(
            body_values=empty_float4,
            body_identities=empty_uint4,
            controller_values=empty_float4,
            controller_identities=empty_uint4,
        ),
    )


def main() -> int:
    sampled = _sampled_state()
    try:
        q = sampled.world.q
        kinematics = franka_kinematics(q)

        epsilon = 1.0e-3
        finite_difference_columns = []
        for joint in range(7):
            perturbation = (
                mx.eye(9, dtype=mx.float32)[joint : joint + 1]
                * epsilon
            )
            perturbed = franka_kinematics(q + perturbation)
            finite_difference_columns.append(
                (perturbed.position - kinematics.position) / epsilon
            )
        finite_difference = mx.stack(
            finite_difference_columns,
            axis=-1,
        )
        jacobian_error = mx.max(
            mx.abs(
                finite_difference - kinematics.linear_jacobian
            )
        )
        adapted = adapt_cartesian_action(
            q,
            mx.array(
                [[0.25, -0.1, 0.2, 0.1, 0.0, -0.1, -1.0]],
                dtype=mx.float32,
            ),
            control_period_seconds=0.05,
        )
        mx.eval(jacobian_error, adapted.joint_targets)
        if float(jacobian_error.item()) > 3.0e-3:
            raise RuntimeError("analytic Franka Jacobian is inconsistent")
        if not bool(mx.all(mx.isfinite(adapted.joint_targets)).item()):
            raise RuntimeError("Cartesian action adapter produced nonfinite q")
        impedance = adapt_operational_space_action(
            q,
            sampled.world.v,
            mx.array(
                [[0.25, -0.1, 0.2, 0.1, 0.0, -0.1, -1.0]],
                dtype=mx.float32,
            ),
            control_period_seconds=0.05,
            payload_mass_kg=0.4,
        )
        mx.eval(impedance.feedforward_torque)
        if not bool(
            mx.all(mx.isfinite(impedance.feedforward_torque)).item()
        ):
            raise RuntimeError(
                "operational-space adapter produced nonfinite torque"
            )

        teacher_task = initial_franka_pick_place_task(sampled)._replace(
            phase=mx.full(
                (1,),
                FrankaTaskPhase.transport,
                dtype=mx.uint32,
            )
        )
        teacher = privileged_franka_teacher_action(
            sampled,
            teacher_task,
        )
        mx.eval(teacher.normalized_action, teacher.detour_active)
        if not bool(teacher.detour_active.item()):
            raise RuntimeError(
                "privileged teacher did not route around blocking clutter"
            )

        tool = kinematics.position
        object_position = mx.concatenate(
            (tool, mx.zeros((1, 1), dtype=mx.float32)),
            axis=-1,
        )
        object_velocity = mx.zeros((1, 4), dtype=mx.float32)
        sampled = _with_scene(
            sampled,
            object_position=object_position,
            object_velocity=object_velocity,
        )
        state = initial_franka_pick_place_task(sampled)
        valid = mx.array([True], dtype=mx.bool_)
        empty = _contacts(())
        bilateral = _contacts(((24, 32), (28, 32)))
        support = _contacts(((32, 34),))

        state, _ = step_franka_pick_place_task(
            state,
            sampled,
            empty,
            physics_valid=valid,
        )
        if _phase(state.phase) != FrankaTaskPhase.pregrasp:
            raise RuntimeError("task did not enter pre-grasp")
        for expected in (
            FrankaTaskPhase.contact,
            FrankaTaskPhase.grasp,
            FrankaTaskPhase.lift,
        ):
            state, _ = step_franka_pick_place_task(
                state,
                sampled,
                bilateral,
                physics_valid=valid,
            )
            if _phase(state.phase) != expected:
                raise RuntimeError(
                    f"task did not enter phase {expected.name}"
                )

        lifted_position = object_position + mx.array(
            [[0.0, 0.0, 0.06, 0.0]],
            dtype=mx.float32,
        )
        sampled = _with_scene(
            sampled,
            object_position=lifted_position,
            object_velocity=object_velocity,
        )
        state, _ = step_franka_pick_place_task(
            state,
            sampled,
            bilateral,
            physics_valid=valid,
        )
        if _phase(state.phase) != FrankaTaskPhase.transport:
            raise RuntimeError("task did not enter transport")

        target = sampled.world.scene_bodies.position[:, 2, :]
        placed_position = target + mx.array(
            [[0.0, 0.0, 0.025, 0.0]],
            dtype=mx.float32,
        )
        sampled = _with_scene(
            sampled,
            object_position=placed_position,
            object_velocity=object_velocity,
        )
        state, _ = step_franka_pick_place_task(
            state,
            sampled,
            bilateral,
            physics_valid=valid,
        )
        if _phase(state.phase) != FrankaTaskPhase.place:
            raise RuntimeError("task did not enter place")

        open_q = mx.concatenate(
            (
                sampled.world.q[:, :7],
                mx.full((1, 2), 0.04, dtype=mx.float32),
            ),
            axis=-1,
        )
        sampled = _with_scene(
            sampled,
            object_position=placed_position,
            object_velocity=object_velocity,
            q=open_q,
        )
        state, _ = step_franka_pick_place_task(
            state,
            sampled,
            support,
            physics_valid=valid,
        )
        if _phase(state.phase) != FrankaTaskPhase.release:
            raise RuntimeError("task did not enter release")

        evidence = None
        for _ in range(6):
            state, evidence = step_franka_pick_place_task(
                state,
                sampled,
                support,
                physics_valid=valid,
            )
        assert evidence is not None
        mx.eval(evidence.success, evidence.reward)
        if _phase(state.phase) != FrankaTaskPhase.settle:
            raise RuntimeError("task did not enter settle")
        if not bool(evidence.success.item()):
            raise RuntimeError("settled placement did not become success")

        _, invalid_evidence = step_franka_pick_place_task(
            state,
            sampled,
            support,
            physics_valid=mx.array([False], dtype=mx.bool_),
        )
        mx.eval(invalid_evidence.reward, invalid_evidence.success)
        if float(invalid_evidence.reward.item()) != 0.0:
            raise RuntimeError("invalid physics contributed task reward")
        if bool(invalid_evidence.success.item()):
            raise RuntimeError("invalid physics contributed task success")

        print(
            f'device="{mx.default_device()}" '
            f"jacobian_max_error={float(jacobian_error.item()):.6g} "
            "phases=approach/pregrasp/contact/grasp/lift/"
            "transport/place/release/settle success=yes "
            "teacher_detour=yes bounded_impedance=yes invalid_reward=zero"
        )
        return 0
    except Exception:
        raise


if __name__ == "__main__":
    raise SystemExit(main())
