#!/usr/bin/env python3
"""Focused active-encoder unified-quality and explicit-rod-state probe."""

from __future__ import annotations

import mlx.core as mx

from metalrobo import RodState, compile_world, initial_state, step


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    world = compile_world(
        "franka",
        scene="cube",
        environment_capacity=4,
        solver_mode="quality_newton",
        ccd_mode="disabled",
        physics_substeps=1,
    )
    state = initial_state(world, 4)
    require(
        world.solver_mode == "quality_newton"
        and isinstance(state.rods, RodState)
        and state.rods.position.shape == (4, 0, 4),
        "quality mode or explicit zero-sized RodState was not published",
    )
    actions = mx.zeros((4, world.nv), dtype=mx.float32)
    compiled_step = mx.compile(
        lambda world_state, controls: step(
            world,
            world_state,
            controls,
        )
    )
    first = compiled_step(state, actions)
    replay = compiled_step(state, actions)
    mx.eval(first, replay)
    require(
        first.status[:, 0].tolist() == [0, 0, 0, 0]
        and first.contacts.counts.tolist() == [2, 2, 2, 2]
        and first.status[:, 40].tolist() == [4, 4, 4, 4]
        and all(
            iteration > 0
            for iteration in first.status[:, 48].tolist()
        ),
        "persistent unified quality solve did not converge through the contact graph",
    )
    require(
        first.next_state.q.tolist() == replay.next_state.q.tolist()
        and first.next_state.v.tolist() == replay.next_state.v.tolist()
        and first.contacts.stable_ids.tolist()
        == replay.contacts.stable_ids.tolist(),
        "same-device quality replay was not deterministic",
    )
    print(
        "mlx_quality_contact=ok "
        f"environments=4 contacts={sum(first.contacts.counts.tolist())} "
        f"worker_packets={first.status[0, 40].item()} "
        f"newton_max={max(first.status[:, 48].tolist())} "
        f"rod_nodes={world.rod_node_count}"
    )


if __name__ == "__main__":
    main()
