#!/usr/bin/env python3
"""Focused active-encoder probe for heterogeneous generalized constraints."""

from __future__ import annotations

import mlx.core as mx

from metalrobo.mlx_multi_articulated import compile_program, step


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    environments = 4
    program = compile_program(
        "dual_psm_g1",
        environment_capacity=environments,
        solver_iterations=192,
        convergence_tolerance=5.0e-5,
    )
    q = mx.broadcast_to(
        mx.array(program.default_q, dtype=mx.float32),
        (environments, program.nq),
    )
    free_velocity = 0.08 * mx.sin(
        mx.arange(
            environments * program.nv,
            dtype=mx.float32,
        ).reshape(environments, program.nv)
        + 1.0
    )

    @mx.compile
    def compiled_step(
        q_value: mx.array,
        velocity_value: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        output = step(program, q_value, velocity_value)
        return output

    next_velocity, impulses, status = compiled_step(
        q,
        free_velocity,
    )
    mx.eval(next_velocity, impulses, status)
    require(
        status.shape == (environments, 12),
        "generalized status ABI shape changed",
    )
    require(
        all(row[0] == 0 for row in status.tolist()),
        "one or more MLX generalized environments failed",
    )
    require(
        float(mx.max(mx.abs(impulses)).item()) > 0.0,
        "constraints emitted no impulse",
    )
    require(
        float(mx.max(mx.abs(next_velocity - free_velocity)).item())
        > 0.0,
        "constraints did not change velocity",
    )

    replay_velocity, replay_impulses, replay_status = compiled_step(
        q,
        free_velocity,
    )
    mx.eval(replay_velocity, replay_impulses, replay_status)
    require(
        bool(mx.array_equal(next_velocity, replay_velocity).item())
        and bool(mx.array_equal(impulses, replay_impulses).item())
        and bool(mx.array_equal(status, replay_status).item()),
        "MLX generalized replay is not bitwise deterministic",
    )

    invalid_location = (
        (mx.arange(environments)[:, None] == 0)
        & (mx.arange(program.nq)[None, :] == 3)
    )
    invalid_q = mx.where(
        invalid_location,
        mx.array(float("nan"), dtype=mx.float32),
        q,
    )
    failed_velocity, failed_impulses, failed_status = compiled_step(
        invalid_q,
        free_velocity,
    )
    mx.eval(failed_velocity, failed_impulses, failed_status)
    failure_rows = failed_status.tolist()
    require(
        failure_rows[0][0] != 0
        and all(row[0] == 0 for row in failure_rows[1:]),
        "MLX generalized failure was not isolated",
    )
    require(
        bool(
            mx.array_equal(
                failed_velocity[0],
                free_velocity[0],
            ).item()
        )
        and float(mx.max(mx.abs(failed_impulses[0])).item())
        == 0.0,
        "failed generalized environment published candidate state",
    )

    print(
        "mlx_multi_articulated=ok",
        f"device={mx.default_device()}",
        f"articulations={program.articulation_count}",
        f"environments={environments}",
        f"nq={program.nq}",
        f"nv={program.nv}",
        f"rows={program.row_count}",
        f"fingerprint={program.fingerprint}",
        "compiled=yes",
        "active_encoder=yes",
        "deterministic=yes",
        f"isolated_failure_code={failure_rows[0][0]}",
        "transactional=yes",
    )


if __name__ == "__main__":
    main()
