"""MLX-native generalized constraints for heterogeneous robot worlds."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from ._mlx_ext import (  # type: ignore[attr-defined]
    MLXCompiledMultiArticulatedProgram,
    compile_multi_articulated_program,
    generalized_constraint_step,
)


class GeneralizedConstraintOutput(NamedTuple):
    next_velocity: mx.array
    impulses: mx.array
    status: mx.array


def compile_program(
    model: str = "dual_psm",
    *,
    environment_capacity: int = 256,
    solver_iterations: int = 192,
    convergence_tolerance: float = 5.0e-5,
    timestep: float = 1.0e-3,
    metallib_path: str = "",
    stream: mx.Stream | None = None,
) -> MLXCompiledMultiArticulatedProgram:
    """Cook immutable ABA/Jacobian packets and prepare one MLX stream."""

    return compile_multi_articulated_program(
        model,
        environment_capacity=environment_capacity,
        solver_iterations=solver_iterations,
        convergence_tolerance=convergence_tolerance,
        timestep=timestep,
        metallib_path=metallib_path,
        stream=stream,
    )


def step(
    program: MLXCompiledMultiArticulatedProgram,
    q: mx.array,
    free_velocity: mx.array,
    *,
    stream: mx.Stream | None = None,
) -> GeneralizedConstraintOutput:
    """Apply the cooked constraints on MLX's active Metal command encoder."""

    values = generalized_constraint_step(
        program,
        q,
        free_velocity,
        stream=stream,
    )
    return GeneralizedConstraintOutput(*values)


__all__ = [
    "GeneralizedConstraintOutput",
    "MLXCompiledMultiArticulatedProgram",
    "compile_program",
    "step",
]
