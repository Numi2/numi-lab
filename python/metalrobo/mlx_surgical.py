"""Pure-MLX surgical command transforms for the PSM contact world."""

from __future__ import annotations

import mlx.core as mx


def psm_physical_position_targets(
    logical_targets: mx.array,
) -> mx.array:
    """Expand six arm targets plus jaw aperture into eight PSM coordinates.

    ``logical_targets[..., :6]`` map directly to the arm. The final
    non-negative aperture is split symmetrically across the independent jaws.
    The physics model intentionally keeps both jaw coordinates explicit until
    tendon/transmission constraints are executable.
    """

    if logical_targets.ndim < 1 or logical_targets.shape[-1] != 7:
        raise ValueError(
            "PSM logical targets must have a final dimension of 7"
        )
    aperture = mx.maximum(
        logical_targets[..., 6:7],
        mx.array(0.0, dtype=logical_targets.dtype),
    )
    return mx.concatenate(
        (
            logical_targets[..., :6],
            -0.5 * aperture,
            0.5 * aperture,
        ),
        axis=-1,
    )


__all__ = ["psm_physical_position_targets"]
