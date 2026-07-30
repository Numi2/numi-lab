"""Fail-fast compatibility gate for the native MLX extension."""

from __future__ import annotations

from types import ModuleType

from . import _mlx_ext


def validate_mlx_extension(module: ModuleType) -> tuple[int, int]:
    """Return ABI metadata or reject an extension predating the guard."""

    try:
        engine_abi = int(module.engine_abi_version)
        runtime_fingerprint = int(module.runtime_abi_fingerprint)
    except (AttributeError, TypeError, ValueError) as error:
        raise ImportError(
            "MetalRobo's MLX extension is stale and has no runtime ABI "
            "guard. Rebuild with `cd python && python setup.py "
            "build_ext --inplace` before running Apple-GPU workloads."
        ) from error
    if engine_abi <= 0 or runtime_fingerprint <= 0:
        raise ImportError(
            "MetalRobo's MLX extension published invalid ABI metadata"
        )
    return engine_abi, runtime_fingerprint


ENGINE_ABI_VERSION, RUNTIME_ABI_FINGERPRINT = (
    validate_mlx_extension(_mlx_ext)
)


__all__ = [
    "ENGINE_ABI_VERSION",
    "RUNTIME_ABI_FINGERPRINT",
    "validate_mlx_extension",
]
