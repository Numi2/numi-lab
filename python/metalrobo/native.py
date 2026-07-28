"""Thin ctypes binding for the stable MetalRobo C ABI."""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]

_LIBRARY_ENV: Final = "METALROBO_LIBRARY"
_METALLIB_ENV: Final = "METALROBO_METALLIB"


class MetalRoboError(RuntimeError):
    """Raised when the native MetalRobo runtime reports an error."""


class _RuntimeStatsC(ct.Structure):
    _fields_ = [
        ("last_gpu_milliseconds", ct.c_double),
        ("total_gpu_milliseconds", ct.c_double),
        ("control_steps", ct.c_uint64),
        ("physics_steps", ct.c_uint64),
    ]


class _WorldFamilyLayoutC(ct.Structure):
    _fields_ = [
        ("capacity", ct.c_uint32),
        ("active_instance_count", ct.c_uint32),
        ("asset_count_per_instance", ct.c_uint32),
        ("sensor_count_per_instance", ct.c_uint32),
        ("appearance_count_per_instance", ct.c_uint32),
        ("variation_count", ct.c_uint32),
        ("categorical_value_count", ct.c_uint32),
        ("asset_binding_count", ct.c_uint32),
        ("binding_index_count", ct.c_uint32),
        ("primary_articulation_index", ct.c_uint32),
        ("nq", ct.c_uint32),
        ("nv", ct.c_uint32),
        ("body_count", ct.c_uint32),
        ("scene_body_count", ct.c_uint32),
        ("articulation_count", ct.c_uint32),
        ("retained_private_bytes", ct.c_size_t),
    ]


class _WorldFamilyStatsC(ct.Structure):
    _fields_ = [
        ("compile_count", ct.c_uint64),
        ("sample_count", ct.c_uint64),
        ("readback_count", ct.c_uint64),
        ("last_sample_milliseconds", ct.c_double),
    ]


@dataclass(frozen=True, slots=True)
class RuntimeStats:
    """A snapshot of native runtime counters."""

    last_gpu_milliseconds: float
    total_gpu_milliseconds: float
    control_steps: int
    physics_steps: int


def _decode(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else ""


def _package_directory() -> Path:
    return Path(__file__).resolve().parent


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.is_file():
            return expanded.resolve()
    return None


def resolve_library_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the native dylib, preferring explicit and environment paths."""

    if path is not None:
        explicit = Path(path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"MetalRobo library does not exist: {explicit}")
        return explicit.resolve()

    configured = os.environ.get(_LIBRARY_ENV)
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            raise FileNotFoundError(
                f"{_LIBRARY_ENV} points to a missing file: {configured_path}"
            )
        return configured_path.resolve()

    package_dir = _package_directory()
    found = _first_existing(
        [
            package_dir / "../../build/lib/libmetalrobo.dylib",
            package_dir / "../lib/libmetalrobo.dylib",
            Path("/usr/local/lib/libmetalrobo.dylib"),
            Path("/opt/homebrew/lib/libmetalrobo.dylib"),
        ]
    )
    if found is not None:
        return found

    system_name = ctypes.util.find_library("metalrobo")
    if system_name:
        return Path(system_name)

    default = (package_dir / "../../build/lib/libmetalrobo.dylib").resolve()
    raise FileNotFoundError(
        "Could not find libmetalrobo.dylib. Build the native runtime or set "
        f"{_LIBRARY_ENV}. Expected the development build at {default}"
    )


def resolve_metallib_path(
    path: str | os.PathLike[str] | None = None,
    *,
    library_path: Path | None = None,
) -> Path | None:
    """Resolve MetalRobo.metallib.

    A missing implicit path returns ``None`` so a development dylib can use its
    compiled-in shader path. Explicit and environment paths fail immediately.
    """

    if path is not None:
        explicit = Path(path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"MetalRobo metallib does not exist: {explicit}")
        return explicit.resolve()

    configured = os.environ.get(_METALLIB_ENV)
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            raise FileNotFoundError(
                f"{_METALLIB_ENV} points to a missing file: {configured_path}"
            )
        return configured_path.resolve()

    package_dir = _package_directory()
    candidates = [
        package_dir / "../../build/shaders/MetalRobo.metallib",
        package_dir / "../../build/lib/metalrobo/MetalRobo.metallib",
        package_dir / "../lib/metalrobo/MetalRobo.metallib",
    ]
    if library_path is not None and library_path.is_absolute():
        candidates.extend(
            [
                library_path.parent.parent / "shaders/MetalRobo.metallib",
                library_path.parent / "metalrobo/MetalRobo.metallib",
            ]
        )
    return _first_existing(candidates)


class _Bindings:
    """Configured ctypes function table for one loaded dylib."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = resolve_library_path(path)
        self.lib = ct.CDLL(str(self.path))

        self.lib.mr_version.argtypes = []
        self.lib.mr_version.restype = ct.c_char_p
        self.lib.mr_last_error.argtypes = []
        self.lib.mr_last_error.restype = ct.c_char_p

        self.lib.mr_create_franka.argtypes = [
            ct.c_uint32,
            ct.c_uint64,
            ct.c_char_p,
        ]
        self.lib.mr_create_franka.restype = ct.c_void_p
        self.lib.mr_destroy.argtypes = [ct.c_void_p]
        self.lib.mr_destroy.restype = None

        self.lib.mr_reset.argtypes = [ct.c_void_p, ct.c_uint64]
        self.lib.mr_reset.restype = ct.c_int
        self.lib.mr_step.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_float),
            ct.c_size_t,
        ]
        self.lib.mr_step.restype = ct.c_int

        for name in (
            "mr_environment_count",
            "mr_action_count",
            "mr_observation_count",
            "mr_link_count",
        ):
            function = getattr(self.lib, name)
            function.argtypes = [ct.c_void_p]
            function.restype = ct.c_uint32

        self.lib.mr_observations.argtypes = [ct.c_void_p]
        self.lib.mr_observations.restype = ct.POINTER(ct.c_float)
        self.lib.mr_rewards.argtypes = [ct.c_void_p]
        self.lib.mr_rewards.restype = ct.POINTER(ct.c_float)
        self.lib.mr_terminated.argtypes = [ct.c_void_p]
        self.lib.mr_terminated.restype = ct.POINTER(ct.c_uint8)
        self.lib.mr_body_positions.argtypes = [ct.c_void_p]
        self.lib.mr_body_positions.restype = ct.POINTER(ct.c_float)
        self.lib.mr_body_rotations.argtypes = [ct.c_void_p]
        self.lib.mr_body_rotations.restype = ct.POINTER(ct.c_float)

        self.lib.mr_stats.argtypes = [ct.c_void_p]
        self.lib.mr_stats.restype = _RuntimeStatsC
        self.lib.mr_device_name.argtypes = [ct.c_void_p]
        self.lib.mr_device_name.restype = ct.c_char_p

        self.lib.mr_create_franka_pick_place_world_family.argtypes = [
            ct.c_uint32,
            ct.c_char_p,
        ]
        self.lib.mr_create_franka_pick_place_world_family.restype = ct.c_void_p
        self.lib.mr_load_world_family_pack.argtypes = [
            ct.c_char_p,
            ct.c_uint32,
            ct.c_char_p,
        ]
        self.lib.mr_load_world_family_pack.restype = ct.c_void_p
        self.lib.mr_world_family_destroy.argtypes = [ct.c_void_p]
        self.lib.mr_world_family_destroy.restype = None
        self.lib.mr_world_family_sample.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
            ct.c_uint64,
        ]
        self.lib.mr_world_family_sample.restype = ct.c_int
        self.lib.mr_world_family_readback.argtypes = [ct.c_void_p]
        self.lib.mr_world_family_readback.restype = ct.c_int
        self.lib.mr_world_family_layout.argtypes = [ct.c_void_p]
        self.lib.mr_world_family_layout.restype = _WorldFamilyLayoutC
        self.lib.mr_world_family_stats.argtypes = [ct.c_void_p]
        self.lib.mr_world_family_stats.restype = _WorldFamilyStatsC
        self.lib.mr_world_family_device_name.argtypes = [ct.c_void_p]
        self.lib.mr_world_family_device_name.restype = ct.c_char_p
        self.lib.mr_world_family_native_buffer.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
        ]
        self.lib.mr_world_family_native_buffer.restype = ct.c_void_p
        for name in (
            "mr_world_family_instance_headers",
            "mr_world_family_asset_instances",
            "mr_world_family_sensor_instances",
            "mr_world_family_appearance_instances",
        ):
            function = getattr(self.lib, name)
            function.argtypes = [ct.c_void_p]
            function.restype = ct.c_void_p

    def last_error(self) -> str:
        return _decode(self.lib.mr_last_error()) or "unknown native error"


_binding_cache: dict[Path, _Bindings] = {}


def _load_bindings(path: str | os.PathLike[str] | None = None) -> _Bindings:
    resolved = resolve_library_path(path)
    bindings = _binding_cache.get(resolved)
    if bindings is None:
        bindings = _Bindings(resolved)
        _binding_cache[resolved] = bindings
    return bindings


def library_version(path: str | os.PathLike[str] | None = None) -> str:
    """Return the loaded native library version string."""

    return _decode(_load_bindings(path).lib.mr_version())


def _readonly_view(
    pointer: ct._Pointer,  # type: ignore[name-defined]
    shape: tuple[int, ...],
    *,
    dtype: npt.DTypeLike,
    label: str,
) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64))
    if size and not bool(pointer):
        raise MetalRoboError(f"Native runtime returned a null {label} pointer")
    view = np.ctypeslib.as_array(pointer, shape=(size,)).view(dtype)
    view = view.reshape(shape)
    view.setflags(write=False)
    return view


class NativeRuntime:
    """Own a batched Franka simulator and its zero-copy NumPy views.

    Returned arrays alias native shared memory. Their addresses remain stable
    until :meth:`close`, while their contents are updated in place on every
    reset and step. The runtime is intentionally not thread-safe.
    """

    def __init__(
        self,
        environment_count: int = 1024,
        *,
        seed: int = 1,
        library_path: str | os.PathLike[str] | None = None,
        metallib_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not 1 <= int(environment_count) <= np.iinfo(np.uint32).max:
            raise ValueError("environment_count must fit in a nonzero uint32")
        if not 0 <= int(seed) <= np.iinfo(np.uint64).max:
            raise ValueError("seed must fit in a uint64")

        self._bindings = _load_bindings(library_path)
        self.library_path = self._bindings.path
        self.metallib_path = resolve_metallib_path(
            metallib_path, library_path=self.library_path
        )
        encoded_metallib = (
            os.fsencode(self.metallib_path) if self.metallib_path is not None else None
        )
        self._handle = self._bindings.lib.mr_create_franka(
            ct.c_uint32(environment_count),
            ct.c_uint64(seed),
            encoded_metallib,
        )
        if not self._handle:
            raise MetalRoboError(
                f"Could not create Franka runtime: {self._bindings.last_error()}"
            )

        try:
            self.environment_count = int(
                self._bindings.lib.mr_environment_count(self._handle)
            )
            self.action_count = int(
                self._bindings.lib.mr_action_count(self._handle)
            )
            self.observation_count = int(
                self._bindings.lib.mr_observation_count(self._handle)
            )
            self.link_count = int(self._bindings.lib.mr_link_count(self._handle))
            if min(
                self.environment_count,
                self.action_count,
                self.observation_count,
                self.link_count,
            ) <= 0:
                raise MetalRoboError("Native runtime reported an invalid model shape")

            lib = self._bindings.lib
            self._observations = _readonly_view(
                lib.mr_observations(self._handle),
                (self.environment_count, self.observation_count),
                dtype=np.float32,
                label="observation",
            )
            self._rewards = _readonly_view(
                lib.mr_rewards(self._handle),
                (self.environment_count,),
                dtype=np.float32,
                label="reward",
            )
            self._terminated_u8 = _readonly_view(
                lib.mr_terminated(self._handle),
                (self.environment_count,),
                dtype=np.uint8,
                label="termination",
            )
            self._terminated = self._terminated_u8.view(np.bool_)
            self._terminated.setflags(write=False)
            self._body_positions = _readonly_view(
                lib.mr_body_positions(self._handle),
                (self.environment_count, self.link_count, 4),
                dtype=np.float32,
                label="body position",
            )
            self._body_rotations = _readonly_view(
                lib.mr_body_rotations(self._handle),
                (self.environment_count, self.link_count, 4),
                dtype=np.float32,
                label="body rotation",
            )
        except BaseException:
            self.close()
            raise

    def _require_open(self) -> ct.c_void_p:
        handle = getattr(self, "_handle", None)
        if not handle:
            raise MetalRoboError("NativeRuntime is closed")
        return handle

    @property
    def observations(self) -> FloatArray:
        self._require_open()
        return self._observations

    @property
    def rewards(self) -> FloatArray:
        self._require_open()
        return self._rewards

    @property
    def terminated(self) -> BoolArray:
        self._require_open()
        return self._terminated

    @property
    def body_positions(self) -> FloatArray:
        self._require_open()
        return self._body_positions

    @property
    def body_rotations(self) -> FloatArray:
        self._require_open()
        return self._body_rotations

    @property
    def version(self) -> str:
        return _decode(self._bindings.lib.mr_version())

    @property
    def device_name(self) -> str:
        handle = self._require_open()
        return _decode(self._bindings.lib.mr_device_name(handle))

    @property
    def stats(self) -> RuntimeStats:
        handle = self._require_open()
        stats = self._bindings.lib.mr_stats(handle)
        return RuntimeStats(
            last_gpu_milliseconds=float(stats.last_gpu_milliseconds),
            total_gpu_milliseconds=float(stats.total_gpu_milliseconds),
            control_steps=int(stats.control_steps),
            physics_steps=int(stats.physics_steps),
        )

    def reset(self, seed: int = 1) -> FloatArray:
        """Reset every environment and return the live observation view."""

        if not 0 <= int(seed) <= np.iinfo(np.uint64).max:
            raise ValueError("seed must fit in a uint64")
        handle = self._require_open()
        result = self._bindings.lib.mr_reset(handle, ct.c_uint64(seed))
        if result != 0:
            raise MetalRoboError(f"Reset failed: {self._bindings.last_error()}")
        return self._observations

    def step(self, actions: npt.ArrayLike) -> FloatArray:
        """Advance one control step using an ``(envs, actions)`` float array."""

        handle = self._require_open()
        contiguous = np.ascontiguousarray(actions, dtype=np.float32)
        expected = self.environment_count * self.action_count
        if contiguous.size != expected:
            raise ValueError(
                f"Expected {expected} actions shaped "
                f"({self.environment_count}, {self.action_count}), got "
                f"{contiguous.shape} ({contiguous.size} values)"
            )
        contiguous = contiguous.reshape(self.environment_count, self.action_count)
        if not np.isfinite(contiguous).all():
            raise ValueError("actions must contain only finite values")
        result = self._bindings.lib.mr_step(
            handle,
            contiguous.ctypes.data_as(ct.POINTER(ct.c_float)),
            ct.c_size_t(expected),
        )
        if result != 0:
            raise MetalRoboError(f"Step failed: {self._bindings.last_error()}")
        return self._observations

    def close(self) -> None:
        """Destroy the native runtime. Existing NumPy views become invalid."""

        handle = getattr(self, "_handle", None)
        if handle:
            self._handle = None
            self._bindings.lib.mr_destroy(handle)

    def __enter__(self) -> NativeRuntime:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
