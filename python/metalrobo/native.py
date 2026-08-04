"""Thin ctypes binding for the stable MetalRobo C ABI."""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt

_LIBRARY_ENV: Final = "METALROBO_LIBRARY"
_METALLIB_ENV: Final = "METALROBO_METALLIB"


class MetalRoboError(RuntimeError):
    """Raised when the native MetalRobo runtime reports an error."""


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


class _ScenarioFeatureC(ct.Structure):
    _fields_ = [
        ("axis", ct.c_uint32),
        ("distribution", ct.c_uint32),
        ("target", ct.c_uint32),
        ("ordinal", ct.c_uint32),
        ("parameters", ct.c_float * 4),
    ]


class _PolicyDenseLayerC(ct.Structure):
    _fields_ = [
        ("input_count", ct.c_uint32),
        ("output_count", ct.c_uint32),
        ("activation", ct.c_uint32),
        ("weights", ct.POINTER(ct.c_float)),
        ("weight_count", ct.c_size_t),
        ("bias", ct.POINTER(ct.c_float)),
        ("bias_count", ct.c_size_t),
    ]


class _PolicyPackC(ct.Structure):
    _fields_ = [
        ("id", ct.c_char_p),
        ("revision", ct.c_uint64),
        ("contract_version", ct.c_uint64),
        ("world_fingerprint", ct.c_uint64),
        ("task_fingerprint", ct.c_uint64),
        ("observation_fingerprint", ct.c_uint64),
        ("action_fingerprint", ct.c_uint64),
        ("observation_mean", ct.POINTER(ct.c_float)),
        ("observation_mean_count", ct.c_size_t),
        (
            "observation_inverse_standard_deviation",
            ct.POINTER(ct.c_float),
        ),
        (
            "observation_inverse_standard_deviation_count",
            ct.c_size_t,
        ),
        ("layers", ct.POINTER(_PolicyDenseLayerC)),
        ("layer_count", ct.c_size_t),
        ("critic_observation_mean", ct.POINTER(ct.c_float)),
        ("critic_observation_mean_count", ct.c_size_t),
        (
            "critic_observation_inverse_standard_deviation",
            ct.POINTER(ct.c_float),
        ),
        (
            "critic_observation_inverse_standard_deviation_count",
            ct.c_size_t,
        ),
        ("critic_layers", ct.POINTER(_PolicyDenseLayerC)),
        ("critic_layer_count", ct.c_size_t),
        (
            "action_log_standard_deviation",
            ct.POINTER(ct.c_float),
        ),
        ("action_log_standard_deviation_count", ct.c_size_t),
        ("action_bias", ct.POINTER(ct.c_float)),
        ("action_bias_count", ct.c_size_t),
        ("action_scale", ct.POINTER(ct.c_float)),
        ("action_scale_count", ct.c_size_t),
        ("observation_clip", ct.c_float),
        ("action_clip", ct.c_float),
    ]


@dataclass(frozen=True, slots=True)
class PolicyDenseLayerArtifact:
    """One output-major dense layer for the generic native PolicyPack."""

    weights: npt.ArrayLike
    bias: npt.ArrayLike
    activation: int


class _HybridRendererLayoutC(ct.Structure):
    _fields_ = [
        ("capacity", ct.c_uint32),
        ("active_environment_count", ct.c_uint32),
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("tile_count_x", ct.c_uint32),
        ("tile_count_y", ct.c_uint32),
        ("gaussian_count", ct.c_uint32),
        ("maximum_gaussians_per_tile", ct.c_uint32),
        ("maximum_mesh_triangles_per_tile", ct.c_uint32),
        ("mesh_vertex_count", ct.c_uint32),
        ("mesh_triangle_count", ct.c_uint32),
        ("mesh_cluster_count", ct.c_uint32),
        ("mesh_primitive_count", ct.c_uint32),
        ("mesh_instance_count", ct.c_uint32),
        ("mesh_index_count", ct.c_uint32),
        ("material_count", ct.c_uint32),
        ("texture_count", ct.c_uint32),
        ("light_count", ct.c_uint32),
        ("body_count", ct.c_uint32),
        ("sensor_binding_count", ct.c_uint32),
        ("shadow_layer_capacity", ct.c_uint32),
        ("ray_instance_count", ct.c_uint32),
        ("shadow_workspace_bytes", ct.c_size_t),
        ("acceleration_structure_bytes", ct.c_size_t),
        ("retained_private_bytes", ct.c_size_t),
        ("last_render_milliseconds", ct.c_double),
    ]

class _VisualFrameMetadataC(ct.Structure):
    _fields_ = [
        ("dimensions", ct.c_uint32 * 4),
        ("identity", ct.c_uint32 * 4),
        ("timing", ct.c_float * 4),
        ("contract", ct.c_uint32 * 4),
    ]

class _TactileLayoutC(ct.Structure):
    _fields_ = [
        ("capacity", ct.c_uint32),
        ("active_environment_count", ct.c_uint32),
        ("body_count", ct.c_uint32),
        ("shape_count", ct.c_uint32),
        ("sensor_count", ct.c_uint32),
        ("sample_count", ct.c_uint32),
        ("target_count", ct.c_uint32),
        ("contact_capacity_per_environment", ct.c_uint32),
        ("query_backend", ct.c_uint32),
        ("hardware_ray_queries_available", ct.c_uint32),
        ("retained_bytes", ct.c_size_t),
        ("bytes_per_environment", ct.c_size_t),
        ("last_observe_milliseconds", ct.c_double),
    ]


class _TactileSummaryC(ct.Structure):
    _fields_ = [
        ("pose_position_and_timestamp", ct.c_float * 4),
        ("pose_orientation", ct.c_float * 4),
        ("net_force_and_contact_area", ct.c_float * 4),
        ("net_torque_and_maximum_depth", ct.c_float * 4),
        ("centroid_local_and_mean_depth", ct.c_float * 4),
        ("centroid_world_and_active_count", ct.c_float * 4),
        (
            "center_of_pressure_local_and_force_weight",
            ct.c_float * 4,
        ),
        (
            "center_of_pressure_world_and_contact_count",
            ct.c_float * 4,
        ),
        ("tangential_motion_and_friction", ct.c_float * 4),
        ("statistics_and_identity", ct.c_uint32 * 4),
    ]


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
        self.lib.mr_unitree_g1_deployment_contract_json.argtypes = []
        self.lib.mr_unitree_g1_deployment_contract_json.restype = (
            ct.c_char_p
        )
        self.lib.mr_write_policy_pack.argtypes = [
            ct.POINTER(_PolicyPackC),
            ct.c_char_p,
        ]
        self.lib.mr_write_policy_pack.restype = ct.c_int
        self.lib.mr_learning_pack_content_hash.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
        ]
        self.lib.mr_learning_pack_content_hash.restype = ct.c_uint64

        self.lib.mr_compile_episode_manifest.argtypes = [
            ct.c_char_p,
            ct.c_char_p,
            ct.c_char_p,
        ]
        self.lib.mr_compile_episode_manifest.restype = ct.c_int

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
        self.lib.mr_world_family_sample_ex.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
            ct.c_uint64,
            ct.c_uint32,
            ct.c_uint64,
        ]
        self.lib.mr_world_family_sample_ex.restype = ct.c_int
        self.lib.mr_world_family_configure_sampling.argtypes = [
            ct.c_void_p,
            ct.c_uint64,
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.c_uint64,
            ct.POINTER(ct.c_uint32),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float),
            ct.c_uint32,
            ct.c_float,
            ct.c_float,
            ct.c_float,
            ct.c_float,
        ]
        self.lib.mr_world_family_configure_sampling.restype = ct.c_int
        self.lib.mr_world_family_scenario_fingerprint.argtypes = [
            ct.c_void_p
        ]
        self.lib.mr_world_family_scenario_fingerprint.restype = (
            ct.c_uint64
        )
        self.lib.mr_world_family_authored_pack_hash.argtypes = [
            ct.c_void_p
        ]
        self.lib.mr_world_family_authored_pack_hash.restype = (
            ct.c_uint64
        )
        self.lib.mr_world_family_scenario_id.argtypes = [ct.c_void_p]
        self.lib.mr_world_family_scenario_id.restype = ct.c_char_p
        self.lib.mr_world_family_scenario_feature_id.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
        ]
        self.lib.mr_world_family_scenario_feature_id.restype = (
            ct.c_char_p
        )
        self.lib.mr_world_family_scenario_target_id.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
        ]
        self.lib.mr_world_family_scenario_target_id.restype = (
            ct.c_char_p
        )
        self.lib.mr_world_family_scenario_feature.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
        ]
        self.lib.mr_world_family_scenario_feature.restype = (
            _ScenarioFeatureC
        )
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
            "mr_world_family_scenario_headers",
            "mr_world_family_scenario_values",
        ):
            function = getattr(self.lib, name)
            function.argtypes = [ct.c_void_p]
            function.restype = ct.c_void_p

        self.lib.mr_hybrid_renderer_create_v3.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_char_p,
            ct.c_char_p,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_char_p,
            ct.c_char_p,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_char_p,
        ]
        self.lib.mr_hybrid_renderer_create_v3.restype = ct.c_void_p
        self.lib.mr_hybrid_renderer_destroy.argtypes = [ct.c_void_p]
        self.lib.mr_hybrid_renderer_destroy.restype = None
        self.lib.mr_hybrid_renderer_render.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_uint32,
            ct.c_uint32,
        ]
        self.lib.mr_hybrid_renderer_render.restype = ct.c_int
        self.lib.mr_hybrid_renderer_readback.argtypes = [ct.c_void_p]
        self.lib.mr_hybrid_renderer_readback.restype = ct.c_int
        self.lib.mr_hybrid_renderer_layout.argtypes = [ct.c_void_p]
        self.lib.mr_hybrid_renderer_layout.restype = (
            _HybridRendererLayoutC
        )
        self.lib.mr_hybrid_renderer_device_name.argtypes = [
            ct.c_void_p
        ]
        self.lib.mr_hybrid_renderer_device_name.restype = ct.c_char_p
        self.lib.mr_hybrid_renderer_native_buffer.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
        ]
        self.lib.mr_hybrid_renderer_native_buffer.restype = ct.c_void_p
        self.lib.mr_hybrid_renderer_rgb.argtypes = [ct.c_void_p]
        self.lib.mr_hybrid_renderer_rgb.restype = ct.POINTER(ct.c_float)
        self.lib.mr_hybrid_renderer_depth.argtypes = [ct.c_void_p]
        self.lib.mr_hybrid_renderer_depth.restype = ct.POINTER(ct.c_float)
        self.lib.mr_hybrid_renderer_segmentation.argtypes = [
            ct.c_void_p
        ]
        self.lib.mr_hybrid_renderer_segmentation.restype = ct.POINTER(
            ct.c_uint32
        )
        self.lib.mr_hybrid_renderer_identities.argtypes = [
            ct.c_void_p
        ]
        self.lib.mr_hybrid_renderer_identities.restype = ct.POINTER(
            ct.c_uint32
        )
        self.lib.mr_hybrid_renderer_normals.argtypes = [ct.c_void_p]
        self.lib.mr_hybrid_renderer_normals.restype = ct.POINTER(
            ct.c_float
        )
        self.lib.mr_hybrid_renderer_motion.argtypes = [ct.c_void_p]
        self.lib.mr_hybrid_renderer_motion.restype = ct.POINTER(
            ct.c_float
        )
        self.lib.mr_hybrid_renderer_validity.argtypes = [
            ct.c_void_p
        ]
        self.lib.mr_hybrid_renderer_validity.restype = ct.POINTER(
            ct.c_uint32
        )
        self.lib.mr_hybrid_renderer_frame_metadata.argtypes = [
            ct.c_void_p
        ]
        self.lib.mr_hybrid_renderer_frame_metadata.restype = (
            _VisualFrameMetadataC
        )

        self.lib.mr_tactile_create_world_pack.argtypes = [
            ct.c_char_p,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_char_p,
        ]
        self.lib.mr_tactile_create_world_pack.restype = ct.c_void_p
        self.lib.mr_tactile_destroy.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_destroy.restype = None
        self.lib.mr_tactile_encode.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_float,
            ct.c_float,
            ct.c_uint64,
            ct.c_double,
            ct.c_void_p,
        ]
        self.lib.mr_tactile_encode.restype = ct.c_int
        self.lib.mr_tactile_readback.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_readback.restype = ct.c_int
        self.lib.mr_tactile_layout.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_layout.restype = _TactileLayoutC
        self.lib.mr_tactile_device_name.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_device_name.restype = ct.c_char_p
        self.lib.mr_tactile_observation_metadata_json.argtypes = [
            ct.c_void_p
        ]
        self.lib.mr_tactile_observation_metadata_json.restype = (
            ct.c_char_p
        )
        self.lib.mr_tactile_native_buffer.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
        ]
        self.lib.mr_tactile_native_buffer.restype = ct.c_void_p
        self.lib.mr_tactile_depth.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_depth.restype = ct.POINTER(ct.c_float)
        self.lib.mr_tactile_depth_velocity.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_depth_velocity.restype = ct.POINTER(
            ct.c_float
        )
        self.lib.mr_tactile_tangential_motion.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_tangential_motion.restype = ct.POINTER(
            ct.c_float
        )
        self.lib.mr_tactile_validity.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_validity.restype = ct.POINTER(ct.c_uint32)
        self.lib.mr_tactile_object_shape_ids.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_object_shape_ids.restype = ct.POINTER(
            ct.c_uint32
        )
        self.lib.mr_tactile_summaries.argtypes = [ct.c_void_p]
        self.lib.mr_tactile_summaries.restype = ct.POINTER(
            _TactileSummaryC
        )

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


def learning_pack_content_hash(
    payload: npt.ArrayLike,
    path: str | os.PathLike[str] | None = None,
) -> int:
    """Hash one contiguous mapped learning-pack payload through native C++."""

    values = np.asarray(payload)
    if values.dtype != np.uint8 or values.ndim != 1:
        raise ValueError(
            "learning-pack payload must be a one-dimensional uint8 array"
        )
    if not values.flags.c_contiguous:
        raise ValueError("learning-pack payload must be contiguous")
    pointer = (
        values.ctypes.data_as(ct.c_void_p)
        if values.size
        else None
    )
    result = int(
        _load_bindings(path).lib.mr_learning_pack_content_hash(
            pointer,
            values.nbytes,
        )
    )
    if result == 0:
        raise MetalRoboError("native learning-pack hash failed")
    return result


def unitree_g1_deployment_contract(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Return the native bundled G1 deployment/mechanics contract."""

    bindings = _load_bindings(path)
    encoded = (
        bindings.lib.mr_unitree_g1_deployment_contract_json()
    )
    if not encoded:
        raise MetalRoboError(
            "G1 deployment contract failed: "
            + bindings.last_error()
        )
    record = json.loads(_decode(encoded))
    if (
        not isinstance(record, dict)
        or record.get("format")
            != "metalrobo.unitree-g1-deployment"
        or record.get("schema") != 1
    ):
        raise MetalRoboError(
            "Native G1 deployment contract is malformed"
        )
    return record


def write_policy_pack(
    output: str | os.PathLike[str],
    *,
    policy_id: str,
    revision: int,
    contract_version: int,
    world_fingerprint: int,
    task_fingerprint: int,
    observation_fingerprint: int,
    action_fingerprint: int,
    layers: Sequence[PolicyDenseLayerArtifact],
    critic_layers: Sequence[PolicyDenseLayerArtifact] = (),
    observation_mean: npt.ArrayLike = (),
    observation_inverse_standard_deviation: npt.ArrayLike = (),
    critic_observation_mean: npt.ArrayLike = (),
    critic_observation_inverse_standard_deviation:
        npt.ArrayLike = (),
    action_log_standard_deviation: npt.ArrayLike = (),
    action_bias: npt.ArrayLike = (),
    action_scale: npt.ArrayLike = (),
    observation_clip: float = 100.0,
    action_clip: float = float(np.finfo(np.float32).max),
    library_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Publish actor/critic weights through the canonical native writer."""

    if not policy_id or not 0 < int(revision) <= np.iinfo(np.uint64).max:
        raise ValueError("policy_id and a nonzero uint64 revision are required")
    if not layers:
        raise ValueError("at least one policy dense layer is required")

    retained: list[np.ndarray] = []

    def values(source: npt.ArrayLike, label: str) -> np.ndarray:
        array = np.ascontiguousarray(source, dtype=np.float32).reshape(-1)
        if not np.isfinite(array).all():
            raise ValueError(f"{label} must contain only finite values")
        retained.append(array)
        return array

    mean = values(observation_mean, "observation_mean")
    inverse_std = values(
        observation_inverse_standard_deviation,
        "observation_inverse_standard_deviation",
    )
    critic_mean = values(
        critic_observation_mean,
        "critic_observation_mean",
    )
    critic_inverse_std = values(
        critic_observation_inverse_standard_deviation,
        "critic_observation_inverse_standard_deviation",
    )
    log_standard_deviation = values(
        action_log_standard_deviation,
        "action_log_standard_deviation",
    )
    bias = values(action_bias, "action_bias")
    scale = values(action_scale, "action_scale")
    def native_layer_table(
        source: Sequence[PolicyDenseLayerArtifact],
        label: str,
    ) -> ct.Array:
        native = (_PolicyDenseLayerC * len(source))()
        for index, layer in enumerate(source):
            weights = np.ascontiguousarray(
                layer.weights,
                dtype=np.float32,
            )
            layer_bias = values(
                layer.bias,
                f"{label}[{index}].bias",
            )
            if (
                weights.ndim != 2
                or weights.shape[0] != layer_bias.size
            ):
                raise ValueError(
                    f"{label}[{index}] weights must be [output, input] "
                    "with one bias per output"
                )
            if not np.isfinite(weights).all():
                raise ValueError(
                    f"{label}[{index}].weights must contain only finite values"
                )
            retained.append(weights)
            if not 0 <= int(layer.activation) <= 4:
                raise ValueError(
                    f"{label}[{index}] activation is unsupported"
                )
            native[index] = _PolicyDenseLayerC(
                input_count=weights.shape[1],
                output_count=weights.shape[0],
                activation=int(layer.activation),
                weights=weights.ctypes.data_as(
                    ct.POINTER(ct.c_float)
                ),
                weight_count=weights.size,
                bias=layer_bias.ctypes.data_as(
                    ct.POINTER(ct.c_float)
                ),
                bias_count=layer_bias.size,
            )
        return native

    native_layers = native_layer_table(layers, "layers")
    native_critic_layers = native_layer_table(
        critic_layers,
        "critic_layers",
    )

    def pointer(array: np.ndarray) -> ct.POINTER(ct.c_float) | None:
        return (
            array.ctypes.data_as(ct.POINTER(ct.c_float))
            if array.size
            else None
        )

    encoded_id = policy_id.encode("utf-8")
    native = _PolicyPackC(
        id=encoded_id,
        revision=int(revision),
        contract_version=int(contract_version),
        world_fingerprint=int(world_fingerprint),
        task_fingerprint=int(task_fingerprint),
        observation_fingerprint=int(observation_fingerprint),
        action_fingerprint=int(action_fingerprint),
        observation_mean=pointer(mean),
        observation_mean_count=mean.size,
        observation_inverse_standard_deviation=pointer(inverse_std),
        observation_inverse_standard_deviation_count=inverse_std.size,
        layers=native_layers,
        layer_count=len(layers),
        critic_observation_mean=pointer(critic_mean),
        critic_observation_mean_count=critic_mean.size,
        critic_observation_inverse_standard_deviation=pointer(
            critic_inverse_std
        ),
        critic_observation_inverse_standard_deviation_count=(
            critic_inverse_std.size
        ),
        critic_layers=native_critic_layers,
        critic_layer_count=len(critic_layers),
        action_log_standard_deviation=pointer(
            log_standard_deviation
        ),
        action_log_standard_deviation_count=(
            log_standard_deviation.size
        ),
        action_bias=pointer(bias),
        action_bias_count=bias.size,
        action_scale=pointer(scale),
        action_scale_count=scale.size,
        observation_clip=float(observation_clip),
        action_clip=float(action_clip),
    )
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    bindings = _load_bindings(library_path)
    status = bindings.lib.mr_write_policy_pack(
        ct.byref(native),
        os.fsencode(target),
    )
    if status != 0:
        raise MetalRoboError(
            f"PolicyPack write failed: {bindings.last_error()}"
        )
    return target
