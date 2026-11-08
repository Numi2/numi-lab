"""Immutable PointWorld artifacts and advisory point-flow planning.

This module deliberately owns no simulator state.  It provides the strict
capture/candidate/forecast boundary shared by the CUDA oracle and the native
Metal executor: PointWorld consumes metric RGB-D and *surface point flows*,
never raw robot actions, and publishes only predicted scene motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_PACK_FORMAT = "numi.pointworld-model-pack.v1"
OBSERVATION_FORMAT = "numi.pointworld-observation.v1"
CANDIDATES_FORMAT = "numi.pointworld-robot-flow-candidates.v1"
FORECAST_FORMAT = "numi.pointworld-forecast.v1"
REFERENCE_FORMAT = "numi.pointworld-cuda-reference.v1"
POINTWORLD_SOURCE_REVISION = "05484826dfef74cbe278a3974179a5a16705d35d"
POINTWORLD_MODEL_REPOSITORY = "nvidia/PointWorld_models"
POINTWORLD_MODEL_REVISION = "b9e2e19a4f2bd65922e1f6d70aa953fe70aa9dba"
POINTWORLD_CHECKPOINT = "large-droid+behavior/model-best.pt"
POINTWORLD_CHECKPOINT_SHA256 = "bb7a5b0d717b79363c75751531c2416b099b8fca28b2f799ff49ace7123af787"
DINO_REVISION = "54694f7627fd815f62a5dcc82944ffa6153bbb76"
DROID_BEHAVIOR_STATS_SHA256 = "9bc88f9ad1402662b6ee24023b022ec515770eeae6aa2a150cfe2b049e0a5f54"
PTV3_BLUEPRINT_SHA256 = "58a5022895a639409b7f2659cdd95348bc7f3eeed4dac626603011093588f8ec"
CHECKPOINT_CONTRACT_SHA256 = "d87286cd119d5dcd5cc60b9697920ad1d5df0ade6d24377de4104fc43e7cdb97"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
CONTEXT_FRAMES = 1
PREDICTION_FRAMES = 10
MAX_SCENE_POINTS = 12000
MAX_ROBOT_POINTS = 500
GRID_SIZE_M = 0.015
DEPTH_THRESHOLD_M = 0.003
VAR_FLOOR = 1e-6
VAR_CEILING = 1e2
SIM_VAR_CONST = 1e-3


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype=np.uint64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"artifact root must be an object: {path}")
    return raw


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return value


def _require_commit(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{name} must be a resolved 40-character commit")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a resolved 40-character commit") from error
    return value


def _finite_array(value: Any, dtype: np.dtype[Any], name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is missing")
    result = np.ascontiguousarray(value, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid NPZ artifact: {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return _sha256_path(path)


def _load_bound_npz(path: Path, manifest: Mapping[str, Any], expected_format: str) -> dict[str, np.ndarray]:
    if manifest.get("format") != expected_format:
        raise ValueError(f"artifact manifest format must be {expected_format}")
    expected_sha = _require_digest(manifest.get("sha256"), "manifest.sha256")
    actual_sha = _sha256_path(path)
    if actual_sha != expected_sha:
        raise ValueError(f"artifact hash mismatch: expected {expected_sha}, found {actual_sha}")
    return _load_npz(path)


@dataclass(frozen=True)
class PointWorldModelPack:
    manifest: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest)).hexdigest()


def validate_model_pack(manifest: Mapping[str, Any]) -> PointWorldModelPack:
    if manifest.get("format") != MODEL_PACK_FORMAT:
        raise ValueError("PointWorld model pack format is invalid")
    source = manifest.get("source")
    checkpoint = manifest.get("checkpoint")
    dino = manifest.get("dino")
    release_assets = manifest.get("release_assets")
    architecture = manifest.get("architecture")
    license_receipts = manifest.get("license_receipts")
    contract = manifest.get("contract")
    if not all(isinstance(value, dict) for value in (source, checkpoint, dino, release_assets, architecture, license_receipts, contract)):
        raise ValueError("PointWorld model pack is missing a required section")
    if source != {
        "repository": "NVlabs/PointWorld",
        "revision": POINTWORLD_SOURCE_REVISION,
        "license": "Apache-2.0",
    }:
        raise ValueError("PointWorld source receipt is not the pinned release")
    if checkpoint.get("repository") != POINTWORLD_MODEL_REPOSITORY:
        raise ValueError("PointWorld checkpoint repository is invalid")
    if checkpoint.get("revision") != POINTWORLD_MODEL_REVISION:
        raise ValueError("PointWorld checkpoint revision is not the pinned release")
    if checkpoint.get("path") != POINTWORLD_CHECKPOINT:
        raise ValueError("full parity requires the large DROID+BEHAVIOR checkpoint")
    if checkpoint.get("license") != "nvidia-open-model-license":
        raise ValueError("PointWorld checkpoint license receipt is invalid")
    if checkpoint.get("sha256") != POINTWORLD_CHECKPOINT_SHA256:
        raise ValueError("PointWorld checkpoint hash is not the released large DROID+BEHAVIOR checkpoint")
    if dino.get("revision") != DINO_REVISION or not str(dino.get("license", "")).strip():
        raise ValueError("DINO receipt is invalid")
    _require_digest(dino.get("weights_sha256"), "dino.weights_sha256")
    expected_assets = {
        "normalization_path": "stats/droid_behavior/norm_stats.json",
        "normalization_sha256": DROID_BEHAVIOR_STATS_SHA256,
        "ptv3_blueprint_path": "ptv3/ptv3_arch.yaml",
        "ptv3_blueprint_sha256": PTV3_BLUEPRINT_SHA256,
        "checkpoint_contract_path": "pointworld/checkpoint_contract.py",
        "checkpoint_contract_sha256": CHECKPOINT_CONTRACT_SHA256,
    }
    if release_assets != expected_assets:
        raise ValueError("PointWorld release assets do not match the pinned source")
    expected_architecture = {
        "scene_encoder": "dinov3_vitl16",
        "scene_encoder_layers": [4, 11, 17, 23],
        "ptv3_size": "large",
        "ptv3_patch_size": 256,
        "predictor_dim": 256,
        "grid_size_m": GRID_SIZE_M,
        "depth_threshold_m": DEPTH_THRESHOLD_M,
        "robot_features": ["robot_flows", "robot_colors", "robot_normals", "gripper_open", "robot_velocity", "robot_acceleration"],
        "scene_features": ["scene_flows", "scene_colors", "scene_normals", "gripper_open", "dist2robot"],
    }
    if architecture != expected_architecture:
        raise ValueError("PointWorld architecture contract differs from the pinned large release")
    if license_receipts != {
        "nvidia_open_model_license_accepted": True,
        "dinov3_access_granted": True,
        "dinov3_license_accepted": True,
    }:
        raise ValueError("PointWorld conversion requires explicit NVIDIA and DINOv3 access/license receipts")
    expected_contract = {
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "context_frames": CONTEXT_FRAMES,
        "prediction_frames": PREDICTION_FRAMES,
        "max_scene_points": MAX_SCENE_POINTS,
        "max_robot_points": MAX_ROBOT_POINTS,
        "domains": ["droid", "behavior"],
    }
    if contract != expected_contract:
        raise ValueError("PointWorld tensor contract differs from the pinned release")
    return PointWorldModelPack(dict(manifest))


def convert_model_pack(
    *, checkpoint: Path, checkpoint_revision: str, dino_weights: Path,
    dino_license: str, normalization_stats: Path, ptv3_blueprint: Path,
    checkpoint_contract: Path, accept_nvidia_license: bool,
    dino_access_granted: bool, accept_dino_license: bool, output: Path,
) -> dict[str, Any]:
    _require_commit(checkpoint_revision, "checkpoint revision")
    required_files = (checkpoint, dino_weights, normalization_stats, ptv3_blueprint, checkpoint_contract)
    if not all(path.is_file() for path in required_files):
        raise ValueError("checkpoint, DINO weights, normalization stats, PTv3 blueprint, and checkpoint contract must be regular files")
    if checkpoint_revision != POINTWORLD_MODEL_REVISION or _sha256_path(checkpoint) != POINTWORLD_CHECKPOINT_SHA256:
        raise ValueError("checkpoint is not the pinned large DROID+BEHAVIOR release")
    if not (accept_nvidia_license and dino_access_granted and accept_dino_license):
        raise ValueError("conversion requires NVIDIA license acceptance plus DINOv3 access and license acceptance")
    expected_release_files = {
        normalization_stats: DROID_BEHAVIOR_STATS_SHA256,
        ptv3_blueprint: PTV3_BLUEPRINT_SHA256,
        checkpoint_contract: CHECKPOINT_CONTRACT_SHA256,
    }
    for path, expected_sha in expected_release_files.items():
        if _sha256_path(path) != expected_sha:
            raise ValueError(f"release asset hash mismatch: {path}")
    manifest = {
        "format": MODEL_PACK_FORMAT,
        "id": "pointworld-large-droid-behavior-v1",
        "source": {"repository": "NVlabs/PointWorld", "revision": POINTWORLD_SOURCE_REVISION, "license": "Apache-2.0"},
        "checkpoint": {
            "repository": POINTWORLD_MODEL_REPOSITORY, "revision": POINTWORLD_MODEL_REVISION,
            "path": POINTWORLD_CHECKPOINT, "sha256": POINTWORLD_CHECKPOINT_SHA256,
            "license": "nvidia-open-model-license",
        },
        "dino": {"revision": DINO_REVISION, "weights_sha256": _sha256_path(dino_weights), "license": dino_license},
        "release_assets": {
            "normalization_path": "stats/droid_behavior/norm_stats.json",
            "normalization_sha256": DROID_BEHAVIOR_STATS_SHA256,
            "ptv3_blueprint_path": "ptv3/ptv3_arch.yaml",
            "ptv3_blueprint_sha256": PTV3_BLUEPRINT_SHA256,
            "checkpoint_contract_path": "pointworld/checkpoint_contract.py",
            "checkpoint_contract_sha256": CHECKPOINT_CONTRACT_SHA256,
        },
        "architecture": {
            "scene_encoder": "dinov3_vitl16", "scene_encoder_layers": [4, 11, 17, 23],
            "ptv3_size": "large", "ptv3_patch_size": 256, "predictor_dim": 256,
            "grid_size_m": GRID_SIZE_M, "depth_threshold_m": DEPTH_THRESHOLD_M,
            "robot_features": ["robot_flows", "robot_colors", "robot_normals", "gripper_open", "robot_velocity", "robot_acceleration"],
            "scene_features": ["scene_flows", "scene_colors", "scene_normals", "gripper_open", "dist2robot"],
        },
        "license_receipts": {
            "nvidia_open_model_license_accepted": True,
            "dinov3_access_granted": True,
            "dinov3_license_accepted": True,
        },
        "contract": {
            "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
            "context_frames": CONTEXT_FRAMES, "prediction_frames": PREDICTION_FRAMES,
            "max_scene_points": MAX_SCENE_POINTS, "max_robot_points": MAX_ROBOT_POINTS,
            "domains": ["droid", "behavior"],
        },
    }
    pack = validate_model_pack(manifest)
    _write_json(output, manifest)
    return {"model_pack": str(output), "fingerprint": pack.fingerprint, "sha256": _sha256_path(output)}


def fetch_release_file(*, repository: str, revision: str, path: str, output: Path) -> dict[str, Any]:
    """Fetch an explicitly requested, immutable release file into a caller path.

    Authentication and license acceptance remain Hugging Face account policy;
    this command neither supplies a token nor bypasses a gated repository.
    """
    if repository not in {"NVlabs/PointWorld", POINTWORLD_MODEL_REPOSITORY}:
        raise ValueError("repository is not an approved PointWorld release source")
    if len(revision) != 40:
        raise ValueError("revision must be a resolved 40-character commit")
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("release path must be a relative repository path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    if temporary.exists():
        raise ValueError(f"stale temporary download exists: {temporary}")
    if repository == "NVlabs/PointWorld":
        if revision != POINTWORLD_SOURCE_REVISION:
            raise ValueError("PointWorld source fetch requires the pinned source revision")
        url = f"https://raw.githubusercontent.com/NVlabs/PointWorld/{revision}/{path}"
        urllib.request.urlretrieve(url, temporary)
    else:
        if revision != POINTWORLD_MODEL_REVISION:
            raise ValueError("PointWorld model fetch requires the pinned model revision")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required for PointWorld model fetch") from error
        downloaded = Path(hf_hub_download(
            repo_id=repository, revision=revision, filename=path,
            token=os.environ.get("HF_TOKEN"),
        ))
        shutil.copyfile(downloaded, temporary)
    downloaded_sha = _sha256_path(temporary)
    if output.exists():
        if _sha256_path(output) != downloaded_sha:
            temporary.unlink()
            raise ValueError(f"refusing to overwrite different existing file: {output}")
        temporary.unlink()
    else:
        temporary.replace(output)
    return {
        "repository": repository,
        "revision": revision,
        "path": path,
        "output": str(output),
        "sha256": downloaded_sha,
    }


def _backproject(depth: np.ndarray, validity: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera intrinsic focal length must be positive")
    z = depth.reshape(-1)
    valid = validity.reshape(-1) & (z > 0.0)
    camera = np.stack(((xx.reshape(-1) - cx) * z / fx, (yy.reshape(-1) - cy) * z / fy, z, np.ones_like(z)), axis=1)
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    # ``einsum`` avoids the Accelerate wide-matrix diagnostic emitted by
    # ``4x4 @ 4x57600`` while preserving the camera-to-world convention.
    world = np.einsum("ij,nj->ni", rotation, camera[:, :3]) + translation
    if not np.isfinite(world[valid]).all():
        raise ValueError("camera calibration produced non-finite world points")
    return world[valid].astype(np.float32), valid


def _depth_normals(depth: np.ndarray, validity: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points, valid = _backproject(depth, validity, intrinsic, np.eye(4, dtype=np.float32))
    normals = np.zeros((depth.size, 3), dtype=np.float32)
    # The dense normal operator must be metric and deterministic.  Border/invalid
    # pixels remain zero; the accompanying validity mask prevents fabricated data.
    xyz = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
    xyz.reshape(-1, 3)[valid] = points
    dx = xyz[:, 2:] - xyz[:, :-2]
    dy = xyz[2:, :] - xyz[:-2, :]
    cross = np.cross(dx[1:-1], dy[:, 1:-1])
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    neighborhood_valid = (
        validity[1:-1, 1:-1] & validity[1:-1, :-2] & validity[1:-1, 2:]
        & validity[:-2, 1:-1] & validity[2:, 1:-1]
    )
    finite = (norm[..., 0] > 1e-8) & neighborhood_valid
    cross[finite] /= norm[finite]
    cross[~finite] = 0.0
    normals.reshape(depth.shape[0], depth.shape[1], 3)[1:-1, 1:-1] = cross
    return normals


def _release_voxel_selection(points: np.ndarray, frame_fingerprint: str) -> np.ndarray:
    """Mirror release test-time 1.5 cm voxel selection, then cap deterministically."""
    grid = np.floor(points / GRID_SIZE_M).astype(np.int64)
    grid -= grid.min(axis=0)
    # Lexicographic ordering groups identical voxels without relying on Python's
    # salted hash. The first source point in each voxel matches release test mode.
    order = np.lexsort((np.arange(len(grid), dtype=np.int64), grid[:, 2], grid[:, 1], grid[:, 0]))
    ordered_grid = grid[order]
    first = np.ones(len(order), dtype=np.bool_)
    first[1:] = np.any(ordered_grid[1:] != ordered_grid[:-1], axis=1)
    selected = np.sort(order[first])
    if len(selected) > MAX_SCENE_POINTS:
        seed = int.from_bytes(hashlib.md5(("42" + frame_fingerprint + "scene").encode("utf-8")).digest()[:4], "little")
        random = np.random.RandomState(seed)
        selected = np.sort(random.choice(selected, MAX_SCENE_POINTS, replace=False))
    return selected.astype(np.int64, copy=False)


def compile_observation(
    *, model_pack: PointWorldModelPack, source: Path, output: Path,
) -> dict[str, Any]:
    arrays = _load_npz(source)
    required = {"rgb", "depth", "depth_validity", "intrinsic", "camera_to_world", "timestamp_ns", "frame_fingerprint"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"PointWorld observation is missing {', '.join(missing)}")
    rgb = np.asarray(arrays["rgb"])
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be sRGB uint8; implicit quantization is forbidden")
    depth = _finite_array(arrays["depth"], np.float32, "depth")
    validity = np.asarray(arrays["depth_validity"])
    if validity.dtype != np.bool_:
        raise ValueError("depth_validity must be bool")
    intrinsic = _finite_array(arrays["intrinsic"], np.float32, "intrinsic")
    camera_to_world = _finite_array(arrays["camera_to_world"], np.float32, "camera_to_world")
    if rgb.ndim != 4 or rgb.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError("rgb must have shape [camera, 180, 320, 3] in sRGB uint8")
    camera_count = rgb.shape[0]
    if not 1 <= camera_count <= 3 or depth.shape != rgb.shape[:3] or validity.shape != depth.shape:
        raise ValueError("camera RGB-D layout is invalid")
    if intrinsic.shape != (camera_count, 3, 3) or camera_to_world.shape != (camera_count, 4, 4):
        raise ValueError("camera calibration layout is invalid")
    if not np.allclose(camera_to_world[:, 3], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6):
        raise ValueError("camera_to_world must be a rigid homogeneous transform")
    rotations = camera_to_world[:, :3, :3].astype(np.float64)
    if not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise ValueError("extrinsic rotation must be orthonormal and right-handed")
    frame = np.asarray(arrays["frame_fingerprint"])
    if frame.size != 1:
        raise ValueError("frame_fingerprint must contain one SHA-256 value")
    frame_fingerprint = str(frame.reshape(-1)[0])
    _require_digest(frame_fingerprint, "frame_fingerprint")
    timestamps = np.asarray(arrays["timestamp_ns"])
    if timestamps.size not in (1, camera_count) or timestamps.dtype.kind not in "ui":
        raise ValueError("timestamp_ns must contain one or one-per-camera unsigned/integer timestamp")
    timestamps = np.broadcast_to(timestamps.reshape(-1), (camera_count,)).astype(np.uint64, copy=False)
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_camera_indices: list[np.ndarray] = []
    all_pixel_indices: list[np.ndarray] = []
    for camera in range(camera_count):
        points, mask = _backproject(depth[camera], validity[camera], intrinsic[camera], camera_to_world[camera])
        all_points.append(points)
        all_colors.append(rgb[camera].reshape(-1, 3)[mask].astype(np.float32) / 255.0)
        camera_normals = _depth_normals(depth[camera], validity[camera], intrinsic[camera])[mask]
        world_normals = np.einsum("ij,nj->ni", camera_to_world[camera, :3, :3], camera_normals)
        all_normals.append(world_normals.astype(np.float32))
        valid_pixels = np.flatnonzero(mask).astype(np.uint32)
        all_camera_indices.append(np.full(len(valid_pixels), camera, dtype=np.uint8))
        all_pixel_indices.append(valid_pixels)
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    if not len(points):
        raise ValueError("PointWorld observation has no valid metric scene points")
    selected = _release_voxel_selection(points, frame_fingerprint)
    camera_indices = np.concatenate(all_camera_indices)
    pixel_indices = np.concatenate(all_pixel_indices)
    world_to_camera = np.empty_like(camera_to_world)
    world_to_camera[:, :3, :3] = np.swapaxes(camera_to_world[:, :3, :3], 1, 2)
    world_to_camera[:, :3, 3] = -np.einsum(
        "cij,cj->ci", world_to_camera[:, :3, :3], camera_to_world[:, :3, 3]
    )
    world_to_camera[:, 3] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    payload = {
        "scene_points": points[selected], "scene_colors": colors[selected],
        "scene_normals": normals[selected], "scene_camera_index": camera_indices[selected],
        "scene_pixel_index": pixel_indices[selected], "camera_rgb": rgb,
        "camera_depth": depth, "camera_validity": validity, "intrinsic": intrinsic,
        "camera_to_world": camera_to_world, "extrinsic": world_to_camera,
        "timestamp_ns": timestamps,
    }
    archive_sha = _write_npz(output, payload)
    manifest = {
        "format": OBSERVATION_FORMAT, "model_pack_fingerprint": model_pack.fingerprint,
        "frame_fingerprint": frame_fingerprint, "timestamp_ns": timestamps.astype(int).tolist(),
        "camera_count": camera_count, "scene_point_count": int(len(selected)),
        "grid_size_m": GRID_SIZE_M, "coordinate_frame": "world",
        "extrinsic_convention": "world_to_camera", "sha256": archive_sha,
    }
    _write_json(output.with_suffix(".json"), manifest)
    return {"observation": str(output), "manifest": str(output.with_suffix('.json')), **manifest}


def _flow_derivatives(flows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros_like(flows)
    acceleration = np.zeros_like(flows)
    velocity[:, 0] = flows[:, 1] - flows[:, 0]
    velocity[:, 1:-1] = (flows[:, 2:] - flows[:, :-2]) * 0.5
    velocity[:, -1] = flows[:, -1] - flows[:, -2]
    acceleration[:, 0] = velocity[:, 1] - velocity[:, 0]
    acceleration[:, 1:-1] = (velocity[:, 2:] - velocity[:, :-2]) * 0.5
    acceleration[:, -1] = velocity[:, -1] - velocity[:, -2]
    return velocity, acceleration


def _scene_robot_distances(scene_points: np.ndarray, robot_flows: np.ndarray) -> np.ndarray:
    candidate_count, frames, _, _ = robot_flows.shape
    distances = np.empty((candidate_count, frames, len(scene_points)), dtype=np.float32)
    # Bound the transient arena: 256 scene points x 500 robot points per
    # candidate/frame. This is artifact compilation, never the inference loop.
    for candidate_begin in range(0, candidate_count, 4):
        candidate_end = min(candidate_begin + 4, candidate_count)
        candidate_flows = robot_flows[candidate_begin:candidate_end]
        for begin in range(0, len(scene_points), 256):
            end = min(begin + 256, len(scene_points))
            delta = scene_points[None, None, begin:end, None, :] - candidate_flows[:, :, None, :, :]
            distances[candidate_begin:candidate_end, :, begin:end] = np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=-1)
    return distances


def compile_robot_flow_candidates(*, source: Path, observation: Path, observation_manifest: Path, output: Path) -> dict[str, Any]:
    observation_meta = _read_json(observation_manifest)
    observation_arrays = _load_bound_npz(observation, observation_meta, OBSERVATION_FORMAT)
    arrays = _load_npz(source)
    forbidden = {"joint_actions", "joint_positions", "joint_trajectory", "actions"} & set(arrays)
    if forbidden:
        raise ValueError("raw joint actions or joint trajectories are not PointWorld input")
    required = {
        "candidate_names", "link_transforms", "surface_points_local",
        "surface_normals_local", "surface_link_indices", "gripper_open",
        "source_fingerprint", "visual_geometry_fingerprint", "robot_topology_fingerprint",
    }
    if set(arrays) != required:
        missing = sorted(required - set(arrays))
        extra = sorted(set(arrays) - required)
        raise ValueError(f"candidate source contract mismatch; missing={missing}, extra={extra}")
    transforms_raw = np.asarray(arrays["link_transforms"])
    points_raw = np.asarray(arrays["surface_points_local"])
    normals_raw = np.asarray(arrays["surface_normals_local"])
    if transforms_raw.dtype != np.float64 or points_raw.dtype != np.float64 or normals_raw.dtype != np.float64:
        raise ValueError("authored surface points, normals, and link transforms must be FP64")
    transforms = _finite_array(transforms_raw, np.float64, "link_transforms")
    local_points = _finite_array(points_raw, np.float64, "surface_points_local")
    local_normals = _finite_array(normals_raw, np.float64, "surface_normals_local")
    link_indices = np.asarray(arrays["surface_link_indices"])
    names = np.asarray(arrays["candidate_names"])
    gripper_open = _finite_array(arrays["gripper_open"], np.float32, "gripper_open")
    if transforms.ndim != 5 or transforms.shape[1] != CONTEXT_FRAMES + PREDICTION_FRAMES or transforms.shape[-2:] != (4, 4):
        raise ValueError("link_transforms must have shape [candidate, 11, link, 4, 4]")
    candidate_count, frames, link_count = transforms.shape[:3]
    if names.shape != (candidate_count,) or len(set(str(name) for name in names)) != candidate_count:
        raise ValueError("candidate_names must be unique and match candidate count")
    if local_points.ndim != 2 or local_points.shape[1] != 3 or not 1 <= len(local_points) <= MAX_ROBOT_POINTS or local_normals.shape != local_points.shape:
        raise ValueError("authored surface geometry must have 1..500 point/normal triples")
    if link_indices.shape != (len(local_points),) or link_indices.dtype.kind not in "ui" or np.any(link_indices >= link_count):
        raise ValueError("surface_link_indices must bind every surface point to a valid link")
    if gripper_open.shape != (candidate_count, frames, 1) or np.any((gripper_open < 0.0) | (gripper_open > 1.0)):
        raise ValueError("gripper_open must have shape [candidate, 11, 1] in [0, 1]")
    bottom = transforms[..., 3, :]
    if not np.allclose(bottom, np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-12):
        raise ValueError("link transforms must be rigid homogeneous transforms")
    rotations = transforms[..., :3, :3]
    if not np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=1e-10) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-10):
        raise ValueError("link transform rotations must be FP64 orthonormal and right-handed")
    point_rotations = np.take(rotations, link_indices, axis=2)
    translations = np.take(transforms[..., :3, 3], link_indices, axis=2)
    flows64 = np.einsum("ctnij,nj->ctni", point_rotations, local_points) + translations
    normals64 = np.einsum("ctnij,nj->ctni", point_rotations, local_normals)
    normal_length = np.linalg.norm(normals64, axis=-1, keepdims=True)
    if np.any(normal_length <= 1e-12):
        raise ValueError("surface normals must be non-zero")
    normals64 /= normal_length
    flows = flows64.astype(np.float32)
    robot_normals = normals64.astype(np.float32)
    velocity, acceleration = _flow_derivatives(flows)
    robot_colors = np.empty_like(flows)
    robot_colors[..., 0] = 1.0
    robot_colors[..., 1] = 0.0
    robot_colors[..., 2] = 1.0
    scene_points = _finite_array(observation_arrays.get("scene_points"), np.float32, "scene_points")
    dist2robot = _scene_robot_distances(scene_points, flows)
    fingerprints: dict[str, str] = {}
    for key in ("source_fingerprint", "visual_geometry_fingerprint", "robot_topology_fingerprint"):
        value = str(np.asarray(arrays[key]).reshape(-1)[0])
        fingerprints[key] = _require_digest(value, key)
    computed_geometry = _fingerprint_arrays({
        "surface_points_local": points_raw,
        "surface_normals_local": normals_raw,
        "surface_link_indices": link_indices,
    })
    computed_topology = _fingerprint_arrays({
        "surface_link_indices": link_indices,
        "link_count": np.asarray([link_count], dtype=np.uint32),
    })
    source_payload = {key: value for key, value in arrays.items() if key != "source_fingerprint"}
    computed_source = _fingerprint_arrays(source_payload)
    if fingerprints["visual_geometry_fingerprint"] != computed_geometry:
        raise ValueError("visual geometry fingerprint does not match authored surface geometry")
    if fingerprints["robot_topology_fingerprint"] != computed_topology:
        raise ValueError("robot topology fingerprint does not match candidate link ownership")
    if fingerprints["source_fingerprint"] != computed_source:
        raise ValueError("candidate source fingerprint does not match the immutable candidate tensors")
    payload = {
        "candidate_names": names.astype("U"), "robot_flows": flows,
        "robot_colors": robot_colors, "robot_normals": robot_normals,
        "robot_velocity": velocity, "robot_acceleration": acceleration,
        "gripper_open": gripper_open, "scene_dist2robot": dist2robot,
    }
    sha = _write_npz(output, payload)
    manifest = {
        "format": CANDIDATES_FORMAT, "candidate_count": candidate_count,
        "frames": frames, "robot_point_count": int(flows.shape[2]),
        "observation_fingerprint": observation_meta["sha256"], **fingerprints,
        "kinematics_precision": "float64", "sha256": sha,
    }
    _write_json(output.with_suffix(".json"), manifest)
    return {"candidates": str(output), "manifest": str(output.with_suffix('.json')), **manifest}


def _release_confidence(log_var: np.ndarray, domains: np.ndarray) -> np.ndarray:
    prepared = np.clip(log_var.astype(np.float64), math.log(VAR_FLOOR), math.log(VAR_CEILING))
    behavior = np.asarray(["behavior" in str(domain) for domain in domains], dtype=np.bool_)
    if behavior.any():
        real = ~behavior
        variance = float(np.exp(prepared[real]).mean()) if real.any() else SIM_VAR_CONST
        prepared[behavior] = math.log(float(np.clip(variance, VAR_FLOOR, VAR_CEILING)))
    variance = np.exp(prepared)
    scalar = variance[..., 0] if variance.shape[-1] == 1 else variance.mean(axis=-1)
    return np.clip(1.0 - (scalar - VAR_FLOOR) / (VAR_CEILING - VAR_FLOOR), 0.0, 1.0).astype(np.float32)


def seal_reference_forecast(
    *, model_pack: PointWorldModelPack, observation: Path,
    observation_manifest: Path, candidates: Path, candidates_manifest: Path,
    source: Path, output: Path,
) -> dict[str, Any]:
    observation_meta = _read_json(observation_manifest)
    candidates_meta = _read_json(candidates_manifest)
    _load_bound_npz(observation, observation_meta, OBSERVATION_FORMAT)
    _load_bound_npz(candidates, candidates_meta, CANDIDATES_FORMAT)
    if observation_meta.get("model_pack_fingerprint") != model_pack.fingerprint:
        raise ValueError("observation was compiled for a different PointWorld model pack")
    if candidates_meta.get("observation_fingerprint") != observation_meta.get("sha256"):
        raise ValueError("robot-flow candidates were compiled for a different observation")
    arrays = _load_npz(source)
    required = {"scene_relative", "log_var", "confidence", "domains", "provider_timing_names", "provider_timings_ms"}
    if not required.issubset(arrays) or any(not key.startswith("stage__") for key in set(arrays) - required):
        raise ValueError("reference forecast must contain release outputs, domains, timings, and optional stage__ tensors only")
    flow = _finite_array(arrays["scene_relative"], np.float32, "scene_relative")
    log_var = _finite_array(arrays["log_var"], np.float32, "log_var")
    expected = (int(candidates_meta["candidate_count"]), PREDICTION_FRAMES, int(observation_meta["scene_point_count"]), 3)
    if flow.shape != expected or log_var.shape not in (expected, expected[:-1] + (1,)):
        raise ValueError("reference forecast tensors do not match the compiled observation/candidates")
    domains = np.asarray(arrays["domains"])
    confidence = _finite_array(arrays["confidence"], np.float32, "confidence")
    if domains.shape != (expected[0],) or any(str(domain) not in {"droid", "behavior"} for domain in domains) or len(set(str(domain) for domain in domains)) != 1:
        raise ValueError("domains must contain one droid/behavior label per candidate")
    derived_confidence = _release_confidence(log_var, domains)
    if confidence.shape != derived_confidence.shape or not np.allclose(confidence, derived_confidence, rtol=1e-6, atol=1e-7):
        raise ValueError("CUDA confidence is inconsistent with the pinned release mapping")
    timing_names = np.asarray(arrays["provider_timing_names"])
    timings = _finite_array(arrays["provider_timings_ms"], np.float64, "provider_timings_ms")
    if timing_names.ndim != 1 or timings.shape != timing_names.shape or len(set(str(name) for name in timing_names)) != len(timing_names) or np.any(timings < 0.0):
        raise ValueError("provider timings must be unique named non-negative millisecond values")
    payload = {"scene_relative": flow, "log_var": log_var, "confidence": confidence}
    stage_fingerprints: dict[str, str] = {}
    for key in sorted(set(arrays) - required):
        stage = _finite_array(arrays[key], np.float32, key)
        payload[key] = stage
        stage_fingerprints[key.removeprefix("stage__")] = _fingerprint_arrays({key: stage})
    sha = _write_npz(output, payload)
    manifest = {
        "format": FORECAST_FORMAT, "provider": REFERENCE_FORMAT,
        "model_pack_fingerprint": model_pack.fingerprint,
        "observation_fingerprint": observation_meta["sha256"],
        "candidate_fingerprint": candidates_meta["sha256"], "candidate_count": expected[0],
        "prediction_frames": PREDICTION_FRAMES, "scene_point_count": expected[2],
        "domains": [str(domain) for domain in domains],
        "provider_timings_ms": {str(name): float(value) for name, value in zip(timing_names, timings)},
        "stage_fingerprints": stage_fingerprints, "sha256": sha,
    }
    _write_json(output.with_suffix(".json"), manifest)
    return {"forecast": str(output), "manifest": str(output.with_suffix('.json')), **manifest}


def rank_forecast(
    *, observation_manifest: Path, observation: Path, forecast_manifest: Path,
    forecast: Path, target: Path, uncertainty_penalty_m: float, output: Path,
) -> dict[str, Any]:
    observation_meta = _read_json(observation_manifest)
    observation_arrays = _load_bound_npz(observation, observation_meta, OBSERVATION_FORMAT)
    manifest = _read_json(forecast_manifest)
    arrays = _load_bound_npz(forecast, manifest, FORECAST_FORMAT)
    if manifest.get("observation_fingerprint") != observation_meta.get("sha256"):
        raise ValueError("forecast and observation fingerprints differ")
    target_arrays = _load_npz(target)
    mask = np.asarray(target_arrays.get("target_mask"))
    target_points = _finite_array(target_arrays.get("target_points"), np.float32, "target_points")
    if mask.dtype != np.bool_:
        raise ValueError("target_mask must be bool")
    flow = _finite_array(arrays.get("scene_relative"), np.float32, "scene_relative")
    confidence = _finite_array(arrays.get("confidence"), np.float32, "confidence")
    scene_points = _finite_array(observation_arrays.get("scene_points"), np.float32, "scene_points")
    if mask.shape != (flow.shape[2],) or not mask.any() or target_points.ndim != 2 or target_points.shape[1] != 3 or len(target_points) == 0:
        raise ValueError("target artifact must contain a non-empty scene mask and 3D target point set")
    if not math.isfinite(uncertainty_penalty_m) or uncertainty_penalty_m < 0.0:
        raise ValueError("uncertainty_penalty_m must be finite and non-negative")
    predicted_endpoint = scene_points[None, mask, :] + flow[:, -1, mask, :]
    confidence_target = np.clip(confidence[:, -1, mask], 0.0, 1.0)
    endpoint_error_sum = np.zeros(predicted_endpoint.shape[0], dtype=np.float64)
    for begin in range(0, predicted_endpoint.shape[1], 256):
        endpoint_chunk = predicted_endpoint[:, begin:begin + 256]
        delta = endpoint_chunk[:, :, None, :] - target_points[None, None, :, :]
        endpoint_error_sum += np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=-1).sum(axis=1)
    endpoint_error = endpoint_error_sum / predicted_endpoint.shape[1]
    confidence_penalty = uncertainty_penalty_m * (1.0 - confidence_target.mean(axis=1))
    score = endpoint_error + confidence_penalty
    winner = int(np.argmin(score))
    result = {
        "format": "numi.pointworld-plan.v1", "forecast_sha256": manifest["sha256"],
        "observation_sha256": observation_meta["sha256"], "target_sha256": _sha256_path(target),
        "selected_candidate": winner, "scores_m": score.astype(float).tolist(),
        "endpoint_error_m": endpoint_error.astype(float).tolist(),
        "confidence_penalty_m": confidence_penalty.astype(float).tolist(),
        "objective": "minimum_target_region_endpoint_error_plus_uncertainty_penalty",
        "advisory_only": True,
    }
    _write_json(output, result)
    return result


def evaluate_forecast(*, forecast_manifest: Path, forecast: Path, ground_truth: Path, output: Path) -> dict[str, Any]:
    manifest = _read_json(forecast_manifest)
    predicted = _finite_array(_load_bound_npz(forecast, manifest, FORECAST_FORMAT).get("scene_relative"), np.float32, "scene_relative")
    truth_arrays = _load_npz(ground_truth)
    truth = _finite_array(truth_arrays.get("scene_relative"), np.float32, "ground_truth.scene_relative")
    if truth.shape != predicted.shape or predicted.ndim != 4 or predicted.shape[-1] != 3:
        raise ValueError("ground truth must match forecast [candidate, 10, point, 3] shape")
    error_m = np.linalg.norm(predicted - truth, axis=-1)
    expected_mask_shape = error_m.shape
    masks: dict[str, np.ndarray] = {}
    for key in ("scene_exists", "scene_supervised_mask", "scene_moved_mask"):
        value = np.asarray(truth_arrays.get(key))
        if value.dtype != np.bool_ or value.shape != expected_mask_shape:
            raise ValueError(f"{key} must be bool with shape [candidate, 10, point]")
        masks[key] = value
    valid = masks["scene_exists"] & masks["scene_supervised_mask"]
    moved = valid & masks["scene_moved_mask"]
    if not valid.any() or not moved.any():
        raise ValueError("official evaluation masks select no valid moved points")
    domains = manifest.get("domains")
    if not isinstance(domains, list) or len(domains) != predicted.shape[0] or len(set(domains)) != 1:
        raise ValueError("evaluation requires one declared domain across the forecast batch")
    domain = domains[0]
    result = {
        "format": "numi.pointworld-evaluation.v1",
        "forecast_sha256": manifest["sha256"],
        "full_eval/test/l2/mean": float(error_m[valid].mean()),
        "full_eval/test/l2_moved/mean": float(error_m[moved].mean()),
        "point_count": int(predicted.shape[2]),
        "candidate_count": int(predicted.shape[0]),
        "valid_prediction_count": int(valid.sum()),
        "domain": domain,
    }
    if domain == "droid":
        filter_mask = np.asarray(truth_arrays.get("scene_filter_mask"))
        if filter_mask.dtype != np.bool_ or filter_mask.shape != expected_mask_shape:
            raise ValueError("DROID evaluation requires the released bool scene_filter_mask")
        filtered = valid & filter_mask
        filtered_moved = filtered & masks["scene_moved_mask"]
        if not filtered_moved.any():
            raise ValueError("released DROID confidence mask selects no moved predictions")
        result["full_eval/test/filtered_l2/mean"] = float(error_m[filtered].mean())
        result["full_eval/test/filtered_l2_moved/mean"] = float(error_m[filtered_moved].mean())
        result["filtered_moved_prediction_count"] = int(filtered_moved.sum())
    _write_json(output, result)
    return result


def unavailable_native_operation(operation: str) -> None:
    raise RuntimeError(
        f"PointWorld {operation} is intentionally unavailable until a converted large "
        "DROID+BEHAVIOR ModelPack, gated DINO receipt, CUDA golden corpus, and "
        "native PTv3/sparse operator parity are present. No dense substitute or "
        "raw PyTorch checkpoint fallback is permitted."
    )


def _model_pack(path: Path) -> PointWorldModelPack:
    return validate_model_pack(_read_json(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile and evaluate immutable PointWorld artifacts")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="validate a PointWorld model pack")
    inspect.add_argument("--model-pack", type=Path, required=True)
    fetch = commands.add_parser("fetch", help="fetch an explicitly requested pinned release file")
    fetch.add_argument("--repository", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--path", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    convert = commands.add_parser("convert", help="seal released checkpoint and DINO receipts")
    convert.add_argument("--checkpoint", type=Path, required=True)
    convert.add_argument("--checkpoint-revision", required=True)
    convert.add_argument("--dino-weights", type=Path, required=True)
    convert.add_argument("--dino-license", required=True)
    convert.add_argument("--normalization-stats", type=Path, required=True)
    convert.add_argument("--ptv3-blueprint", type=Path, required=True)
    convert.add_argument("--checkpoint-contract", type=Path, required=True)
    convert.add_argument("--accept-nvidia-license", action="store_true")
    convert.add_argument("--dino-access-granted", action="store_true")
    convert.add_argument("--accept-dino-license", action="store_true")
    convert.add_argument("--output", type=Path, required=True)
    observation = commands.add_parser("observation", help="compile calibrated RGB-D into a PointWorld observation")
    observation.add_argument("--model-pack", type=Path, required=True)
    observation.add_argument("--source", type=Path, required=True)
    observation.add_argument("--output", type=Path, required=True)
    candidates = commands.add_parser("candidates", help="validate native robot surface-flow candidates")
    candidates.add_argument("--source", type=Path, required=True)
    candidates.add_argument("--observation", type=Path, required=True)
    candidates.add_argument("--observation-manifest", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True)
    reference = commands.add_parser("reference", help="seal CUDA-oracle PointWorld outputs")
    reference.add_argument("--model-pack", type=Path, required=True)
    reference.add_argument("--observation", type=Path, required=True)
    reference.add_argument("--observation-manifest", type=Path, required=True)
    reference.add_argument("--candidates", type=Path, required=True)
    reference.add_argument("--candidates-manifest", type=Path, required=True)
    reference.add_argument("--source", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("infer", help="run the parity-qualified native Metal PointWorld provider")
    infer.add_argument("--model-pack", type=Path, required=True)
    train = commands.add_parser("train", help="run the parity-qualified MLX PointWorld trainer")
    train.add_argument("--model-pack", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="evaluate PointWorld flows against released-format ground truth")
    evaluate.add_argument("--forecast-manifest", type=Path, required=True)
    evaluate.add_argument("--forecast", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan", help="rank PointWorld forecast candidates")
    plan.add_argument("--observation-manifest", type=Path, required=True)
    plan.add_argument("--observation", type=Path, required=True)
    plan.add_argument("--forecast-manifest", type=Path, required=True)
    plan.add_argument("--forecast", type=Path, required=True)
    plan.add_argument("--target", type=Path, required=True)
    plan.add_argument("--uncertainty-penalty-m", type=float, default=0.05)
    plan.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        pack = _model_pack(args.model_pack)
        print(json.dumps({
            "model_pack": str(args.model_pack), "fingerprint": pack.fingerprint,
            "release_receipts_valid": True, "apple_runtime": "Metal/MPSGraph + MLX",
            "native_preprocessing": "implemented",
            "native_large_inference": "unqualified",
            "mlx_training_resume": "unqualified",
            "full_parity": False,
        }, sort_keys=True))
    elif args.command == "fetch":
        print(json.dumps(fetch_release_file(repository=args.repository, revision=args.revision, path=args.path, output=args.output), sort_keys=True))
    elif args.command == "convert":
        print(json.dumps(convert_model_pack(
            checkpoint=args.checkpoint,
            checkpoint_revision=args.checkpoint_revision,
            dino_weights=args.dino_weights,
            dino_license=args.dino_license,
            normalization_stats=args.normalization_stats,
            ptv3_blueprint=args.ptv3_blueprint,
            checkpoint_contract=args.checkpoint_contract,
            accept_nvidia_license=args.accept_nvidia_license,
            dino_access_granted=args.dino_access_granted,
            accept_dino_license=args.accept_dino_license,
            output=args.output,
        ), sort_keys=True))
    elif args.command == "observation":
        print(json.dumps(compile_observation(model_pack=_model_pack(args.model_pack), source=args.source, output=args.output), sort_keys=True))
    elif args.command == "candidates":
        print(json.dumps(compile_robot_flow_candidates(source=args.source, observation=args.observation, observation_manifest=args.observation_manifest, output=args.output), sort_keys=True))
    elif args.command == "reference":
        print(json.dumps(seal_reference_forecast(model_pack=_model_pack(args.model_pack), observation=args.observation, observation_manifest=args.observation_manifest, candidates=args.candidates, candidates_manifest=args.candidates_manifest, source=args.source, output=args.output), sort_keys=True))
    elif args.command == "infer":
        _model_pack(args.model_pack)
        unavailable_native_operation("inference")
    elif args.command == "train":
        _model_pack(args.model_pack)
        unavailable_native_operation("MLX training")
    elif args.command == "evaluate":
        print(json.dumps(evaluate_forecast(forecast_manifest=args.forecast_manifest, forecast=args.forecast, ground_truth=args.ground_truth, output=args.output), sort_keys=True))
    elif args.command == "plan":
        print(json.dumps(rank_forecast(observation_manifest=args.observation_manifest, observation=args.observation, forecast_manifest=args.forecast_manifest, forecast=args.forecast, target=args.target, uncertainty_penalty_m=args.uncertainty_penalty_m, output=args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
