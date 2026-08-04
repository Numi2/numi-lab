"""Apple-native host integration for TREE Industries' ARDY ONNX exports.

ARDY proposes motion. It does not own robot control or Numi physics.
This module executes the exported diffusion/decoder graphs and publishes a
versioned, fingerprinted skeleton-motion proposal for downstream consumers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ARDY_MOTION_FORMAT = "numi.motion-proposal.v1"
ARDY_REPOSITORY = "TREEIndustries/ARDY-Core-RP-20FPS-Horizon40-ONNX"
ARDY_REVISION = "99da2ef6605784967141d3fa91754c0f99e8a65d"
ARDY_GRAPH_HASHES = {
    "encoder": "44c751ebcf11fad1b088c16c087cbfcf72e43ac2641e6f18ea9dc57b9ad25137",
    "denoiser": "c0745f310b373c93858eb02bb7f30bae0ee139258e608ef4a317e8882e9bfac3",
    "decoder": "5a9ba4e1d1f537c61f91d63d63685f7f2176dffc8b9dd12cc88783d205987d2e",
}
ARDY_G1_REPOSITORY = "TREEIndustries/ARDY-G1-RP-25FPS-Horizon52-ONNX"
ARDY_G1_REVISION = "d36d7069d514f9e6534c44c9fcf2463733298326"
ARDY_G1_GRAPH_HASHES = {
    "encoder": "d29bf5c592200f5fd557fb4e39fdc53099c1703f23df41854a694c834b7a8de1",
    "denoiser": "a9e3c463f0807bec94629048ecdf66cf5291837a99efd475b3677836ad345b13",
    "decoder": "45fb14fd8fb85d01204341c16fefab82812c48ec4d82fb7e3ecf5138d14506a0",
}

_ARDY_MODEL_SPECS = {
    "ARDY-Core-RP-20FPS-Horizon40": {
        "repository": ARDY_REPOSITORY,
        "revision": ARDY_REVISION,
        "skeleton": "cskel27",
        "fps": 20,
        "horizon_frames": 40,
        "motion_dim": 330,
        "body_dim": 325,
        "joint_count": 27,
        "graph_hashes": ARDY_GRAPH_HASHES,
    },
    "ARDY-G1-RP-25FPS-Horizon52": {
        "repository": ARDY_G1_REPOSITORY,
        "revision": ARDY_G1_REVISION,
        "skeleton": "g1skel34",
        "fps": 25,
        "horizon_frames": 52,
        "motion_dim": 414,
        "body_dim": 409,
        "joint_count": 34,
        "graph_hashes": ARDY_G1_GRAPH_HASHES,
    },
}


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _array_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _physical_tracking_outcome(
    reference: Mapping[str, np.ndarray],
    solver: Mapping[str, np.ndarray],
    reference_fps: float,
    solver_fps: float,
) -> dict[str, Any]:
    """Measure physical execution against motion intent without gating it."""

    reference_root = np.asarray(
        reference["root_position_quaternion_xyzw"], dtype=np.float64
    )
    solver_root = np.asarray(
        solver["root_position_quaternion_xyzw"], dtype=np.float64
    )
    reference_joints = np.asarray(reference["joint_positions"], dtype=np.float64)
    solver_joints = np.asarray(solver["joint_positions"], dtype=np.float64)
    solver_time = (np.arange(solver_root.shape[0], dtype=np.float64) + 1.0) / solver_fps
    reference_time = np.arange(reference_root.shape[0], dtype=np.float64) / reference_fps
    sample_time = np.minimum(solver_time, reference_time[-1])
    sampled_root_position = np.column_stack(
        [np.interp(sample_time, reference_time, reference_root[:, axis]) for axis in range(3)]
    )
    sampled_joints = np.column_stack(
        [np.interp(sample_time, reference_time, reference_joints[:, axis]) for axis in range(reference_joints.shape[1])]
    )
    position_error = solver_root[:, :3] - sampled_root_position
    joint_error = solver_joints - sampled_joints
    intended_displacement = reference_root[-1, :3] - reference_root[0, :3]
    achieved_displacement = solver_root[-1, :3] - solver_root[0, :3]
    intended_distance = float(np.linalg.norm(intended_displacement))
    if intended_distance > 1.0e-8:
        direction = intended_displacement / intended_distance
        projected_progress = float(np.dot(achieved_displacement, direction))
        progress_ratio: float | None = projected_progress / intended_distance
    else:
        projected_progress = 0.0
        progress_ratio = None
    return {
        "semantics": "measurement only; not a promotion or rejection gate",
        "reference_root_displacement_xyz_m": intended_displacement.tolist(),
        "solver_root_displacement_xyz_m": achieved_displacement.tolist(),
        "projected_progress_m": projected_progress,
        "projected_progress_ratio": progress_ratio,
        "root_position_rmse_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "root_position_final_error_m": float(np.linalg.norm(position_error[-1])),
        "joint_position_rmse_rad": float(np.sqrt(np.mean(np.square(joint_error)))),
        "joint_position_final_rmse_rad": float(np.sqrt(np.mean(np.square(joint_error[-1])))),
    }


def _model_spec(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    model = contract.get("model")
    if model not in _ARDY_MODEL_SPECS:
        raise ValueError(f"unsupported ARDY model contract: {model}")
    return _ARDY_MODEL_SPECS[str(model)]


def _validate_contract(contract: Mapping[str, Any]) -> None:
    spec = _model_spec(contract)
    expected = {
        "model": contract["model"],
        "skeleton": spec["skeleton"],
        "fps": spec["fps"],
        "horizon_frames": spec["horizon_frames"],
        "frames_per_token": 4,
        "max_tokens": 64,
        "max_frames": 256,
        "diffusion_steps": 10,
        "hybrid_dim": 148,
        "root_dim": 5,
        "latent_dim": 128,
        "motion_dim": spec["motion_dim"],
        "body_dim": spec["body_dim"],
        "local_root_dim": 4,
        "text_dim": 4096,
        "joint_count": spec["joint_count"],
    }
    mismatches = [
        key for key, value in expected.items() if contract.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "ARDY model contract is incompatible: " + ", ".join(mismatches)
        )
    if (
        len(contract.get("joint_names", ())) != spec["joint_count"]
        or len(contract.get("joint_parents", ())) != spec["joint_count"]
        or len(contract.get("motion_mean", ())) != spec["motion_dim"]
        or len(contract.get("motion_std", ())) != spec["motion_dim"]
        or len(contract.get("local_root_mean", ())) != 4
        or len(contract.get("local_root_std", ())) != 4
    ):
        raise ValueError("ARDY skeleton or normalization contract is incomplete")


def inspect_model(model_directory: Path, verify_hashes: bool) -> dict[str, Any]:
    contract_path = model_directory / "model_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("ARDY model contract must be a JSON object")
    _validate_contract(contract)
    spec = _model_spec(contract)
    graphs: dict[str, Any] = {}
    for name, expected_hash in spec["graph_hashes"].items():
        path = model_directory / "fp32" / f"{name}.onnx"
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = _sha256(path) if verify_hashes else None
        if actual_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"ARDY {name} graph sha256 mismatch")
        graphs[name] = {
            "path": str(path.relative_to(model_directory)),
            "bytes": path.stat().st_size,
            "sha256": actual_hash or expected_hash,
            "hash_verified": actual_hash is not None,
        }
    return {
        "format": ARDY_MOTION_FORMAT,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "contract": {
            "path": contract_path.name,
            "sha256": _sha256(contract_path),
            "skeleton": contract["skeleton"],
            "joint_count": contract["joint_count"],
            "fps": contract["fps"],
            "horizon_frames": contract["horizon_frames"],
        },
        "graphs": graphs,
    }


def _load_text_feature(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = [
                key for key in ("text_embedding", "text_feature", "embedding")
                if key in archive.files
            ]
            if len(keys) != 1:
                raise ValueError(
                    "ARDY text-feature NPZ needs one text_embedding array"
                )
            value = np.asarray(archive[keys[0]])
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
        value = document.get("embedding") if isinstance(document, dict) else None
    feature = np.asarray(value, dtype=np.float32)
    if feature.shape == (4096,):
        feature = feature.reshape(1, 1, 4096)
    elif feature.shape == (1, 4096):
        feature = feature.reshape(1, 1, 4096)
    if feature.shape != (1, 1, 4096) or not np.isfinite(feature).all():
        raise ValueError("ARDY text feature must be finite with shape [1, 1, 4096]")
    return feature


def load_reference_text_feature(path: Path, prompt: str) -> np.ndarray:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else None
    if not isinstance(records, list):
        raise ValueError("ARDY reference text-feature file has no records")
    matches = [record for record in records if record.get("prompt") == prompt]
    if len(matches) != 1:
        raise ValueError(f"ARDY reference file has no unique prompt: {prompt}")
    return _load_text_feature_from_value(matches[0].get("embedding"))


def _load_text_feature_from_value(value: Any) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32)
    if feature.shape != (4096,) or not np.isfinite(feature).all():
        raise ValueError("ARDY reference embedding is invalid")
    return feature.reshape(1, 1, 4096)


def _cosine_diffusion() -> tuple[np.ndarray, np.ndarray]:
    betas = []
    for index in range(10):
        first = math.cos(
            (index / 10.0 + 0.008) / 1.008 * math.pi / 2.0
        ) ** 2
        second = math.cos(
            ((index + 1) / 10.0 + 0.008) / 1.008 * math.pi / 2.0
        ) ** 2
        betas.append(min(1.0 - second / first, 0.999))
    cumulative = np.cumprod(
        1.0 - np.asarray(betas, dtype=np.float32)
    )
    previous = np.concatenate(
        (np.ones(1, dtype=np.float32), cumulative[:-1])
    )
    return cumulative, previous


def _local_root_condition(
    root_normalized: np.ndarray, contract: Mapping[str, Any]
) -> np.ndarray:
    mean = np.asarray(contract["motion_mean"][:5], dtype=np.float32)
    standard_deviation = np.asarray(
        contract["motion_std"][:5], dtype=np.float32
    )
    root = root_normalized * standard_deviation + mean
    positions = root[..., :3]
    heading = np.arctan2(root[..., 4], root[..., 3])
    heading_delta = np.arctan2(
        np.sin(heading[:, 1:] - heading[:, :-1]),
        np.cos(heading[:, 1:] - heading[:, :-1]),
    ) * float(contract["fps"])
    planar_velocity = (
        positions[:, 1:, (0, 2)] - positions[:, :-1, (0, 2)]
    ) * float(contract["fps"])
    heading_delta = np.concatenate(
        (heading_delta, heading_delta[:, -1:]), axis=1
    )
    planar_velocity = np.concatenate(
        (planar_velocity, planar_velocity[:, -1:]), axis=1
    )
    local = np.concatenate(
        (heading_delta[..., None], planar_velocity, positions[..., 1:2]),
        axis=2,
    )
    local_mean = np.asarray(contract["local_root_mean"], dtype=np.float32)
    local_std = np.asarray(contract["local_root_std"], dtype=np.float32)
    return ((local - local_mean) / local_std).astype(np.float32)


def _provider_attempts(
    requested: str,
    available: Sequence[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    cpu = ("cpu", ("CPUExecutionProvider",))
    coreml = (
        "coreml",
        ("CoreMLExecutionProvider", "CPUExecutionProvider"),
    )
    if requested == "cpu":
        return (cpu,)
    if requested not in {"auto", "coreml"}:
        raise ValueError(f"unsupported ARDY provider: {requested}")
    if "CoreMLExecutionProvider" not in available:
        if requested == "coreml":
            raise RuntimeError("ONNX Runtime has no CoreMLExecutionProvider")
        return (cpu,)
    return (coreml,) if requested == "coreml" else (coreml, cpu)


def _output_parity(
    candidate: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
    *,
    absolute_tolerance: float = 5.0e-4,
    relative_tolerance: float = 5.0e-3,
) -> dict[str, Any]:
    if len(candidate) != len(reference):
        return {
            "passed": False,
            "reason": "output-count mismatch",
            "candidate_output_count": len(candidate),
            "reference_output_count": len(reference),
        }
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    squared_error_sum = 0.0
    reference_squared_sum = 0.0
    candidate_reference_dot = 0.0
    candidate_squared_sum = 0.0
    element_count = 0
    shapes: list[list[int]] = []
    passed = True
    for candidate_value, reference_value in zip(candidate, reference, strict=True):
        candidate_array = np.asarray(candidate_value, dtype=np.float32)
        reference_array = np.asarray(reference_value, dtype=np.float32)
        shapes.append(list(reference_array.shape))
        if (
            candidate_array.shape != reference_array.shape
            or not np.isfinite(candidate_array).all()
            or not np.isfinite(reference_array).all()
        ):
            passed = False
            continue
        absolute_error = np.abs(candidate_array - reference_array)
        relative_error = absolute_error / np.maximum(
            np.abs(reference_array), 1.0e-6
        )
        maximum_absolute_error = max(
            maximum_absolute_error,
            float(np.max(absolute_error, initial=0.0)),
        )
        maximum_relative_error = max(
            maximum_relative_error,
            float(np.max(relative_error, initial=0.0)),
        )
        squared_error_sum += float(np.sum(absolute_error.astype(np.float64) ** 2))
        reference_squared_sum += float(
            np.sum(reference_array.astype(np.float64) ** 2)
        )
        candidate_squared_sum += float(
            np.sum(candidate_array.astype(np.float64) ** 2)
        )
        candidate_reference_dot += float(np.sum(
            candidate_array.astype(np.float64)
            * reference_array.astype(np.float64)
        ))
        element_count += int(reference_array.size)
        passed = passed and bool(np.allclose(
            candidate_array,
            reference_array,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ))
    return {
        "passed": passed,
        "reference_provider": "cpu",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "root_mean_square_error": (
            math.sqrt(squared_error_sum / element_count)
            if element_count else None
        ),
        "relative_l2_error": (
            math.sqrt(squared_error_sum / reference_squared_sum)
            if reference_squared_sum > 0.0 else None
        ),
        "cosine_similarity": (
            candidate_reference_dot
            / math.sqrt(candidate_squared_sum * reference_squared_sum)
            if candidate_squared_sum > 0.0 and reference_squared_sum > 0.0
            else 0.0
        ),
        "output_shapes": shapes,
    }


def _run_denoiser_stage(
    ort: Any,
    graph: Path,
    providers: Sequence[str],
    initial_hybrid: np.ndarray,
    fixed_feeds: Mapping[str, np.ndarray],
    cumulative: np.ndarray,
    previous: np.ndarray,
    generation_tokens: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    session = ort.InferenceSession(str(graph), providers=list(providers))
    loaded = time.perf_counter()
    hybrid = initial_hybrid.copy()
    step_seconds: list[float] = []
    for timestep in range(9, -1, -1):
        step_started = time.perf_counter()
        predicted = session.run(
            None,
            {
                **fixed_feeds,
                "x": hybrid,
                "timesteps": np.asarray([timestep], dtype=np.int64),
            },
        )[0]
        alpha = cumulative[timestep]
        epsilon = (
            hybrid[:, :generation_tokens] / np.sqrt(alpha)
            - predicted[:, :generation_tokens]
        ) / np.sqrt((1.0 - alpha) / alpha)
        hybrid[:, :generation_tokens] = (
            predicted[:, :generation_tokens] * np.sqrt(previous[timestep])
            + np.sqrt(1.0 - previous[timestep]) * epsilon
        )
        if not np.isfinite(hybrid[:, :generation_tokens]).all():
            raise RuntimeError(f"ARDY denoising step {timestep} is non-finite")
        step_seconds.append(time.perf_counter() - step_started)
    finished = time.perf_counter()
    return hybrid, {
        "session_providers": session.get_providers(),
        "load_seconds": loaded - started,
        "inference_seconds": finished - loaded,
        "step_seconds": step_seconds,
    }


def _run_decoder_stage(
    ort: Any,
    graph: Path,
    providers: Sequence[str],
    latent_tokens: np.ndarray,
    local_root: np.ndarray,
    frame_generation_mask: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    session = ort.InferenceSession(str(graph), providers=list(providers))
    loaded = time.perf_counter()
    decoded = session.run(
        None,
        {
            "latent_tokens": latent_tokens,
            "external_cond": local_root,
            "motion_pad_mask": frame_generation_mask,
        },
    )
    finished = time.perf_counter()
    arrays = [np.asarray(value, dtype=np.float32) for value in decoded]
    if not arrays or any(not np.isfinite(value).all() for value in arrays):
        raise RuntimeError("ARDY decoder produced non-finite output")
    return arrays, {
        "session_providers": session.get_providers(),
        "load_seconds": loaded - started,
        "inference_seconds": finished - loaded,
    }


def _run_ardy(
    ort: Any,
    model_directory: Path,
    contract: Mapping[str, Any],
    text_feature: np.ndarray,
    seed: int,
    requested_provider: str,
    available_providers: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    generator = np.random.default_rng(seed)
    horizon_frames = int(contract["horizon_frames"])
    frames_per_token = int(contract["frames_per_token"])
    generation_tokens = int(math.ceil(horizon_frames / frames_per_token))
    motion_dim = int(contract["motion_dim"])
    joint_count = int(contract["joint_count"])
    initial_hybrid = np.zeros((1, 64, 148), dtype=np.float32)
    initial_hybrid[:, :generation_tokens] = generator.standard_normal(
        (1, generation_tokens, 148), dtype=np.float32
    )
    frame_generation_mask = np.zeros((1, 256), dtype=np.float32)
    frame_generation_mask[:, :horizon_frames] = 1.0
    token_generation_mask = np.zeros((1, 64), dtype=np.float32)
    token_generation_mask[:, :generation_tokens] = 1.0
    fixed_feeds = {
        "cfg_weight_text": np.asarray([2.0], dtype=np.float32),
        "cfg_weight_cstr": np.asarray([0.0], dtype=np.float32),
        "history_len": np.asarray([0], dtype=np.int64),
        "generation_len": np.asarray([horizon_frames], dtype=np.int64),
        "history_mask": np.zeros((1, 256), dtype=np.float32),
        "generation_mask": frame_generation_mask,
        "history_token_mask": np.zeros((1, 64), dtype=np.float32),
        "generation_token_mask": token_generation_mask,
        "future_token_mask": np.zeros((1, 64), dtype=np.float32),
        "text_feat": text_feature,
        "first_heading_angle": np.zeros(1, dtype=np.float32),
        "motion_mask": np.zeros((1, 256, motion_dim), dtype=np.float32),
        "observed_motion": np.zeros((1, 256, motion_dim), dtype=np.float32),
    }
    cumulative, previous = _cosine_diffusion()
    attempts = _provider_attempts(requested_provider, available_providers)
    denoiser_graph = model_directory / "fp32" / "denoiser.onnx"
    denoiser_failures: list[dict[str, Any]] = []
    denoiser_cpu_cache: tuple[np.ndarray, dict[str, Any]] | None = None
    denoiser_result: np.ndarray | None = None
    denoiser_evidence: dict[str, Any] | None = None
    for name, providers in attempts:
        parity = None
        try:
            if name == "cpu" and denoiser_cpu_cache is not None:
                candidate, stage = denoiser_cpu_cache
            else:
                candidate, stage = _run_denoiser_stage(
                    ort,
                    denoiser_graph,
                    providers,
                    initial_hybrid,
                    fixed_feeds,
                    cumulative,
                    previous,
                    generation_tokens,
                )
            if name == "coreml":
                denoiser_cpu_cache = _run_denoiser_stage(
                    ort,
                    denoiser_graph,
                    ("CPUExecutionProvider",),
                    initial_hybrid,
                    fixed_feeds,
                    cumulative,
                    previous,
                    generation_tokens,
                )
                parity = _output_parity(
                    (candidate[:, :generation_tokens],),
                    (denoiser_cpu_cache[0][:, :generation_tokens],),
                )
                if not parity["passed"]:
                    raise RuntimeError(
                        "CoreML denoiser failed CPU parity: "
                        f"max_abs={parity['maximum_absolute_error']:.6g}"
                    )
            denoiser_result = candidate
            denoiser_evidence = {
                **stage,
                "selected_provider": name,
                "provider_failures": list(denoiser_failures),
                "cpu_parity": parity,
            }
            break
        except Exception as error:
            failure: dict[str, Any] = {"provider": name, "error": str(error)}
            if parity is not None:
                failure["cpu_parity"] = parity
            denoiser_failures.append(failure)
            if requested_provider != "auto":
                raise
    if denoiser_result is None or denoiser_evidence is None:
        raise RuntimeError("ARDY denoiser failed on every available provider")

    root_normalized = denoiser_result[:, :, :20].reshape(1, 256, 5)
    local_root = _local_root_condition(root_normalized, contract)
    decoder_graph = model_directory / "fp32" / "decoder.onnx"
    decoder_failures: list[dict[str, Any]] = []
    decoder_cpu_cache: tuple[list[np.ndarray], dict[str, Any]] | None = None
    decoded: list[np.ndarray] | None = None
    decoder_evidence: dict[str, Any] | None = None
    for name, providers in attempts:
        parity = None
        try:
            if name == "cpu" and decoder_cpu_cache is not None:
                candidate, stage = decoder_cpu_cache
            else:
                candidate, stage = _run_decoder_stage(
                    ort,
                    decoder_graph,
                    providers,
                    denoiser_result[:, :, 20:],
                    local_root,
                    frame_generation_mask,
                )
            if name == "coreml":
                decoder_cpu_cache = _run_decoder_stage(
                    ort,
                    decoder_graph,
                    ("CPUExecutionProvider",),
                    denoiser_result[:, :, 20:],
                    local_root,
                    frame_generation_mask,
                )
                parity = _output_parity(candidate, decoder_cpu_cache[0])
                if not parity["passed"]:
                    raise RuntimeError(
                        "CoreML decoder failed CPU parity: "
                        f"max_abs={parity['maximum_absolute_error']:.6g}"
                    )
            decoded = candidate
            decoder_evidence = {
                **stage,
                "selected_provider": name,
                "provider_failures": list(decoder_failures),
                "cpu_parity": parity,
            }
            break
        except Exception as error:
            failure = {"provider": name, "error": str(error)}
            if parity is not None:
                failure["cpu_parity"] = parity
            decoder_failures.append(failure)
            if requested_provider != "auto":
                raise
    if decoded is None or decoder_evidence is None:
        raise RuntimeError("ARDY decoder failed on every available provider")

    body_normalized = np.asarray(decoded[1], dtype=np.float32)
    motion_normalized = np.concatenate(
        (root_normalized, body_normalized), axis=2
    )[:, :horizon_frames]
    mean = np.asarray(contract["motion_mean"], dtype=np.float32)
    standard_deviation = np.asarray(contract["motion_std"], dtype=np.float32)
    motion = motion_normalized * standard_deviation + mean
    if motion.shape != (1, horizon_frames, motion_dim) or not np.isfinite(motion).all():
        raise RuntimeError("ARDY decoder produced an invalid motion tensor")
    local_position_end = 5 + (joint_count - 1) * 3
    rotation_end = local_position_end + joint_count * 6
    velocity_end = rotation_end + joint_count * 3
    arrays = {
        "normalized_motion": motion_normalized[0].astype(np.float32),
        "root_positions": motion[0, :, :3].astype(np.float32),
        "root_heading": motion[0, :, 3:5].astype(np.float32),
        "local_joint_positions": motion[0, :, 5:local_position_end]
        .reshape(horizon_frames, joint_count - 1, 3).astype(np.float32),
        "global_rotations_6d": motion[0, :, local_position_end:rotation_end]
        .reshape(horizon_frames, joint_count, 6).astype(np.float32),
        "joint_velocities": motion[0, :, rotation_end:velocity_end]
        .reshape(horizon_frames, joint_count, 3).astype(np.float32),
        "foot_contact_scores": motion[0, :, velocity_end:velocity_end + 4]
        .astype(np.float32),
        "foot_contacts": (motion[0, :, velocity_end:velocity_end + 4] > 0.5)
        .astype(np.uint8),
    }
    stages = {
        "denoiser": denoiser_evidence,
        "decoder": decoder_evidence,
    }
    return arrays, stages


@dataclass(frozen=True)
class ARDYMotionResult:
    arrays: Mapping[str, np.ndarray]
    evidence: Mapping[str, Any]

    def write(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        motion_path = output_directory / "motion_proposal.npz"
        np.savez_compressed(motion_path, **self.arrays)
        evidence = dict(self.evidence)
        evidence["motion_proposal"] = {
            "path": motion_path.name,
            "sha256": _sha256(motion_path),
            "arrays_fingerprint": _array_fingerprint(self.arrays),
            "shapes": {
                name: list(value.shape) for name, value in self.arrays.items()
            },
        }
        (output_directory / "evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def infer_motion(
    model_directory: Path,
    text_feature: np.ndarray,
    prompt: str,
    seed: int,
    provider: str,
    verify_hashes: bool,
    text_encoder_evidence: Mapping[str, Any] | None = None,
) -> ARDYMotionResult:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for ARDY inference") from error
    validated = inspect_model(model_directory, verify_hashes)
    contract = json.loads(
        (model_directory / "model_contract.json").read_text(encoding="utf-8")
    )
    available = ort.get_available_providers()
    started = time.perf_counter()
    arrays, stages = _run_ardy(
        ort,
        model_directory,
        contract,
        text_feature,
        seed,
        provider,
        available,
    )
    stage_providers = {
        name: str(stage["selected_provider"])
        for name, stage in stages.items()
    }
    if text_encoder_evidence is not None:
        stage_providers = {
            "text_encoder": str(text_encoder_evidence["selected_provider"]),
            **stage_providers,
        }
    unique_stage_providers = set(stage_providers.values())
    selected = (
        next(iter(unique_stage_providers))
        if len(unique_stage_providers) == 1
        else "mixed"
    )
    failures = [
        {"stage": stage_name, **failure}
        for stage_name, stage in stages.items()
        for failure in stage["provider_failures"]
    ]
    if text_encoder_evidence is not None:
        failures = [
            {
                "stage": "text_encoder",
                **failure,
            }
            for failure in text_encoder_evidence["provider_failures"]
        ] + failures
    evidence = {
        "format": ARDY_MOTION_FORMAT,
        "status": "runtime-qualified",
        "model_role": "motion proposal only; downstream simulation remains authoritative",
        "model_repository": validated["repository"],
        "model_revision": validated["revision"],
        "model_artifacts": validated,
        "prompt": prompt,
        "text_feature_sha256": _array_fingerprint({"text_feature": text_feature}),
        "text_encoder": text_encoder_evidence,
        "seed": seed,
        "requested_provider": provider,
        "selected_provider": selected,
        "selected_provider_by_stage": stage_providers,
        "provider_failures": failures,
        "available_providers": available,
        "stages": stages,
        "elapsed_seconds": time.perf_counter() - started,
        "skeleton": {
            "id": contract["skeleton"],
            "joint_names": contract["joint_names"],
            "joint_parents": contract["joint_parents"],
            "coordinate_system": "ARDY y-up skeleton coordinates",
        },
        "fps": contract["fps"],
        "frame_count": contract["horizon_frames"],
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
        },
    }
    return ARDYMotionResult(arrays, evidence)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ARDY as a fingerprinted Numi motion provider"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--model-directory", type=Path, required=True)
    inspect_parser.add_argument("--verify-hashes", action="store_true")
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--model-directory", type=Path, required=True)
    infer_parser.add_argument("--output-directory", type=Path, required=True)
    feature = infer_parser.add_mutually_exclusive_group(required=True)
    feature.add_argument("--text-feature", type=Path)
    feature.add_argument("--reference-text-features", type=Path)
    feature.add_argument("--text-encoder-directory", type=Path)
    feature.add_argument("--unconditioned", action="store_true")
    infer_parser.add_argument("--prompt", default="")
    infer_parser.add_argument("--seed", type=int, default=0)
    infer_parser.add_argument(
        "--provider", choices=("auto", "coreml", "cpu"), default="auto"
    )
    infer_parser.add_argument("--verify-hashes", action="store_true")
    retarget_parser = subparsers.add_parser("retarget-g1")
    retarget_parser.add_argument("--motion-proposal", type=Path, required=True)
    retarget_parser.add_argument("--g1-urdf", type=Path, required=True)
    retarget_parser.add_argument("--output-directory", type=Path, required=True)
    imagine_parser = subparsers.add_parser("imagine-g1")
    imagine_parser.add_argument("--prompt", required=True)
    imagine_parser.add_argument(
        "--model-family", choices=("g1", "core"), default="g1"
    )
    imagine_parser.add_argument("--model-directory", type=Path)
    imagine_parser.add_argument(
        "--text-encoder-directory", type=Path
    )
    imagine_parser.add_argument("--g1-urdf", type=Path)
    imagine_parser.add_argument("--output-directory", type=Path)
    imagine_parser.add_argument("--seed", type=int, default=0)
    imagine_parser.add_argument(
        "--provider", choices=("auto", "coreml", "cpu"), default="auto"
    )
    imagine_parser.add_argument("--verify-hashes", action="store_true")
    imagine_parser.add_argument("--width", type=int, default=960)
    imagine_parser.add_argument("--height", type=int, default=1200)
    imagine_parser.add_argument("--samples", type=int, default=64)
    arguments = parser.parse_args(argv)
    if arguments.command == "inspect":
        print(json.dumps(inspect_model(
            arguments.model_directory, arguments.verify_hashes
        ), indent=2, sort_keys=True))
        return 0
    if arguments.command == "retarget-g1":
        from .g1_motion_retarget import retarget_g1, write_retarget

        arrays, evidence = retarget_g1(
            arguments.motion_proposal,
            arguments.g1_urdf,
        )
        write_retarget(arguments.output_directory, arrays, evidence)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    if arguments.command == "imagine-g1":
        from .ardy_text_encoder import encode_prompt
        from .ardy_interaction_convert import write_retarget_interaction_pack
        from .ardy_g1 import (
            native_g1_mechanism,
            write_native_g1_interaction_pack,
        )
        from .g1_motion_retarget import (
            retarget_g1,
            solver_trace_to_g1,
            write_retarget,
        )

        if not arguments.prompt.strip():
            parser.error("--prompt must not be empty")
        numi_root = Path(os.environ.get("NUMI_LAB_ROOT", Path.cwd()))
        model_root = Path(
            os.environ.get(
                "NUMI_MODELS_DIR",
                Path.home() / "MetalRobo-training",
            )
        )
        if arguments.model_directory is not None:
            model_directory = arguments.model_directory
        elif arguments.model_family == "g1":
            model_directory = Path(os.environ.get(
                "NUMI_ARDY_G1_MODEL",
                model_root / "ardy-g1-rp-h52-onnx",
            ))
        else:
            model_directory = Path(os.environ.get(
                "NUMI_ARDY_CORE_MODEL",
                model_root / "ardy-core-rp-onnx",
            ))
        text_encoder_directory = arguments.text_encoder_directory or Path(
            os.environ.get(
                "NUMI_ARDY_TEXT_ENCODER",
                model_root / "ardy-text-encoder-int4",
            )
        )
        g1_urdf = arguments.g1_urdf or Path(
            os.environ.get(
                "NUMI_G1_URDF",
                numi_root
                / "build"
                / "unitree_ros"
                / "robots"
                / "g1_description"
                / "g1_29dof_rev_1_0.urdf",
            )
        )
        prompt_slug = re.sub(
            r"[^a-z0-9]+", "-", arguments.prompt.strip().lower()
        ).strip("-")[:48] or "motion"
        default_output = (
            numi_root
            / ".numi"
            / "runs"
            / (
                f"ardy-g1-{prompt_slug}-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )
        )
        output = (arguments.output_directory or default_output).resolve()
        motion_directory = output / "motion"
        retarget_directory = output / "g1-retarget"
        interaction_pack = output / "ardy-g1.interactionpack"
        physical_run_directory = output / "physical-run"
        state_trace = output / "solver-state.tsv"
        solver_motion_directory = output / "solver-motion"
        frames_directory = output / "frames"
        text_feature, encoder_evidence = encode_prompt(
            text_encoder_directory,
            arguments.prompt.strip(),
            arguments.provider,
            arguments.verify_hashes,
        )
        motion = infer_motion(
            model_directory,
            text_feature,
            arguments.prompt.strip(),
            arguments.seed,
            arguments.provider,
            arguments.verify_hashes,
            encoder_evidence,
        )
        motion.write(motion_directory)
        actual_skeleton = str(motion.evidence["skeleton"]["id"])
        expected_skeleton = (
            "g1skel34" if arguments.model_family == "g1" else "cskel27"
        )
        if actual_skeleton != expected_skeleton:
            raise ValueError(
                f"--model-family {arguments.model_family} requires "
                f"{expected_skeleton}, model supplies {actual_skeleton}"
            )
        if actual_skeleton == "g1skel34":
            arrays, retarget_evidence = native_g1_mechanism(
                motion_directory, g1_urdf
            )
        else:
            arrays, retarget_evidence = retarget_g1(
                motion_directory, g1_urdf
            )
        write_retarget(retarget_directory, arrays, retarget_evidence)

        if actual_skeleton == "g1skel34":
            _, interaction_fingerprint = write_native_g1_interaction_pack(
                output=interaction_pack,
                arrays=arrays,
                evidence=retarget_evidence,
                desired_outcome=arguments.prompt.strip(),
            )
        else:
            _, interaction_fingerprint = write_retarget_interaction_pack(
                output=interaction_pack,
                root_targets=arrays["root_position_quaternion_xyzw"],
                joint_targets=arrays["joint_positions"],
                frames_per_second=float(retarget_evidence["fps"]),
                desired_outcome=arguments.prompt.strip(),
                source_repository=retarget_evidence["source_motion"][
                    "model_repository"
                ],
                source_revision=retarget_evidence["source_motion"][
                    "model_revision"
                ],
                pack_id=(
                    "ardy-core-g1-"
                    + retarget_evidence["source_motion"]["arrays_fingerprint"][:16]
                ),
                foot_contacts=arrays["foot_contacts"],
                foot_contact_scores=arrays["foot_contact_scores"],
            )
        numi = numi_root / "tools" / "numi"
        if not os.access(numi, os.X_OK):
            raise FileNotFoundError(numi)
        physical_timestep = 0.02
        # Stop one transaction before the non-looping clip timeout resets the
        # episode. The final rendered frame must remain the physical outcome,
        # not the next episode's initialized pose.
        physical_steps = max(2, int(math.ceil(
            int(retarget_evidence["frame_count"])
            / float(retarget_evidence["fps"])
            / physical_timestep
        )) - 1)
        evaluation_environment = os.environ.copy()
        evaluation_environment["NUMI_RUN_DIR"] = str(physical_run_directory)
        subprocess.run(
            [
                str(numi),
                "evaluate",
                "--interaction-pack", str(interaction_pack),
                "--interaction-clip", "ardy-g1",
                "--zero-actions",
                "--interaction-student-authority", "0",
                "--task", "velocity",
                "--scene", "ground",
                "--envs", "1",
                "--steps", str(physical_steps),
                "--repeats", "1",
                "--chunk", "1",
                "--physics-substeps", "8",
                "--velocity-iterations", "8",
                "--final-velocity-iterations", "4",
                "--continue-after-termination",
                "--no-scheduled-resets",
                "--state-trace", str(state_trace),
            ],
            check=True,
            env=evaluation_environment,
        )
        physical_report = json.loads(
            (physical_run_directory / "evidence.json").read_text(
                encoding="utf-8"
            )
        )
        solver_arrays, solver_evidence = solver_trace_to_g1(
            state_trace, g1_urdf
        )
        physical_tracking = _physical_tracking_outcome(
            arrays,
            solver_arrays,
            float(retarget_evidence["fps"]),
            float(solver_evidence["fps"]),
        )
        write_retarget(
            solver_motion_directory, solver_arrays, solver_evidence
        )

        blender = shutil.which("blender")
        ffmpeg = shutil.which("ffmpeg")
        if blender is None or ffmpeg is None:
            raise RuntimeError("numi motion imagine-g1 needs Blender and ffmpeg")
        renderer = numi_root / "tools" / "render_g1_motion.py"
        if not renderer.is_file():
            raise FileNotFoundError(renderer)
        if frames_directory.exists():
            shutil.rmtree(frames_directory)
        frames_directory.mkdir(parents=True, exist_ok=True)
        render_started = time.perf_counter()
        subprocess.run(
            [
                blender,
                "--background",
                "--python",
                str(renderer),
                "--",
                "--retarget-directory",
                str(solver_motion_directory),
                "--g1-urdf",
                str(g1_urdf),
                "--frames",
                str(frames_directory),
                "--width",
                str(arguments.width),
                "--height",
                str(arguments.height),
                "--samples",
                str(arguments.samples),
            ],
            check=True,
        )
        render_seconds = time.perf_counter() - render_started
        render_evidence_path = output / "render-evidence.json"
        render_evidence = json.loads(render_evidence_path.read_text(encoding="utf-8"))
        palette = output / "palette.png"
        gif = output / "ardy-g1.gif"
        video = output / "ardy-g1.mp4"
        frame_pattern = str(frames_directory / "g1-%04d.png")
        physical_fps = float(solver_evidence["fps"])
        physical_fps_text = f"{physical_fps:.9g}"
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate",
                physical_fps_text,
                "-i", frame_pattern, "-vf",
                "fps=25,scale=720:-1:flags=lanczos,"
                "palettegen=max_colors=256:stats_mode=diff",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate",
                physical_fps_text,
                "-i", frame_pattern, "-i", str(palette), "-lavfi",
                "fps=25,scale=720:-1:flags=lanczos[x];"
                "[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                "-loop", "0", str(gif),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate",
                physical_fps_text,
                "-i", frame_pattern, "-c:v", "libx264", "-preset", "slow",
                "-crf", "16", "-pix_fmt", "yuv420p", "-movflags",
                "+faststart", str(video),
            ],
            check=True,
        )
        pipeline_evidence = {
            "format": "numi.imagine-g1.v2",
            "prompt": arguments.prompt.strip(),
            "seed": arguments.seed,
            "model_family": arguments.model_family,
            "motion_to_mechanism": (
                "native-g1-joint-frames"
                if actual_skeleton == "g1skel34"
                else "core-endpoint-ik-retarget"
            ),
            "source_motion": str(motion_directory / "evidence.json"),
            "g1_retarget": str(retarget_directory / "evidence.json"),
            "interaction_pack": {
                "path": interaction_pack.name,
                "sha256": _sha256(interaction_pack),
                "content_fingerprint": interaction_fingerprint,
                "contact_fields": "unknown; no force or pressure synthesized",
            },
            "physical_run": {
                "path": str(physical_run_directory),
                "state_trace": str(state_trace),
                "device": physical_report["device"],
                "solver": physical_report["solver_mode"],
                "gravity": "compiled world gravity; integrated every substep",
                "physics_substeps": physical_report["physics_substeps"],
                "control_steps": physical_steps,
                "failed_environment_steps": physical_report[
                    "failed_environment_steps"
                ],
                "maximum_root_height": physical_report["maximum_root_height"],
                "maximum_tilt": physical_report["maximum_tilt"],
                "termination_count": physical_report["termination_count"],
                "mean_tracking_score": physical_report[
                    "mean_tracking_score"
                ],
                "clean_horizon_semantics": (
                    "solver transactions completed without an authored "
                    "termination; this is not motion-success evidence"
                ),
                "reference_tracking_outcome": physical_tracking,
            },
            "solver_motion": str(
                solver_motion_directory / "evidence.json"
            ),
            "render": {
                "renderer": str(renderer),
                "blender": blender,
                "width": arguments.width,
                "height": arguments.height,
                "fps": physical_fps,
                "frame_count": int(solver_evidence["frame_count"]),
                "elapsed_seconds": render_seconds,
                "presentation": render_evidence,
            },
            "gif": {"path": gif.name, "sha256": _sha256(gif)},
            "video": {"path": video.name, "sha256": _sha256(video)},
            "authority": (
                "ARDY joint intent executed by native G1 drives; every rendered "
                "pose comes from an accepted NumiSolver state"
            ),
        }
        (output / "evidence.json").write_text(
            json.dumps(pipeline_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(pipeline_evidence, indent=2, sort_keys=True))
        return 0
    if arguments.text_feature is not None:
        text_feature = _load_text_feature(arguments.text_feature)
        text_encoder_evidence = None
    elif arguments.reference_text_features is not None:
        if not arguments.prompt.strip():
            parser.error("--reference-text-features requires --prompt")
        text_feature = load_reference_text_feature(
            arguments.reference_text_features, arguments.prompt
        )
        text_encoder_evidence = None
    elif arguments.text_encoder_directory is not None:
        if not arguments.prompt.strip():
            parser.error("--text-encoder-directory requires --prompt")
        from .ardy_text_encoder import encode_prompt

        text_feature, text_encoder_evidence = encode_prompt(
            arguments.text_encoder_directory,
            arguments.prompt.strip(),
            arguments.provider,
            arguments.verify_hashes,
        )
    else:
        text_feature = np.zeros((1, 1, 4096), dtype=np.float32)
        text_encoder_evidence = None
    result = infer_motion(
        arguments.model_directory,
        text_feature,
        arguments.prompt.strip(),
        arguments.seed,
        arguments.provider,
        arguments.verify_hashes,
        text_encoder_evidence,
    )
    result.write(arguments.output_directory)
    print(json.dumps(result.evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
