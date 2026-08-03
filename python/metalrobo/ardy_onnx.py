"""Apple-native host integration for TREE Industries' ARDY Core ONNX export.

ARDY proposes human motion. It does not own robot control or Numi physics.
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


def _validate_contract(contract: Mapping[str, Any]) -> None:
    expected = {
        "model": "ARDY-Core-RP-20FPS-Horizon40",
        "skeleton": "cskel27",
        "fps": 20,
        "horizon_frames": 40,
        "frames_per_token": 4,
        "max_tokens": 64,
        "max_frames": 256,
        "diffusion_steps": 10,
        "hybrid_dim": 148,
        "root_dim": 5,
        "latent_dim": 128,
        "motion_dim": 330,
        "body_dim": 325,
        "local_root_dim": 4,
        "text_dim": 4096,
        "joint_count": 27,
    }
    mismatches = [
        key for key, value in expected.items() if contract.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "ARDY model contract is incompatible: " + ", ".join(mismatches)
        )
    if (
        len(contract.get("joint_names", ())) != 27
        or len(contract.get("joint_parents", ())) != 27
        or len(contract.get("motion_mean", ())) != 330
        or len(contract.get("motion_std", ())) != 330
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
    graphs: dict[str, Any] = {}
    for name, expected_hash in ARDY_GRAPH_HASHES.items():
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
        "repository": ARDY_REPOSITORY,
        "revision": ARDY_REVISION,
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
            hybrid[:, :10] / np.sqrt(alpha) - predicted[:, :10]
        ) / np.sqrt((1.0 - alpha) / alpha)
        hybrid[:, :10] = (
            predicted[:, :10] * np.sqrt(previous[timestep])
            + np.sqrt(1.0 - previous[timestep]) * epsilon
        )
        if not np.isfinite(hybrid[:, :10]).all():
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
    initial_hybrid = np.zeros((1, 64, 148), dtype=np.float32)
    initial_hybrid[:, :10] = generator.standard_normal(
        (1, 10, 148), dtype=np.float32
    )
    frame_generation_mask = np.zeros((1, 256), dtype=np.float32)
    frame_generation_mask[:, :40] = 1.0
    token_generation_mask = np.zeros((1, 64), dtype=np.float32)
    token_generation_mask[:, :10] = 1.0
    fixed_feeds = {
        "cfg_weight_text": np.asarray([2.0], dtype=np.float32),
        "cfg_weight_cstr": np.asarray([0.0], dtype=np.float32),
        "history_len": np.asarray([0], dtype=np.int64),
        "generation_len": np.asarray([40], dtype=np.int64),
        "history_mask": np.zeros((1, 256), dtype=np.float32),
        "generation_mask": frame_generation_mask,
        "history_token_mask": np.zeros((1, 64), dtype=np.float32),
        "generation_token_mask": token_generation_mask,
        "future_token_mask": np.zeros((1, 64), dtype=np.float32),
        "text_feat": text_feature,
        "first_heading_angle": np.zeros(1, dtype=np.float32),
        "motion_mask": np.zeros((1, 256, 330), dtype=np.float32),
        "observed_motion": np.zeros((1, 256, 330), dtype=np.float32),
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
                )
                parity = _output_parity(
                    (candidate[:, :10],),
                    (denoiser_cpu_cache[0][:, :10],),
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
    )[:, :40]
    mean = np.asarray(contract["motion_mean"], dtype=np.float32)
    standard_deviation = np.asarray(contract["motion_std"], dtype=np.float32)
    motion = motion_normalized * standard_deviation + mean
    if motion.shape != (1, 40, 330) or not np.isfinite(motion).all():
        raise RuntimeError("ARDY decoder produced an invalid motion tensor")
    arrays = {
        "normalized_motion": motion_normalized[0].astype(np.float32),
        "root_positions": motion[0, :, :3].astype(np.float32),
        "root_heading": motion[0, :, 3:5].astype(np.float32),
        "local_joint_positions": motion[0, :, 5:83].reshape(40, 26, 3).astype(np.float32),
        "global_rotations_6d": motion[0, :, 83:245].reshape(40, 27, 6).astype(np.float32),
        "joint_velocities": motion[0, :, 245:326].reshape(40, 27, 3).astype(np.float32),
        "foot_contact_scores": motion[0, :, 326:330].astype(np.float32),
        "foot_contacts": (motion[0, :, 326:330] > 0.5).astype(np.uint8),
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
        "model_repository": ARDY_REPOSITORY,
        "model_revision": ARDY_REVISION,
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
        from .g1_motion_retarget import retarget_g1, write_retarget

        if not arguments.prompt.strip():
            parser.error("--prompt must not be empty")
        numi_root = Path(os.environ.get("NUMI_LAB_ROOT", Path.cwd()))
        model_root = Path(
            os.environ.get(
                "NUMI_MODELS_DIR",
                Path.home() / "MetalRobo-training",
            )
        )
        model_directory = arguments.model_directory or Path(
            os.environ.get(
                "NUMI_ARDY_CORE_MODEL",
                model_root / "ardy-core-rp-onnx",
            )
        )
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
        arrays, retarget_evidence = retarget_g1(
            motion_directory,
            g1_urdf,
        )
        write_retarget(retarget_directory, arrays, retarget_evidence)

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
                str(retarget_directory),
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
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate", "20",
                "-i", frame_pattern, "-vf",
                "fps=20,scale=720:-1:flags=lanczos,"
                "palettegen=max_colors=256:stats_mode=diff",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate", "20",
                "-i", frame_pattern, "-i", str(palette), "-lavfi",
                "fps=20,scale=720:-1:flags=lanczos[x];"
                "[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                "-loop", "0", str(gif),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate", "20",
                "-i", frame_pattern, "-c:v", "libx264", "-preset", "slow",
                "-crf", "16", "-pix_fmt", "yuv420p", "-movflags",
                "+faststart", str(video),
            ],
            check=True,
        )
        pipeline_evidence = {
            "format": "numi.imagine-g1.v1",
            "prompt": arguments.prompt.strip(),
            "seed": arguments.seed,
            "source_motion": str(motion_directory / "evidence.json"),
            "g1_retarget": str(retarget_directory / "evidence.json"),
            "render": {
                "renderer": str(renderer),
                "blender": blender,
                "width": arguments.width,
                "height": arguments.height,
                "fps": 20,
                "frame_count": int(retarget_evidence["frame_count"]),
                "elapsed_seconds": render_seconds,
                "presentation": render_evidence,
            },
            "gif": {"path": gif.name, "sha256": _sha256(gif)},
            "video": {"path": video.name, "sha256": _sha256(video)},
            "authority": (
                "ARDY-conditioned G1 kinematic preview; NumiSolver physics "
                "has not been applied"
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
