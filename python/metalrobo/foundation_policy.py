"""Provider-neutral foundation-policy action chunks for Numi Lab.

Foundation models propose finite-horizon robot intent. They do not own physics,
contacts, safety, or rollout scheduling; those remain native Numi Lab concerns.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_CHUNK_FORMAT = "numi.foundation-action-chunk.v1"
GROOT_APPLE_PNP_REPOSITORY = "nvidia/GR00T-N1.7-ApplePnP-V1"
GROOT_STAGES = (
    "preprocess_video",
    "preprocess_state",
    "backbone",
    "action_head",
    "decode_action",
)
G1_STATE_SHAPES = {
    "left_leg": (1, 6),
    "right_leg": (1, 6),
    "waist": (1, 3),
    "left_arm": (1, 7),
    "right_arm": (1, 7),
    "left_hand": (1, 7),
    "right_hand": (1, 7),
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


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read exported_leapp.yaml") from error
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("models"), dict):
        raise ValueError(f"invalid LEAPP manifest: {path}")
    return document


def _validate_model_directory(model_directory: Path, verify_hashes: bool) -> dict[str, Any]:
    manifest_path = model_directory / "exported_leapp.yaml"
    manifest = _load_manifest(manifest_path)
    models = manifest["models"]
    missing = [stage for stage in GROOT_STAGES if stage not in models]
    if missing:
        raise ValueError(f"manifest is missing stages: {', '.join(missing)}")

    artifacts: dict[str, Any] = {
        "manifest": {
            "path": manifest_path.name,
            "sha256": _sha256(manifest_path),
        }
    }
    source_path = model_directory / ".numi-foundation-source.json"
    if source_path.is_file():
        artifacts["source"] = json.loads(source_path.read_text(encoding="utf-8"))
    for stage in GROOT_STAGES:
        parameters = models[stage].get("parameters", {})
        model_name = parameters.get("model_path")
        expected = parameters.get("sha256sum")
        if not isinstance(model_name, str) or not isinstance(expected, str):
            raise ValueError(f"stage {stage} has no fingerprinted model path")
        model_path = model_directory / model_name
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        actual = _sha256(model_path) if verify_hashes else None
        if actual is not None and actual != expected:
            raise ValueError(f"stage {stage} sha256 mismatch")
        external = model_path.with_name(model_path.name + ".data")
        artifacts[stage] = {
            "path": model_name,
            "bytes": model_path.stat().st_size,
            "sha256": actual or expected,
            "hash_verified": actual is not None,
        }
        if stage in {"backbone", "action_head"} and not external.is_file():
            raise FileNotFoundError(external)
        if external.is_file():
            artifacts[stage]["external_data"] = {
                "path": external.name,
                "bytes": external.stat().st_size,
                "sha256": _sha256(external) if verify_hashes else None,
            }
    return {"manifest": manifest, "artifacts": artifacts}


def _provider_order(requested: str, available: Sequence[str]) -> list[Any]:
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested not in {"auto", "coreml"}:
        raise ValueError(f"unsupported provider: {requested}")
    if "CoreMLExecutionProvider" not in available:
        if requested == "coreml":
            raise RuntimeError("ONNX Runtime has no CoreMLExecutionProvider")
        return ["CPUExecutionProvider"]
    # CPU is an intentional correctness fallback for unsupported ONNX operators.
    return ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def _run_stage(
    ort: Any,
    model_path: Path,
    providers: Sequence[Any],
    feeds: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    session = ort.InferenceSession(str(model_path), providers=list(providers))
    loaded = time.perf_counter()
    output_names = [output.name for output in session.get_outputs()]
    values = session.run(output_names, dict(feeds))
    finished = time.perf_counter()
    if len(output_names) != len(values):
        raise RuntimeError("ONNX Runtime returned an unexpected output count")
    outputs = {name: values[index] for index, name in enumerate(output_names)}
    evidence = {
        "providers": session.get_providers(),
        "load_seconds": loaded - started,
        "inference_seconds": finished - loaded,
        "input_shapes": {key: list(value.shape) for key, value in feeds.items()},
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
    }
    del session
    gc.collect()
    return outputs, evidence


def _synthetic_observation(seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # A deterministic neutral observation is a runtime probe, not policy-quality evidence.
    generator = np.random.default_rng(seed)
    image = generator.integers(0, 256, size=(1, 480, 640, 3), dtype=np.uint8).astype(np.float32)
    state = {name: np.zeros(shape, dtype=np.float32) for name, shape in G1_STATE_SHAPES.items()}
    return image, state


def _load_observation(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"ego_view", *G1_STATE_SHAPES}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"observation is missing: {', '.join(missing)}")
        image = np.asarray(archive["ego_view"], dtype=np.float32)
        state = {name: np.asarray(archive[name], dtype=np.float32) for name in G1_STATE_SHAPES}
    if image.shape != (1, 480, 640, 3):
        raise ValueError(f"ego_view has shape {image.shape}, expected (1, 480, 640, 3)")
    for name, shape in G1_STATE_SHAPES.items():
        if state[name].shape != shape:
            raise ValueError(f"{name} has shape {state[name].shape}, expected {shape}")
    return image, state


@dataclass(frozen=True)
class FoundationInferenceResult:
    action_arrays: Mapping[str, np.ndarray]
    evidence: Mapping[str, Any]

    def write(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        archive_path = output_directory / "action_chunk.npz"
        np.savez_compressed(archive_path, **self.action_arrays)
        evidence = dict(self.evidence)
        evidence["action_chunk"] = {
            "path": archive_path.name,
            "sha256": _sha256(archive_path),
            "arrays_fingerprint": _array_fingerprint(self.action_arrays),
            "shapes": {name: list(value.shape) for name, value in self.action_arrays.items()},
        }
        (output_directory / "evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_groot_onnx(
    model_directory: Path,
    image: np.ndarray,
    state: Mapping[str, np.ndarray],
    provider: str,
    seed: int,
    verify_hashes: bool,
    observation_kind: str = "provided",
) -> FoundationInferenceResult:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for foundation inference") from error

    validated = _validate_model_directory(model_directory, verify_hashes)
    providers = _provider_order(provider, ort.get_available_providers())
    stages: dict[str, Any] = {}
    total_started = time.perf_counter()

    video, stages["preprocess_video"] = _run_stage(
        ort, model_directory / "preprocess_video.onnx", providers, {"ego_view": image}
    )
    normalized_state, stages["preprocess_state"] = _run_stage(
        ort, model_directory / "preprocess_state.onnx", providers, state
    )
    backbone, stages["backbone"] = _run_stage(
        ort,
        model_directory / "backbone.onnx",
        providers,
        {
            "vl_input_input_ids": video["input_ids"],
            "vl_input_attention_mask": video["attention_mask"],
            "vl_input_pixel_values": video["pixel_values"],
            "vl_input_image_grid_thw": video["image_grid_thw"],
        },
    )
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal((1, 40, 132), dtype=np.float32)
    head, stages["action_head"] = _run_stage(
        ort,
        model_directory / "action_head.onnx",
        providers,
        {
            "backbone_outputs_backbone_features": backbone["converted_outputs_backbone_features"],
            "backbone_outputs_backbone_attention_mask": backbone["converted_outputs_backbone_attention_mask"],
            "backbone_outputs_image_mask": backbone["converted_outputs_image_mask"],
            "action_inputs_state": normalized_state["state"],
            "action_inputs_embodiment_id": video["embodiment_id"],
            "action_inputs_input_ids": video["input_ids"],
            "action_inputs_attention_mask": video["attention_mask"],
            "action_inputs_pixel_values": video["pixel_values"],
            "action_inputs_image_grid_thw": video["image_grid_thw"],
            "initial_noise": noise,
        },
    )
    actions, stages["decode_action"] = _run_stage(
        ort,
        model_directory / "decode_action.onnx",
        providers,
        {
            "normalized_action": head["output1_action_pred"],
            "state_0_left_arm": normalized_state["reference_0_left_arm"],
            "state_0_right_arm": normalized_state["reference_0_right_arm"],
        },
    )
    for name, value in actions.items():
        if value.shape[1] != 16 or not np.isfinite(value).all():
            raise RuntimeError(f"invalid decoded action {name}: shape={value.shape}")

    observation = {"ego_view": image, **state}
    evidence = {
        "format": ACTION_CHUNK_FORMAT,
        "status": "runtime-qualified",
        "policy_role": "action-chunk proposal only; Numi Metal remains physics authority",
        "model_repository": GROOT_APPLE_PNP_REPOSITORY,
        "model_artifacts": validated["artifacts"],
        "requested_provider": provider,
        "available_providers": ort.get_available_providers(),
        "stages": stages,
        "seed": seed,
        "noise_sha256": _array_fingerprint({"initial_noise": noise}),
        "observation_fingerprint": _array_fingerprint(observation),
        "observation_kind": observation_kind,
        "adapter": {
            "path": Path(__file__).name,
            "sha256": _sha256(Path(__file__)),
        },
        "elapsed_seconds": time.perf_counter() - total_started,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
        },
    }
    return FoundationInferenceResult(actions, evidence)


def _inspect(model_directory: Path, verify_hashes: bool) -> dict[str, Any]:
    validated = _validate_model_directory(model_directory, verify_hashes)
    return {
        "format": ACTION_CHUNK_FORMAT,
        "model_repository": GROOT_APPLE_PNP_REPOSITORY,
        "model_artifacts": validated["artifacts"],
        "stages": list(GROOT_STAGES),
    }


def _fetch(repository: str, revision: str, model_directory: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to fetch a foundation model") from error
    resolved_revision = HfApi().model_info(repository, revision=revision).sha
    model_directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        revision=resolved_revision,
        local_dir=model_directory,
    )
    source = {
        "repository": repository,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
    }
    (model_directory / ".numi-foundation-source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run foundation policies as Numi action-chunk proposers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate a staged ONNX model directory")
    inspect_parser.add_argument("--model-directory", type=Path, required=True)
    inspect_parser.add_argument("--verify-hashes", action="store_true")

    fetch_parser = subparsers.add_parser("fetch", help="fetch a revision-pinned model snapshot")
    fetch_parser.add_argument("--repository", default=GROOT_APPLE_PNP_REPOSITORY)
    fetch_parser.add_argument("--revision", default="main")
    fetch_parser.add_argument("--model-directory", type=Path, required=True)

    infer_parser = subparsers.add_parser("infer", help="produce one fingerprinted action chunk")
    infer_parser.add_argument("--model-directory", type=Path, required=True)
    infer_parser.add_argument("--output-directory", type=Path, required=True)
    infer_parser.add_argument("--observation", type=Path)
    infer_parser.add_argument("--synthetic-observation", action="store_true")
    infer_parser.add_argument("--provider", choices=("auto", "coreml", "cpu"), default="auto")
    infer_parser.add_argument("--seed", type=int, default=0)
    infer_parser.add_argument("--verify-hashes", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.command == "inspect":
        print(json.dumps(_inspect(arguments.model_directory, arguments.verify_hashes), indent=2, sort_keys=True))
        return 0
    if arguments.command == "fetch":
        print(json.dumps(_fetch(arguments.repository, arguments.revision, arguments.model_directory), indent=2, sort_keys=True))
        return 0
    if bool(arguments.observation) == bool(arguments.synthetic_observation):
        parser.error("infer requires exactly one of --observation or --synthetic-observation")
    image, state = (
        _load_observation(arguments.observation)
        if arguments.observation
        else _synthetic_observation(arguments.seed)
    )
    result = run_groot_onnx(
        arguments.model_directory,
        image,
        state,
        arguments.provider,
        arguments.seed,
        arguments.verify_hashes,
        "provided" if arguments.observation else "deterministic-synthetic-runtime-probe",
    )
    result.write(arguments.output_directory)
    print(json.dumps(result.evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
