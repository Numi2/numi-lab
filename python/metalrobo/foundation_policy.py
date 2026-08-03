"""Provider-neutral foundation-policy action chunks for Numi Lab.

Foundation models propose finite-horizon robot intent. They do not own physics,
contacts, safety, or rollout scheduling; those remain native Numi Lab concerns.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metalrobo.ardy_interaction_convert import write_interaction_pack


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
G1_GROUP_JOINTS = {
    "left_leg": (
        "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ),
    "right_leg": (
        "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ),
    "waist": (
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ),
    "left_arm": (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right_arm": (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
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


def _read_ppm(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"P6":
            raise ValueError(f"camera frame is not binary PPM: {path}")
        dimensions = stream.readline()
        while dimensions.startswith(b"#"):
            dimensions = stream.readline()
        fields = dimensions.split()
        if len(fields) != 2:
            raise ValueError(f"camera frame dimensions are malformed: {path}")
        width, height = (int(value) for value in fields)
        if stream.readline().strip() != b"255":
            raise ValueError("camera frame must use 8-bit PPM samples")
        pixels = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = width * height * 3
    if pixels.size != expected:
        raise ValueError(f"camera frame has {pixels.size} bytes, expected {expected}")
    if (width, height) != (640, 480):
        raise ValueError(
            f"foundation capture must be 640x480, received {width}x{height}"
        )
    return pixels.reshape(1, height, width, 3).astype(np.float32)


def _read_state_trace(path: Path, selected_step: int | None) -> tuple[int, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# step nq="):
        raise ValueError("Numi state trace header is missing")
    header = dict(
        field.split("=", 1)
        for field in lines[0][2:].split()
        if "=" in field
    )
    nq = int(header.get("nq", "0"))
    samples: list[tuple[int, np.ndarray]] = []
    for line in lines[1:]:
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        step = int(fields[0])
        values = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
        if values.size < nq:
            raise ValueError(f"state trace step {step} is truncated")
        samples.append((step, values[:nq]))
    if not samples:
        raise ValueError("Numi state trace contains no samples")
    if selected_step is None:
        return samples[-1]
    for sample in samples:
        if sample[0] == selected_step:
            return sample
    raise ValueError(f"Numi state trace has no step {selected_step}")


def compile_numi_observation(
    camera_frame: Path,
    state_trace: Path,
    output: Path,
    evidence_path: Path,
    step: int | None,
) -> dict[str, Any]:
    image = _read_ppm(camera_frame)
    selected_step, q = _read_state_trace(state_trace, step)
    if q.size < 36:
        raise ValueError(f"bundled G1 state must have at least 36 q values, received {q.size}")
    joints = q[7:36]
    state: dict[str, np.ndarray] = {}
    offset = 0
    for name in ("left_leg", "right_leg", "waist", "left_arm", "right_arm"):
        count = len(G1_GROUP_JOINTS[name])
        state[name] = joints[offset : offset + count].reshape(1, count).copy()
        offset += count
    # Numi's bundled 29-DoF G1 has no dexterous-hand joints. The model's hand
    # channels remain explicit zero-valid placeholders and are never mapped to
    # simulator actuators.
    state["left_hand"] = np.zeros((1, 7), dtype=np.float32)
    state["right_hand"] = np.zeros((1, 7), dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ego_view=image,
        numi_root_q=q[:7].reshape(1, 7),
        **state,
    )
    evidence = {
        "format": "numi.foundation-observation.v1",
        "camera_frame": {"path": str(camera_frame), "sha256": _sha256(camera_frame)},
        "state_trace": {"path": str(state_trace), "sha256": _sha256(state_trace)},
        "selected_step": selected_step,
        "output": {"path": str(output), "sha256": _sha256(output)},
        "camera_shape": list(image.shape),
        "root_state_shape": [1, 7],
        "state_shapes": {name: list(value.shape) for name, value in state.items()},
        "hand_state_semantics": "zero placeholders; bundled Numi G1 has no hand actuators",
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def compile_numi_interaction_pack(
    action_chunk: Path,
    observation: Path,
    native_library: Path,
    output: Path,
    evidence_path: Path,
    pack_id: str,
    desired_outcome: str,
    source_hz: float,
    prefix_hold_frames: int,
) -> dict[str, Any]:
    """Compile generated upper-body intent into Numi's native motion teacher."""
    if not pack_id.strip() or not desired_outcome.strip():
        raise ValueError("interaction identity and desired outcome are required")
    if prefix_hold_frames < 0:
        raise ValueError("prefix_hold_frames must be non-negative")
    with np.load(observation, allow_pickle=False) as archive:
        if "numi_root_q" not in archive.files:
            raise ValueError(
                "foundation observation lacks numi_root_q; recompile it from "
                "the synchronized Numi state trace"
            )
        root = np.asarray(archive["numi_root_q"], dtype=np.float32).reshape(-1)
    if root.shape != (7,) or not np.isfinite(root).all():
        raise ValueError("Numi root state must contain seven finite q values")
    quaternion_norm = float(np.linalg.norm(root[3:]))
    if quaternion_norm <= 1.0e-6:
        raise ValueError("Numi root quaternion is degenerate")
    root[3:] /= quaternion_norm

    contract = _native_g1_contract(native_library)
    default_pose = np.asarray(contract["default_pose"], dtype=np.float32)
    action_scale = np.asarray(contract["task_action_scale"], dtype=np.float32)
    _, groups = _load_observation(observation)
    captured_joints = default_pose.copy()
    index_by_name = {
        str(name): index for index, name in enumerate(contract["joint_order"])
    }
    for group, names in G1_GROUP_JOINTS.items():
        values = groups[group].reshape(-1)
        for source_index, name in enumerate(names):
            captured_joints[index_by_name[name]] = values[source_index]

    with tempfile.TemporaryDirectory(prefix="numi-foundation-interaction-") as directory:
        temporary = Path(directory)
        action_stream = temporary / "actions.f32"
        action_evidence = temporary / "actions.evidence.json"
        compiled = compile_numi_action_stream(
            action_chunk,
            observation,
            native_library,
            action_stream,
            action_evidence,
            source_hz,
            0,
        )
        normalized = np.fromfile(action_stream, dtype="<f4").reshape(-1, 29)

    joint_targets = default_pose[None, :] + normalized * action_scale[None, :]
    if prefix_hold_frames:
        joint_targets = np.concatenate(
            (
                np.repeat(
                    captured_joints[None, :], prefix_hold_frames, axis=0
                ),
                joint_targets,
            ),
            axis=0,
        )
    frame_count = int(joint_targets.shape[0])
    root_targets = np.repeat(root[None, :], frame_count, axis=0)
    tracks = (
        ("left_foot", "left_foot_contact", "locomotion_ground"),
        ("right_foot", "right_foot_contact", "locomotion_ground"),
    )
    contact_modes = np.full((frame_count, 2), 2, dtype=np.uint32)
    contact_confidence = np.full((frame_count, 2), 0.5, dtype=np.float32)
    target, content_hash = write_interaction_pack(
        output=output,
        pack_id=pack_id.strip(),
        clip_id=pack_id.strip(),
        desired_outcome=desired_outcome.strip(),
        source_repository="NumiLab/foundation-policy",
        source_revision=_sha256(action_chunk),
        license_name="generated-intent; upstream model terms apply",
        frames_per_second=float(compiled["control_hz"]),
        root_targets=root_targets,
        joint_targets=joint_targets,
        tracks=tracks,
        contact_modes=contact_modes,
        contact_confidence=contact_confidence,
    )
    evidence = {
        "format": "numi.foundation-interaction.v1",
        "pack_id": pack_id.strip(),
        "desired_outcome": desired_outcome.strip(),
        "interaction_pack": {
            "path": str(target),
            "sha256": _sha256(target),
            "content_hash": content_hash,
        },
        "action_chunk": {
            "path": str(action_chunk),
            "sha256": _sha256(action_chunk),
        },
        "observation": {
            "path": str(observation),
            "sha256": _sha256(observation),
        },
        "native_library": {
            "path": str(native_library),
            "sha256": _sha256(native_library),
        },
        "frames_per_second": float(compiled["control_hz"]),
        "frame_count": frame_count,
        "prefix_hold_frames": prefix_hold_frames,
        "mapped_joint_groups": compiled["mapped_joint_groups"],
        "ignored_model_outputs": compiled["ignored_model_outputs"],
        "lower_body_semantics": (
            "captured stance held initially, then native default-pose targets"
        ),
        "contact_semantics": (
            "bilateral support intent authored by the Numi bridge; no generated "
            "wrench, center-of-pressure, or pressure claims"
        ),
        "position_clamps": compiled["position_clamps"],
        "velocity_clamps": compiled["velocity_clamps"],
        "normalized_clamps": compiled["normalized_clamps"],
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _native_g1_contract(native_library: Path) -> dict[str, Any]:
    library = ctypes.CDLL(str(native_library))
    function = library.mr_unitree_g1_deployment_contract_json
    function.argtypes = []
    function.restype = ctypes.c_char_p
    encoded = function()
    if not encoded:
        raise RuntimeError("native G1 deployment contract is unavailable")
    contract = json.loads(encoded.decode("utf-8"))
    required = ("joint_order", "default_pose", "task_action_scale", "velocity_limits", "position_limits")
    if any(name not in contract for name in required):
        raise ValueError("native G1 deployment contract lacks action mapping fields")
    return contract


def compile_numi_action_stream(
    action_chunk: Path,
    observation: Path,
    native_library: Path,
    output: Path,
    evidence_path: Path,
    source_hz: float,
    prefix_zero_steps: int,
) -> dict[str, Any]:
    if not np.isfinite(source_hz) or source_hz <= 0.0:
        raise ValueError("source_hz must be finite and positive")
    if prefix_zero_steps < 0:
        raise ValueError("prefix_zero_steps must be non-negative")
    contract = _native_g1_contract(native_library)
    joint_order = tuple(str(name) for name in contract["joint_order"])
    default_pose = np.asarray(contract["default_pose"], dtype=np.float32)
    action_scale = np.asarray(contract["task_action_scale"], dtype=np.float32)
    velocity_limits = np.asarray(contract["velocity_limits"], dtype=np.float32)
    position_limits = np.asarray(contract["position_limits"], dtype=np.float32)
    joint_count = len(joint_order)
    if (
        joint_count != 29
        or default_pose.shape != (joint_count,)
        or action_scale.shape != (joint_count,)
        or velocity_limits.shape != (joint_count,)
        or position_limits.shape != (joint_count, 2)
        or np.any(action_scale <= 0.0)
    ):
        raise ValueError("native G1 deployment arrays are inconsistent")
    index_by_name = {name: index for index, name in enumerate(joint_order)}

    _, current_groups = _load_observation(observation)
    current = default_pose.copy()
    for group, names in G1_GROUP_JOINTS.items():
        values = current_groups[group].reshape(-1)
        for source_index, name in enumerate(names):
            current[index_by_name[name]] = values[source_index]

    with np.load(action_chunk, allow_pickle=False) as archive:
        mapped = {
            group: np.asarray(archive[group], dtype=np.float32)[0]
            for group in ("waist", "left_arm", "right_arm")
        }
        ignored_outputs = sorted(
            set(archive.files).difference(mapped)
        )
    source_steps = next(iter(mapped.values())).shape[0]
    if source_steps < 1 or any(
        values.shape != (source_steps, len(G1_GROUP_JOINTS[group]))
        for group, values in mapped.items()
    ):
        raise ValueError("foundation action chunk has incompatible G1 arrays")
    control_hz = 1.0 / float(contract["policy_timestep_seconds"])
    duration = (source_steps - 1) / source_hz
    destination_steps = int(round(duration * control_hz)) + 1
    source_time = np.arange(source_steps, dtype=np.float64) / source_hz
    destination_time = np.arange(destination_steps, dtype=np.float64) / control_hz
    desired = np.repeat(default_pose[None, :], destination_steps, axis=0)
    for group, names in G1_GROUP_JOINTS.items():
        if group not in mapped:
            continue
        for source_index, name in enumerate(names):
            joint = index_by_name[name]
            desired[:, joint] = np.interp(
                destination_time,
                source_time,
                mapped[group][:, source_index],
            )

    position_clamps = 0
    velocity_clamps = 0
    normalized_clamps = 0
    accepted_targets = desired.copy()
    previous = current.copy()
    timestep = 1.0 / control_hz
    mapped_indices = [
        index_by_name[name]
        for group in ("waist", "left_arm", "right_arm")
        for name in G1_GROUP_JOINTS[group]
    ]
    for step_index in range(destination_steps):
        for joint in mapped_indices:
            target = float(accepted_targets[step_index, joint])
            limited = float(np.clip(target, position_limits[joint, 0], position_limits[joint, 1]))
            position_clamps += int(limited != target)
            maximum_delta = float(velocity_limits[joint]) * timestep
            rate_limited = float(np.clip(limited, previous[joint] - maximum_delta, previous[joint] + maximum_delta))
            velocity_clamps += int(rate_limited != limited)
            accepted_targets[step_index, joint] = rate_limited
            previous[joint] = rate_limited

    normalized = (accepted_targets - default_pose[None, :]) / action_scale[None, :]
    unclamped = normalized.copy()
    normalized = np.clip(normalized, -1.0, 1.0).astype("<f4")
    normalized_clamps = int(np.count_nonzero(normalized != unclamped))
    if prefix_zero_steps:
        normalized = np.concatenate(
            (np.zeros((prefix_zero_steps, joint_count), dtype="<f4"), normalized),
            axis=0,
        )
    if not np.isfinite(normalized).all():
        raise RuntimeError("compiled Numi action stream is non-finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(normalized.tobytes(order="C"))
    evidence = {
        "format": "numi.foundation-action-stream.v1",
        "action_chunk": {"path": str(action_chunk), "sha256": _sha256(action_chunk)},
        "observation": {"path": str(observation), "sha256": _sha256(observation)},
        "native_library": {"path": str(native_library), "sha256": _sha256(native_library)},
        "output": {"path": str(output), "sha256": _sha256(output)},
        "source_hz": source_hz,
        "control_hz": control_hz,
        "source_steps": source_steps,
        "control_steps": int(normalized.shape[0]),
        "prefix_zero_steps": prefix_zero_steps,
        "joint_order": list(joint_order),
        "mapped_joint_groups": ["waist", "left_arm", "right_arm"],
        "ignored_model_outputs": ignored_outputs,
        "lower_body_semantics": "zero normalized action; native default-pose balance controller",
        "position_clamps": position_clamps,
        "velocity_clamps": velocity_clamps,
        "normalized_clamps": normalized_clamps,
        "maximum_absolute_action": float(np.max(np.abs(normalized))),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


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

    observation_parser = subparsers.add_parser(
        "observation", help="compile synchronized Numi camera and G1 state"
    )
    observation_parser.add_argument("--camera-frame", type=Path, required=True)
    observation_parser.add_argument("--state-trace", type=Path, required=True)
    observation_parser.add_argument("--step", type=int)
    observation_parser.add_argument("--output", type=Path, required=True)
    observation_parser.add_argument("--evidence", type=Path, required=True)

    actions_parser = subparsers.add_parser(
        "compile-actions", help="map a foundation chunk into Numi's G1 controller"
    )
    actions_parser.add_argument("--action-chunk", type=Path, required=True)
    actions_parser.add_argument("--observation", type=Path, required=True)
    actions_parser.add_argument("--native-library", type=Path, required=True)
    actions_parser.add_argument("--output", type=Path, required=True)
    actions_parser.add_argument("--evidence", type=Path, required=True)
    actions_parser.add_argument("--source-hz", type=float, default=30.0)
    actions_parser.add_argument("--prefix-zero-steps", type=int, default=1)

    interaction_parser = subparsers.add_parser(
        "compile-interaction",
        help="compile a foundation chunk into a native InteractionPack teacher",
    )
    interaction_parser.add_argument("--action-chunk", type=Path, required=True)
    interaction_parser.add_argument("--observation", type=Path, required=True)
    interaction_parser.add_argument("--native-library", type=Path, required=True)
    interaction_parser.add_argument("--output", type=Path, required=True)
    interaction_parser.add_argument("--evidence", type=Path, required=True)
    interaction_parser.add_argument("--id", required=True)
    interaction_parser.add_argument("--desired-outcome", required=True)
    interaction_parser.add_argument("--source-hz", type=float, default=30.0)
    interaction_parser.add_argument("--prefix-hold-frames", type=int, default=1)

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
    if arguments.command == "observation":
        print(json.dumps(compile_numi_observation(
            arguments.camera_frame,
            arguments.state_trace,
            arguments.output,
            arguments.evidence,
            arguments.step,
        ), indent=2, sort_keys=True))
        return 0
    if arguments.command == "compile-actions":
        print(json.dumps(compile_numi_action_stream(
            arguments.action_chunk,
            arguments.observation,
            arguments.native_library,
            arguments.output,
            arguments.evidence,
            arguments.source_hz,
            arguments.prefix_zero_steps,
        ), indent=2, sort_keys=True))
        return 0
    if arguments.command == "compile-interaction":
        print(json.dumps(compile_numi_interaction_pack(
            arguments.action_chunk,
            arguments.observation,
            arguments.native_library,
            arguments.output,
            arguments.evidence,
            arguments.id,
            arguments.desired_outcome,
            arguments.source_hz,
            arguments.prefix_hold_frames,
        ), indent=2, sort_keys=True))
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
