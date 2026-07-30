"""Apple-native dual-timescale tactile imitation policy.

MLX owns both optimization and inference. The policy combines diffusion
denoising for a coherent action tube with a streaming feedback vector field
that can react to each fresh tactile observation without discarding the tube.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import numpy.typing as npt
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .lerobot_dataset import OrigamiBatch, RobotDatasetManifest
from .tactile_stream import TactileStreamContract


TACTILE_TUBE_CHECKPOINT_SCHEMA = (
    "metalrobo.tactile_tube_policy_checkpoint"
)
TACTILE_TUBE_CHECKPOINT_FORMAT = 1


def _sinusoidal_embedding(
    value: mx.array,
    dimensions: int,
) -> mx.array:
    half = dimensions // 2
    frequencies = mx.exp(
        -math.log(10_000.0)
        * mx.arange(half, dtype=mx.float32)
        / max(half - 1, 1)
    )
    angles = value[..., None].astype(mx.float32) * frequencies
    result = mx.concatenate(
        (mx.sin(angles), mx.cos(angles)),
        axis=-1,
    )
    if dimensions % 2:
        result = mx.concatenate(
            (
                result,
                mx.zeros(
                    (*result.shape[:-1], 1),
                    dtype=result.dtype,
                ),
            ),
            axis=-1,
        )
    return result


def _masked_mean_squared_error(
    prediction: mx.array,
    target: mx.array,
    mask: mx.array,
) -> mx.array:
    weights = mask.astype(prediction.dtype)
    while weights.ndim < prediction.ndim:
        weights = weights[..., None]
    dimensions = math.prod(prediction.shape[mask.ndim :])
    denominator = mx.maximum(
        mx.sum(weights) * dimensions,
        mx.array(1.0, dtype=prediction.dtype),
    )
    return mx.sum(mx.square(prediction - target) * weights) / (
        denominator
    )


@dataclass(frozen=True, slots=True)
class TactileTubePolicyConfig:
    state_dimensions: int = 65
    action_dimensions: int = 65
    sensor_count: int = 10
    wrench_dimensions: int = 6
    history: int = 4
    horizon: int = 16
    model_dimensions: int = 256
    transformer_layers: int = 6
    attention_heads: int = 8
    mlp_dimensions: int = 1024
    dropout: float = 0.0
    vision_views: int = 0
    vision_keys: tuple[str, ...] = ()
    vision_height: int = 128
    vision_width: int = 128
    diffusion_steps: int = 16
    stream_rate_hz: float = 30.0
    stream_attraction: float = 8.0
    stream_noise: float = 0.2
    stream_noise_decay: float = 4.0
    tactile_mask_probability: float = 0.25
    diffusion_loss_weight: float = 1.0
    stream_loss_weight: float = 1.0
    tactile_reconstruction_weight: float = 0.1
    next_tactile_weight: float = 0.2
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    max_gradient_norm: float = 1.0
    ema_decay: float = 0.999
    seed: int = 0

    def validate(self) -> None:
        positive_integers = (
            self.state_dimensions,
            self.action_dimensions,
            self.sensor_count,
            self.wrench_dimensions,
            self.history,
            self.horizon,
            self.model_dimensions,
            self.transformer_layers,
            self.attention_heads,
            self.mlp_dimensions,
            self.diffusion_steps,
        )
        if (
            any(value <= 0 for value in positive_integers)
            or self.horizon <= 1
            or self.vision_views < 0
            or self.vision_views != len(self.vision_keys)
            or len(set(self.vision_keys)) != len(self.vision_keys)
            or any(
                not key.startswith("observation.images.")
                for key in self.vision_keys
            )
            or self.vision_height <= 0
            or self.vision_width <= 0
            or self.model_dimensions % self.attention_heads
            or not 0.0 <= self.dropout < 1.0
            or not 0.0 <= self.tactile_mask_probability < 1.0
            or not 0.0 < self.ema_decay < 1.0
        ):
            raise ValueError("tactile tube policy dimensions are invalid")
        positive_floats = (
            self.stream_rate_hz,
            self.stream_attraction,
            self.stream_noise,
            self.stream_noise_decay,
            self.learning_rate,
            self.max_gradient_norm,
        )
        loss_weights = (
            self.diffusion_loss_weight,
            self.stream_loss_weight,
            self.tactile_reconstruction_weight,
            self.next_tactile_weight,
            self.weight_decay,
        )
        if (
            any(
                not math.isfinite(value) or value <= 0.0
                for value in positive_floats
            )
            or any(
                not math.isfinite(value) or value < 0.0
                for value in loss_weights
            )
        ):
            raise ValueError(
                "tactile tube policy optimization values are invalid"
            )


class TactileTubeOutput(NamedTuple):
    field: mx.array
    reconstructed_wrench: mx.array
    next_wrench: mx.array


class TactileStreamDecision(NamedTuple):
    action_tube: npt.NDArray[np.float32]
    reconstructed_wrench: npt.NDArray[np.float32]
    predicted_next_wrench: npt.NDArray[np.float32]
    correction_rms: npt.NDArray[np.float32]


class _VisionStem(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        width = max(dimensions // 4, 16)
        self.conv1 = nn.Conv2d(3, width, 7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(
            width,
            width * 2,
            5,
            stride=2,
            padding=2,
        )
        self.conv3 = nn.Conv2d(
            width * 2,
            dimensions,
            3,
            stride=2,
            padding=1,
        )
        self.norm = nn.LayerNorm(dimensions)

    def __call__(self, image: mx.array) -> mx.array:
        hidden = nn.silu(self.conv1(image))
        hidden = nn.silu(self.conv2(hidden))
        hidden = nn.silu(self.conv3(hidden))
        return self.norm(mx.mean(hidden, axis=(1, 2)))


class TactileTubePolicy(nn.Module):
    """Transformer over proprioception, fingertip wrench, vision, and tube."""

    def __init__(self, config: TactileTubePolicyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dimensions = config.model_dimensions
        self.state_stem = nn.Linear(
            config.state_dimensions,
            dimensions,
        )
        self.tactile_stem = nn.Linear(
            config.wrench_dimensions + 1,
            dimensions,
        )
        self.action_stem = nn.Linear(
            config.action_dimensions + 1,
            dimensions,
        )
        self.time_stem = nn.Sequential(
            nn.Linear(dimensions * 2 + 1, dimensions),
            nn.SiLU(),
            nn.Linear(dimensions, dimensions),
        )
        self.history_embedding = (
            mx.random.normal((config.history, dimensions)) * 0.02
        )
        self.sensor_embedding = (
            mx.random.normal((config.sensor_count, dimensions)) * 0.02
        )
        self.horizon_embedding = (
            mx.random.normal((config.horizon, dimensions)) * 0.02
        )
        self.view_embedding = (
            mx.random.normal(
                (max(config.vision_views, 1), dimensions)
            )
            * 0.02
        )
        self.summary_token = (
            mx.random.normal((1, 1, dimensions)) * 0.02
        )
        self.missing_tactile = (
            mx.random.normal((1, 1, 1, dimensions)) * 0.02
        )
        self.vision_stem = (
            _VisionStem(dimensions)
            if config.vision_views
            else None
        )
        self.transformer = [
            nn.TransformerEncoderLayer(
                dimensions,
                config.attention_heads,
                mlp_dims=config.mlp_dimensions,
                dropout=config.dropout,
                activation=nn.silu,
                norm_first=True,
            )
            for _ in range(config.transformer_layers)
        ]
        self.final_norm = nn.LayerNorm(dimensions)
        self.field_head = nn.Linear(
            dimensions,
            config.action_dimensions,
        )
        self.wrench_head = nn.Linear(
            dimensions,
            2
            * config.sensor_count
            * config.wrench_dimensions,
        )

    def __call__(
        self,
        state_history: mx.array,
        wrench_history: mx.array,
        wrench_presence: mx.array,
        action_tube: mx.array,
        action_mask: mx.array,
        diffusion_time: mx.array,
        stream_time: mx.array,
        mode: mx.array,
        vision: mx.array,
    ) -> TactileTubeOutput:
        config = self.config
        batch_size = state_history.shape[0]
        if state_history.shape != (
            batch_size,
            config.history,
            config.state_dimensions,
        ):
            raise ValueError("policy state history shape is invalid")
        if wrench_history.shape != (
            batch_size,
            config.history,
            config.sensor_count,
            config.wrench_dimensions,
        ):
            raise ValueError("policy wrench history shape is invalid")
        if wrench_presence.shape != (
            batch_size,
            config.history,
            config.sensor_count,
        ):
            raise ValueError("policy wrench presence shape is invalid")
        if action_tube.shape != (
            batch_size,
            config.horizon,
            config.action_dimensions,
        ) or action_mask.shape != (
            batch_size,
            config.horizon,
        ):
            raise ValueError("policy action tube shape is invalid")
        if (
            diffusion_time.shape != (batch_size,)
            or stream_time.shape != (batch_size,)
            or mode.shape != (batch_size,)
        ):
            raise ValueError("policy time shape is invalid")

        history_embedding = self.history_embedding[None, :, :]
        state_tokens = self.state_stem(state_history)
        state_tokens = state_tokens + history_embedding

        presence = wrench_presence.astype(wrench_history.dtype)
        tactile_input = mx.concatenate(
            (wrench_history, presence[..., None]),
            axis=-1,
        )
        tactile_tokens = self.tactile_stem(tactile_input)
        tactile_tokens = (
            tactile_tokens
            + history_embedding[:, :, None, :]
            + self.sensor_embedding[None, None, :, :]
        )
        tactile_tokens = mx.where(
            wrench_presence[..., None],
            tactile_tokens,
            tactile_tokens + self.missing_tactile,
        ).reshape(
            (
                batch_size,
                config.history * config.sensor_count,
                config.model_dimensions,
            )
        )

        action_input = mx.concatenate(
            (
                action_tube,
                action_mask.astype(action_tube.dtype)[..., None],
            ),
            axis=-1,
        )
        action_tokens = self.action_stem(action_input)
        action_tokens = (
            action_tokens + self.horizon_embedding[None, :, :]
        )

        time_input = mx.concatenate(
            (
                _sinusoidal_embedding(
                    diffusion_time,
                    config.model_dimensions,
                ),
                _sinusoidal_embedding(
                    stream_time,
                    config.model_dimensions,
                ),
                mode[:, None],
            ),
            axis=-1,
        )
        time_token = self.time_stem(time_input)
        action_tokens = action_tokens + time_token[:, None, :]
        summary = mx.broadcast_to(
            self.summary_token,
            (batch_size, 1, config.model_dimensions),
        ) + time_token[:, None, :]

        tokens = [summary, state_tokens, tactile_tokens]
        if self.vision_stem is not None:
            if (
                vision.ndim != 6
                or vision.shape[:3]
                != (
                    batch_size,
                    config.history,
                    config.vision_views,
                )
                or vision.shape[3:5]
                != (config.vision_height, config.vision_width)
                or vision.shape[-1] != 3
            ):
                raise ValueError("policy vision history shape is invalid")
            flattened = vision.reshape(
                (
                    batch_size
                    * config.history
                    * config.vision_views,
                    vision.shape[3],
                    vision.shape[4],
                    3,
                )
            )
            vision_tokens = self.vision_stem(flattened).reshape(
                (
                    batch_size,
                    config.history,
                    config.vision_views,
                    config.model_dimensions,
                )
            )
            vision_tokens = (
                vision_tokens
                + history_embedding[:, :, None, :]
                + self.view_embedding[
                    None,
                    None,
                    : config.vision_views,
                    :,
                ]
            ).reshape(
                (
                    batch_size,
                    config.history * config.vision_views,
                    config.model_dimensions,
                )
            )
            tokens.append(vision_tokens)
        tokens.append(action_tokens)
        hidden = mx.concatenate(tokens, axis=1)
        for layer in self.transformer:
            hidden = layer(hidden, None)
        hidden = self.final_norm(hidden)
        action_hidden = hidden[:, -config.horizon :, :]
        field = self.field_head(action_hidden)
        wrench = self.wrench_head(hidden[:, 0, :]).reshape(
            (
                batch_size,
                2,
                config.sensor_count,
                config.wrench_dimensions,
            )
        )
        return TactileTubeOutput(
            field=field,
            reconstructed_wrench=wrench[:, 0, :, :],
            next_wrench=wrench[:, 1, :, :],
        )


def _mlx_version() -> str:
    try:
        from importlib.metadata import version

        return version("mlx")
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _config_payload(
    config: TactileTubePolicyConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    payload["vision_keys"] = list(config.vision_keys)
    return payload


def _config_from_payload(
    payload: dict[str, Any],
) -> TactileTubePolicyConfig:
    value = dict(payload)
    value["vision_keys"] = tuple(value.get("vision_keys", ()))
    config = TactileTubePolicyConfig(**value)
    config.validate()
    return config


class TactileTubeTrainer:
    """Compiled AdamW trainer whose mutable state stays inside MLX."""

    def __init__(
        self,
        config: TactileTubePolicyConfig,
        *,
        manifest: RobotDatasetManifest,
        stream_contract: TactileStreamContract,
    ) -> None:
        config.validate()
        manifest.validate()
        stream_contract.validate()
        expected = (
            manifest.state.dimensions,
            manifest.action.dimensions,
            manifest.wrench.dimensions,
        )
        actual = (
            config.state_dimensions,
            config.action_dimensions,
            config.sensor_count * config.wrench_dimensions,
        )
        if actual != expected:
            raise ValueError(
                f"policy dimensions {actual} do not match dataset "
                f"features {expected}"
            )
        if (
            manifest.stream_fingerprint
            != stream_contract.fingerprint
            or manifest.source_revision
            != stream_contract.source_revision
        ):
            raise ValueError(
                "dataset and physical stream provenance do not match"
            )
        self.config = config
        self.manifest = manifest
        self.stream_contract = stream_contract
        mx.random.seed(config.seed)
        self.model = TactileTubePolicy(config)
        self.ema_model = TactileTubePolicy(config)
        self.ema_model.update(self.model.parameters())
        self.ema_model.eval()
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            bias_correction=True,
        )
        self.optimizer.init(self.model.trainable_parameters())
        mx.eval(
            self.model.parameters(),
            self.ema_model.parameters(),
            self.optimizer.state,
        )
        self.step = 0
        self.examples = 0
        self._loss_and_grad = nn.value_and_grad(
            self.model,
            self._loss,
        )

        def update(
            state_history: mx.array,
            wrench_history: mx.array,
            action_chunk: mx.array,
            action_mask: mx.array,
            stream_state_history: mx.array,
            stream_wrench_history: mx.array,
            stream_action_chunk: mx.array,
            stream_action_derivative: mx.array,
            stream_action_mask: mx.array,
            stream_time: mx.array,
            next_wrench: mx.array,
            vision: mx.array,
            stream_vision: mx.array,
        ) -> dict[str, mx.array]:
            (loss, metrics), gradients = self._loss_and_grad(
                state_history,
                wrench_history,
                action_chunk,
                action_mask,
                stream_state_history,
                stream_wrench_history,
                stream_action_chunk,
                stream_action_derivative,
                stream_action_mask,
                stream_time,
                next_wrench,
                vision,
                stream_vision,
            )
            gradients, gradient_norm = optim.clip_grad_norm(
                gradients,
                self.config.max_gradient_norm,
            )
            self.optimizer.update(self.model, gradients)
            self.ema_model.update(
                tree_map(
                    lambda average, current: (
                        self.config.ema_decay * average
                        + (1.0 - self.config.ema_decay) * current
                    ),
                    self.ema_model.parameters(),
                    self.model.parameters(),
                )
            )
            metrics["gradient_norm"] = gradient_norm
            metrics["loss"] = loss
            return metrics

        captured_state = [
            self.model.state,
            self.ema_model.state,
            self.optimizer.state,
            mx.random.state,
        ]
        self._compiled_update = mx.compile(
            update,
            inputs=captured_state,
            outputs=captured_state,
        )

    def _loss(
        self,
        state_history: mx.array,
        wrench_history: mx.array,
        action_chunk: mx.array,
        action_mask: mx.array,
        stream_state_history: mx.array,
        stream_wrench_history: mx.array,
        stream_action_chunk: mx.array,
        stream_action_derivative: mx.array,
        stream_action_mask: mx.array,
        stream_time: mx.array,
        next_wrench: mx.array,
        vision: mx.array,
        stream_vision: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        config = self.config
        batch_size = action_chunk.shape[0]
        tactile_keep = (
            mx.random.uniform(
                shape=(
                    batch_size,
                    config.history,
                    config.sensor_count,
                )
            )
            >= config.tactile_mask_probability
        )
        stream_tactile_keep = (
            mx.random.uniform(
                shape=(
                    batch_size,
                    config.history,
                    config.sensor_count,
                )
            )
            >= config.tactile_mask_probability
        )

        diffusion_time = mx.random.uniform(
            shape=(batch_size,),
            low=1.0e-3,
            high=1.0,
        )
        noise = mx.random.normal(action_chunk.shape)
        angle = diffusion_time * (math.pi * 0.5)
        alpha = mx.cos(angle)[:, None, None]
        sigma = mx.sin(angle)[:, None, None]
        noisy_action = alpha * action_chunk + sigma * noise
        diffusion_output = self.model(
            state_history,
            wrench_history
            * tactile_keep[..., None].astype(
                wrench_history.dtype
            ),
            tactile_keep,
            noisy_action,
            action_mask,
            diffusion_time,
            mx.zeros_like(diffusion_time),
            mx.zeros_like(diffusion_time),
            vision,
        )
        diffusion_loss = _masked_mean_squared_error(
            diffusion_output.field,
            noise,
            action_mask,
        )

        normalized_stream_time = mx.clip(
            stream_time
            * config.stream_rate_hz
            / config.horizon,
            0.0,
            1.0,
        )
        stream_sigma = (
            config.stream_noise
            * mx.exp(
                -config.stream_noise_decay
                * normalized_stream_time
            )
        )
        stream_noise = mx.random.normal(stream_action_chunk.shape)
        perturbed_stream = (
            stream_action_chunk
            + stream_sigma[:, None, None] * stream_noise
        )
        stream_target = (
            stream_action_derivative
            - config.stream_attraction
            * (perturbed_stream - stream_action_chunk)
        )
        stream_output = self.model(
            stream_state_history,
            stream_wrench_history
            * stream_tactile_keep[..., None].astype(
                stream_wrench_history.dtype
            ),
            stream_tactile_keep,
            perturbed_stream,
            stream_action_mask,
            mx.zeros_like(normalized_stream_time),
            normalized_stream_time,
            mx.ones_like(normalized_stream_time),
            stream_vision,
        )
        stream_loss = _masked_mean_squared_error(
            stream_output.field,
            stream_target,
            stream_action_mask,
        )
        current_wrench = stream_wrench_history[:, -1, :, :]
        tactile_loss = mx.mean(
            mx.square(
                stream_output.reconstructed_wrench - current_wrench
            )
        )
        next_tactile_loss = mx.mean(
            mx.square(stream_output.next_wrench - next_wrench)
        )
        total = (
            config.diffusion_loss_weight * diffusion_loss
            + config.stream_loss_weight * stream_loss
            + config.tactile_reconstruction_weight * tactile_loss
            + config.next_tactile_weight * next_tactile_loss
        )
        return total, {
            "diffusion_loss": diffusion_loss,
            "stream_loss": stream_loss,
            "tactile_reconstruction_loss": tactile_loss,
            "next_tactile_loss": next_tactile_loss,
        }

    def _normalize_batch(
        self,
        batch: OrigamiBatch,
    ) -> tuple[np.ndarray, ...]:
        state_normalization = self.manifest.state_normalization
        action_normalization = self.manifest.action_normalization
        wrench_normalization = self.manifest.wrench_normalization
        action_scale = np.maximum(
            np.asarray(
                action_normalization.std,
                dtype=np.float32,
            ),
            np.float32(1.0e-6),
        )

        def wrench(value: np.ndarray) -> np.ndarray:
            normalized = wrench_normalization.normalize(value)
            return normalized.reshape(
                (
                    *normalized.shape[:-1],
                    self.config.sensor_count,
                    self.config.wrench_dimensions,
                )
            )

        return (
            state_normalization.normalize(batch.state_history),
            wrench(batch.wrench_history),
            action_normalization.normalize(batch.action_chunk),
            np.asarray(batch.action_mask, dtype=np.bool_),
            state_normalization.normalize(
                batch.stream_state_history
            ),
            wrench(batch.stream_wrench_history),
            action_normalization.normalize(
                batch.stream_action_chunk
            ),
            np.ascontiguousarray(
                batch.stream_action_derivative / action_scale
            ),
            np.asarray(batch.stream_action_mask, dtype=np.bool_),
            np.asarray(batch.stream_time, dtype=np.float32),
            wrench_normalization.normalize(
                batch.next_wrench
            ).reshape(
                (
                    len(batch.next_wrench),
                    self.config.sensor_count,
                    self.config.wrench_dimensions,
                )
            ),
        )

    def train_batch(
        self,
        batch: OrigamiBatch,
        *,
        vision: npt.ArrayLike | None = None,
        stream_vision: npt.ArrayLike | None = None,
    ) -> dict[str, float]:
        """Run one compiled Metal-backed optimization step."""

        normalized = self._normalize_batch(batch)
        batch_size = normalized[0].shape[0]
        if self.config.vision_views:
            if vision is None or stream_vision is None:
                raise ValueError(
                    "vision-enabled policy requires both histories"
                )
            vision_array = np.asarray(vision, dtype=np.float32)
            stream_vision_array = np.asarray(
                stream_vision,
                dtype=np.float32,
            )
            if vision_array.max(initial=0.0) > 1.0:
                vision_array = vision_array / np.float32(255.0)
            if stream_vision_array.max(initial=0.0) > 1.0:
                stream_vision_array = (
                    stream_vision_array / np.float32(255.0)
                )
        else:
            vision_array = np.zeros(
                (
                    batch_size,
                    self.config.history,
                    0,
                    1,
                    1,
                    3,
                ),
                dtype=np.float32,
            )
            stream_vision_array = vision_array
        self.model.train()
        started = time.perf_counter()
        metrics = self._compiled_update(
            *(mx.array(value) for value in normalized),
            mx.array(vision_array),
            mx.array(stream_vision_array),
        )
        mx.eval(
            metrics,
            self.model.parameters(),
            self.optimizer.state,
        )
        self.step += 1
        self.examples += batch_size
        result = {
            key: float(value.item())
            for key, value in metrics.items()
        }
        result["step_seconds"] = time.perf_counter() - started
        result["examples"] = float(self.examples)
        return result

    def save_checkpoint(
        self,
        directory: str | os.PathLike[str],
    ) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.ema_model.save_weights(
            str(destination / "model.safetensors")
        )
        self.model.save_weights(
            str(destination / "training_model.safetensors")
        )
        mx.save_safetensors(
            str(destination / "optimizer.safetensors"),
            dict(tree_flatten(self.optimizer.state)),
        )
        config_path = destination / "config.json"
        config_path.write_text(
            json.dumps(
                _config_payload(self.config),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_hashes = {
            name: _sha256_file(destination / name)
            for name in (
                "model.safetensors",
                "training_model.safetensors",
                "optimizer.safetensors",
                "config.json",
            )
        }
        promotion_blockers = []
        if not self.stream_contract.action.verified:
            promotion_blockers.append(
                "action_semantics_unverified"
            )
        if not self.stream_contract.wrench_verified:
            promotion_blockers.append(
                "wrench_units_or_frame_unverified"
            )
        provenance = {
            "schema": TACTILE_TUBE_CHECKPOINT_SCHEMA,
            "format_version": TACTILE_TUBE_CHECKPOINT_FORMAT,
            "step": self.step,
            "examples": self.examples,
            "dataset_manifest_fingerprint": (
                self.manifest.fingerprint
            ),
            "stream_fingerprint": self.stream_contract.fingerprint,
            "source_repository": (
                self.stream_contract.source_repository
            ),
            "source_revision": self.stream_contract.source_revision,
            "canonical_target_fingerprint": (
                self.stream_contract.canonical_target_fingerprint
            ),
            "action_semantics_verified": (
                self.stream_contract.action.verified
            ),
            "wrench_contract_verified": (
                self.stream_contract.wrench_verified
            ),
            "hardware_deployable": (
                not promotion_blockers
            ),
            "promotion_blockers": promotion_blockers,
            "artifact_sha256": artifact_hashes,
            "capabilities": {
                "modalities": [
                    "joint_state",
                    "fingertip_wrench",
                    *(
                        ["multi_view_rgb"]
                        if self.config.vision_views
                        else []
                    ),
                ],
                "inference": [
                    "ddim_action_tube",
                    "streaming_feedback_field",
                    "next_wrench_prediction",
                    "ema_inference_weights",
                ],
                "execution_backend": "apple_mlx_metal",
                "sensor_order": list(
                    self.stream_contract.sensors
                ),
                "normalization_source": "training_seasons_only",
                "evaluation_partition": "whole_season",
            },
            "runtime": {
                "framework": "mlx",
                "mlx_version": _mlx_version(),
                "device": str(mx.default_device()),
                "machine": platform.machine(),
            },
        }
        (destination / "provenance.json").write_text(
            json.dumps(
                provenance,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination

    def load_checkpoint(
        self,
        directory: str | os.PathLike[str],
    ) -> None:
        checkpoint = Path(directory).expanduser()
        config = json.loads(
            (checkpoint / "config.json").read_text(
                encoding="utf-8"
            )
        )
        if config != _config_payload(self.config):
            raise ValueError(
                "checkpoint policy configuration does not match"
            )
        provenance = _validated_provenance(
            checkpoint,
            manifest=self.manifest,
            stream_contract=self.stream_contract,
        )
        self.model.load_weights(
            str(checkpoint / "training_model.safetensors")
        )
        self.ema_model.load_weights(
            str(checkpoint / "model.safetensors")
        )
        optimizer_flat = mx.load(
            str(checkpoint / "optimizer.safetensors")
        )
        self.optimizer.state = tree_unflatten(
            list(optimizer_flat.items())
        )
        self.step = int(provenance["step"])
        self.examples = int(provenance["examples"])
        mx.eval(
            self.model.parameters(),
            self.ema_model.parameters(),
            self.optimizer.state,
        )


def _validated_provenance(
    checkpoint: Path,
    *,
    manifest: RobotDatasetManifest,
    stream_contract: TactileStreamContract,
) -> dict[str, Any]:
    provenance = json.loads(
        (checkpoint / "provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        provenance.get("schema")
        != TACTILE_TUBE_CHECKPOINT_SCHEMA
        or provenance.get("format_version")
        != TACTILE_TUBE_CHECKPOINT_FORMAT
        or provenance.get("dataset_manifest_fingerprint")
        != manifest.fingerprint
        or provenance.get("stream_fingerprint")
        != stream_contract.fingerprint
        or provenance.get("source_revision")
        != stream_contract.source_revision
    ):
        raise ValueError(
            "checkpoint provenance does not match the requested dataset"
        )
    hashes = provenance.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("checkpoint artifact hashes are missing")
    for name in (
        "model.safetensors",
        "training_model.safetensors",
        "optimizer.safetensors",
        "config.json",
    ):
        expected = hashes.get(name)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256_file(checkpoint / name) != expected
        ):
            raise ValueError(
                f"checkpoint artifact hash differs for {name}"
            )
    return provenance


class TactileTubeRuntime:
    """DDIM tube generation plus one-frame streaming tactile correction."""

    def __init__(
        self,
        model: TactileTubePolicy,
        *,
        manifest: RobotDatasetManifest,
        stream_contract: TactileStreamContract,
    ) -> None:
        manifest.validate()
        stream_contract.validate()
        if manifest.stream_fingerprint != stream_contract.fingerprint:
            raise ValueError(
                "runtime dataset and tactile stream do not match"
            )
        self.model = model
        self.config = model.config
        self.manifest = manifest
        self.stream_contract = stream_contract
        self.model.eval()

        def forward(
            state: mx.array,
            wrench: mx.array,
            presence: mx.array,
            tube: mx.array,
            mask: mx.array,
            diffusion_time: mx.array,
            stream_time: mx.array,
            mode: mx.array,
            vision: mx.array,
        ) -> TactileTubeOutput:
            return self.model(
                state,
                wrench,
                presence,
                tube,
                mask,
                diffusion_time,
                stream_time,
                mode,
                vision,
            )

        self._compiled_forward = mx.compile(
            forward,
            inputs=[self.model.state],
        )

    def _bounded_action(
        self,
        normalized: npt.ArrayLike,
    ) -> npt.NDArray[np.float32]:
        physical = self.manifest.action_normalization.denormalize(
            normalized
        )
        minimum = np.asarray(
            self.manifest.action_normalization.minimum,
            dtype=np.float32,
        )
        maximum = np.asarray(
            self.manifest.action_normalization.maximum,
            dtype=np.float32,
        )
        return np.ascontiguousarray(
            np.clip(physical, minimum, maximum),
            dtype=np.float32,
        )

    @classmethod
    def from_checkpoint(
        cls,
        directory: str | os.PathLike[str],
        *,
        manifest: RobotDatasetManifest,
        stream_contract: TactileStreamContract,
    ) -> "TactileTubeRuntime":
        checkpoint = Path(directory).expanduser()
        _validated_provenance(
            checkpoint,
            manifest=manifest,
            stream_contract=stream_contract,
        )
        config_payload = json.loads(
            (checkpoint / "config.json").read_text(
                encoding="utf-8"
            )
        )
        config = _config_from_payload(config_payload)
        model = TactileTubePolicy(config)
        model.load_weights(str(checkpoint / "model.safetensors"))
        mx.eval(model.parameters())
        return cls(
            model,
            manifest=manifest,
            stream_contract=stream_contract,
        )

    def assert_hardware_ready(self) -> None:
        if (
            not self.stream_contract.action.verified
            or not self.stream_contract.wrench_verified
        ):
            raise RuntimeError(
                "hardware execution is blocked: action semantics and "
                "wrench units/frame have not been verified"
            )

    def _observation(
        self,
        state_history: npt.ArrayLike,
        wrench_history: npt.ArrayLike,
        vision: npt.ArrayLike | None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        config = self.config
        state = self.manifest.state_normalization.normalize(
            state_history
        )
        wrench = self.manifest.wrench_normalization.normalize(
            wrench_history
        ).reshape(
            (
                -1,
                config.history,
                config.sensor_count,
                config.wrench_dimensions,
            )
        )
        state = state.reshape(
            (-1, config.history, config.state_dimensions)
        )
        presence = np.ones(
            (
                state.shape[0],
                config.history,
                config.sensor_count,
            ),
            dtype=np.bool_,
        )
        if config.vision_views:
            if vision is None:
                raise ValueError(
                    "vision-enabled policy requires image history"
                )
            vision_value = np.asarray(vision, dtype=np.float32)
            if vision_value.max(initial=0.0) > 1.0:
                vision_value = vision_value / np.float32(255.0)
        else:
            vision_value = np.zeros(
                (
                    state.shape[0],
                    config.history,
                    0,
                    1,
                    1,
                    3,
                ),
                dtype=np.float32,
            )
        return (
            mx.array(state),
            mx.array(wrench),
            mx.array(presence),
            mx.array(vision_value),
        )

    def initial_tube(
        self,
        state_history: npt.ArrayLike,
        wrench_history: npt.ArrayLike,
        *,
        vision: npt.ArrayLike | None = None,
        seed: int | None = None,
    ) -> npt.NDArray[np.float32]:
        config = self.config
        if seed is not None:
            mx.random.seed(seed)
        state, wrench, presence, vision_value = self._observation(
            state_history,
            wrench_history,
            vision,
        )
        batch_size = state.shape[0]
        tube = mx.random.normal(
            (
                batch_size,
                config.horizon,
                config.action_dimensions,
            )
        )
        mask = mx.ones(
            (batch_size, config.horizon),
            dtype=mx.bool_,
        )
        schedule = np.linspace(
            1.0,
            0.0,
            config.diffusion_steps + 1,
            dtype=np.float32,
        )
        for index in range(config.diffusion_steps):
            current = float(schedule[index])
            following = float(schedule[index + 1])
            time_value = mx.full(
                (batch_size,),
                current,
                dtype=mx.float32,
            )
            prediction = self._compiled_forward(
                state,
                wrench,
                presence,
                tube,
                mask,
                time_value,
                mx.zeros_like(time_value),
                mx.zeros_like(time_value),
                vision_value,
            ).field
            angle = current * (math.pi * 0.5)
            next_angle = following * (math.pi * 0.5)
            alpha = max(math.cos(angle), 1.0e-4)
            sigma = math.sin(angle)
            clean = (tube - sigma * prediction) / alpha
            tube = (
                math.cos(next_angle) * clean
                + math.sin(next_angle) * prediction
            )
        mx.eval(tube)
        normalized = np.asarray(tube, dtype=np.float32)
        return self._bounded_action(normalized)

    def stream_decision(
        self,
        action_tube: npt.ArrayLike,
        state_history: npt.ArrayLike,
        wrench_history: npt.ArrayLike,
        *,
        elapsed_seconds: npt.ArrayLike,
        vision: npt.ArrayLike | None = None,
    ) -> TactileStreamDecision:
        config = self.config
        state, wrench, presence, vision_value = self._observation(
            state_history,
            wrench_history,
            vision,
        )
        physical = np.asarray(action_tube, dtype=np.float32).reshape(
            (
                state.shape[0],
                config.horizon,
                config.action_dimensions,
            )
        )
        shifted = np.concatenate(
            (physical[:, 1:, :], physical[:, -1:, :]),
            axis=1,
        )
        normalized = self.manifest.action_normalization.normalize(
            shifted
        )
        tube = mx.array(normalized)
        batch_size = tube.shape[0]
        elapsed = np.asarray(elapsed_seconds, dtype=np.float32)
        if elapsed.ndim == 0:
            elapsed = np.full(
                (batch_size,),
                elapsed,
                dtype=np.float32,
            )
        if (
            elapsed.shape != (batch_size,)
            or not np.all(np.isfinite(elapsed))
            or np.any(elapsed < 0.0)
        ):
            raise ValueError("stream elapsed time is invalid")
        stream_time = np.clip(
            elapsed * config.stream_rate_hz / config.horizon,
            0.0,
            1.0,
        )
        output = self._compiled_forward(
            state,
            wrench,
            presence,
            tube,
            mx.ones(
                (batch_size, config.horizon),
                dtype=mx.bool_,
            ),
            mx.zeros((batch_size,), dtype=mx.float32),
            mx.array(stream_time),
            mx.ones((batch_size,), dtype=mx.float32),
            vision_value,
        )
        corrected = tube + output.field / config.stream_rate_hz
        mx.eval(
            corrected,
            output.reconstructed_wrench,
            output.next_wrench,
        )
        action = self._bounded_action(
            np.asarray(corrected, dtype=np.float32)
        )
        reconstructed = (
            self.manifest.wrench_normalization.denormalize(
                np.asarray(
                    output.reconstructed_wrench,
                    dtype=np.float32,
                ).reshape((batch_size, -1))
            ).reshape(
                (
                    batch_size,
                    config.sensor_count,
                    config.wrench_dimensions,
                )
            )
        )
        predicted_next = (
            self.manifest.wrench_normalization.denormalize(
                np.asarray(
                    output.next_wrench,
                    dtype=np.float32,
                ).reshape((batch_size, -1))
            ).reshape(
                (
                    batch_size,
                    config.sensor_count,
                    config.wrench_dimensions,
                )
            )
        )
        return TactileStreamDecision(
            action_tube=action,
            reconstructed_wrench=np.ascontiguousarray(
                reconstructed,
                dtype=np.float32,
            ),
            predicted_next_wrench=np.ascontiguousarray(
                predicted_next,
                dtype=np.float32,
            ),
            correction_rms=np.sqrt(
                np.mean(
                    np.square(action - shifted),
                    axis=(1, 2),
                )
            ).astype(np.float32),
        )

    def stream_update(
        self,
        action_tube: npt.ArrayLike,
        state_history: npt.ArrayLike,
        wrench_history: npt.ArrayLike,
        *,
        elapsed_seconds: npt.ArrayLike,
        vision: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float32]:
        return self.stream_decision(
            action_tube,
            state_history,
            wrench_history,
            elapsed_seconds=elapsed_seconds,
            vision=vision,
        ).action_tube


__all__ = [
    "TACTILE_TUBE_CHECKPOINT_FORMAT",
    "TACTILE_TUBE_CHECKPOINT_SCHEMA",
    "TactileTubeOutput",
    "TactileTubePolicy",
    "TactileTubePolicyConfig",
    "TactileTubeRuntime",
    "TactileTubeTrainer",
    "TactileStreamDecision",
]
