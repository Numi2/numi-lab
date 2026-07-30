"""MLX camera-to-metric-depth translator for real tactile sensors.

This model is never used by simulation. It consumes physical camera frames and
predicts the exact metric penetration representation defined by a cooked
MetalRobo tactile observation contract.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import numpy.typing as npt

from .tactile import (
    TactileCalibrationRecord,
    TactileObservationContract,
)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, value: mx.array) -> mx.array:
        hidden = nn.relu(self.conv1(value))
        return nn.relu(value + self.conv2(hidden))


class MetricDepthTranslator(nn.Module):
    """Small Core-ML-friendly residual encoder-decoder with NHWC tensors."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        base_channels: int = 32,
        maximum_depth_m: float,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels <= 0:
            raise ValueError("translator channel counts must be positive")
        if not np.isfinite(maximum_depth_m) or maximum_depth_m <= 0.0:
            raise ValueError("maximum_depth_m must be positive")
        self.maximum_depth_m = float(maximum_depth_m)
        self.stem = nn.Conv2d(input_channels, base_channels, 5, padding=2)
        self.resolution1 = _ResidualBlock(base_channels)
        self.down1 = nn.Conv2d(
            base_channels,
            2 * base_channels,
            4,
            stride=2,
            padding=1,
        )
        self.resolution2 = _ResidualBlock(2 * base_channels)
        self.down2 = nn.Conv2d(
            2 * base_channels,
            4 * base_channels,
            4,
            stride=2,
            padding=1,
        )
        self.bottleneck = _ResidualBlock(4 * base_channels)
        self.up1 = nn.ConvTranspose2d(
            4 * base_channels,
            2 * base_channels,
            4,
            stride=2,
            padding=1,
        )
        self.decode1 = nn.Conv2d(
            4 * base_channels,
            2 * base_channels,
            3,
            padding=1,
        )
        self.up2 = nn.ConvTranspose2d(
            2 * base_channels,
            base_channels,
            4,
            stride=2,
            padding=1,
        )
        self.decode2 = nn.Conv2d(
            2 * base_channels,
            base_channels,
            3,
            padding=1,
        )
        self.output = nn.Conv2d(base_channels, 1, 1)

    def __call__(self, raw_frame: mx.array) -> mx.array:
        if raw_frame.ndim != 4:
            raise ValueError("translator input must be [batch,height,width,channels]")
        height = raw_frame.shape[1]
        width = raw_frame.shape[2]
        padded_height = (-height) % 4
        padded_width = (-width) % 4
        if padded_height or padded_width:
            raw_frame = mx.pad(
                raw_frame,
                (
                    (0, 0),
                    (0, padded_height),
                    (0, padded_width),
                    (0, 0),
                ),
            )
        first = self.resolution1(nn.relu(self.stem(raw_frame)))
        second = self.resolution2(nn.relu(self.down1(first)))
        latent = self.bottleneck(nn.relu(self.down2(second)))
        decoded = nn.relu(
            self.decode1(
                mx.concatenate((self.up1(latent), second), axis=-1)
            )
        )
        decoded = nn.relu(
            self.decode2(
                mx.concatenate((self.up2(decoded), first), axis=-1)
            )
        )
        translated = (
            mx.sigmoid(self.output(decoded))
            * self.maximum_depth_m
        )
        return translated[:, :height, :width, :]


@dataclass(frozen=True, slots=True)
class TranslatorTrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3.0e-4
    base_channels: int = 32
    input_channels: int = 3
    seed: int = 1
    gradient_loss_weight: float = 0.0

    def validate(self) -> None:
        if (
            self.epochs <= 0
            or self.batch_size <= 0
            or self.learning_rate <= 0.0
            or self.base_channels <= 0
            or self.input_channels <= 0
            or self.gradient_loss_weight < 0.0
        ):
            raise ValueError("tactile translator training config is invalid")


def _load_array(uri: str, *, raw_frame: bool) -> npt.NDArray[np.float32]:
    path = Path(uri).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            preferred = "raw_frame" if raw_frame else "depth_m"
            if preferred in archive:
                array = archive[preferred]
            elif len(archive.files) == 1:
                array = archive[archive.files[0]]
            else:
                raise ValueError(
                    f"{path} must contain {preferred!r} or one array"
                )
    else:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Pillow is required for non-NPY tactile camera frames"
            ) from error
        array = np.asarray(Image.open(path))
    result = np.asarray(array, dtype=np.float32)
    if raw_frame:
        if result.ndim == 2:
            result = result[..., None]
        if result.ndim != 3:
            raise ValueError(f"raw tactile frame has invalid shape: {result.shape}")
        if np.max(result, initial=0.0) > 1.0:
            result /= np.float32(255.0)
    else:
        result = np.squeeze(result)
        if result.ndim != 2:
            raise ValueError(f"metric depth target has invalid shape: {result.shape}")
        result = result[..., None]
    return result


def load_calibration_manifest(
    manifest: str | os.PathLike[str],
    *,
    sensor_id: str | None = None,
) -> tuple[TactileCalibrationRecord, ...]:
    manifest_path = Path(manifest).expanduser().resolve()

    def resolved_uri(uri: str | None) -> str | None:
        if uri is None:
            return None
        path = Path(uri).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        return str(path.resolve())

    records: list[TactileCalibrationRecord] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = TactileCalibrationRecord(**json.loads(line))
            record.validate()
            record = replace(
                record,
                raw_frame_uri=resolved_uri(record.raw_frame_uri),
                target_depth_uri=resolved_uri(record.target_depth_uri),
                validity_uri=resolved_uri(record.validity_uri),
                force_torque_uri=resolved_uri(record.force_torque_uri),
            )
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                f"invalid tactile calibration row {line_number}: {error}"
            ) from error
        if sensor_id is None or record.sensor_id == sensor_id:
            records.append(record)
    if not records:
        raise ValueError("tactile calibration manifest contains no selected rows")
    return tuple(records)


class TactileTranslatorTrainer:
    def __init__(
        self,
        contract: TactileObservationContract,
        sensor_id: str,
        config: TranslatorTrainingConfig = TranslatorTrainingConfig(),
    ) -> None:
        config.validate()
        matching = [sensor for sensor in contract.sensors if sensor.id == sensor_id]
        if len(matching) != 1:
            raise ValueError(f"unknown or duplicate tactile sensor {sensor_id!r}")
        self.contract = contract
        self.sensor = matching[0]
        self.config = config
        self.model = MetricDepthTranslator(
            input_channels=config.input_channels,
            maximum_depth_m=self.sensor.maximum_depth_m,
            base_channels=config.base_channels,
        )
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)

    def _loss(
        self,
        raw_frames: mx.array,
        targets: mx.array,
        validity: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        predicted = self.model(raw_frames)
        squared = mx.square(predicted - targets) * validity
        denominator = mx.maximum(mx.sum(validity), mx.array(1.0))
        pixel_mse = mx.sum(squared) / denominator
        gradient_loss = mx.array(0.0)
        if self.config.gradient_loss_weight > 0.0:
            predicted_x = predicted[:, :, 1:, :] - predicted[:, :, :-1, :]
            target_x = targets[:, :, 1:, :] - targets[:, :, :-1, :]
            predicted_y = predicted[:, 1:, :, :] - predicted[:, :-1, :, :]
            target_y = targets[:, 1:, :, :] - targets[:, :-1, :, :]
            gradient_loss = (
                mx.mean(mx.abs(predicted_x - target_x))
                + mx.mean(mx.abs(predicted_y - target_y))
            )
        loss = pixel_mse + self.config.gradient_loss_weight * gradient_loss
        return loss, {
            "loss": loss,
            "pixel_mse_m2": pixel_mse,
            "gradient_l1_m": gradient_loss,
        }

    def _batches(
        self,
        records: Sequence[TactileCalibrationRecord],
        *,
        epoch: int,
    ) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
        generator = np.random.default_rng(self.config.seed + epoch)
        indices = generator.permutation(len(records))
        for start in range(0, len(records), self.config.batch_size):
            selected = indices[start : start + self.config.batch_size]
            raw_batch: list[npt.NDArray[np.float32]] = []
            target_batch: list[npt.NDArray[np.float32]] = []
            validity_batch: list[npt.NDArray[np.float32]] = []
            for index in selected:
                record = records[int(index)]
                raw = _load_array(record.raw_frame_uri, raw_frame=True)
                target = _load_array(
                    record.target_depth_uri,
                    raw_frame=False,
                )
                if target.shape[:2] != (
                    self.sensor.height,
                    self.sensor.width,
                ):
                    raise ValueError(
                        f"{record.target_depth_uri} shape {target.shape[:2]} "
                        f"does not match {(self.sensor.height, self.sensor.width)}"
                    )
                if raw.shape[:2] != target.shape[:2]:
                    raise ValueError(
                        "camera frame and metric depth target must already be "
                        "registered to the canonical atlas"
                    )
                if raw.shape[-1] != self.config.input_channels:
                    raise ValueError(
                        f"{record.raw_frame_uri} has {raw.shape[-1]} "
                        f"channels; expected {self.config.input_channels}"
                    )
                valid = (
                    np.isfinite(target)
                    & (target >= 0.0)
                    & (target <= self.sensor.maximum_depth_m)
                ).astype(np.float32)
                if record.validity_uri is not None:
                    supplied_validity = _load_array(
                        record.validity_uri,
                        raw_frame=False,
                    )
                    if supplied_validity.shape != target.shape:
                        raise ValueError(
                            f"{record.validity_uri} shape "
                            f"{supplied_validity.shape} does not match "
                            f"{target.shape}"
                        )
                    valid *= (
                        np.isfinite(supplied_validity)
                        & (supplied_validity > 0.5)
                    ).astype(np.float32)
                target = np.nan_to_num(
                    target,
                    nan=0.0,
                    posinf=self.sensor.maximum_depth_m,
                    neginf=0.0,
                )
                raw_batch.append(raw)
                target_batch.append(target)
                validity_batch.append(valid)
            yield (
                mx.array(np.stack(raw_batch)),
                mx.array(np.stack(target_batch)),
                mx.array(np.stack(validity_batch)),
            )

    def train(
        self,
        records: Sequence[TactileCalibrationRecord],
    ) -> tuple[dict[str, float], ...]:
        if not records:
            raise ValueError("translator training requires calibration records")
        for record in records:
            if record.sensor_id != self.sensor.id:
                raise ValueError(
                    "translator record sensor does not match the selected sensor"
                )
        history: list[dict[str, float]] = []
        for epoch in range(self.config.epochs):
            totals: dict[str, float] = {}
            batches = 0
            for raw, target, validity in self._batches(records, epoch=epoch):
                (_, metrics), gradients = self._loss_and_grad(
                    raw,
                    target,
                    validity,
                )
                self.optimizer.update(self.model, gradients)
                mx.eval(self.model.parameters(), self.optimizer.state, metrics)
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value.item())
                batches += 1
            history.append(
                {
                    "epoch": float(epoch + 1),
                    **{
                        name: value / max(batches, 1)
                        for name, value in totals.items()
                    },
                }
            )
        return tuple(history)

    def save(self, directory: str | os.PathLike[str]) -> Path:
        output = Path(directory).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(output / "model.safetensors"))
        translator_contract = {
            "schema": "metalrobo.metric_depth_translator",
            "model": "metalrobo.metric_depth_translator.residual_encoder_decoder",
            "training_runtime": "MLX",
            "simulation_dependency": False,
            "sensor_id": self.sensor.id,
            "tactile_observation_fingerprint": self.contract.fingerprint,
            "input": {
                "layout": "NHWC",
                "dtype": "float32",
                "range": [0.0, 1.0],
            },
            "output": {
                "name": "penetration_depth_m",
                "layout": "NHWC",
                "shape": [1, self.sensor.height, self.sensor.width, 1],
                "dtype": "float32",
                "unit": "m",
                "range": [0.0, self.sensor.maximum_depth_m],
            },
            "coreml_export": {
                "compatible_operators": [
                    "convolution",
                    "convolution_transpose",
                    "relu",
                    "sigmoid",
                    "concatenation",
                ],
                "conversion_validated": False,
            },
            "training_config": asdict(self.config),
        }
        path = output / "translator_contract.json"
        path.write_text(
            json.dumps(
                translator_contract,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return path


__all__ = [
    "MetricDepthTranslator",
    "TactileTranslatorTrainer",
    "TranslatorTrainingConfig",
    "load_calibration_manifest",
]
