"""Shared dense controller and low-rank adapter-basis artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import (
    HYPER_BASE_FORMAT, _ACTIVATION_ELU, _ACTIVATION_IDENTITY,
    _array_bytes, _atomic_directory, _canonical_json,
    _verify_file_hashes, _write_array_files,
)

@dataclass(frozen=True, slots=True)
class HyperBaseLayer:
    input_count: int
    output_count: int
    rank: int
    activation: int
    weight: np.ndarray
    bias: np.ndarray
    adapter_down: np.ndarray
    adapter_up: np.ndarray
    adapter_bias_basis: np.ndarray

    def validate(self) -> None:
        if (
            self.input_count <= 0
            or self.output_count <= 0
            or self.rank <= 0
            or self.rank > min(self.input_count, self.output_count)
            or self.activation not in (_ACTIVATION_IDENTITY, _ACTIVATION_ELU)
            or self.weight.shape != (self.output_count, self.input_count)
            or self.bias.shape != (self.output_count,)
            or self.adapter_down.shape != (self.rank, self.input_count)
            or self.adapter_up.shape != (self.output_count, self.rank)
            or self.adapter_bias_basis.shape != (self.output_count, self.rank)
        ):
            raise ValueError("hyper-base layer topology is invalid")
        if not all(
            np.isfinite(value).all()
            for value in (
                self.weight,
                self.bias,
                self.adapter_down,
                self.adapter_up,
                self.adapter_bias_basis,
            )
        ):
            raise ValueError("hyper-base layer contains non-finite parameters")


@dataclass(frozen=True, slots=True)
class HyperBasePolicy:
    id: str
    revision: int
    world_fingerprint: int
    task_fingerprint: int
    observation_fingerprint: int
    action_fingerprint: int
    observation_mean: np.ndarray
    observation_inverse_standard_deviation: np.ndarray
    layers: tuple[HyperBaseLayer, ...]
    coefficient_limits: np.ndarray
    action_bias: np.ndarray
    action_scale: np.ndarray
    observation_clip: float
    action_clip: float
    fingerprint: str = ""

    @property
    def observation_count(self) -> int:
        return self.layers[0].input_count

    @property
    def action_count(self) -> int:
        return self.layers[-1].output_count

    @property
    def coefficient_count(self) -> int:
        return sum(layer.rank for layer in self.layers)

    @property
    def coefficient_slices(self) -> tuple[slice, ...]:
        result: list[slice] = []
        offset = 0
        for layer in self.layers:
            result.append(slice(offset, offset + layer.rank))
            offset += layer.rank
        return tuple(result)

    def validate(self, *, require_fingerprint: bool = False) -> None:
        if (
            not self.id
            or self.revision <= 0
            or min(
                self.world_fingerprint,
                self.task_fingerprint,
                self.observation_fingerprint,
                self.action_fingerprint,
            ) <= 0
            or not self.layers
            or self.observation_mean.shape != (self.observation_count,)
            or self.observation_inverse_standard_deviation.shape != (
                self.observation_count,
            )
            or self.coefficient_limits.shape != (self.coefficient_count,)
            or self.action_bias.shape != (self.action_count,)
            or self.action_scale.shape != (self.action_count,)
            or not math.isfinite(self.observation_clip)
            or self.observation_clip <= 0.0
            or not math.isfinite(self.action_clip)
            or self.action_clip <= 0.0
            or (require_fingerprint and len(self.fingerprint) != 64)
        ):
            raise ValueError("hyper-base policy contract is invalid")
        for index, layer in enumerate(self.layers):
            layer.validate()
            if index and layer.input_count != self.layers[index - 1].output_count:
                raise ValueError("hyper-base layer chain is disconnected")
            final = index + 1 == len(self.layers)
            if final != (layer.activation == _ACTIVATION_IDENTITY):
                raise ValueError(
                    "only the final hyper-base layer may use identity activation"
                )
        if not all(
            np.isfinite(value).all()
            for value in (
                self.observation_mean,
                self.observation_inverse_standard_deviation,
                self.coefficient_limits,
                self.action_bias,
                self.action_scale,
            )
        ):
            raise ValueError("hyper-base policy contains non-finite values")
        if (
            np.any(self.observation_inverse_standard_deviation <= 0.0)
            or np.any(self.coefficient_limits <= 0.0)
            or np.any(np.abs(self.action_scale) <= 1.0e-12)
        ):
            raise ValueError("hyper-base normalization, limits, or action scale is invalid")

    def computed_fingerprint(self) -> str:
        self.validate(require_fingerprint=False)
        digest = hashlib.sha256()
        metadata = {
            "format": HYPER_BASE_FORMAT,
            "id": self.id,
            "revision": int(self.revision),
            "world_fingerprint": int(self.world_fingerprint),
            "task_fingerprint": int(self.task_fingerprint),
            "observation_fingerprint": int(self.observation_fingerprint),
            "action_fingerprint": int(self.action_fingerprint),
            "observation_clip": float(self.observation_clip),
            "action_clip": float(self.action_clip),
            "layers": [
                {
                    "input": layer.input_count,
                    "output": layer.output_count,
                    "rank": layer.rank,
                    "activation": layer.activation,
                }
                for layer in self.layers
            ],
        }
        digest.update(_canonical_json(metadata))
        for value in (
            self.observation_mean,
            self.observation_inverse_standard_deviation,
            self.coefficient_limits,
            self.action_bias,
            self.action_scale,
        ):
            digest.update(_array_bytes(value))
        for layer in self.layers:
            for value in (
                layer.weight,
                layer.bias,
                layer.adapter_down,
                layer.adapter_up,
                layer.adapter_bias_basis,
            ):
                digest.update(_array_bytes(value))
        return digest.hexdigest()

    def with_fingerprint(self) -> "HyperBasePolicy":
        return HyperBasePolicy(
            id=self.id,
            revision=self.revision,
            world_fingerprint=self.world_fingerprint,
            task_fingerprint=self.task_fingerprint,
            observation_fingerprint=self.observation_fingerprint,
            action_fingerprint=self.action_fingerprint,
            observation_mean=self.observation_mean,
            observation_inverse_standard_deviation=(
                self.observation_inverse_standard_deviation
            ),
            layers=self.layers,
            coefficient_limits=self.coefficient_limits,
            action_bias=self.action_bias,
            action_scale=self.action_scale,
            observation_clip=self.observation_clip,
            action_clip=self.action_clip,
            fingerprint=self.computed_fingerprint(),
        )

    def write(self, directory: str | Path) -> Path:
        policy = self.with_fingerprint()
        target = Path(directory)
        with _atomic_directory(target) as staging:
            arrays: dict[str, np.ndarray] = {
                "observation_mean": policy.observation_mean,
                "observation_inverse_standard_deviation": (
                    policy.observation_inverse_standard_deviation
                ),
                "coefficient_limits": policy.coefficient_limits,
                "action_bias": policy.action_bias,
                "action_scale": policy.action_scale,
            }
            for index, layer in enumerate(policy.layers):
                prefix = f"layer_{index:02d}"
                arrays[f"{prefix}_weight"] = layer.weight
                arrays[f"{prefix}_bias"] = layer.bias
                arrays[f"{prefix}_adapter_down"] = layer.adapter_down
                arrays[f"{prefix}_adapter_up"] = layer.adapter_up
                arrays[f"{prefix}_adapter_bias_basis"] = (
                    layer.adapter_bias_basis
                )
            file_hashes = _write_array_files(staging, arrays)
            manifest = {
                "format": HYPER_BASE_FORMAT,
                "id": policy.id,
                "revision": policy.revision,
                "world_fingerprint": policy.world_fingerprint,
                "task_fingerprint": policy.task_fingerprint,
                "observation_fingerprint": policy.observation_fingerprint,
                "action_fingerprint": policy.action_fingerprint,
                "observation_clip": policy.observation_clip,
                "action_clip": policy.action_clip,
                "fingerprint": policy.fingerprint,
                "layers": [
                    {
                        "input_count": layer.input_count,
                        "output_count": layer.output_count,
                        "rank": layer.rank,
                        "activation": layer.activation,
                    }
                    for layer in policy.layers
                ],
                "files": file_hashes,
            }
            (staging / "manifest.json").write_bytes(
                _canonical_json(manifest) + b"\n"
            )
        return target

    @classmethod
    def read(cls, directory: str | Path) -> "HyperBasePolicy":
        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest.get("format") != HYPER_BASE_FORMAT:
            raise ValueError("hyper-base format is unsupported")
        _verify_file_hashes(source, manifest["files"])
        arrays = {
            name: np.load(source / f"{name}.npy", allow_pickle=False)
            for name in manifest["files"]
        }
        layers = tuple(
            HyperBaseLayer(
                input_count=int(spec["input_count"]),
                output_count=int(spec["output_count"]),
                rank=int(spec["rank"]),
                activation=int(spec["activation"]),
                weight=np.asarray(
                    arrays[f"layer_{index:02d}_weight"], dtype=np.float32
                ),
                bias=np.asarray(
                    arrays[f"layer_{index:02d}_bias"], dtype=np.float32
                ),
                adapter_down=np.asarray(
                    arrays[f"layer_{index:02d}_adapter_down"], dtype=np.float32
                ),
                adapter_up=np.asarray(
                    arrays[f"layer_{index:02d}_adapter_up"], dtype=np.float32
                ),
                adapter_bias_basis=np.asarray(
                    arrays[f"layer_{index:02d}_adapter_bias_basis"],
                    dtype=np.float32,
                ),
            )
            for index, spec in enumerate(manifest["layers"])
        )
        policy = cls(
            id=str(manifest["id"]),
            revision=int(manifest["revision"]),
            world_fingerprint=int(manifest["world_fingerprint"]),
            task_fingerprint=int(manifest["task_fingerprint"]),
            observation_fingerprint=int(manifest["observation_fingerprint"]),
            action_fingerprint=int(manifest["action_fingerprint"]),
            observation_mean=np.asarray(
                arrays["observation_mean"], dtype=np.float32
            ),
            observation_inverse_standard_deviation=np.asarray(
                arrays["observation_inverse_standard_deviation"],
                dtype=np.float32,
            ),
            layers=layers,
            coefficient_limits=np.asarray(
                arrays["coefficient_limits"], dtype=np.float32
            ),
            action_bias=np.asarray(arrays["action_bias"], dtype=np.float32),
            action_scale=np.asarray(arrays["action_scale"], dtype=np.float32),
            observation_clip=float(manifest["observation_clip"]),
            action_clip=float(manifest["action_clip"]),
            fingerprint=str(manifest["fingerprint"]),
        )
        policy.validate(require_fingerprint=True)
        if policy.computed_fingerprint() != policy.fingerprint:
            raise ValueError("hyper-base semantic fingerprint is invalid")
        return policy
