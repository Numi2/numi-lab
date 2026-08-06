"""Phase-varying low-rank feedback actor used by generated ARDY policies."""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import HyperBaseLayer, HyperBasePolicy
from .mlx_model import ARDYHyperNetwork, ARDYHyperPolicyConfiguration, HyperNetworkOutput

class _LowRankFeedbackLayer(nn.Module):
    def __init__(self, source: HyperBaseLayer) -> None:
        super().__init__()
        source.validate()
        self.input_count = source.input_count
        self.output_count = source.output_count
        self.rank = source.rank
        self.activation = source.activation
        self.weight = mx.array(source.weight, dtype=mx.float32)
        self.bias = mx.array(source.bias, dtype=mx.float32)
        self.adapter_down = mx.array(source.adapter_down, dtype=mx.float32)
        self.adapter_up = mx.array(source.adapter_up, dtype=mx.float32)
        self.adapter_bias_basis = mx.array(
            source.adapter_bias_basis, dtype=mx.float32
        )
        self.freeze(
            recurse=False,
            keys=["weight", "bias"],
            strict=True,
        )

    def __call__(self, value: mx.array, coefficient: mx.array) -> mx.array:
        base = value @ mx.transpose(self.weight) + self.bias
        down = value @ mx.transpose(self.adapter_down)
        adapted = (down * coefficient) @ mx.transpose(self.adapter_up)
        adapted = adapted + coefficient @ mx.transpose(
            self.adapter_bias_basis
        )
        result = base + adapted
        if self.activation == 3:
            negative = mx.exp(mx.minimum(result, 0.0)) - 1.0
            result = mx.where(result >= 0.0, result, negative)
        return result


class PhaseVaryingLowRankActor(nn.Module):
    """Shared physical actor with phase-dependent generated adapter gates."""

    def __init__(self, hyper_base: HyperBasePolicy) -> None:
        super().__init__()
        hyper_base = hyper_base.with_fingerprint()
        hyper_base.validate(require_fingerprint=True)
        self.hyper_base_fingerprint = hyper_base.fingerprint
        self.layers = [
            _LowRankFeedbackLayer(layer) for layer in hyper_base.layers
        ]
        self.coefficient_offsets: tuple[tuple[int, int], ...] = tuple(
            (value.start or 0, value.stop or 0)
            for value in hyper_base.coefficient_slices
        )
        self.observation_mean = mx.array(
            hyper_base.observation_mean, dtype=mx.float32
        )
        self.observation_inverse_standard_deviation = mx.array(
            hyper_base.observation_inverse_standard_deviation,
            dtype=mx.float32,
        )
        self.observation_clip = hyper_base.observation_clip
        self.freeze(
            recurse=False,
            keys=[
                "observation_mean",
                "observation_inverse_standard_deviation",
            ],
            strict=True,
        )

    def actor_mean(
        self,
        observations: mx.array,
        coefficients: mx.array,
    ) -> mx.array:
        value = mx.clip(
            (
                observations - self.observation_mean
            ) * self.observation_inverse_standard_deviation,
            -self.observation_clip,
            self.observation_clip,
        )
        for layer, (start, end) in zip(
            self.layers, self.coefficient_offsets, strict=True
        ):
            value = layer(value, coefficients[..., start:end])
        return value

    def policy_actions(
        self,
        observations: mx.array,
        coefficients: mx.array,
        authority: mx.array,
        reference_actions: mx.array,
    ) -> mx.array:
        return reference_actions + authority * mx.tanh(
            self.actor_mean(observations, coefficients)
        )


class ARDYMotionConditionedPolicy(nn.Module):
    def __init__(
        self,
        hypernetwork: ARDYHyperNetwork,
        actor: PhaseVaryingLowRankActor,
    ) -> None:
        super().__init__()
        if (
            hypernetwork.configuration.coefficient_count
            != sum(layer.rank for layer in actor.layers)
            or hypernetwork.configuration.action_count
            != actor.layers[-1].output_count
        ):
            raise ValueError("hypernetwork and feedback actor contracts differ")
        self.hypernetwork = hypernetwork
        self.actor = actor

    def generate(
        self,
        features: mx.array,
        valid_mask: mx.array,
        frame_phases: mx.array,
        knot_phases: mx.array,
        knot_mask: mx.array,
        knot_event_features: mx.array,
    ) -> HyperNetworkOutput:
        return self.hypernetwork(
            features,
            valid_mask,
            frame_phases,
            knot_phases,
            knot_mask,
            knot_event_features,
        )



def _initialize_module(
    module: nn.Module,
    *,
    generator: np.random.Generator,
) -> None:
    for _, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        output_count, input_count = (
            int(child.weight.shape[0]),
            int(child.weight.shape[1]),
        )
        bound = math.sqrt(6.0 / float(input_count + output_count))
        child.weight = mx.array(
            generator.uniform(
                -bound, bound, (output_count, input_count)
            ).astype(np.float32),
            dtype=mx.float32,
        )
        if getattr(child, "bias", None) is not None:
            child.bias = mx.zeros((output_count,), dtype=mx.float32)


def _sinusoidal_encoding(frames: int, width: int) -> np.ndarray:
    position = np.arange(frames, dtype=np.float64)[:, None]
    half = width // 2
    frequency = np.exp(
        -math.log(10_000.0)
        * np.arange(half, dtype=np.float64)
        / max(half - 1, 1)
    )[None, :]
    encoding = np.concatenate(
        (np.sin(position * frequency), np.cos(position * frequency)), axis=1
    )
    if encoding.shape[1] < width:
        encoding = np.pad(encoding, ((0, 0), (0, width - encoding.shape[1])))
    return encoding[:, :width].astype(np.float32)


def _phase_fourier_mx(phases: mx.array, harmonics: int) -> mx.array:
    values = []
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * math.pi * harmonic * phases
        values.extend((mx.sin(angle), mx.cos(angle)))
    return mx.stack(values, axis=-1)


def _sigmoid(value: mx.array) -> mx.array:
    return 1.0 / (1.0 + mx.exp(mx.clip(-value, -80.0, 80.0)))


def _softplus(value: mx.array) -> mx.array:
    absolute = mx.abs(value)
    return mx.maximum(value, 0.0) + mx.log1p(mx.exp(-absolute))
