"""MLX temporal encoder, hypernetwork decoder, and phase-varying actor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

@dataclass(frozen=True, slots=True)
class ARDYHyperPolicyConfiguration:
    feature_count: int
    action_count: int
    coefficient_count: int
    event_feature_count: int
    model_width: int = 384
    transformer_blocks: int = 6
    attention_heads: int = 8
    feed_forward_width: int = 1536
    maximum_frames: int = 512
    local_phase_sigma: float = 0.055
    decoder_hidden_width: int = 768
    decoder_layers: int = 3
    coefficient_limits: tuple[float, ...] = ()
    minimum_uncertainty: float = 1.0e-4
    maximum_uncertainty: float = 0.50
    maximum_gradient_norm: float = 1.0
    learning_rate: float = 3.0e-4
    seed: int = 1

    def validate(self) -> None:
        if (
            min(
                self.feature_count,
                self.action_count,
                self.coefficient_count,
                self.event_feature_count,
                self.model_width,
                self.transformer_blocks,
                self.attention_heads,
                self.feed_forward_width,
                self.maximum_frames,
                self.decoder_hidden_width,
                self.decoder_layers,
            ) <= 0
            or self.model_width % self.attention_heads != 0
            or self.local_phase_sigma <= 0.0
            or self.minimum_uncertainty <= 0.0
            or self.maximum_uncertainty < self.minimum_uncertainty
            or self.maximum_gradient_norm <= 0.0
            or self.learning_rate <= 0.0
            or self.seed < 0
        ):
            raise ValueError("ARDY hyper-policy configuration is invalid")
        if self.coefficient_limits and (
            len(self.coefficient_limits) != self.coefficient_count
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.coefficient_limits
            )
        ):
            raise ValueError("hyper-policy coefficient limits are invalid")

    @property
    def effective_coefficient_limits(self) -> np.ndarray:
        return np.asarray(
            self.coefficient_limits
            if self.coefficient_limits
            else (1.0,) * self.coefficient_count,
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class HyperNetworkOutput:
    coefficient_mean: mx.array
    coefficient_uncertainty: mx.array
    authority: mx.array
    phase_rate_multiplier: mx.array
    failure_probability: mx.array
    out_of_distribution_score: mx.array
    global_code: mx.array
    local_codes: mx.array


class _SiLU(nn.Module):
    def __call__(self, value: mx.array) -> mx.array:
        return value / (1.0 + mx.exp(mx.clip(-value, -80.0, 80.0)))


class _StableELU(nn.Module):
    def __call__(self, value: mx.array) -> mx.array:
        negative = mx.exp(mx.minimum(value, 0.0)) - 1.0
        return mx.where(value >= 0.0, value, negative)


class _RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = mx.ones((width,), dtype=mx.float32)
        self.bias = mx.zeros((width,), dtype=mx.float32)
        self.epsilon = epsilon

    def __call__(self, value: mx.array) -> mx.array:
        inverse = 1.0 / mx.sqrt(
            mx.mean(mx.square(value), axis=-1, keepdims=True)
            + self.epsilon
        )
        return value * inverse * self.weight + self.bias


class _SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.output = nn.Linear(width, width, bias=True)

    def __call__(self, value: mx.array, valid_mask: mx.array) -> mx.array:
        batch, frames, _ = value.shape
        projected = self.qkv(value)
        query = projected[..., : self.width]
        key = projected[..., self.width : 2 * self.width]
        payload = projected[..., 2 * self.width :]

        def split_heads(table: mx.array) -> mx.array:
            return mx.transpose(
                mx.reshape(
                    table,
                    (batch, frames, self.heads, self.head_width),
                ),
                (0, 2, 1, 3),
            )

        query = split_heads(query)
        key = split_heads(key)
        payload = split_heads(payload)
        score = (
            query @ mx.transpose(key, (0, 1, 3, 2))
        ) * (1.0 / math.sqrt(float(self.head_width)))
        key_mask = valid_mask[:, None, None, :]
        score = score + (1.0 - key_mask) * -1.0e4
        weight = mx.softmax(score, axis=-1)
        attended = weight @ payload
        attended = mx.reshape(
            mx.transpose(attended, (0, 2, 1, 3)),
            (batch, frames, self.width),
        )
        # Invalid padded queries remain exactly zero and cannot leak through
        # residual connections or the global pooling head.
        return self.output(attended) * valid_mask[..., None]


class _FeedForward(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.input = nn.Linear(width, 2 * hidden_width, bias=True)
        self.output = nn.Linear(hidden_width, width, bias=True)
        self.hidden_width = hidden_width

    def __call__(self, value: mx.array) -> mx.array:
        projected = self.input(value)
        first = projected[..., : self.hidden_width]
        gate = projected[..., self.hidden_width :]
        gate = gate / (1.0 + mx.exp(mx.clip(-gate, -80.0, 80.0)))
        return self.output(first * gate)


class _TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, feed_forward_width: int) -> None:
        super().__init__()
        self.attention_norm = _RMSNorm(width)
        self.attention = _SelfAttention(width, heads)
        self.feed_forward_norm = _RMSNorm(width)
        self.feed_forward = _FeedForward(width, feed_forward_width)

    def __call__(self, value: mx.array, valid_mask: mx.array) -> mx.array:
        value = value + self.attention(
            self.attention_norm(value), valid_mask
        )
        value = value + self.feed_forward(
            self.feed_forward_norm(value)
        ) * valid_mask[..., None]
        return value


class ARDYMotionEncoder(nn.Module):
    """Bidirectional temporal encoder with event-local knot pooling."""

    def __init__(
        self,
        configuration: ARDYHyperPolicyConfiguration,
        *,
        feature_mean: np.ndarray | None = None,
        feature_inverse_standard_deviation: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        configuration.validate()
        self.configuration = configuration
        feature_mean = (
            np.zeros(configuration.feature_count, dtype=np.float32)
            if feature_mean is None
            else np.asarray(feature_mean, dtype=np.float32)
        )
        feature_inverse_standard_deviation = (
            np.ones(configuration.feature_count, dtype=np.float32)
            if feature_inverse_standard_deviation is None
            else np.asarray(
                feature_inverse_standard_deviation, dtype=np.float32
            )
        )
        if (
            feature_mean.shape != (configuration.feature_count,)
            or feature_inverse_standard_deviation.shape
            != (configuration.feature_count,)
            or not np.isfinite(feature_mean).all()
            or not np.isfinite(
                feature_inverse_standard_deviation
            ).all()
            or np.any(feature_inverse_standard_deviation <= 0.0)
        ):
            raise ValueError("motion feature normalization is invalid")
        self.feature_mean = mx.array(feature_mean, dtype=mx.float32)
        self.feature_inverse_standard_deviation = mx.array(
            feature_inverse_standard_deviation, dtype=mx.float32
        )
        self.freeze(
            recurse=False,
            keys=[
                "feature_mean",
                "feature_inverse_standard_deviation",
            ],
            strict=True,
        )
        self.input_projection = nn.Linear(
            configuration.feature_count,
            configuration.model_width,
            bias=True,
        )
        self.blocks = [
            _TransformerBlock(
                configuration.model_width,
                configuration.attention_heads,
                configuration.feed_forward_width,
            )
            for _ in range(configuration.transformer_blocks)
        ]
        self.final_norm = _RMSNorm(configuration.model_width)
        self.pool_query = mx.zeros(
            (configuration.model_width,), dtype=mx.float32
        )
        self.position_encoding = mx.array(
            _sinusoidal_encoding(
                configuration.maximum_frames,
                configuration.model_width,
            ),
            dtype=mx.float32,
        )
        self.freeze(
            recurse=False,
            keys=["position_encoding"],
            strict=True,
        )
        _initialize_module(
            self,
            generator=np.random.default_rng(configuration.seed),
        )

    def __call__(
        self,
        features: mx.array,
        valid_mask: mx.array,
        frame_phases: mx.array,
        knot_phases: mx.array,
        knot_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if features.ndim != 3:
            raise ValueError("motion encoder expects [batch, frames, features]")
        batch, frames, width = features.shape
        if (
            int(width) != self.configuration.feature_count
            or int(frames) > self.configuration.maximum_frames
        ):
            raise ValueError("motion encoder input width or horizon is invalid")
        normalized = (
            features - self.feature_mean
        ) * self.feature_inverse_standard_deviation
        value = self.input_projection(mx.clip(normalized, -12.0, 12.0))
        value = (
            value
            + self.position_encoding[: int(frames)][None, :, :]
        ) * valid_mask[..., None]
        for block in self.blocks:
            value = block(value, valid_mask)
        value = self.final_norm(value) * valid_mask[..., None]

        pool_score = mx.sum(
            value * self.pool_query[None, None, :], axis=-1
        ) * (1.0 / math.sqrt(float(self.configuration.model_width)))
        pool_score = pool_score + (1.0 - valid_mask) * -1.0e4
        pool_weight = mx.softmax(pool_score, axis=1) * valid_mask
        pool_weight = pool_weight / mx.maximum(
            mx.sum(pool_weight, axis=1, keepdims=True), 1.0e-6
        )
        global_code = mx.sum(value * pool_weight[..., None], axis=1)

        phase_delta = (
            knot_phases[:, :, None] - frame_phases[:, None, :]
        ) / self.configuration.local_phase_sigma
        local_weight = mx.exp(-0.5 * mx.square(phase_delta))
        local_weight = (
            local_weight
            * valid_mask[:, None, :]
            * knot_mask[:, :, None]
        )
        local_weight = local_weight / mx.maximum(
            mx.sum(local_weight, axis=-1, keepdims=True), 1.0e-6
        )
        local_codes = (local_weight @ value) * knot_mask[..., None]
        return value, global_code, local_codes


class MotionAdapterDecoder(nn.Module):
    """Generate bounded coefficient distributions and execution controls."""

    def __init__(self, configuration: ARDYHyperPolicyConfiguration) -> None:
        super().__init__()
        self.configuration = configuration
        phase_feature_count = 8
        input_count = (
            2 * configuration.model_width
            + phase_feature_count
            + configuration.event_feature_count
        )
        layers: list[nn.Module] = []
        previous = input_count
        for _ in range(configuration.decoder_layers):
            layers.extend(
                (
                    nn.Linear(
                        previous,
                        configuration.decoder_hidden_width,
                        bias=True,
                    ),
                    _SiLU(),
                )
            )
            previous = configuration.decoder_hidden_width
        output_count = (
            2 * configuration.coefficient_count
            + configuration.action_count
            + 1
        )
        layers.append(nn.Linear(previous, output_count, bias=True))
        self.knot_decoder = nn.Sequential(*layers)
        self.failure_head = nn.Sequential(
            nn.Linear(configuration.model_width, 256, bias=True),
            _SiLU(),
            nn.Linear(256, 1, bias=True),
        )
        self.ood_head = nn.Sequential(
            nn.Linear(configuration.model_width, 256, bias=True),
            _SiLU(),
            nn.Linear(256, 1, bias=True),
        )
        self.coefficient_limits = mx.array(
            configuration.effective_coefficient_limits,
            dtype=mx.float32,
        )
        self.freeze(
            recurse=False,
            keys=["coefficient_limits"],
            strict=True,
        )
        _initialize_module(
            self,
            generator=np.random.default_rng(configuration.seed + 1),
        )
        # A zero final mean gives exact base-policy behavior before training.
        final = [
            layer
            for layer in self.knot_decoder.layers
            if isinstance(layer, nn.Linear)
        ][-1]
        final.weight = mx.zeros_like(final.weight)
        final.bias = mx.zeros_like(final.bias)

    def __call__(
        self,
        global_code: mx.array,
        local_codes: mx.array,
        knot_phases: mx.array,
        knot_mask: mx.array,
        knot_event_features: mx.array,
    ) -> HyperNetworkOutput:
        batch, knots, _ = local_codes.shape
        global_expanded = mx.broadcast_to(
            global_code[:, None, :],
            (batch, knots, self.configuration.model_width),
        )
        phase_features = _phase_fourier_mx(knot_phases, harmonics=4)
        decoder_input = mx.concatenate(
            (
                global_expanded,
                local_codes,
                phase_features,
                knot_event_features,
            ),
            axis=-1,
        )
        decoded = self.knot_decoder(decoder_input) * knot_mask[..., None]
        coefficient_count = self.configuration.coefficient_count
        action_count = self.configuration.action_count
        mean_raw = decoded[..., :coefficient_count]
        uncertainty_raw = decoded[
            ..., coefficient_count : 2 * coefficient_count
        ]
        authority_raw = decoded[
            ...,
            2 * coefficient_count : 2 * coefficient_count + action_count,
        ]
        phase_rate_raw = decoded[..., -1]

        mean = mx.tanh(mean_raw) * self.coefficient_limits
        uncertainty = self.configuration.minimum_uncertainty + (
            self.configuration.maximum_uncertainty
            - self.configuration.minimum_uncertainty
        ) * _sigmoid(uncertainty_raw)
        authority = _sigmoid(authority_raw)
        phase_rate = 1.5 * _sigmoid(phase_rate_raw)
        failure = _sigmoid(self.failure_head(global_code).squeeze(-1))
        out_of_distribution = _softplus(
            self.ood_head(global_code).squeeze(-1)
        )
        return HyperNetworkOutput(
            coefficient_mean=mean * knot_mask[..., None],
            coefficient_uncertainty=uncertainty * knot_mask[..., None],
            authority=authority * knot_mask[..., None],
            phase_rate_multiplier=phase_rate * knot_mask,
            failure_probability=failure,
            out_of_distribution_score=out_of_distribution,
            global_code=global_code,
            local_codes=local_codes,
        )


class ARDYHyperNetwork(nn.Module):
    def __init__(
        self,
        configuration: ARDYHyperPolicyConfiguration,
        *,
        feature_mean: np.ndarray | None = None,
        feature_inverse_standard_deviation: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.configuration = configuration
        self.encoder = ARDYMotionEncoder(
            configuration,
            feature_mean=feature_mean,
            feature_inverse_standard_deviation=(
                feature_inverse_standard_deviation
            ),
        )
        self.decoder = MotionAdapterDecoder(configuration)

    def __call__(
        self,
        features: mx.array,
        valid_mask: mx.array,
        frame_phases: mx.array,
        knot_phases: mx.array,
        knot_mask: mx.array,
        knot_event_features: mx.array,
    ) -> HyperNetworkOutput:
        _, global_code, local_codes = self.encoder(
            features,
            valid_mask,
            frame_phases,
            knot_phases,
            knot_mask,
        )
        return self.decoder(
            global_code,
            local_codes,
            knot_phases,
            knot_mask,
            knot_event_features,
        )



from .mlx_actor import (
    ARDYMotionConditionedPolicy, PhaseVaryingLowRankActor,
    _initialize_module, _phase_fourier_mx, _sigmoid, _sinusoidal_encoding,
    _softplus,
)
