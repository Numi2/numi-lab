"""Safety-initialized production surface for the ARDY hypernetwork.

The underlying architecture remains in :mod:`mlx_model`. This module changes
only the initial decoder state: before meta-training it generates zero adapter
coefficients, one-percent residual authority, small uncertainty, and nominal
phase progression. An untrained model therefore cannot inject a half-authority
residual merely because sigmoid(0) equals 0.5.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .mlx_model import (
    ARDYHyperPolicyConfiguration,
    ARDYMotionConditionedPolicy,
    ARDYMotionEncoder,
    HyperNetworkOutput,
    MotionAdapterDecoder as _MotionAdapterDecoder,
    PhaseVaryingLowRankActor,
)


class MotionAdapterDecoder(_MotionAdapterDecoder):
    """The standard decoder with a physically neutral initial output."""

    def __init__(self, configuration: ARDYHyperPolicyConfiguration) -> None:
        super().__init__(configuration)
        final = [
            layer for layer in self.knot_decoder.layers if isinstance(layer, nn.Linear)
        ][-1]
        output_count = int(final.bias.shape[0])
        coefficient_count = configuration.coefficient_count
        action_count = configuration.action_count
        bias = np.zeros((output_count,), dtype=np.float32)

        uncertainty_target = min(
            configuration.maximum_uncertainty,
            max(configuration.minimum_uncertainty, 0.01),
        )
        uncertainty_span = (
            configuration.maximum_uncertainty - configuration.minimum_uncertainty
        )
        if uncertainty_span > 0.0:
            probability = np.clip(
                (uncertainty_target - configuration.minimum_uncertainty)
                / uncertainty_span,
                1.0e-4,
                1.0 - 1.0e-4,
            )
            bias[coefficient_count : 2 * coefficient_count] = math.log(
                float(probability) / (1.0 - float(probability))
            )

        # Generated residuals begin almost disabled. The immutable reference
        # action still executes, while physical teacher data earns authority.
        bias[2 * coefficient_count : 2 * coefficient_count + action_count] = math.log(
            0.01 / 0.99
        )

        # Decoder emits 1.5 * sigmoid(raw); log(2) maps exactly to 1.0.
        bias[-1] = math.log(2.0)
        final.weight = mx.zeros_like(final.weight)
        final.bias = mx.array(bias, dtype=mx.float32)


class ARDYHyperNetwork(nn.Module):
    """Production hypernetwork with neutral pre-training behavior."""

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
        self.encoder = ARDYMotionEncoder(
            configuration,
            feature_mean=feature_mean,
            feature_inverse_standard_deviation=(feature_inverse_standard_deviation),
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


__all__ = [
    "ARDYHyperNetwork",
    "ARDYHyperPolicyConfiguration",
    "ARDYMotionConditionedPolicy",
    "ARDYMotionEncoder",
    "HyperNetworkOutput",
    "MotionAdapterDecoder",
    "PhaseVaryingLowRankActor",
]
