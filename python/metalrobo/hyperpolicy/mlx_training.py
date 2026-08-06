"""Meta-supervision and end-to-end MLX optimization for generated policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .mlx_model import ARDYHyperPolicyConfiguration, ARDYMotionConditionedPolicy

@dataclass(frozen=True, slots=True)
class HyperPolicySupervisionBatch:
    motion_features: np.ndarray
    motion_valid_mask: np.ndarray
    frame_phases: np.ndarray
    knot_phases: np.ndarray
    knot_mask: np.ndarray
    knot_event_features: np.ndarray
    specialist_coefficients: np.ndarray
    specialist_authority: np.ndarray
    specialist_phase_rate_multiplier: np.ndarray
    actor_observations: np.ndarray
    sample_motion_indices: np.ndarray
    sample_phases: np.ndarray
    reference_actions: np.ndarray
    teacher_actions: np.ndarray
    sample_weights: np.ndarray
    failure_targets: np.ndarray

    def validate(self, configuration: ARDYHyperPolicyConfiguration) -> None:
        batch, frames, features = self.motion_features.shape
        knots = self.knot_phases.shape[1]
        samples = self.actor_observations.shape[0]
        if (
            features != configuration.feature_count
            or self.motion_valid_mask.shape != (batch, frames)
            or self.frame_phases.shape != (batch, frames)
            or self.knot_mask.shape != (batch, knots)
            or self.knot_event_features.shape
            != (batch, knots, configuration.event_feature_count)
            or self.specialist_coefficients.shape
            != (batch, knots, configuration.coefficient_count)
            or self.specialist_authority.shape
            != (batch, knots, configuration.action_count)
            or self.specialist_phase_rate_multiplier.shape != (batch, knots)
            or self.sample_motion_indices.shape != (samples,)
            or self.sample_phases.shape != (samples,)
            or self.reference_actions.shape
            != (samples, configuration.action_count)
            or self.teacher_actions.shape
            != (samples, configuration.action_count)
            or self.sample_weights.shape != (samples,)
            or self.failure_targets.shape != (batch,)
            or samples == 0
            or batch == 0
        ):
            raise ValueError("hyper-policy supervision batch shape is invalid")
        tables = (
            self.motion_features,
            self.motion_valid_mask,
            self.frame_phases,
            self.knot_phases,
            self.knot_mask,
            self.knot_event_features,
            self.specialist_coefficients,
            self.specialist_authority,
            self.specialist_phase_rate_multiplier,
            self.actor_observations,
            self.sample_phases,
            self.reference_actions,
            self.teacher_actions,
            self.sample_weights,
            self.failure_targets,
        )
        if not all(np.isfinite(value).all() for value in tables):
            raise ValueError("hyper-policy supervision batch is non-finite")
        if (
            np.any(self.sample_motion_indices < 0)
            or np.any(self.sample_motion_indices >= batch)
            or np.any(self.sample_phases < 0.0)
            or np.any(self.sample_phases > 1.0)
            or np.any(self.sample_weights < 0.0)
            or np.any(self.knot_mask < 0.0)
            or np.any(self.knot_mask > 1.0)
            or np.any(self.motion_valid_mask < 0.0)
            or np.any(self.motion_valid_mask > 1.0)
            or np.any(self.failure_targets < 0.0)
            or np.any(self.failure_targets > 1.0)
        ):
            raise ValueError("hyper-policy supervision batch range is invalid")

    def as_mx(self) -> dict[str, mx.array]:
        return {
            "motion_features": mx.array(self.motion_features, dtype=mx.float32),
            "motion_valid_mask": mx.array(
                self.motion_valid_mask, dtype=mx.float32
            ),
            "frame_phases": mx.array(self.frame_phases, dtype=mx.float32),
            "knot_phases": mx.array(self.knot_phases, dtype=mx.float32),
            "knot_mask": mx.array(self.knot_mask, dtype=mx.float32),
            "knot_event_features": mx.array(
                self.knot_event_features, dtype=mx.float32
            ),
            "specialist_coefficients": mx.array(
                self.specialist_coefficients, dtype=mx.float32
            ),
            "specialist_authority": mx.array(
                self.specialist_authority, dtype=mx.float32
            ),
            "specialist_phase_rate_multiplier": mx.array(
                self.specialist_phase_rate_multiplier, dtype=mx.float32
            ),
            "actor_observations": mx.array(
                self.actor_observations, dtype=mx.float32
            ),
            "sample_motion_indices": mx.array(
                self.sample_motion_indices, dtype=mx.int32
            ),
            "sample_phases": mx.array(self.sample_phases, dtype=mx.float32),
            "reference_actions": mx.array(
                self.reference_actions, dtype=mx.float32
            ),
            "teacher_actions": mx.array(
                self.teacher_actions, dtype=mx.float32
            ),
            "sample_weights": mx.array(self.sample_weights, dtype=mx.float32),
            "failure_targets": mx.array(
                self.failure_targets, dtype=mx.float32
            ),
        }


@dataclass(frozen=True, slots=True)
class HyperPolicyLossWeights:
    coefficient: float = 1.0
    uncertainty: float = 0.10
    action: float = 2.0
    authority: float = 0.20
    phase_rate: float = 0.20
    smoothness: float = 0.02
    coefficient_norm: float = 0.001
    failure: float = 0.25

    def validate(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.coefficient,
                self.uncertainty,
                self.action,
                self.authority,
                self.phase_rate,
                self.smoothness,
                self.coefficient_norm,
                self.failure,
            )
        ):
            raise ValueError("hyper-policy loss weights are invalid")


class HyperPolicyMetaLearner:
    """Jointly trains encoder, hypernetwork, adapter bases, and failure head."""

    def __init__(
        self,
        model: ARDYMotionConditionedPolicy,
        configuration: ARDYHyperPolicyConfiguration,
        loss_weights: HyperPolicyLossWeights = HyperPolicyLossWeights(),
    ) -> None:
        configuration.validate()
        loss_weights.validate()
        self.model = model
        self.configuration = configuration
        self.loss_weights = loss_weights
        self.optimizer = optim.Adam(
            learning_rate=configuration.learning_rate,
            bias_correction=True,
        )
        self.optimizer.init(self.model.trainable_parameters())
        self._loss_and_gradient = nn.value_and_grad(
            self.model,
            self._loss,
        )
        mx.eval(self.model.parameters(), self.optimizer.state)

    def _loss(
        self,
        model: ARDYMotionConditionedPolicy,
        batch: Mapping[str, mx.array],
    ) -> tuple[mx.array, dict[str, mx.array]]:
        output = model.generate(
            batch["motion_features"],
            batch["motion_valid_mask"],
            batch["frame_phases"],
            batch["knot_phases"],
            batch["knot_mask"],
            batch["knot_event_features"],
        )
        knot_weight = batch["knot_mask"][..., None]
        coefficient_error = _huber(
            output.coefficient_mean
            - batch["specialist_coefficients"],
            0.05,
        )
        coefficient_loss = _weighted_mean(
            coefficient_error, knot_weight
        )
        normalized_error = (
            output.coefficient_mean
            - batch["specialist_coefficients"]
        ) / mx.maximum(output.coefficient_uncertainty, 1.0e-5)
        uncertainty_loss = _weighted_mean(
            0.5 * mx.square(normalized_error)
            + mx.log(
                mx.maximum(output.coefficient_uncertainty, 1.0e-5)
            ),
            knot_weight,
        )
        authority_loss = _weighted_mean(
            _huber(
                output.authority - batch["specialist_authority"],
                0.05,
            ),
            knot_weight,
        )
        phase_rate_loss = _weighted_mean(
            _huber(
                output.phase_rate_multiplier
                - batch["specialist_phase_rate_multiplier"],
                0.05,
            ),
            batch["knot_mask"],
        )

        selected_coefficients = output.coefficient_mean[
            batch["sample_motion_indices"]
        ]
        selected_authority = output.authority[
            batch["sample_motion_indices"]
        ]
        selected_knot_phases = batch["knot_phases"][
            batch["sample_motion_indices"]
        ]
        selected_knot_mask = batch["knot_mask"][
            batch["sample_motion_indices"]
        ]
        sample_coefficients = _rbf_interpolate_mx(
            selected_knot_phases,
            selected_coefficients,
            selected_knot_mask,
            batch["sample_phases"],
            sigma=self.configuration.local_phase_sigma,
        )
        sample_authority = _rbf_interpolate_mx(
            selected_knot_phases,
            selected_authority,
            selected_knot_mask,
            batch["sample_phases"],
            sigma=self.configuration.local_phase_sigma,
        )
        predicted_action = model.actor.policy_actions(
            batch["actor_observations"],
            sample_coefficients,
            sample_authority,
            batch["reference_actions"],
        )
        action_loss = _weighted_mean(
            mx.mean(
                _huber(
                    predicted_action - batch["teacher_actions"],
                    0.05,
                ),
                axis=-1,
            ),
            batch["sample_weights"],
        )

        coefficient_delta = (
            output.coefficient_mean[:, 1:]
            - output.coefficient_mean[:, :-1]
        )
        phase_width = mx.maximum(
            batch["knot_phases"][:, 1:]
            - batch["knot_phases"][:, :-1],
            1.0e-4,
        )
        pair_mask = (
            batch["knot_mask"][:, 1:]
            * batch["knot_mask"][:, :-1]
        )
        smoothness_loss = _weighted_mean(
            mx.mean(
                mx.square(
                    coefficient_delta / phase_width[..., None]
                ),
                axis=-1,
            ),
            pair_mask,
        )
        coefficient_norm_loss = _weighted_mean(
            mx.mean(mx.square(output.coefficient_mean), axis=-1),
            batch["knot_mask"],
        )
        failure_loss = mx.mean(
            -batch["failure_targets"]
            * mx.log(mx.maximum(output.failure_probability, 1.0e-6))
            - (1.0 - batch["failure_targets"])
            * mx.log(
                mx.maximum(1.0 - output.failure_probability, 1.0e-6)
            )
        )

        weights = self.loss_weights
        loss = (
            weights.coefficient * coefficient_loss
            + weights.uncertainty * uncertainty_loss
            + weights.action * action_loss
            + weights.authority * authority_loss
            + weights.phase_rate * phase_rate_loss
            + weights.smoothness * smoothness_loss
            + weights.coefficient_norm * coefficient_norm_loss
            + weights.failure * failure_loss
        )
        metrics = {
            "hyper_policy_loss": loss,
            "hyper_coefficient_loss": coefficient_loss,
            "hyper_uncertainty_loss": uncertainty_loss,
            "hyper_action_loss": action_loss,
            "hyper_authority_loss": authority_loss,
            "hyper_phase_rate_loss": phase_rate_loss,
            "hyper_smoothness_loss": smoothness_loss,
            "hyper_coefficient_norm": coefficient_norm_loss,
            "hyper_failure_loss": failure_loss,
            "hyper_predicted_failure": mx.mean(
                output.failure_probability
            ),
            "hyper_ood_score": mx.mean(
                output.out_of_distribution_score
            ),
        }
        return loss, metrics

    def update(self, batch: HyperPolicySupervisionBatch) -> dict[str, float]:
        batch.validate(self.configuration)
        mx_batch = batch.as_mx()
        (loss, metrics), gradients = self._loss_and_gradient(mx_batch)
        gradients, gradient_norm = optim.clip_grad_norm(
            gradients,
            max_norm=self.configuration.maximum_gradient_norm,
        )
        self.optimizer.update(self.model, gradients)
        mx.eval(
            loss,
            gradient_norm,
            *metrics.values(),
            self.model.parameters(),
            self.optimizer.state,
        )
        result = {
            name: float(value.item()) for name, value in metrics.items()
        }
        result["hyper_gradient_norm"] = float(gradient_norm.item())
        return result



def _huber(value: mx.array, delta: float) -> mx.array:
    absolute = mx.abs(value)
    quadratic = mx.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * mx.square(quadratic) + delta * linear


def _weighted_mean(value: mx.array, weight: mx.array) -> mx.array:
    expanded = weight
    while expanded.ndim < value.ndim:
        expanded = expanded[..., None]
    return mx.sum(value * expanded) / mx.maximum(
        mx.sum(mx.ones_like(value) * expanded), 1.0e-6
    )


def _rbf_interpolate_mx(
    knot_phases: mx.array,
    knot_values: mx.array,
    knot_mask: mx.array,
    query_phases: mx.array,
    *,
    sigma: float,
) -> mx.array:
    delta = (knot_phases - query_phases[:, None]) / sigma
    weight = mx.exp(-0.5 * mx.square(delta)) * knot_mask
    weight = weight / mx.maximum(mx.sum(weight, axis=1, keepdims=True), 1.0e-6)
    return mx.sum(knot_values * weight[..., None], axis=1)
