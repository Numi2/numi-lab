"""Canonical per-motion adapter optimization and bounded deployment repair."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .base import HyperBasePolicy
from .mlx_model import ARDYHyperPolicyConfiguration, PhaseVaryingLowRankActor


@dataclass(frozen=True, slots=True)
class SpecialistAdapterBatch:
    """Solver-qualified teacher actions for one or more fixed motions.

    Teacher actions use the deployment actor's policy coordinates, matching
    ``PolicyRolloutPack.teacherActions``. Failed physics transitions should be
    assigned zero weight by the caller.
    """

    knot_phases: np.ndarray
    knot_mask: np.ndarray
    actor_observations: np.ndarray
    sample_motion_indices: np.ndarray
    sample_phases: np.ndarray
    reference_actions: np.ndarray
    teacher_actions: np.ndarray
    teacher_weights: np.ndarray

    def validate(self, configuration: ARDYHyperPolicyConfiguration) -> None:
        motions, knots = self.knot_phases.shape
        samples = self.actor_observations.shape[0]
        if (
            motions <= 0
            or knots < 2
            or samples <= 0
            or self.knot_mask.shape != (motions, knots)
            or self.sample_motion_indices.shape != (samples,)
            or self.sample_phases.shape != (samples,)
            or self.reference_actions.shape != (samples, configuration.action_count)
            or self.teacher_actions.shape != (samples, configuration.action_count)
            or self.teacher_weights.shape != (samples,)
        ):
            raise ValueError("specialist adapter batch shape is invalid")
        tables = (
            self.knot_phases,
            self.knot_mask,
            self.actor_observations,
            self.sample_phases,
            self.reference_actions,
            self.teacher_actions,
            self.teacher_weights,
        )
        if not all(np.isfinite(value).all() for value in tables):
            raise ValueError("specialist adapter batch contains non-finite values")
        if (
            np.any(self.sample_motion_indices < 0)
            or np.any(self.sample_motion_indices >= motions)
            or np.any(self.sample_phases < 0.0)
            or np.any(self.sample_phases > 1.0)
            or np.any(self.knot_mask < 0.0)
            or np.any(self.knot_mask > 1.0)
            or np.any(self.teacher_weights < 0.0)
            or np.any(self.teacher_weights > 1.0)
        ):
            raise ValueError("specialist adapter batch range is invalid")
        for motion in range(motions):
            count = int(np.count_nonzero(self.knot_mask[motion] > 0.5))
            active = self.knot_phases[motion, :count]
            if (
                count < 2
                or abs(float(active[0])) > 1.0e-6
                or abs(float(active[-1]) - 1.0) > 1.0e-6
                or not np.all(np.diff(active) > 0.0)
            ):
                raise ValueError("specialist knot phases are invalid")

    def as_mx(self) -> dict[str, mx.array]:
        return {
            "knot_phases": mx.array(self.knot_phases, dtype=mx.float32),
            "knot_mask": mx.array(self.knot_mask, dtype=mx.float32),
            "actor_observations": mx.array(self.actor_observations, dtype=mx.float32),
            "sample_motion_indices": mx.array(
                self.sample_motion_indices, dtype=mx.int32
            ),
            "sample_phases": mx.array(self.sample_phases, dtype=mx.float32),
            "reference_actions": mx.array(self.reference_actions, dtype=mx.float32),
            "teacher_actions": mx.array(self.teacher_actions, dtype=mx.float32),
            "teacher_weights": mx.array(self.teacher_weights, dtype=mx.float32),
        }


@dataclass(frozen=True, slots=True)
class SpecialistAdapterProgram:
    coefficients: np.ndarray
    authority: np.ndarray
    phase_rate_multiplier: np.ndarray


class _SpecialistCodeBook(nn.Module):
    def __init__(
        self,
        *,
        motion_count: int,
        maximum_knot_count: int,
        configuration: ARDYHyperPolicyConfiguration,
        initial: SpecialistAdapterProgram | None,
    ) -> None:
        super().__init__()
        shape = (
            motion_count,
            maximum_knot_count,
            configuration.coefficient_count,
        )
        authority_shape = (
            motion_count,
            maximum_knot_count,
            configuration.action_count,
        )
        limits = configuration.effective_coefficient_limits
        self.coefficient_limits = mx.array(limits, dtype=mx.float32)
        self.freeze(
            recurse=False,
            keys=["coefficient_limits"],
            strict=True,
        )
        if initial is None:
            raw_coefficients = np.zeros(shape, dtype=np.float32)
            raw_authority = np.full(
                authority_shape,
                _logit(0.10),
                dtype=np.float32,
            )
            raw_phase_rate = np.full(
                (motion_count, maximum_knot_count),
                math.log(2.0),
                dtype=np.float32,
            )
        else:
            if (
                initial.coefficients.shape != shape
                or initial.authority.shape != authority_shape
                or initial.phase_rate_multiplier.shape
                != (motion_count, maximum_knot_count)
            ):
                raise ValueError("initial specialist program shape is invalid")
            normalized = np.clip(
                initial.coefficients / limits[None, None, :],
                -1.0 + 1.0e-5,
                1.0 - 1.0e-5,
            )
            raw_coefficients = np.arctanh(normalized).astype(np.float32)
            raw_authority = _logit_array(initial.authority)
            raw_phase_rate = _logit_array(
                np.asarray(initial.phase_rate_multiplier, dtype=np.float32)
                / np.float32(1.5)
            )
        self.raw_coefficients = mx.array(raw_coefficients, dtype=mx.float32)
        self.raw_authority = mx.array(raw_authority, dtype=mx.float32)
        self.raw_phase_rate = mx.array(raw_phase_rate, dtype=mx.float32)
        # Phase evolution is owned by Metal contact/signature alignment.  A
        # fixed-phase action regression has no causal signal for changing its
        # rate, so preserve the compiler value instead of fitting a surrogate.
        self.freeze(recurse=False, keys=["raw_phase_rate"], strict=True)

    def values(self) -> tuple[mx.array, mx.array, mx.array]:
        coefficients = mx.tanh(self.raw_coefficients) * self.coefficient_limits
        authority = _sigmoid(self.raw_authority)
        phase_rate = 1.5 * _sigmoid(self.raw_phase_rate)
        return coefficients, authority, phase_rate


class MLXSpecialistAdapterLearner:
    """Optimizes adapter bases and canonical knots around a fixed base actor.

    The same class serves the offline specialist stage and bounded compile-time
    repair. Passing the generated program as ``initial`` starts repair from the
    hypernetwork prediction instead of from zero.
    """

    def __init__(
        self,
        *,
        actor: PhaseVaryingLowRankActor,
        hyper_base: HyperBasePolicy,
        configuration: ARDYHyperPolicyConfiguration,
        motion_count: int,
        maximum_knot_count: int,
        initial: SpecialistAdapterProgram | None = None,
        learning_rate: float = 2.0e-4,
        maximum_gradient_norm: float = 1.0,
        action_coefficient: float = 1.0,
        smoothness_coefficient: float = 0.02,
        norm_coefficient: float = 0.001,
        authority_coefficient: float = 0.0005,
        phase_rate_coefficient: float = 0.01,
    ) -> None:
        hyper_base.validate(require_fingerprint=True)
        configuration.validate()
        if (
            motion_count <= 0
            or maximum_knot_count < 2
            or learning_rate <= 0.0
            or maximum_gradient_norm <= 0.0
            or min(
                action_coefficient,
                smoothness_coefficient,
                norm_coefficient,
                authority_coefficient,
                phase_rate_coefficient,
            )
            < 0.0
        ):
            raise ValueError("specialist adapter learner configuration is invalid")
        if (
            configuration.coefficient_count != hyper_base.coefficient_count
            or configuration.action_count != hyper_base.action_count
        ):
            raise ValueError("specialist learner and hyper-base contracts differ")
        self.actor = actor
        self.hyper_base = hyper_base
        self.configuration = configuration
        self.maximum_gradient_norm = maximum_gradient_norm
        self.action_coefficient = action_coefficient
        self.smoothness_coefficient = smoothness_coefficient
        self.norm_coefficient = norm_coefficient
        self.authority_coefficient = authority_coefficient
        self.phase_rate_coefficient = phase_rate_coefficient
        self.codebook = _SpecialistCodeBook(
            motion_count=motion_count,
            maximum_knot_count=maximum_knot_count,
            configuration=configuration,
            initial=initial,
        )
        self.optimizer = optim.Adam(
            learning_rate=learning_rate,
            bias_correction=True,
        )
        # Dense actor weights and observation normalization are frozen by the
        # actor.  The low-rank bases are deliberately trainable: optimizing a
        # codebook against tiny random bases has no useful physical authority.
        self.model = _SpecialistModel(actor=self.actor, codebook=self.codebook)
        self.optimizer.init(self.model.trainable_parameters())
        self._loss_and_gradient = nn.value_and_grad(
            self.model,
            self._loss,
        )
        mx.eval(
            self.actor.parameters(),
            self.codebook.parameters(),
            self.optimizer.state,
        )

    def _loss(
        self,
        batch: Mapping[str, mx.array],
    ) -> tuple[mx.array, dict[str, mx.array]]:
        coefficients, authority, phase_rate = self.model.codebook.values()
        selected_coefficients = coefficients[batch["sample_motion_indices"]]
        selected_authority = authority[batch["sample_motion_indices"]]
        selected_phases = batch["knot_phases"][batch["sample_motion_indices"]]
        selected_mask = batch["knot_mask"][batch["sample_motion_indices"]]
        sample_coefficients = _rbf_interpolate(
            selected_phases,
            selected_coefficients,
            selected_mask,
            batch["sample_phases"],
            self.configuration.local_phase_sigma,
        )
        sample_authority = _rbf_interpolate(
            selected_phases,
            selected_authority,
            selected_mask,
            batch["sample_phases"],
            self.configuration.local_phase_sigma,
        )
        prediction = self.model.actor.policy_actions(
            batch["actor_observations"],
            sample_coefficients,
            sample_authority,
            batch["reference_actions"],
        )
        action_loss = _weighted_mean(
            mx.mean(_huber(prediction - batch["teacher_actions"], 0.05), axis=-1),
            batch["teacher_weights"],
        )
        pair_mask = batch["knot_mask"][:, 1:] * batch["knot_mask"][:, :-1]
        phase_width = mx.maximum(
            batch["knot_phases"][:, 1:] - batch["knot_phases"][:, :-1],
            1.0e-4,
        )
        coefficient_delta = coefficients[:, 1:] - coefficients[:, :-1]
        smoothness = _weighted_mean(
            mx.mean(
                mx.square(coefficient_delta / phase_width[..., None]),
                axis=-1,
            ),
            pair_mask,
        )
        norm = _weighted_mean(
            mx.mean(mx.square(coefficients), axis=-1),
            batch["knot_mask"],
        )
        authority_cost = _weighted_mean(
            mx.mean(mx.square(authority), axis=-1),
            batch["knot_mask"],
        )
        phase_rate_cost = _weighted_mean(
            mx.square(phase_rate - 1.0),
            batch["knot_mask"],
        )
        loss = (
            self.action_coefficient * action_loss
            + self.smoothness_coefficient * smoothness
            + self.norm_coefficient * norm
            + self.authority_coefficient * authority_cost
            + self.phase_rate_coefficient * phase_rate_cost
        )
        return loss, {
            "specialist_loss": loss,
            "specialist_action_loss": action_loss,
            "specialist_smoothness_loss": smoothness,
            "specialist_coefficient_norm": norm,
            "specialist_authority_cost": authority_cost,
            "specialist_phase_rate_cost": phase_rate_cost,
        }

    def update(self, batch: SpecialistAdapterBatch) -> dict[str, float]:
        batch.validate(self.configuration)
        (loss, metrics), gradients = self._loss_and_gradient(batch.as_mx())
        gradients, gradient_norm = optim.clip_grad_norm(
            gradients,
            max_norm=self.maximum_gradient_norm,
        )
        self.optimizer.update(self.model, gradients)
        mx.eval(
            loss,
            gradient_norm,
            *metrics.values(),
            self.codebook.parameters(),
            self.actor.parameters(),
            self.optimizer.state,
        )
        result = {name: float(value.item()) for name, value in metrics.items()}
        result["specialist_gradient_norm"] = float(gradient_norm.item())
        return result

    def export(self) -> SpecialistAdapterProgram:
        coefficients, authority, phase_rate = self.codebook.values()
        mx.eval(coefficients, authority, phase_rate)
        return SpecialistAdapterProgram(
            coefficients=np.asarray(coefficients, dtype=np.float32).copy(),
            authority=np.asarray(authority, dtype=np.float32).copy(),
            phase_rate_multiplier=np.asarray(phase_rate, dtype=np.float32).copy(),
        )


class _SpecialistModel(nn.Module):
    def __init__(
        self,
        *,
        actor: PhaseVaryingLowRankActor,
        codebook: _SpecialistCodeBook,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.codebook = codebook


def _sigmoid(value: mx.array) -> mx.array:
    return 1.0 / (1.0 + mx.exp(mx.clip(-value, -80.0, 80.0)))


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1.0e-5, 1.0 - 1.0e-5))
    return math.log(value / (1.0 - value))


def _logit_array(probability: np.ndarray) -> np.ndarray:
    value = np.clip(
        np.asarray(probability, dtype=np.float32),
        1.0e-5,
        1.0 - 1.0e-5,
    )
    return np.log(value / (1.0 - value)).astype(np.float32)


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


def _rbf_interpolate(
    knot_phases: mx.array,
    knot_values: mx.array,
    knot_mask: mx.array,
    query_phases: mx.array,
    sigma: float,
) -> mx.array:
    delta = (knot_phases - query_phases[:, None]) / sigma
    weight = mx.exp(-0.5 * mx.square(delta)) * knot_mask
    weight = weight / mx.maximum(mx.sum(weight, axis=1, keepdims=True), 1.0e-6)
    return mx.sum(knot_values * weight[..., None], axis=1)
