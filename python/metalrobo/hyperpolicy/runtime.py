"""Event-synchronised phase tracking and executable feedback-policy reference."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .common import _ACTIVATION_ELU, _ACTIVATION_IDENTITY, _sample_rows_linear, evaluate_event_safe_cubic
from .generated import GeneratedMotionPolicy, MotionPolicyBundle

@dataclass(slots=True)
class EventSynchronizedPhaseTracker:
    policy: GeneratedMotionPolicy
    alignment_blend: float = 0.35
    maximum_phase_advance_per_step: float = 0.04
    forward_search_frames: int = 12
    huber_delta: float = 2.0
    phase: float = 0.0
    aligned_phase: float = 0.0
    reference_index: int = 0
    next_event_index: int = 0
    event_dwell_steps: int = 0
    completed: bool = False

    def __post_init__(self) -> None:
        self.policy.validate()
        if (
            not 0.0 <= self.alignment_blend <= 1.0
            or self.maximum_phase_advance_per_step <= 0.0
            or self.forward_search_frames <= 0
            or self.huber_delta <= 0.0
        ):
            raise ValueError("phase tracker configuration is invalid")

    def reset(self, phase: float = 0.0) -> None:
        if not math.isfinite(phase) or not 0.0 <= phase <= 1.0:
            raise ValueError("phase reset must be in [0, 1]")
        self.phase = phase
        self.aligned_phase = phase
        self.reference_index = int(
            np.searchsorted(self.policy.reference_phases, phase, side="right") - 1
        )
        self.reference_index = int(
            np.clip(self.reference_index, 0, self.policy.frame_count - 1)
        )
        self.next_event_index = next(
            (
                index
                for index, event in enumerate(self.policy.events)
                if event.phase >= phase - 1.0e-7
            ),
            len(self.policy.events),
        )
        self.event_dwell_steps = 0
        self.completed = phase >= 1.0

    def update(
        self,
        signature: np.ndarray,
        measured_contact_mask: int,
        control_time_step: float,
    ) -> float:
        if self.completed:
            return 1.0
        value = np.asarray(signature, dtype=np.float32)
        if (
            value.shape != self.policy.signature_weights.shape
            or not np.isfinite(value).all()
            or control_time_step <= 0.0
            or not math.isfinite(control_time_step)
        ):
            raise ValueError("phase tracker input is invalid")
        contact_limit = (1 << self.policy.contact_modes.shape[1]) - 1
        if measured_contact_mask & ~contact_limit:
            raise ValueError("measured contact mask exceeds policy contract")

        start = self.reference_index
        end = min(
            self.policy.frame_count,
            start + self.forward_search_frames + 1,
        )
        candidates = self.policy.reference_signature[start:end]
        delta = (candidates - value[None, :]) * self.policy.signature_weights[None, :]
        absolute = np.abs(delta)
        robust = np.where(
            absolute <= self.huber_delta,
            0.5 * np.square(delta),
            self.huber_delta * (absolute - 0.5 * self.huber_delta),
        )
        costs = np.mean(robust, axis=1)
        reference_masks = np.sum(
            self.policy.contact_modes[start:end].astype(np.uint32)
            << np.arange(
                self.policy.contact_modes.shape[1], dtype=np.uint32
            )[None, :],
            axis=1,
            dtype=np.uint32,
        )
        mismatch = np.asarray(
            [
                int(int(mask) ^ int(measured_contact_mask)).bit_count()
                for mask in reference_masks
            ],
            dtype=np.float32,
        )
        costs = costs + 4.0 * mismatch
        best_local = int(np.argmin(costs))
        best_index = start + best_local
        best_phase = float(self.policy.reference_phases[best_index])

        phase_rate = float(
            evaluate_event_safe_cubic(
                self.policy.knot_phases,
                self.policy.phase_rate_knots[:, None],
                self.policy.phase_rate_tangents[:, None],
                self.phase,
            )[0]
        )
        predicted = self.phase + phase_rate * control_time_step
        aligned = max(self.phase, best_phase)
        candidate = (
            (1.0 - self.alignment_blend) * predicted
            + self.alignment_blend * aligned
        )
        candidate = min(
            candidate,
            self.phase + self.maximum_phase_advance_per_step,
            1.0,
        )

        while self.next_event_index < len(self.policy.events):
            event = self.policy.events[self.next_event_index]
            if candidate + 1.0e-7 < event.phase:
                break
            on_satisfied = (
                measured_contact_mask & event.required_contact_on_mask
            ) == event.required_contact_on_mask
            off_satisfied = (
                (~measured_contact_mask) & event.required_contact_off_mask
            ) == event.required_contact_off_mask
            if on_satisfied and off_satisfied:
                self.event_dwell_steps += 1
                if self.event_dwell_steps >= event.minimum_dwell_steps:
                    self.next_event_index += 1
                    self.event_dwell_steps = 0
                    continue
            else:
                self.event_dwell_steps = 0
            candidate = min(candidate, event.phase)
            break

        self.phase = float(np.clip(max(self.phase, candidate), 0.0, 1.0))
        self.aligned_phase = best_phase
        self.reference_index = int(
            np.searchsorted(
                self.policy.reference_phases, self.phase, side="right"
            ) - 1
        )
        self.reference_index = int(
            np.clip(self.reference_index, 0, self.policy.frame_count - 1)
        )
        self.completed = self.phase >= 1.0 - 1.0e-7
        return self.phase


@dataclass(slots=True)
class PhaseVaryingFeedbackPolicy:
    """Executable NumPy reference for the generated deployment policy."""

    bundle: MotionPolicyBundle
    phase_tracker: EventSynchronizedPhaseTracker = field(init=False)
    previous_action: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.bundle.validate()
        self.phase_tracker = EventSynchronizedPhaseTracker(
            self.bundle.motion_policy
        )
        reference = self.bundle.motion_policy.reference_actions[0]
        self.previous_action = np.clip(
            self.bundle.hyper_base.action_scale * reference
            + self.bundle.hyper_base.action_bias,
            self.bundle.motion_policy.action_lower,
            self.bundle.motion_policy.action_upper,
        ).astype(np.float32)

    def reset(self, phase: float = 0.0) -> None:
        self.phase_tracker.reset(phase)
        reference = _sample_rows_linear(
            self.bundle.motion_policy.reference_phases,
            self.bundle.motion_policy.reference_actions,
            np.asarray([phase], dtype=np.float32),
        )[0]
        self.previous_action = np.clip(
            self.bundle.hyper_base.action_scale * reference
            + self.bundle.hyper_base.action_bias,
            self.bundle.motion_policy.action_lower,
            self.bundle.motion_policy.action_upper,
        ).astype(np.float32)

    def act(
        self,
        observation: np.ndarray,
        signature: np.ndarray,
        measured_contact_mask: int,
        control_time_step: float,
    ) -> np.ndarray:
        base = self.bundle.hyper_base
        policy = self.bundle.motion_policy
        observation_value = np.asarray(observation, dtype=np.float32)
        if observation_value.shape != (base.observation_count,):
            raise ValueError("hyper-policy observation width is invalid")
        phase = self.phase_tracker.update(
            signature,
            measured_contact_mask,
            control_time_step,
        )
        coefficients = evaluate_event_safe_cubic(
            policy.knot_phases,
            policy.coefficient_knots,
            policy.coefficient_tangents,
            phase,
        )
        coefficients = np.clip(
            coefficients,
            -base.coefficient_limits,
            base.coefficient_limits,
        )
        authority = np.clip(
            evaluate_event_safe_cubic(
                policy.knot_phases,
                policy.authority_knots,
                policy.authority_tangents,
                phase,
            ),
            0.0,
            1.0,
        )
        reference = _sample_rows_linear(
            policy.reference_phases,
            policy.reference_actions,
            np.asarray([phase], dtype=np.float32),
        )[0]
        latent = self._actor_mean(observation_value, coefficients)
        policy_action = reference + authority * np.tanh(latent)
        physical_action = np.clip(
            base.action_scale * policy_action + base.action_bias,
            -base.action_clip,
            base.action_clip,
        )
        maximum_delta = policy.maximum_action_rate * control_time_step
        physical_action = np.clip(
            physical_action,
            self.previous_action - maximum_delta,
            self.previous_action + maximum_delta,
        )
        physical_action = np.clip(
            physical_action,
            policy.action_lower,
            policy.action_upper,
        ).astype(np.float32)
        self.previous_action = physical_action
        return physical_action

    def _actor_mean(
        self,
        observation: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        base = self.bundle.hyper_base
        value = np.clip(
            (
                observation - base.observation_mean
            ) * base.observation_inverse_standard_deviation,
            -base.observation_clip,
            base.observation_clip,
        ).astype(np.float32)
        for layer, coefficient_slice in zip(
            base.layers, base.coefficient_slices, strict=True
        ):
            gates = coefficients[coefficient_slice]
            down = layer.adapter_down @ value
            adapter = layer.adapter_up @ (down * gates)
            adapter += layer.adapter_bias_basis @ gates
            value = layer.weight @ value + layer.bias + adapter
            if layer.activation == _ACTIVATION_ELU:
                value = np.where(
                    value >= 0.0,
                    value,
                    np.expm1(np.minimum(value, 0.0)),
                )
        return value.astype(np.float32, copy=False)
