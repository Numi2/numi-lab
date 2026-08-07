"""Physical event extraction and adaptive policy-knot placement."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .common import MotionEvent, MotionEventKind


def _extract_motion_events(
    *,
    phases: np.ndarray,
    joint_velocity: np.ndarray,
    root_linear_velocity: np.ndarray,
    contact_modes: np.ndarray,
    contact_confidence: np.ndarray,
) -> tuple[MotionEvent, ...]:
    frames = phases.size
    contacts = contact_modes.shape[1]
    all_contact_mask = (1 << contacts) - 1
    events: list[MotionEvent] = []
    speed = np.linalg.norm(joint_velocity, axis=1) + np.linalg.norm(
        root_linear_velocity, axis=1
    )
    threshold = max(float(np.percentile(speed, 25.0)) * 1.5, 0.05)
    active = np.flatnonzero(speed > threshold)
    start_frame = int(active[0]) if active.size else 0
    stop_frame = int(active[-1]) if active.size else frames - 1
    events.append(
        MotionEvent(
            phase=float(phases[start_frame]),
            frame=start_frame,
            kind=MotionEventKind.start,
        )
    )

    previous_mask = sum(
        int(contact_modes[0, contact]) << contact for contact in range(contacts)
    )
    for frame in range(1, frames):
        current_mask = sum(
            int(contact_modes[frame, contact]) << contact for contact in range(contacts)
        )
        if current_mask == previous_mask:
            continue
        enabled = current_mask & ~previous_mask
        disabled = previous_mask & ~current_mask
        confidence = float(np.min(contact_confidence[frame]))
        if previous_mask and current_mask == 0:
            kind = MotionEventKind.takeoff
        elif previous_mask == 0 and current_mask:
            kind = MotionEventKind.landing
        elif enabled and not disabled:
            kind = MotionEventKind.contact_on
        elif disabled and not enabled:
            kind = MotionEventKind.contact_off
        else:
            kind = MotionEventKind.support_change
        events.append(
            MotionEvent(
                phase=float(phases[frame]),
                frame=frame,
                kind=kind,
                required_contact_on_mask=current_mask,
                required_contact_off_mask=(~current_mask) & all_contact_mask,
                confidence=confidence,
                minimum_dwell_steps=3
                if kind
                in (
                    MotionEventKind.landing,
                    MotionEventKind.support_change,
                )
                else 2,
            )
        )
        previous_mask = current_mask

    vertical = root_linear_velocity[:, 2]
    apex_candidates = np.flatnonzero((vertical[:-1] > 0.0) & (vertical[1:] <= 0.0))
    for candidate in apex_candidates:
        frame = int(candidate + 1)
        airborne = np.sum(contact_modes[max(frame - 1, 0) : frame + 1]) == 0
        if airborne:
            events.append(
                MotionEvent(
                    phase=float(phases[frame]),
                    frame=frame,
                    kind=MotionEventKind.apex,
                )
            )

    events.append(
        MotionEvent(
            phase=float(phases[stop_frame]),
            frame=stop_frame,
            kind=MotionEventKind.stop,
            required_contact_on_mask=sum(
                int(contact_modes[stop_frame, contact]) << contact
                for contact in range(contacts)
            ),
            minimum_dwell_steps=3,
        )
    )
    events.sort(key=lambda event: (event.phase, int(event.kind)))
    deduplicated: list[MotionEvent] = []
    for event in events:
        if (
            deduplicated
            and event.kind == deduplicated[-1].kind
            and abs(event.phase - deduplicated[-1].phase) <= 0.5 / max(frames - 1, 1)
        ):
            if event.confidence > deduplicated[-1].confidence:
                deduplicated[-1] = event
            continue
        deduplicated.append(event)
    return tuple(deduplicated)


def _adaptive_knot_phases(
    *,
    phases: np.ndarray,
    events: Sequence[MotionEvent],
    duration_seconds: float,
    maximum_interval_seconds: float,
    maximum_knot_count: int,
) -> np.ndarray:
    mandatory = {0.0, 1.0}
    mandatory.update(float(event.phase) for event in events)
    knots = sorted(mandatory)
    if len(knots) > maximum_knot_count:
        raise ValueError(
            "physical event count exceeds the configured adapter knot capacity"
        )
    target_interval = maximum_interval_seconds / max(duration_seconds, 1.0e-6)
    while len(knots) < maximum_knot_count:
        gaps = np.diff(knots)
        index = int(np.argmax(gaps))
        if gaps[index] <= target_interval + 1.0e-8:
            break
        knots.insert(index + 1, 0.5 * (knots[index] + knots[index + 1]))
    result = np.asarray(knots, dtype=np.float64)
    result[0] = 0.0
    result[-1] = 1.0
    return result


def _event_frame_features(
    frame_count: int,
    events: Sequence[MotionEvent],
) -> np.ndarray:
    values = np.zeros((frame_count, len(MotionEventKind)), dtype=np.float64)
    for event in events:
        channel = int(event.kind) - 1
        values[event.frame, channel] = max(
            values[event.frame, channel], event.confidence
        )
        # Give the encoder one frame of symmetric context without changing the
        # exact event index used by the runtime guard.
        for neighbor, scale in ((event.frame - 1, 0.5), (event.frame + 1, 0.5)):
            if 0 <= neighbor < frame_count:
                values[neighbor, channel] = max(
                    values[neighbor, channel], event.confidence * scale
                )
    return values


def _phase_fourier_features(phases: np.ndarray, harmonics: int) -> np.ndarray:
    columns = []
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * math.pi * harmonic * phases
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.stack(columns, axis=1)


def _select_tracked_links(
    names: tuple[str, ...], maximum_count: int
) -> tuple[int, ...]:
    if maximum_count == 0:
        return ()
    priorities = (
        "left_foot",
        "right_foot",
        "left_ankle",
        "right_ankle",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
        "head",
        "torso",
        "pelvis",
    )
    selected: list[int] = []
    lowered = tuple(name.lower() for name in names)
    for token in priorities:
        for index, name in enumerate(lowered):
            if index not in selected and token in name:
                selected.append(index)
                break
        if len(selected) >= maximum_count:
            break
    return tuple(selected)
