#!/usr/bin/env python3
"""Select the earliest certified representable proposal among ray-union gaps.

This is a small construction helper. It preserves the exact event ordering
supplied by the ray auditor: only bit-identical floating distances are grouped;
no tolerance-based event merging is performed. The selected result is the
first deterministic clearance/midpoint proposal interval whose Float32 point
stays strictly inside that free interval and passes the complete endpoint
classifier. Earlier skipped intervals are returned verbatim as diagnostics.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np


def empty_union_intervals(initial_inside: Iterable[Any],
                          events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return exact-float event intervals where the active bank union is empty.

    An event toggles one bank's inside/outside parity. Simultaneous events are
    grouped only when their distance floats compare exactly equal, matching
    the ray-event convention of the source auditor.
    """
    grouped: dict[float, list[Any]] = defaultdict(list)
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or 'distance_m' not in event or 'bank' not in event:
            raise ValueError(f'event {index} must have distance_m and bank')
        distance = float(event['distance_m'])
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(f'event {index} distance must be finite and nonnegative')
        bank = event['bank']
        try:
            hash(bank)
        except TypeError as exc:
            raise ValueError(f'event {index} bank key must be hashable') from exc
        grouped[distance].append(bank)

    distances = sorted(grouped)
    active = set(initial_inside)
    if not active:
        raise ValueError('initial_inside must identify at least one containing bank')
    intervals: list[dict[str, Any]] = []
    free_start: float | None = None
    for distance in distances:
        was_empty = not active
        event_counts: dict[Any, int] = defaultdict(int)
        for bank in grouped[distance]:
            event_counts[bank] += 1
        for bank, count in event_counts.items():
            if count % 2:
                if bank in active:
                    active.remove(bank)
                else:
                    active.add(bank)
        is_empty = not active
        if not was_empty and is_empty:
            free_start = distance
        elif was_empty and not is_empty:
            if free_start is None:
                raise AssertionError('empty-union interval closed without an opening event')
            gap = distance - free_start
            if gap > 0.0:
                intervals.append({
                    'exit_distance_m': free_start,
                    'next_event_distance_m': distance,
                    'free_gap_m': gap,
                })
            free_start = None
    if not active and free_start is not None:
        intervals.append({
            'exit_distance_m': free_start,
            'next_event_distance_m': None,
            'free_gap_m': math.inf,
        })
    return intervals


def first_representable_union_target_300(
    initial_inside: Iterable[Any],
    events: Iterable[Mapping[str, Any]],
    *,
    start_world_m: np.ndarray,
    unit_direction_world: np.ndarray,
    clearance_m: float,
    classify_endpoint: Callable[[np.ndarray], tuple[Mapping[Any, str], Iterable[Any]]],
    expected_bank_keys: Iterable[Any],
) -> dict[str, Any]:
    """Find the first certified deterministic Float32 proposal interval.

    For each open union-free interval, request the lesser of the configured
    clearance and half its finite width (or the configured clearance for an
    unbounded interval). The requested point is rounded to Float32, then its
    emitted ray projection must be strictly inside that same interval. The
    endpoint classifier must report every expected bank and all must be
    outside; unresolved, missing, or inside results reject this interval and
    are retained before the next interval is considered.

    This routine does not claim to find every possible Float32 point within an
    interval. Its selection claim is deliberately limited to the deterministic
    clearance/midpoint proposal policy above.
    """
    start = np.asarray(start_world_m, dtype=np.float64)
    direction = np.asarray(unit_direction_world, dtype=np.float64)
    clearance = float(clearance_m)
    if start.shape != (3,) or not np.isfinite(start).all():
        raise ValueError('start_world_m must be finite shape (3,)')
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError('unit_direction_world must be finite shape (3,)')
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0 or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('unit_direction_world must have unit length')
    if not math.isfinite(clearance) or clearance <= 0.0:
        raise ValueError('clearance_m must be positive and finite')
    expected_keys = tuple(expected_bank_keys)
    try:
        expected_set = set(expected_keys)
    except TypeError as exc:
        raise ValueError('expected_bank_keys must be hashable identifiers') from exc
    if len(expected_keys) != 185 or len(expected_set) != 185:
        raise ValueError('expected_bank_keys must name exactly 185 unique banks')
    intervals = empty_union_intervals(initial_inside, events)
    skipped: list[dict[str, Any]] = []
    if not intervals:
        return {
            'status': 'NO_EMPTY_UNION_INTERVAL',
            'target_world_f32_m': None,
            'selected_interval_index': None,
            'skipped_intervals': skipped,
            'selection_policy': _POLICY,
        }

    for interval_index, interval in enumerate(intervals):
        exit_distance = interval['exit_distance_m']
        next_distance = interval['next_event_distance_m']
        gap = interval['free_gap_m']
        margin = clearance if math.isinf(gap) else min(clearance, 0.5 * gap)
        requested_distance = exit_distance + margin
        target = np.asarray(start + requested_distance * direction, dtype=np.float32)
        emitted_distance = float(np.dot(target.astype(np.float64) - start, direction))
        witness = {
            'interval_index': interval_index,
            **interval,
            'requested_margin_m': margin,
            'requested_target_distance_m': requested_distance,
            'emitted_target_distance_m': emitted_distance,
            'target_world_f32_m': target.tolist(),
        }
        if not np.isfinite(target).all():
            skipped.append({**witness, 'reason': 'NONFINITE_FLOAT32_TARGET'})
            continue
        if not emitted_distance > exit_distance or (
            next_distance is not None and not emitted_distance < next_distance
        ):
            skipped.append({**witness, 'reason': 'F32_TARGET_LEFT_FREE_INTERVAL'})
            continue

        classes, unresolved = classify_endpoint(target.astype(np.float64))
        if not isinstance(classes, Mapping):
            raise ValueError('endpoint classifier must return a bank->classification mapping')
        unresolved_list = list(unresolved)
        classified_keys = set(classes)
        missing_keys = expected_set - classified_keys
        extra_keys = classified_keys - expected_set
        inside = [key for key, value in classes.items() if value == 'inside']
        unknown = [key for key, value in classes.items() if value not in ('outside', 'outside_aabb', 'inside')]
        if missing_keys or extra_keys or unresolved_list or inside or unknown:
            skipped.append({
                **witness,
                'reason': 'F32_ENDPOINT_NOT_EXACTLY_OUTSIDE_COMPLETE_UNION',
                'classified_bank_count': len(classes),
                'expected_bank_count': len(expected_keys),
                'missing_bank_count': len(missing_keys),
                'missing_banks': sorted(missing_keys, key=repr),
                'unexpected_bank_count': len(extra_keys),
                'unexpected_banks': sorted(extra_keys, key=repr),
                'inside_banks': list(inside),
                'unknown_classifications': [{'bank': key, 'classification': classes[key]} for key in unknown],
                'unresolved': unresolved_list,
            })
            continue
        return {
            'status': 'PASS_FIRST_CERTIFIED_PROPOSAL_INTERVAL',
            **witness,
            'selected_interval_index': interval_index,
            'target_world_f32_m': target,
            'target_exactly_outside_all_banks': True,
            'classified_bank_count': len(classes),
            'skipped_intervals': skipped,
            'selection_policy': _POLICY,
        }

    return {
        'status': 'NO_CERTIFIED_FLOAT32_PROPOSAL_INTERVAL',
        'target_world_f32_m': None,
        'selected_interval_index': None,
        'skipped_intervals': skipped,
        'selection_policy': _POLICY,
    }


_POLICY = (
    'first event-defined union-empty interval whose deterministic clearance/midpoint '
    'proposal rounds to a Float32 endpoint strictly inside that interval and passes '
    'the complete exact bank classifier; earlier interval failures are retained; '
    'this does not claim exhaustive search of every Float32 point in each interval'
)
