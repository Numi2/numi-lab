from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    'first_representable_union_target_300', HERE / 'first_representable_union_target_300.py')
owner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owner)


EXPECTED_BANKS = tuple(('body', i) for i in range(185))

def all_outside(_point, count=185):
    return ({('body', i): 'outside' for i in range(count)}, [])


def test_earliest_interval_with_float32_inside_and_complete_exact_audit():
    events = [{'distance_m': 0.002, 'bank': 'A'},
              {'distance_m': 0.5, 'bank': 'B'},
              {'distance_m': 0.501, 'bank': 'B'}]
    result = owner.first_representable_union_target_300(
        {'A'}, events, start_world_m=np.zeros(3), unit_direction_world=np.array([1., 0., 0.]),
        clearance_m=5e-5, classify_endpoint=all_outside, expected_bank_keys=EXPECTED_BANKS)
    assert result['status'] == 'PASS_FIRST_CERTIFIED_PROPOSAL_INTERVAL'
    assert result['selected_interval_index'] == 0
    assert result['target_exactly_outside_all_banks'] is True
    assert result['classified_bank_count'] == 185
    assert result['exit_distance_m'] < result['emitted_target_distance_m'] < result['next_event_distance_m']
    assert result['skipped_intervals'] == []


def test_float32_unrepresentable_tiny_first_gap_is_recorded_then_next_gap_selected():
    exit1 = 0.000980375951753978
    enter2 = 0.00098037595175444
    events = [
        {'distance_m': exit1, 'bank': 'A'},
        {'distance_m': enter2, 'bank': 'B'},
        {'distance_m': 0.0012, 'bank': 'B'},
    ]
    result = owner.first_representable_union_target_300(
        {'A'}, events, start_world_m=np.array([0.3, 0., 0.]),
        unit_direction_world=np.array([1., 0., 0.]), clearance_m=5e-5,
        classify_endpoint=all_outside, expected_bank_keys=EXPECTED_BANKS)
    assert result['status'] == 'PASS_FIRST_CERTIFIED_PROPOSAL_INTERVAL'
    assert result['selected_interval_index'] == 1
    assert len(result['skipped_intervals']) == 1
    skipped = result['skipped_intervals'][0]
    assert skipped['reason'] == 'F32_TARGET_LEFT_FREE_INTERVAL'
    assert skipped['next_event_distance_m'] == enter2
    assert skipped['free_gap_m'] == enter2 - exit1
    assert result['exit_distance_m'] == 0.0012
    assert result['target_exactly_outside_all_banks']


def test_first_interval_exact_endpoint_failure_is_retained_before_later_success():
    events = [
        {'distance_m': 0.001, 'bank': 'A'},
        {'distance_m': 0.01, 'bank': 'B'},
        {'distance_m': 0.02, 'bank': 'B'},
    ]
    def classify(point):
        if point[0] < 0.015:
            return ({0: 'inside', **{i: 'outside' for i in range(1, 185)}}, [])
        return all_outside(point)
    result = owner.first_representable_union_target_300(
        {'A'}, events, start_world_m=np.zeros(3), unit_direction_world=np.array([1., 0., 0.]),
        clearance_m=0.002, classify_endpoint=classify, expected_bank_keys=EXPECTED_BANKS)
    assert result['status'] == 'PASS_FIRST_CERTIFIED_PROPOSAL_INTERVAL'
    assert result['selected_interval_index'] == 1
    assert result['skipped_intervals'][0]['reason'] == 'F32_ENDPOINT_NOT_EXACTLY_OUTSIDE_COMPLETE_UNION'
    assert result['skipped_intervals'][0]['inside_banks'] == [0]


def test_exact_event_grouping_does_not_merge_nearby_distances():
    exit1 = 0.1
    enter = np.nextafter(exit1, np.inf)
    events = [
        {'distance_m': exit1, 'bank': 'A'},
        {'distance_m': enter, 'bank': 'B'},
        {'distance_m': 0.2, 'bank': 'B'},
    ]
    intervals = owner.empty_union_intervals({'A'}, events)
    assert intervals[0]['exit_distance_m'] == exit1
    assert intervals[0]['next_event_distance_m'] == enter
    assert intervals[0]['free_gap_m'] == enter - exit1
    assert intervals[1]['exit_distance_m'] == 0.2


def test_simultaneous_bank_events_are_grouped_only_at_exact_equal_distance():
    intervals = owner.empty_union_intervals(
        {'A'}, [
            {'distance_m': 0.1, 'bank': 'A'},
            {'distance_m': 0.1, 'bank': 'B'},
            {'distance_m': 0.2, 'bank': 'B'},
        ])
    assert len(intervals) == 1
    assert intervals[0]['exit_distance_m'] == 0.2
    assert intervals[0]['next_event_distance_m'] is None


def test_missing_or_unresolved_bank_classification_fails_closed_and_is_recorded():
    def incomplete(_point):
        return ({EXPECTED_BANKS[0]: 'outside'}, [])
    result = owner.first_representable_union_target_300(
        {'A'}, [{'distance_m': 0.01, 'bank': 'A'}],
        start_world_m=np.zeros(3), unit_direction_world=np.array([1., 0., 0.]),
        clearance_m=1e-4, classify_endpoint=incomplete, expected_bank_keys=EXPECTED_BANKS)
    assert result['status'] == 'NO_CERTIFIED_FLOAT32_PROPOSAL_INTERVAL'
    assert result['skipped_intervals'][0]['reason'] == 'F32_ENDPOINT_NOT_EXACTLY_OUTSIDE_COMPLETE_UNION'
    assert result['skipped_intervals'][0]['missing_bank_count'] == 184


def test_wrong_bank_identity_set_fails_even_when_classification_count_is_185():
    def substituted(_point):
        classes = {key: 'outside' for key in EXPECTED_BANKS}
        classes.pop(EXPECTED_BANKS[0])
        classes[('substituted', 0)] = 'outside'
        return classes, []
    result = owner.first_representable_union_target_300(
        {'A'}, [{'distance_m': 0.01, 'bank': 'A'}],
        start_world_m=np.zeros(3), unit_direction_world=np.array([1., 0., 0.]),
        clearance_m=1e-4, classify_endpoint=substituted,
        expected_bank_keys=EXPECTED_BANKS)
    assert result['status'] == 'NO_CERTIFIED_FLOAT32_PROPOSAL_INTERVAL'
    witness = result['skipped_intervals'][0]
    assert witness['missing_banks'] == [EXPECTED_BANKS[0]]
    assert witness['unexpected_banks'] == [('substituted', 0)]


def test_non_union_events_do_not_split_one_maximal_free_interval():
    intervals = owner.empty_union_intervals(
        {'A'}, [
            {'distance_m': 0.1, 'bank': 'A'},
            # Two exact same-distance toggles leave the union empty.
            {'distance_m': 0.15, 'bank': 'B'},
            {'distance_m': 0.15, 'bank': 'B'},
            {'distance_m': 0.2, 'bank': 'C'},
            {'distance_m': 0.21, 'bank': 'C'},
        ])
    assert intervals == [
        {'exit_distance_m': 0.1, 'next_event_distance_m': 0.2, 'free_gap_m': 0.1},
        {'exit_distance_m': 0.21, 'next_event_distance_m': None, 'free_gap_m': float('inf')},
    ]
