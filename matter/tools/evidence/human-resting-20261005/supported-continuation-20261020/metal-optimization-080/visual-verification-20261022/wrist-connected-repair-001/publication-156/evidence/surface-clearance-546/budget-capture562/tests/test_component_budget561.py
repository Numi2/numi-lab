from pathlib import Path
import importlib.util

import numpy as np
import pytest

MODULE = Path(__file__).with_name("component_lp_budget561.py")
spec = importlib.util.spec_from_file_location("component_budget561_test", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_workload_weights_include_control_rows_and_nnz_and_exclude_trivial():
    got = m.component_workload_weights(
        control_counts=[2, 10, 4],
        hard_row_counts=[3, 7, 1],
        clearance_row_counts=[1, 2, 0],
        hard_nnz=[6, 30, 2],
        clearance_nnz=[2, 8, 0],
        output_map_nnz=[2, 10, 4],
        trivial=[False, False, True],
    )
    assert got.tolist() == [16.0, 67.0, 0.0]


def test_weighted_limits_follow_workload_and_fit_global_budget():
    got = m.weighted_component_stage_limits(
        remaining_s=120.0,
        time_limit_s=90.0,
        workloads=[1.0, 3.0],
        active=[True, True],
    )
    assert got.tolist() == [15.0, 45.0]
    assert got[1] == pytest.approx(3.0 * got[0])
    assert 2.0 * got.sum() <= 120.0


def test_trivial_components_do_not_dilute_budget():
    got = m.weighted_component_stage_limits(
        remaining_s=100.0,
        time_limit_s=90.0,
        workloads=[1.0, 0.0, 3.0],
        active=[True, False, True],
    )
    assert got.tolist() == [12.5, 0.0, 37.5]
    assert 2.0 * got.sum() <= 100.0


def test_per_stage_cap_leaves_unused_budget_for_later_dynamic_redistribution():
    got = m.weighted_component_stage_limits(
        remaining_s=120.0,
        time_limit_s=20.0,
        workloads=[1.0, 3.0],
        active=[True, True],
    )
    assert got.tolist() == [15.0, 20.0]
    assert 2.0 * got.sum() <= 120.0


def test_no_active_components_receive_no_lp_time():
    got = m.weighted_component_stage_limits(
        remaining_s=10.0,
        time_limit_s=5.0,
        workloads=[0.0, 0.0],
        active=[False, False],
    )
    assert got.tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"control_counts": [1], "hard_row_counts": [0, 1]}, "lengths differ"),
        ({"control_counts": [-1]}, "nonnegative integers"),
        ({"control_counts": [1.5]}, "nonnegative integers"),
        ({"control_counts": [float("nan")]}, "nonnegative integers"),
        ({"trivial": [0]}, "trivial mask"),
    ],
)
def test_workload_builder_rejects_invalid_inputs(kwargs, expected):
    values = {
        "control_counts": [1],
        "hard_row_counts": [1],
        "clearance_row_counts": [1],
        "hard_nnz": [1],
        "clearance_nnz": [1],
        "output_map_nnz": [1],
        "trivial": [False],
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=expected):
        m.component_workload_weights(**values)


@pytest.mark.parametrize(
    "remaining, cap, weights, active, expected",
    [
        (0.0, 1.0, [1.0], [True], "remaining wall budget"),
        (1.0, 0.0, [1.0], [True], "per-stage cap"),
        (1.0, 1.0, [1.0, 2.0], [True], "aligned vectors"),
        (1.0, 1.0, [-1.0], [True], "finite and nonnegative"),
        (1.0, 1.0, [0.0], [True], "positive workload"),
        (1.0, 1.0, [1.0], [1], "active mask"),
    ],
)
def test_allocator_rejects_invalid_inputs(remaining, cap, weights, active, expected):
    with pytest.raises(ValueError, match=expected):
        m.weighted_component_stage_limits(remaining, cap, weights, active)


def test_wrapper_pins_budget_adapter_and_original_545_source():
    import hashlib
    wrapper = Path(__file__).with_name("run_joint_full_coordinate_fit561.py")
    text = wrapper.read_text()
    adapter_sha = hashlib.sha256(MODULE.read_bytes()).hexdigest()
    assert f'EXPECTED_ACTIVE_SHA = "{adapter_sha}"' in text
    assert 'EXPECTED_BASE_ACTIVE_SHA = "37c8a7e22f240be1a669e0f29712b0dfb068861fd89c9b6b2a8955c8ffdf39ed"' in text
    assert 'EXPECTED_ENTRY_SHA = "7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693"' in text
    assert 'EXPECTED_SURFACE_HELPER_SHA = "dd6f70afab9cb51f3658d8f4fc9d65a13a6f70517e536a9e81c4c2a098750516"' in text
    assert 'EXPECTED_JOINT_HELPER_SHA = "09ce0c4f47e69142de6ce394576edd3ef0ee6f4b946824a476f5425919c492c8"' in text
    assert "max_total_wall_s_unchanged" in text
