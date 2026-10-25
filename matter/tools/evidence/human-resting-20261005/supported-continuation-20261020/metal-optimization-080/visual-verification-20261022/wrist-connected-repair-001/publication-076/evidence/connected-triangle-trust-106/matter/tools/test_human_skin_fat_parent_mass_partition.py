import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).with_name("human_skin_fat_mass_transfer_geometry.py")
spec = importlib.util.spec_from_file_location("mass_transfer_geometry_partition_test", SOURCE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def test_parent_local_partition_is_separate_from_raw_geometry_fraction():
    result = geometry.build_parent_local_mass_partition(
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([0.2, 0.3, 0.2, 0.3]),
        np.array([0.5, 0.5]),
        np.array([0.4, 0.6]),
        np.array([15, 15]),
        np.array([15, 15]),
    )
    np.testing.assert_allclose(
        result["parent_local_mass_fraction_overlap_over_covered_volume"],
        [0.5, 0.5, 0.5, 0.5],
        rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        result["raw_parent_geometric_fraction_overlap_over_parent_volume"],
        [0.5, 0.5, 0.5, 0.5],
        rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(result["parent_local_normalization_factor_geometry_over_covered"], [1., 1.])
    assert result["maximum_absolute_parent_local_correction"] == 0.0


def test_parent_local_roundoff_partition_reports_correction_without_changing_coverage():
    result = geometry.build_parent_local_mass_partition(
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([0.5 - 2.5e-8, 0.5 + 2.5e-8]),
        np.array([1.0]),
        np.array([0.5, 0.5]),
        np.array([16]),
        np.array([16, 16]),
        coverage_tolerance=1e-7,
    )
    assert abs(result["child_coverage_relative_error"][0]) < 1e-15
    np.testing.assert_allclose(result["parent_coverage_relative_error"], [-5e-8, 5e-8], rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        result["parent_local_mass_fraction_overlap_over_covered_volume"],
        [1.0, 1.0],
        rtol=0, atol=1e-15,
    )
    assert result["maximum_absolute_parent_local_correction"] > 0
    assert np.all(result["raw_parent_geometric_fraction_overlap_over_parent_volume"] !=
                  result["parent_local_mass_fraction_overlap_over_covered_volume"])


def test_coverage_failure_is_rejected_before_partition():
    with pytest.raises(ValueError, match="raw geometric coverage fails"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0.9]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([2]),
            np.array([2]),
        )


def test_cross_material_overlap_is_rejected():
    with pytest.raises(ValueError, match="cross-material overlap"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([15]),
            np.array([16]),
        )


def test_zero_or_inactive_rows_fail_closed_without_nan_partition():
    with pytest.raises(ValueError, match="no positive overlap support"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1]),
            np.array([1]),
        )
    with pytest.raises(ValueError, match="no positive overlap support"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0, 1.0]),
            np.array([1]),
            np.array([1, 1]),
        )


def test_invalid_csr_and_nonfinite_inputs_fail_closed():
    with pytest.raises(ValueError, match="monotone"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 2, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0]),
            np.array([1, 1]),
            np.array([1]),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([np.nan]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1]),
            np.array([1]),
        )



def test_duplicate_parent_pair_is_rejected():
    with pytest.raises(ValueError, match="duplicate parent row"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 2], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([0.5, 0.5]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1]),
            np.array([1]),
        )

def test_material_ids_must_be_positive_integers():
    with pytest.raises(ValueError, match="integer arrays"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1]),
        )
    with pytest.raises(ValueError, match="positive"):
        geometry.build_parent_local_mass_partition(
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([0]),
            np.array([1]),
        )
