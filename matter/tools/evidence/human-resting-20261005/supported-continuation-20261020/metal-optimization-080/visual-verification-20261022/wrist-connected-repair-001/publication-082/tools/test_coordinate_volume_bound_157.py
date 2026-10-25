from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

from coordinate_volume_bound_157 import certify_parent_volume_partition

HERE = Path(__file__).resolve().parent.parent
RUN = HERE / "run-005"
WITNESS = RUN / "volume-diagnostic-004" / "clip-failure-witness-subset.npz"
SOURCE = RUN / "outer-positive-input-checkpoint.npz"


def load_witness():
    z = np.load(WITNESS, allow_pickle=False)
    s = np.load(SOURCE, allow_pickle=False)
    W = sparse.csr_matrix(
        (z["W_data"], z["W_indices"], z["W_indptr"]),
        shape=tuple(z["W_shape"]),
    )
    return {
        "source_points": s["points_world_m_f64"],
        "source_tets": z["source_tets"],
        "points": z["points"],
        "children": z["children"],
        "parents": z["parents"],
        "W": W,
    }


def test_exact_16_volume_witnesses_pass_reference_and_bounded_coordinate_checks():
    x = load_witness()
    report = certify_parent_volume_partition(
        x["source_points"], x["source_tets"], x["points"], x["children"],
        x["parents"], x["W"],
    )
    assert report["status"] == "ARITHMETICALLY_BOUNDED_REFERENCE_PARTITION"
    assert report["parent_count"] == 16
    assert report["child_count"] == 50
    assert report["strict_f64_positive_child_count"] == 50
    assert report["strict_emitted_f32_positive_child_count"] == 50
    assert report["mass_fraction_rule"] == "raw barycentric determinant fractions; no normalization and no child drop"
    assert all(p["children_omitted"] == 0 and not p["fractions_normalized"] for p in report["parents"])
    assert max(p["reference_partition_residual_m3"] for p in report["parents"]) <= max(
        p["fixed_4096eps_tolerance_m3"] for p in report["parents"]
    )
    assert report["max_stored_coordinate_partition_residual_m3"] > 0.0
    assert report["max_derived_arithmetic_bound_m3"] > 0.0
    assert report["point_residual_used_to_set_determinant_bound"] is False
    assert report["max_measured_point_vs_W_source_residual_m"] <= report["max_point_encoding_residual_allowance_m"]
    assert report["max_actual_vs_reference_relative_residual"] > 0.0
    assert report["max_derived_arithmetic_bound_relative_to_parent"] > 0.0
    assert report["max_stored_coordinate_partition_residual_m3"] <= max(
        p["fixed_4096eps_tolerance_m3"] + p["derived_arithmetic_bound_m3"]
        for p in report["parents"]
    )


def test_unexplained_point_perturbation_fails_the_encoding_bound():
    x = load_witness()
    p = x["points"].copy()
    generated = len(x["source_points"])
    p[generated, 0] += 1.0e-8
    with pytest.raises(ValueError, match="stored point exceeds the declared edge-interpolation rounding bound"):
        certify_parent_volume_partition(
            x["source_points"], x["source_tets"], p, x["children"],
            x["parents"], x["W"],
        )


def test_original_rows_require_bitwise_identity():
    x = load_witness()
    p = x["points"].copy()
    p[0, 0] = np.nextafter(p[0, 0], np.inf)
    with pytest.raises(ValueError, match="original point rows must be bitwise identity copies"):
        certify_parent_volume_partition(
            x["source_points"], x["source_tets"], p, x["children"],
            x["parents"], x["W"],
        )


def test_f32_omitted_side_is_ledgered_but_required_side_stays_strict():
    source = np.array([
        [0.1, 0.1, 0.1],
        [0.100001, 0.1, 0.1],
        [0.1, 0.100001, 0.1],
        [0.1, 0.1, 0.100001],
    ], dtype=np.float64)
    t = 1.0e-5
    cut = source[0] + t * (source[1] - source[0])
    points = np.vstack([source, cut])
    W = sparse.csr_matrix((
        np.array([1., 1., 1., 1., 1. - t, t]),
        np.array([0, 1, 2, 3, 0, 1]),
        np.array([0, 1, 2, 3, 4, 6]),
    ), shape=(5, 4))
    children = np.array([[0, 4, 2, 3], [4, 1, 2, 3]], dtype=np.int64)
    parent_rows = np.array([0, 0], dtype=np.int64)
    f32 = points.astype(np.float32).astype(np.float64)
    assert np.linalg.det(np.stack([f32[children[0, i]] - f32[children[0, 0]] for i in (1, 2, 3)])) == 0.0
    report = certify_parent_volume_partition(
        source, np.array([[0, 1, 2, 3]], dtype=np.int64), points, children,
        parent_rows, W, emitted_f32_required_mask=np.array([False, True]),
    )
    assert report["strict_f64_positive_child_count"] == 2
    assert report["strict_emitted_f32_positive_child_count"] == 1
    assert report["strict_required_emitted_f32_positive_child_count"] == 1
    assert report["emitted_f32_nonpositive_excluded_side_child_count"] == 1
    with pytest.raises(ValueError, match="required emitted Float32 child geometry contains nonpositive"):
        certify_parent_volume_partition(
            source, np.array([[0, 1, 2, 3]], dtype=np.int64), points, children,
            parent_rows, W, emitted_f32_required_mask=np.array([True, True]),
        )
