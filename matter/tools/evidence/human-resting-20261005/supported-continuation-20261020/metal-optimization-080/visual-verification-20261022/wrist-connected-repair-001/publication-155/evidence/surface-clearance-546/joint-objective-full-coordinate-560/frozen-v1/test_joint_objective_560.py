from pathlib import Path
import importlib.util

import numpy as np
import pytest
from scipy import sparse

MODULE_PATH = Path(__file__).with_name("joint_objective_560.py")
spec = importlib.util.spec_from_file_location("joint_objective_560_test", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _builder(C, h, weights, metadata):
    def build(*args, **kwargs):
        return C, h, weights, metadata
    return build


def test_keeps_point_and_face_rows_weights_order_and_metadata_exactly():
    C = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    h = np.array([0.1, 0.2, -0.3], dtype=np.float64)
    weights = np.array([0.25, 0.20, 0.55], dtype=np.float64)
    metadata = [
        {"kind": "face", "target": 4},
        {"kind": "point", "target": 8},
        {"kind": "face", "target": 9},
    ]
    result = m.retain_joint_rows560(
        _builder(C, h, weights, metadata), None, None, None, None, {}, {}, None
    )
    out_C, out_h, out_w, out_meta, proof = result
    assert out_C is C
    assert out_h is h
    assert out_w is weights
    assert out_meta is metadata
    assert out_C.shape == (3, 2)
    assert out_h.tobytes() == h.tobytes()
    assert out_w.tobytes() == weights.tobytes()
    assert out_meta == metadata
    assert proof["surface_soft_rows_retained"] == 2
    assert proof["point_soft_rows_retained"] == 1
    assert proof["surface_weight_mass_retained"] == pytest.approx(0.8)
    assert proof["point_weight_mass_retained"] == pytest.approx(0.2)
    assert proof["total_weight_mass_retained"] == pytest.approx(1.0)
    assert proof["weights_recomputed_or_rebalanced"] is False


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ([{"kind": "face"}], "metadata rows do not align"),
        ([{"kind": "face"}, {"kind": "inside"}], "unsupported row kinds"),
        ([{"kind": "point"}, {"target": 1}], "requires a string kind"),
    ],
)
def test_rejects_bad_metadata_alignment_or_kind(metadata, expected):
    C = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match=expected):
        m.retain_joint_rows560(
            _builder(C, np.ones(2), np.ones(2), metadata),
            None, None, None, None, {}, {}, None,
        )


def test_allows_empty_point_family_without_renormalizing_surface_mass():
    C = sparse.csr_matrix([[1.0, 2.0]])
    h = np.array([0.4])
    weights = np.array([0.5])
    metadata = [{"kind": "face", "target": 3}]
    out_C, out_h, out_w, out_meta, proof = m.retain_joint_rows560(
        _builder(C, h, weights, metadata), None, None, None, None, {}, {}, None
    )
    assert out_C is C and out_h is h and out_w is weights and out_meta is metadata
    assert proof["point_soft_rows_retained"] == 0
    assert proof["surface_soft_rows_retained"] == 1
    assert proof["surface_weight_mass_retained"] == 0.5
    assert proof["total_weight_mass_retained"] == 0.5


def test_allows_empty_surface_family_without_renormalizing_point_mass():
    C = sparse.csr_matrix([[2.0, 1.0]])
    h = np.array([0.2])
    weights = np.array([0.5])
    metadata = [{"kind": "point", "target": 7}]
    out_C, out_h, out_w, out_meta, proof = m.retain_joint_rows560(
        _builder(C, h, weights, metadata), None, None, None, None, {}, {}, None
    )
    assert out_C is C and out_h is h and out_w is weights and out_meta is metadata
    assert proof["point_soft_rows_retained"] == 1
    assert proof["surface_soft_rows_retained"] == 0
    assert proof["point_weight_mass_retained"] == 0.5
    assert proof["total_weight_mass_retained"] == 0.5


@pytest.mark.parametrize(
    "C, h, weights, metadata, expected",
    [
        (sparse.csr_matrix([[1.0, 0.0]]), np.array([0.0, 1.0]), np.ones(1), [{"kind": "face"}], "row counts differ"),
        (sparse.csr_matrix([[np.nan, 0.0]]), np.ones(1), np.ones(1), [{"kind": "face"}], "nonfinite"),
        (sparse.csr_matrix([[1.0, 0.0]]), np.ones(1), np.array([0.0]), [{"kind": "face"}], "finite and positive"),
    ],
)
def test_rejects_bad_row_shapes_nonfinite_values_and_weights(C, h, weights, metadata, expected):
    with pytest.raises(ValueError, match=expected):
        m.retain_joint_rows560(
            _builder(C, h, weights, metadata),
            None, None, None, None, {}, {}, None,
        )


def test_wrapper_pins_helper_and_existing_owner_inputs():
    wrapper = Path(__file__).with_name("run_joint_full_coordinate_fit560.py")
    text = wrapper.read_text()
    helper_sha = "09ce0c4f47e69142de6ce394576edd3ef0ee6f4b946824a476f5425919c492c8"
    assert f'EXPECTED_JOINT_HELPER_SHA = "{helper_sha}"' in text
    import hashlib
    actual = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    assert actual == helper_sha
    assert 'EXPECTED_ENTRY_SHA = "7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693"' in text
    assert 'EXPECTED_ACTIVE_SHA = "37c8a7e22f240be1a669e0f29712b0dfb068861fd89c9b6b2a8955c8ffdf39ed"' in text
    assert 'EXPECTED_SURFACE_HELPER_SHA = "dd6f70afab9cb51f3658d8f4fc9d65a13a6f70517e536a9e81c4c2a098750516"' in text
