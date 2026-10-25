import importlib.util
from pathlib import Path
import numpy as np
import pytest

SOURCE = Path(__file__).with_name('build_native_parent_transfer_085c-003.py')
spec = importlib.util.spec_from_file_location('transfer_085c_003', SOURCE)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

A = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=np.float64)


def translated(x, d):
    return np.asarray(x, dtype=np.float64) + np.asarray(d, dtype=np.float64)


def test_full_dimensional_interpenetration_still_fails_closed():
    b = translated(A, [0.25, 0.25, 0.25])
    r = owner.classify_exact_cross_material_pair(A, b)
    assert r['affine_rank'] == 3
    assert r['classification'] == 'exact_positive_volume_intersection'
    assert r['transfer_action'] == 'retain_and_fail_closed_genuine_cross_material_overlap'


def test_face_contact_is_zero_measure_boundary_diagnostic():
    b = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]], dtype=np.float64)
    r = owner.classify_exact_cross_material_pair(A, b)
    assert r['affine_rank'] == 2 and r['vertex_count'] == 3
    assert r['transfer_action'] == 'omit_zero_measure_pair_retain_boundary_contact_diagnostic'


def test_edge_contact_is_zero_measure_boundary_diagnostic():
    b = np.array([[0.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=np.float64)
    r = owner.classify_exact_cross_material_pair(A, b)
    assert r['affine_rank'] == 1 and r['vertex_count'] == 2
    assert r['transfer_action'] == 'omit_zero_measure_pair_retain_boundary_contact_diagnostic'


def test_point_contact_is_zero_measure_boundary_diagnostic():
    b = np.array([[0.,0.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=np.float64)
    r = owner.classify_exact_cross_material_pair(A, b)
    assert r['affine_rank'] == 0 and r['vertex_count'] == 1
    assert r['transfer_action'] == 'omit_zero_measure_pair_retain_boundary_contact_diagnostic'


def test_exactly_disjoint_pair_is_distinguished_from_contact():
    b = translated(A, [3.,3.,3.])
    r = owner.classify_exact_cross_material_pair(A, b)
    assert r['affine_rank'] == -1 and r['vertex_count'] == 0
    assert r['transfer_action'] == 'omit_tolerance_only_positive_exactly_disjoint_pair'


def test_exact_rank_is_translation_invariant_for_dyadic_translation():
    b = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]], dtype=np.float64)
    base = owner.exact_binary64_intersection_rank(A,b)
    shifted = owner.exact_binary64_intersection_rank(translated(A,[16.,-8.,4.]), translated(b,[16.,-8.,4.]))
    assert (base['affine_rank'],base['vertex_count']) == (shifted['affine_rank'],shifted['vertex_count']) == (2,3)


def test_rejects_bad_shape_and_nonfinite_input():
    with pytest.raises(ValueError): owner.exact_binary64_intersection_rank(np.zeros((3,3)), A)
    bad=A.copy(); bad[0,0]=np.nan
    with pytest.raises(ValueError): owner.exact_binary64_intersection_rank(bad, A)
