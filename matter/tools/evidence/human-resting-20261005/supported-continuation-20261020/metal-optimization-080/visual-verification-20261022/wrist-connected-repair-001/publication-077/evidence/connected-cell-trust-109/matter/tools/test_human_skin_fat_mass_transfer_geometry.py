import importlib.util
from pathlib import Path
import numpy as np
import pytest

SOURCE = Path(__file__).with_name('human_skin_fat_mass_transfer_geometry.py')
spec = importlib.util.spec_from_file_location('mass_transfer_geometry', SOURCE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)

BASE_TET = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=np.float64)


def shift(points, delta):
    return np.asarray(points, dtype=np.float64) + np.asarray(delta, dtype=np.float64)


def test_full_dimensional_overlap_is_retained_and_fails_closed():
    other = shift(BASE_TET, [0.25,0.25,0.25])
    result = geometry.classify_exact_cross_material_pair(BASE_TET, other)
    assert result['affine_rank'] == 3
    assert result['vertex_count'] >= 4
    assert result['classification'] == 'exact_positive_volume_intersection'
    assert result['transfer_action'] == 'retain_and_fail_closed_genuine_cross_material_overlap'


def test_face_contact_has_rank_two_and_is_omitted_as_zero_measure():
    other = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]], dtype=np.float64)
    result = geometry.classify_exact_cross_material_pair(BASE_TET, other)
    assert (result['affine_rank'], result['vertex_count']) == (2,3)
    assert result['transfer_action'] == 'omit_zero_measure_pair_retain_boundary_contact_diagnostic'


def test_edge_contact_has_rank_one():
    other = np.array([[0.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=np.float64)
    result = geometry.exact_binary64_intersection_rank(BASE_TET, other)
    assert (result['affine_rank'], result['vertex_count']) == (1,2)
    assert result['classification'] == 'exact_edge_contact'


def test_point_contact_has_rank_zero():
    other = np.array([[0.,0.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=np.float64)
    result = geometry.exact_binary64_intersection_rank(BASE_TET, other)
    assert (result['affine_rank'], result['vertex_count']) == (0,1)
    assert result['classification'] == 'exact_point_contact'


def test_obviously_disjoint_full_rank_tetrahedra_return_empty_sentinel():
    other = shift(BASE_TET, [4.,4.,4.])
    result = geometry.classify_exact_cross_material_pair(BASE_TET, other)
    assert (result['affine_rank'], result['vertex_count']) == (-1,0)
    assert result['classification'] == 'exactly_disjoint'
    assert result['transfer_action'] == 'omit_tolerance_only_positive_exactly_disjoint_pair'


def test_translation_invariance_for_dyadic_contact_and_overlap():
    face = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]], dtype=np.float64)
    overlap = shift(BASE_TET, [0.25,0.25,0.25])
    delta = [16.,-8.,4.]
    for other, expected_rank in ((face,2),(overlap,3)):
        before = geometry.exact_binary64_intersection_rank(BASE_TET, other)
        after = geometry.exact_binary64_intersection_rank(shift(BASE_TET,delta), shift(other,delta))
        assert before['affine_rank'] == after['affine_rank'] == expected_rank
        assert before['vertex_count'] == after['vertex_count']


def test_bad_shape_and_nonfinite_input_fail_closed():
    with pytest.raises(ValueError):
        geometry.exact_binary64_intersection_rank(np.zeros((3,3)), BASE_TET)
    bad = BASE_TET.copy(); bad[0,0] = np.nan
    with pytest.raises(ValueError):
        geometry.exact_binary64_intersection_rank(bad, BASE_TET)
