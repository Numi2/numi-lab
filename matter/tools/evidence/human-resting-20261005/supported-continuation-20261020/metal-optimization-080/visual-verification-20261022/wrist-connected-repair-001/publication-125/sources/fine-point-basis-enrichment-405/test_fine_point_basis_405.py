import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from fine_point_basis_405 import enrich_point_basis

def base():
    return sparse.csr_matrix([[1.,0.],[.5,.5],[0.,1.],[0.,0.]]), np.array([2,5,7,9]), np.array([9])

def test_default_parent_span_embedded_exactly():
    B, scope, anchors = base()
    Q, r = enrich_point_basis(B, scope, anchors, np.array([5,5,9]))
    assert r["fine_output_node_rows"] == [5]
    np.testing.assert_array_equal(Q[:,:2].toarray(), B.toarray())
    np.testing.assert_array_equal(Q[:,2].toarray().ravel(), [0,1,0,0])
    x=np.array([.31,-.42])
    np.testing.assert_array_equal(Q@np.r_[x,0.],B@x)

def test_fine_coordinate_moves_only_its_typed_row():
    B,scope,anchors=base()
    Q,r=enrich_point_basis(B,scope,anchors,np.array([5,7]))
    d=Q@np.array([0.,0.,.12,-.03])
    np.testing.assert_array_equal(d,[0,.12,-.03,0])
    assert Q[np.isin(scope,anchors)].nnz==0

def test_output_space_operator_is_projected_without_dropping_rows():
    B,scope,anchors=base()
    Q,r=enrich_point_basis(B,scope,anchors,np.array([5]))
    M=sparse.kron(Q,sparse.eye(3),format="csr")
    rng=np.random.default_rng(405)
    A=sparse.csr_matrix(rng.normal(size=(17,12)))
    y=rng.normal(size=M.shape[1])
    np.testing.assert_allclose((A@M)@y,A@(M@y),atol=2e-14)
    assert (A@M).shape[0]==A.shape[0]
    assert M[np.r_[9:12]].nnz==0

def test_empty_targets_exact_parity():
    B,scope,anchors=base()
    Q,r=enrich_point_basis(B,scope,anchors,np.array([],dtype=np.int64))
    np.testing.assert_array_equal(Q.toarray(),B.toarray())
    assert r["fine_column_count"]==0

def test_anchor_targets_excluded():
    B,scope,anchors=base()
    Q,r=enrich_point_basis(B,scope,anchors,np.array([9,9]))
    assert Q.shape==B.shape
    assert r["excluded_anchor_rows"]==[9]

def test_isolated_identity_is_not_duplicated():
    B=sparse.eye(3,format="csr")
    Q,r=enrich_point_basis(B,np.array([2,5,7]),np.array([],dtype=np.int64),np.array([5]))
    assert Q.shape==B.shape
    assert r["excluded_isolated_identity_rows"]==[5]

@pytest.mark.parametrize("target", [np.array([6]),np.array([-1]),np.array([5.0]),np.array([[5]])])
def test_reject_untyped_or_out_of_scope(target):
    B,s,a=base()
    with pytest.raises(ValueError):enrich_point_basis(B,s,a,target)

def test_reject_input_anchor_movement():
    B,s,a=base()
    with pytest.raises(ValueError):enrich_point_basis(B,s,np.array([5]),np.array([7]))

def test_reject_unsorted_scope():
    B,s,a=base()
    with pytest.raises(ValueError):enrich_point_basis(B,s[::-1],a,np.array([5]))

def test_reject_nonfinite_basis():
    B,s,a=base();B.data[0]=np.nan
    with pytest.raises(ValueError):enrich_point_basis(B,s,a,np.array([5]))
