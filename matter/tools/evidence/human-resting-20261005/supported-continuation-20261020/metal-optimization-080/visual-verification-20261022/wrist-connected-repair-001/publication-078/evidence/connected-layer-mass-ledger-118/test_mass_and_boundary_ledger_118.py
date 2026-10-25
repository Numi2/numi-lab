import importlib.util
from pathlib import Path
import numpy as np
import pytest

SRC=Path(__file__).with_name("build_mass_and_boundary_audit_118.py")
SPEC=importlib.util.spec_from_file_location("ledger118",SRC)
LEDGER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEDGER)

def test_parent_overlap_measure_conserves_by_material_and_computes_child_density():
    parent_v=np.array([.2,.8,.5])
    parent_m=np.array([1,1,2])
    rho=np.array([1000.,1000.,900.])
    child_v=np.array([.4,.6,.5])
    child_m=np.array([1,1,2])
    overlap=np.array([[.2,0,0],[.2,.6,0],[0,0,.5]])
    got=LEDGER.conservative_parent_transfer(parent_v,parent_m,rho,child_v,child_m,overlap)
    np.testing.assert_allclose(got["child_mass_kg"],[400.,600.,450.])
    np.testing.assert_allclose(got["child_density_kg_m3"],[1000.,1000.,900.])
    assert abs(got["by_material"]["1"]["residual_kg"])<1e-12
    assert abs(got["by_material"]["2"]["residual_kg"])<1e-12

def test_overlap_transfer_rejects_unclosed_parent_measure():
    with pytest.raises(ValueError,match="not closed"):
        LEDGER.conservative_parent_transfer([1.],[1],[1000.],[1.],[1],[[.9]])

def test_overlap_transfer_rejects_cross_material_overlap():
    with pytest.raises(ValueError,match="cross-material"):
        LEDGER.conservative_parent_transfer([1.],[1],[1000.],[1.],[2],[[1.]])

def test_overlap_transfer_rejects_negative_or_nonfinite_measures():
    with pytest.raises(ValueError,match="positive measures"):
        LEDGER.conservative_parent_transfer([1.],[1],[1000.],[1.],[1],[[-.1]])
    with pytest.raises(ValueError,match="finite"):
        LEDGER.conservative_parent_transfer([1.],[1],[1000.],[1.],[1],[[float("nan")]])

def test_closed_shell_checks_edge_incidence_and_orientability():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    got=LEDGER.closed_shell(p,f,"tetra")
    assert got["boundary_edges"]==0
    assert got["nonmanifold_edges"]==0
    assert got["closed_two_incidence"]
    assert got["orientable_closed_boundary_chain_zero_after_consistent_orientation"]
    open_shell=LEDGER.closed_shell(p,f[:-1],"open")
    assert open_shell["boundary_edges"]>0
    assert not open_shell["closed_two_incidence"]
