import sys
import numpy as np
from scipy import sparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"canonical/matter/tools"))
import normal_equation_lsmr_adapter as adapter

def test_well_conditioned_system():
    A=sparse.csr_matrix([[1.,0.],[1.,1.],[0.,1.]])
    b=np.array([1.,2.,1.])
    out=adapter.lsmr(A,b)
    assert out[1]==2 and np.isfinite(out[0]).all()
    rel=np.linalg.norm(A.T@(A@out[0]-b))/np.linalg.norm(A.T@b)
    assert rel<1e-12

def test_nearly_collinear_normal_system_fails_closed():
    # Normal equations lose the distinction at this condition scale; splu must reject.
    eps=1e-8
    A=sparse.csr_matrix([[1.,1.],[1.,1.+eps],[0.,eps]])
    b=np.array([1.,2.,1.])
    try:
        adapter.lsmr(A,b)
    except RuntimeError as e:
        assert "singular" in str(e).lower()
    else:
        raise AssertionError("near-collinear matrix was not rejected as singular")

def test_nonfinite_input_fails_closed():
    A=sparse.csr_matrix([[1.,0.],[0.,1.]])
    try:
        adapter.lsmr(A,np.array([1.,np.nan]))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite input was accepted")

if __name__=="__main__":
    test_well_conditioned_system()
    test_nearly_collinear_normal_system_fails_closed()
    test_nonfinite_input_fails_closed()
    print("3 passed")
