import importlib.util
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('contour122',Path(__file__).with_name('build_connected_source_contour_122.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
F=np.array([[0,1,2],[1,3,2],[2,3,4],[3,5,4],[4,5,6]])
O=np.zeros((7,3))
def test_connected_patch_plane_and_exact_outside_anchor():
    U,r=m.solve_translations(O,F,[{'column_row':0,'normal':[1,0,0],'deficit_m':.001}],rings=1)
    assert U[0,0]>=.001-2e-9
    assert 0<U[1,0]<U[0,0]
    np.testing.assert_array_equal(U[3:],np.zeros((4,3)))
    assert r['outside_patch_exact_zero']
def test_duplicate_plane_does_not_double_the_displacement():
    p={'column_row':0,'normal':[1,0,0],'deficit_m':.001}
    a,_=m.solve_translations(O,F,[p],rings=1);b,_=m.solve_translations(O,F,[p,p],rings=1)
    np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-10)
def test_two_directions_and_inactive_constraint():
    ps=[{'column_row':0,'normal':[1,0,0],'deficit_m':.001},{'column_row':0,'normal':[0,1,0],'deficit_m':.002},{'column_row':1,'normal':[1,0,0],'deficit_m':-.001}]
    U,r=m.solve_translations(O,F,ps,rings=1)
    assert U[0,0]>=.001-2e-9 and U[0,1]>=.002-2e-9
    assert np.isfinite(U).all()
