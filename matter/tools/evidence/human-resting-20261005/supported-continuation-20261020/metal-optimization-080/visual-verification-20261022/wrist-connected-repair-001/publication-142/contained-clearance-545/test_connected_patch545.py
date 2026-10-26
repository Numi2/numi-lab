import numpy as np
from build_connected_endpoint_patch545 import solve_patch

def test_patch_dirichlet_and_fixed_protected_coordinates():
 x=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,0]],np.float32)
 f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
 y,r=solve_patch(x,f,np.array([4]),np.array([[.5,.5,.1]],np.float32),np.array([0]),np.array([2]),1)
 assert np.array_equal(y[4],np.array([.5,.5,.1],np.float32))
 assert np.array_equal(y[[0,2]],x[[0,2]])
 assert np.all(y[[1,3],2]>0)and np.all(y[[1,3],2]<.1)
 assert r['linear_solve_residual_max']<1e-14
