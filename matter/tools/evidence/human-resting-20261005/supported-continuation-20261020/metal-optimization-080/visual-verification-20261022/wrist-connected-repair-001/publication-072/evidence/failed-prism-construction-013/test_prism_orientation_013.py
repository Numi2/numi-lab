import numpy as np
from build_fit_o1245_lp_013_run003 import prism_cells, oriented_boundary_chain, perm_sign

def det(x,t):
 q=np.asarray(x,float)[np.asarray(t,int)]
 return np.linalg.det(np.stack((q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]),axis=2))

def cap_coefficient(chain,tri):
 return chain[tuple(sorted(map(int,tri)))]*perm_sign(tri)

def test_regular_prism_cells_positive_and_volume_matches():
 x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,0,1],[0,1,1]],float)
 t=prism_cells(0,1,2,3,4,5,relative_parity=1,low_cap_sign=-1)
 ds=det(x,t)
 assert np.array_equal(ds,np.array([1.,1.,1.])),ds
 assert abs(ds.sum()/6 - .5)<1e-15

def test_regular_prism_has_eight_oriented_boundary_facets_and_caps():
 t=prism_cells(0,1,2,3,4,5,relative_parity=1,low_cap_sign=-1)
 chain=oriented_boundary_chain(t)
 assert len(chain)==8,chain
 assert all(abs(v)==1 for v in chain.values()),chain
 assert cap_coefficient(chain,(0,1,2))==-1
 assert cap_coefficient(chain,(3,4,5))==1

def test_odd_triangle_order_is_coherently_oriented_by_boundary_not_geometry():
 x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,0,1],[0,1,1]],float)
 t=prism_cells(0,2,1,3,5,4,relative_parity=-1,low_cap_sign=-1)
 ds=det(x,t)
 assert np.array_equal(ds,np.array([1.,1.,1.])),ds
 chain=oriented_boundary_chain(t)
 assert cap_coefficient(chain,(0,1,2))==-1
 assert cap_coefficient(chain,(3,4,5))==1

def test_original_middle_tet_is_opposite_orientation():
 x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,0,1],[0,1,1]],float)
 old=np.array([[0,1,2,5],[0,1,4,5],[0,3,4,5]],int)
 assert np.array_equal(det(x,old),np.array([1.,-1.,1.]))
