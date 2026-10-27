import numpy as np,pytest
from component_patch_centers599 import component_patch_centers
def test_connected_components_and_ring_complete():
 f=np.array([[0,1,2],[2,1,3],[2,3,4],[4,3,5],[8,9,10],[10,9,11]])
 p={(1,1):{(0,0),(1,1),(4,2)}}
 centers,meta=component_patch_centers(f,p,rings=1)
 assert (0,1)in centers and (0,1,2)in centers
 assert (4,5)in centers
 assert meta[(0,1)]["ring"]==0 and meta[(0,1,2)]["ring"]==1
 assert len(centers)==len(set(centers))
 assert all(set(rr)<=set(range(len(f)))for rr in centers)
def test_maximum_and_deterministic():
 f=np.array([[0,1,2],[2,1,3],[2,3,4]])
 p={(1,2):{(0,0),(1,1)}}
 a,m=component_patch_centers(f,p,max_centers=1,max_faces=2)
 b,_=component_patch_centers(f,p,max_centers=1,max_faces=2)
 assert a==b==[(0,1)]
def test_bad_witness_fails():
 with pytest.raises(ValueError):component_patch_centers(np.array([[0,1,2]]),{(1,1):{(1,0)}})

