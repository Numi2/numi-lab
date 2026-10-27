from pathlib import Path
import importlib.util,numpy as np,pytest
import build_joint_outward_targets597 as m
@pytest.fixture
def depth():
 assert m.sha(m.DEPTH)==m.DEPTH_SHA
 return m.load("depth597tests",m.DEPTH)
def test_aligned(depth):
 d,p=m.joint_direction(depth,np.array([0.,0.,1.]),np.array([0.,0.,1.]))
 assert np.array_equal(d,[0.,0.,1.])
def test_oblique(depth):
 s=np.array([1.,0.,0.]);c=np.array([-.5,0.,np.sqrt(.75)])
 d,p=m.joint_direction(depth,s,c)
 assert d@s>=.05 and d@c>=.05
def test_opposite_fails(depth):
 with pytest.raises(ValueError):m.joint_direction(depth,np.array([0.,0.,1.]),np.array([0.,0.,-1.]))
def test_current_f32_sign():
 x=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32);f=np.array([[0,1,2]])
 assert np.array_equal(m.current_face_outward(x,f,0,-1),[0,0,-1])
 with pytest.raises(ValueError):m.current_face_outward(x.astype(float),f,0,-1)
 with pytest.raises(ValueError):m.current_face_outward(x,f,0,0)
def test_zero_current_face_fails():
 with pytest.raises(ValueError):m.current_face_outward(np.zeros((3,3),np.float32),np.array([[0,1,2]]),0,-1)

