import numpy as np
from project_patch_quality545 import oriented_grad

def test_oriented_area_gradient_matches_finite_difference():
 p=np.array([[.2,.3,.1],[1.,.2,.3],[.1,1.,.1]])
 n=np.array([.1,.2,1.]);den=1.3;g=oriented_grad(p,n,den)
 for i in range(3):
  for k in range(3):
   d=np.zeros((3,3));d[i,k]=1e-6
   def f(x):return np.cross(x[1]-x[0],x[2]-x[0])@n/den
   assert abs((f(p+d)-f(p-d))/2e-6-g[i,k])<1e-9
