import importlib.util,sys,tempfile
from pathlib import Path
import numpy as np
p=Path(__file__).with_name("fit.py")
spec=importlib.util.spec_from_file_location("layer_guard_owner",p);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
O=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
I=O+np.array([0.,0.,-.4]);D=O+np.array([0.,0.,-1.5])
faces=np.array([[0,2,1]],dtype=np.int64)
def volumes(x):
 out,mid,deep=[q+x for q in (O,I,D)]
 ans=[]
 for u,v in ((out,mid),(mid,deep)):
  a,b,c=0,1,2
  for tet,sign in [([u[a],u[b],u[c],v[c]],-1.),([u[a],u[b],v[b],v[c]],1.),([u[a],v[a],v[b],v[c]],-1.)]:
   aa,bb,cc,dd=tet
   ans.append(sign*np.dot(bb-aa,np.cross(cc-aa,dd-aa))/6)
 return np.asarray(ans)
with tempfile.TemporaryDirectory() as td:
 file=Path(td)/"layers.npz";np.savez(file,outer_points_f64=O/1000,inner_points_f64=I/1000,deep_points_f64=D/1000,outer_faces=faces[:,[0,2,1]])
 G,lower,report=mod.make_layer_guard(file,faces,np.array([0]),np.array([0,1,2]))
 assert report["rows"]==6 and report["baseline_nonpositive_rows"]==0
 baseline=volumes(np.zeros((3,3)));num=np.empty((6,9));eps=1e-5
 for j in range(9):
  x=np.zeros((3,3));x.ravel()[j]=eps
  num[:,j]=(volumes(x)-volumes(-x))/(2*eps)
 norms=np.linalg.norm(num,axis=1)
 assert np.all(norms>0) and np.allclose(G.toarray(),num/norms[:,None],rtol=2e-6,atol=2e-7)
 assert np.allclose(lower,-.8*baseline/norms,rtol=2e-6,atol=2e-7)
 print("six layer-volume gradient and bound checks pass")
