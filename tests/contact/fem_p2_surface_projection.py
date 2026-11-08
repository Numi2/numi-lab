"""Cross-mesh P2 surface transfer at fixed retained states."""
import json,sys,numpy as np
from numpy.polynomial.legendre import leggauss
from skfem import FacetBasis,ElementTetP2
from fem_p2_dynamics import Model
data={n:json.load(open(sys.argv[j])) for j,n in enumerate((4,8),start=1)}
models={n:Model(n,18) for n in (4,8)}
def duffy(n):
 x,w=leggauss(n);x=(x+1)/2;w=w/2;u,v=np.meshgrid(x,x,indexing='ij');ww=w[:,None]*w[None,:]*(1-u)
 return np.array([u.ravel(),((1-u)*v).ravel()]),ww.ravel()
faces={}
for n,m in models.items():
 facets=m.mesh.facets_satisfying(lambda x:np.abs(x[2]-m.z0)<1e-12,boundaries_only=True)
 faces[n]=FacetBasis(m.mesh,ElementTetP2(),facets=facets,quadrature=duffy(64))
def force(n,z):
 m=models[n];f=faces[n];g=f.global_coordinates()[2]+f.interpolate(z).value
 if np.min(g)<=0:raise ValueError('nonpositive projected sampled gap')
 _,pressure=m.plane(g,m.d,m.k)
 return float(np.sum(pressure*f.dx)),float(np.min(g))
out={'samples':[],'inside_offset_m':1e-12}
for idx in (46,50,60,80,95,100):
 row={'time_s':float(data[4]['trace'][idx][0]),'directions':{}}
 for src,dst in ((4,8),(8,4)):
  a,b=models[src],models[dst]
  z=np.array(data[src]['displacements'][idx])[:,2]
  old,g0=force(src,z)
  boundary=np.abs(b.ref[:,2]-b.z0)<1e-12
  pts=b.ref[boundary].copy();pts[:,2]+=1e-12
  projected=np.zeros(b.b.N);projected[boundary]=a.b.interpolator(z)(pts.T)
  new,g1=force(dst,projected)
  row['directions'][str(src)+'_to_'+str(dst)]={'original_force_N':old,'projected_force_N':new,
   'recorded_duffy64_force_N':data[src]['trace'][idx][3],'original_min_gap_m':g0,'projected_min_gap_m':g1}
 out['samples'].append(row)
print(json.dumps(out))
