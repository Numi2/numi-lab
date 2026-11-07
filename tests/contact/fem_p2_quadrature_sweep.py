"""Evaluate independently constructed surface quadratures at retained P2 states."""
import json,sys,numpy as np
from numpy.polynomial.legendre import leggauss
from skfem import FacetBasis,ElementTetP2
from fem_p2_dynamics import Model
data=json.load(open(sys.argv[1]));m=Model(2);facets=m.mesh.facets_satisfying(lambda x:np.abs(x[2]-m.z0)<1e-12,boundaries_only=True)
def duffy(n):
 x,w=leggauss(n);x=(x+1)/2;w=w/2
 u,v=np.meshgrid(x,x,indexing='ij');ww=w[:,None]*w[None,:]*(1-u)
 return np.array([u.ravel(),((1-u)*v).ravel()]),ww.ravel()
rules={}
for n in (8,12,14,16,18):
 f=FacetBasis(m.mesh,ElementTetP2(),facets=facets,intorder=n)
 rules['triangle'+str(n)]=f
for n in (48,64):
 f=FacetBasis(m.mesh,ElementTetP2(),facets=facets,quadrature=duffy(n))
 rules['duffy'+str(n)]=f
out={'input_sha256':__import__('hashlib').sha256(open(sys.argv[1],'rb').read()).hexdigest(),'rules':{},'samples':[]}
for key,f in rules.items():out['rules'][key]={'points_per_face':int(f.dx.shape[1]),'reference_weight_sum':float(f.W.sum())}
for i,disp in enumerate(data['displacements']):
 u=np.array(disp);r={'time_s':data['trace'][i][0],'force_N':{},'min_gap_m':{}}
 for key,f in rules.items():
  gap=f.global_coordinates()[2]+f.interpolate(u[:,2]).value
  if np.min(gap)<=0:raise RuntimeError('nonpositive sampled gap')
  _,pressure=m.plane(gap,m.d,m.k)
  r['force_N'][key]=float(np.sum(pressure*f.dx));r['min_gap_m'][key]=float(gap.min())
 out['samples'].append(r)
print(json.dumps(out))
