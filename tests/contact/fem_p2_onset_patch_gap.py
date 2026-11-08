"""Compare actual and restricted-prolonged n16 gap on common quadrature points."""
import json,sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from skfem import Basis,FacetBasis,MeshTet,ElementTetP2
d16=json.load(open(sys.argv[1]));previous=json.load(open(sys.argv[2]))
L=.02;z0=.0002;d=1e-4;k=1e16
models={}
for n in (8,16):
 g=np.linspace(0,L,n+1);mesh=MeshTet.init_tensor(g,g,g+z0)
 b=Basis(mesh,ElementTetP2());x=b.doflocs.T
 mask=np.abs(x[:,2]-z0)<1e-12
 facets=mesh.facets_satisfying(lambda x:np.abs(x[2]-z0)<1e-12,boundaries_only=True)
 models[n]=(mesh,b,x,mask,facets)
def transfer(src,dst,z,eps):
 _,a,_,_,_=models[src];_,b,x,mask,_=models[dst]
 pts=x[mask].copy();pts[:,2]+=eps;pts[:,:2]=np.clip(pts[:,:2],eps,L-eps);fn=a.interpolator(z)
 val=np.concatenate([fn(pts[j:j+32].T) for j in range(0,len(pts),32)])
 ret=np.zeros(b.N);ret[mask]=val;return ret
def duffy(n):
 x,w=leggauss(n);x=(x+1)/2;w=w/2;u,v=np.meshgrid(x,x,indexing='ij')
 return np.array([u.ravel(),((1-u)*v).ravel()]),(w[:,None]*w[None,:]*(1-u)).ravel()
def pressure(g):
 q=g*g-d*d;lg=np.log(g*g/(d*d))
 return np.where(g<d,2*g*k*(2*q*lg+q*q/g**2),0.)
actual=np.array(d16['displacements'][3])[:,2]
projected={eps:transfer(8,16,transfer(16,8,actual,eps),eps) for eps in (1e-12,1e-13)}
out={'time_s':.0048,'peak_N':.08743226330345472,'results':{}}
mesh,_,_,_,facets=models[16]
for order in (48,64):
 f=FacetBasis(mesh,ElementTetP2(),facets=facets,quadrature=duffy(order))
 xyz=f.global_coordinates();ix=np.minimum((xyz[0]/(L/4)).astype(int),3)
 iy=np.minimum((xyz[1]/(L/4)).astype(int),3);patch=(ix*4+iy).ravel();w=f.dx.ravel()
 gaps={'actual':(xyz[2]+f.interpolate(actual).value).ravel()}
 for eps,z in projected.items():gaps[str(eps)]=(xyz[2]+f.interpolate(z).value).ravel()
 if min(np.min(g) for g in gaps.values())<=0:raise ValueError('nonpositive quadrature gap')
 areas=np.bincount(patch,weights=w,minlength=16)
 measures={}
 for name,g in gaps.items():
  force=np.bincount(patch,weights=pressure(g)*w,minlength=16)
  avg=np.bincount(patch,weights=g*w,minlength=16)/areas
  measures[name]={'force_N':force.tolist(),'mean_gap_m':avg.tolist(),'min_gap_m':float(np.min(g))}
 diff=gaps['1e-12']-gaps['actual']
 mean_diff=np.bincount(patch,weights=diff*w,minlength=16)/areas
 rms=np.sqrt(np.bincount(patch,weights=diff*diff*w,minlength=16)/areas)
 max_abs=np.zeros(16);np.maximum.at(max_abs,patch,np.abs(diff))
 pconst=float(pressure(np.array([.99*d]))[0])
 out['results'][str(order)]={'areas_m2':areas.tolist(),'area_sum_m2':float(sum(areas)),
  'constant_force_N':float(sum(areas)*pconst),'constant_expected_N':L*L*pconst,
  'measures':measures,'gap_difference_mean_m':mean_diff.tolist(),
  'gap_difference_rms_m':rms.tolist(),'gap_difference_max_abs_m':max_abs.tolist()}
 del f,xyz,ix,iy,patch,w,gaps,diff
print(json.dumps(out))
