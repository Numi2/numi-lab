"""P2 cross-mesh transfer and positive Duffy quadrature on retained onset states."""
import json,sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from skfem import Basis,FacetBasis,MeshTet,ElementTetP2

data={8:json.load(open(sys.argv[1])),16:json.load(open(sys.argv[2]))}
L=.02;z0=.0002;d=1e-4;k=1e16; peak=.08743226330345472
models={}
for n in (8,16):
 grid=np.linspace(0,L,n+1);mesh=MeshTet.init_tensor(grid,grid,grid+z0)
 b=Basis(mesh,ElementTetP2());xyz=b.doflocs.T
 boundary=np.abs(xyz[:,2]-z0)<1e-12
 facets=mesh.facets_satisfying(lambda x:np.abs(x[2]-z0)<1e-12,boundaries_only=True)
 models[n]=(mesh,b,xyz,boundary,facets)
def transfer(src,dst,z,eps):
 _,a,_,_,_=models[src];_,b,x,boundary,_=models[dst]
 pts=x[boundary].copy();pts[:,2]+=eps;pts[:,:2]=np.clip(pts[:,:2],eps,L-eps); fn=a.interpolator(z)
 val=np.concatenate([fn(pts[j:j+32].T) for j in range(0,len(pts),32)])
 result=np.zeros(b.N);result[boundary]=val
 return result
def poly(x,i):
 a,b,c=x.T
 if i==0:return np.ones(len(x))*1e-6
 if i==1:return 1e-6*(1+a/L+2*b/L+3*(c-z0)/L)
 return 1e-6*(1+(a/L)**2+(b/L)**2+(a*b/L**2)+((c-z0)/L)**2)
out={'peak_N':peak,'offsets_m':[1e-12,1e-13],'polynomial_controls':{},'samples':[]}
for src,dst in ((8,16),(16,8)):
 _,_,x,_,_=models[src];_,_,y,mask,_=models[dst]
 for i in range(3):
  zz=transfer(src,dst,poly(x,i),1e-12);q=y[mask].copy();q[:,2]+=1e-12;q[:,:2]=np.clip(q[:,:2],1e-12,L-1e-12)
  out['polynomial_controls'][f'{src}_to_{dst}_degree_{i}_max_abs_m']=float(np.max(abs(zz[mask]-poly(q,i))))
states={}
for n in (8,16):
 for j in range(6):
  idx=45+j if n==8 else j
  s=np.array(data[n]['displacements'][idx])[:,2]
  states[(n,j,'original')]=s
  for dst in (8,16):
   if n!=dst:
    for eps in (1e-12,1e-13):
     states[(dst,j,f'from{n}_eps{eps}')]=transfer(n,dst,s,eps)
def duffy(order):
 x,w=leggauss(order);x=(x+1)/2;w=w/2
 u,v=np.meshgrid(x,x,indexing='ij')
 return np.array([u.ravel(),((1-u)*v).ravel()]),(w[:,None]*w[None,:]*(1-u)).ravel()
for n in (8,16):
 mesh,_,_,_,facets=models[n]
 for order in (48,64):
  f=FacetBasis(mesh,ElementTetP2(),facets=facets,quadrature=duffy(order))
  for j in range(6):
   for tag in ('original',f'from{24-n}_eps1e-12',f'from{24-n}_eps1e-13'):
    g=f.global_coordinates()[2]+f.interpolate(states[(n,j,tag)]).value
    if np.min(g)<=0:raise ValueError('nonpositive sampled gap')
    q=g*g-d*d;lg=np.log(g*g/(d*d));pressure=np.where(g<d,2*g*k*(2*q*lg+q*q/g**2),0.)
    out['samples'].append({'mesh':n,'index':j,'time_s':.0045+j*.0001,'kind':tag,'order':order,
     'force_N':float(np.sum(pressure*f.dx)),'min_gap_m':float(np.min(g)),'max_gap_m':float(np.max(g))})
  del f
print(json.dumps(out))
