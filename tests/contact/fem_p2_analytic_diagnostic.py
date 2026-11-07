"""P1/P2 on smooth prescribed states versus independently integrated laws."""
import json,sys,numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
from skfem import MeshTet,Basis,FacetBasis,ElementTetP1,ElementTetP2
L=.02;z0=.0002;mu=2000.;lam=5000.;k=1e16;d=1e-4;A=.0002
def W(s):
 j=1+s;l=np.log(j)
 return mu*((2+j*j)-3)/2-mu*l+lam*l*l/2
exact_volume=L*L*quad(lambda z:W(A*np.pi/L*np.cos(np.pi*z/L)),0,L,
                     epsabs=1e-14,epsrel=1e-13)[0]
def pressure(g):
 q=g*g-d*d;return np.where(g<d,2*g*k*(2*q*np.log(g*g/d**2)+q*q/g**2),0.)
def density(g):
 q=g*g-d*d;return np.where(g<d,-k*q*q*np.log(g*g/d**2),0.)
def face_exact(order):
 x,w=leggauss(order);x=(x+1)*L/2;w=w*L/2
 xx,yy=np.meshgrid(x,x,indexing='ij');ww=w[:,None]*w[None,:]
 g=95e-6+2e-6*np.cos(np.pi*xx/L)*np.cos(np.pi*yy/L)
 return np.sum(density(g)*ww),np.sum(pressure(g)*ww)
ref48=face_exact(48);ref96=face_exact(96)
if max(abs(a-b)/abs(b) for a,b in zip(ref48,ref96))>1e-10:raise RuntimeError('reference quadrature mismatch')
n=int(sys.argv[1])
if n not in (2,4,8,16):raise ValueError('invalid n')
grid=np.linspace(0,L,n+1)
mesh=MeshTet.init_tensor(grid,grid,grid+z0)
result={'n':n,'tetrahedra':mesh.nelements,'reference_volume_energy_J':exact_volume,
 'reference_surface_energy_J':ref96[0],'reference_force_N':ref96[1],
 'reference_quadrature_relative':max(abs(a-b)/abs(b) for a,b in zip(ref48,ref96)),'orders':{}}
for degree,elem in ((1,ElementTetP1()),(2,ElementTetP2())):
 basis=Basis(mesh,elem,intorder=8)
 xyz=basis.doflocs;uz=A*np.sin(np.pi*(xyz[2]-z0)/L)
 field=basis.interpolate(uz);F=np.zeros((basis.nelems,basis.dx.shape[1],3,3))
 F[:]=np.eye(3)
 for j in range(3):F[:,:,2,j]+=field.grad[j]
 J=np.linalg.det(F)
 if np.min(J)<=0:raise RuntimeError('invalid deformation')
 logj=np.log(J)
 energy=np.sum((mu*(np.sum(F*F,axis=(-1,-2))-3)/2-mu*logj+lam*logj**2/2)*basis.dx)
 facets=mesh.facets_satisfying(lambda x:np.abs(x[2]-z0)<1e-12,boundaries_only=True)
 face=FacetBasis(mesh,elem,facets=facets,intorder=8)
 xyz=face.doflocs;x,y,z=xyz
 u=(-105e-6+2e-6*np.cos(np.pi*x/L)*np.cos(np.pi*y/L))*(1-(z-z0)/L)
 gap=face.global_coordinates()[2]+face.interpolate(u).value
 if np.min(gap)<=0:raise RuntimeError('nonpositive contact gap')
 se=np.sum(density(gap)*face.dx);sf=np.sum(pressure(gap)*face.dx)
 result['orders'][str(degree)]={'dofs':basis.N,'min_J':float(J.min()),'min_gap_m':float(gap.min()),
  'volume_energy_J':float(energy),'volume_energy_relative_error':float(abs(energy-exact_volume)/abs(exact_volume)),
  'surface_energy_J':float(se),'surface_energy_relative_error':float(abs(se-ref96[0])/abs(ref96[0])),
  'force_N':float(sf),'force_relative_error':float(abs(sf-ref96[1])/abs(ref96[1]))}
print(json.dumps(result))
