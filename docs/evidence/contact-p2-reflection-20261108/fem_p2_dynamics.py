"""Independent P2 consistent-mass FEM with continuum plane contact energy."""
import json,sys,numpy as np
from skfem import MeshTet,Basis,FacetBasis,ElementTetP2,LinearForm,BilinearForm,asm
from skfem.helpers import grad
from scipy.sparse.linalg import splu
class Model:
 def __init__(self,n,face_order=8):
  self.L=.02;self.z0=.0002;self.d=1e-4;self.k=1e16;self.mu=2000.;self.lam=5000.
  grid=np.linspace(0,self.L,n+1)
  self.mesh=MeshTet.init_tensor(grid,grid,grid+self.z0)
  self.b=Basis(self.mesh,ElementTetP2(),intorder=8)
  facets=self.mesh.facets_satisfying(lambda x:np.abs(x[2]-self.z0)<1e-12,boundaries_only=True)
  self.f=FacetBasis(self.mesh,ElementTetP2(),facets=facets,intorder=face_order)
  @BilinearForm
  def mass(u,v,w):return 1100*u*v
  self.M=asm(mass,self.b).tocsc()
  self.factor=splu(self.M)
  self.massload=self.M@np.ones(self.b.N)
  self.ref=self.b.doflocs.T
 @staticmethod
 def plane(g,d,k):
  q=g*g-d*d;lg=np.log(g*g/(d*d))
  return np.where(g<d,-k*q*q*lg,0.),np.where(g<d,2*g*k*(2*q*lg+q*q/g**2),0.)
 def evaluate(self,u):
  b,f=self.b,self.f;mu,lam=self.mu,self.lam
  fields=[b.interpolate(u[:,i]) for i in range(3)]
  F=np.zeros((b.nelems,b.dx.shape[1],3,3));F[:]=np.eye(3)
  for i in range(3):
   for j in range(3):F[:,:,i,j]+=fields[i].grad[j]
  J=np.linalg.det(F)
  if np.any(J<=0):raise ValueError('nonpositive Jacobian at volume quadrature')
  l=np.log(J);P=mu*F+(lam*l-mu)[...,None,None]*np.swapaxes(np.linalg.inv(F),-1,-2)
  energy=float(np.sum((mu*(np.sum(F*F,axis=(-1,-2))-3)/2-mu*l+lam*l*l/2)*b.dx))
  @LinearForm
  def row(v,w):return w.p0*grad(v)[0]+w.p1*grad(v)[1]+w.p2*grad(v)[2]
  r=np.column_stack([asm(row,b,p0=P[:,:,i,0],p1=P[:,:,i,1],p2=P[:,:,i,2]) for i in range(3)])
  gap=f.global_coordinates()[2]+f.interpolate(u[:,2]).value
  if np.any(gap<=0):raise ValueError('nonpositive surface quadrature gap')
  density,pressure=self.plane(gap,self.d,self.k)
  @LinearForm
  def load(v,w):return w.pressure*v
  cf=asm(load,f,pressure=pressure)
  ce=float(np.sum(density*f.dx))
  r[:,2]+=9.81*self.massload-cf
  energy+=float(9.81*np.dot(self.massload,self.ref[:,2]+u[:,2]))+ce
  return energy,r,cf,ce,float(np.min(gap)),float(np.min(J))
def checks():
 from scipy.linalg import eigvalsh
 m=Model(2);m12=Model(2,12);N=m.b.N
 x,y,z=m.ref.T;zr=(z-m.z0)/m.L
 a=np.zeros((N,3));a[:,2]=(-105e-6+2e-6*np.cos(np.pi*x/m.L)*np.cos(np.pi*y/m.L))*(1-zr)
 a[:,0]=1e-6*np.sin(np.pi*y/m.L)*np.sin(np.pi*zr)
 rng=np.random.default_rng(91);vec=rng.normal(size=(N,3));vec/=np.max(abs(vec))
 initial=np.zeros_like(a);results={}
 for name,u in [('initial',initial),('active',a)]:
  energy,r,cf,ce,gap,j=m.evaluate(u);eps=1e-9
  ep,_,_,cep,_,_=m.evaluate(u+eps*vec);em,_,_,cem,_,_=m.evaluate(u-eps*vec)
  gradfd=(ep-em)/(2*eps);directional=float(np.sum(r*vec))
  power=float(np.dot(cf,vec[:,2]));powerfd=-(cep-cem)/(2*eps)
  results[name]={'energy_directional_fd':gradfd,'energy_directional_exact':directional,
   'energy_relative':abs(gradfd-directional)/max(abs(directional),1e-30),
   'contact_power_fd':powerfd,'contact_power_exact':power,
   'contact_power_relative':abs(powerfd-power)/max(abs(power),1e-30) if name=='active' else 0.,
   'min_gap_m':gap,'min_jacobian':j}
 _,_,f8,_,_,_=m.evaluate(a);_,_,f12,_,_,_=m12.evaluate(a)
 zvec=np.zeros_like(a);_,r,_,_,_,_=m.evaluate(zvec)
 accel=np.column_stack([m.factor.solve(-r[:,i]) for i in range(3)])
 mass=float(m.massload.sum());sym=np.linalg.norm((m.M-m.M.T).toarray());mineig=float(eigvalsh(m.M.toarray(),subset_by_index=[0,0])[0])
 results['mass']={'kg':mass,'relative_error':abs(mass-.0088)/.0088,'symmetry_l2':sym,'min_eigenvalue':mineig,
  'freefall_relative':float(np.max(abs(accel-np.array([0.,0.,-9.81])))/9.81)}
 results['quadrature']={'force_sum_relative':float(abs(f8.sum()-f12.sum())/abs(f12.sum())),
  'force_l2_relative':float(np.linalg.norm(f8-f12)/np.linalg.norm(f12))}
 return results
if __name__=='__main__':
 if len(sys.argv)!=2 or sys.argv[1]!='check':raise ValueError('usage: check')
 print(json.dumps(checks()))
