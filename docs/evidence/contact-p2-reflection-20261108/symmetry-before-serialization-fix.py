"""Independent reflected P2 mesh comparison for a frictionless plane."""
import sys,json,numpy as np
from pathlib import Path
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skfem import MeshTet, Basis, FacetBasis, ElementTetP2, BilinearForm, asm
sys.path.insert(0,'/Users/numi/numi-lab-contact-reference-20261107/tests/contact')
from fem_p2_dynamics import Model

L=.02;z0=.0002
@BilinearForm
def mass(u,v,w): return 1100*u*v

def construct(mesh):
 m=Model.__new__(Model);m.L=L;m.z0=z0;m.d=1e-4;m.k=1e16;m.mu=2000.;m.lam=5000.
 m.mesh=mesh;m.b=Basis(mesh,ElementTetP2(),intorder=8)
 facets=mesh.facets_satisfying(lambda x:np.abs(x[2]-z0)<1e-12,boundaries_only=True)
 m.f=FacetBasis(mesh,ElementTetP2(),facets=facets,intorder=18)
 m.M=asm(mass,m.b).tocsc();m.massload=m.M@np.ones(m.b.N);m.ref=m.b.doflocs.T
 return m

def build(n):
 grid=np.linspace(0,L/2,n//2+1);z=np.linspace(z0,z0+L,n+1)
 q=MeshTet.init_tensor(grid,grid,z)
 points=[];tet=[];base=q.p.shape[1]
 for sx in (-1,1):
  for sy in (-1,1):
   p=q.p.copy();p[0]=L/2+sx*p[0];p[1]=L/2+sy*p[1]
   tet.append(q.t+len(points)*base);points.append(p)
 pts=np.concatenate(points,axis=1);cells=np.concatenate(tet,axis=1)
 pts=np.round(pts,decimals=14)
 unique,indices=np.unique(pts,axis=1,return_inverse=True)
 full=MeshTet(unique,indices[cells])
 return construct(q),construct(full)

def acceleration(model,residual,constrained=False):
 result=np.zeros_like(residual)
 for axis in range(3):
  if constrained and axis<2:
   free=np.flatnonzero(np.abs(model.ref[:,axis])>1e-13)
  else:free=np.arange(model.b.N)
  result[free,axis]=splu(model.M[free,:][:,free].tocsc()).solve(-residual[free,axis])
 return result

def state(model,quarter,broken=False):
 x,y,z=model.ref.T
 if not quarter:x=x-L/2;y=y-L/2
 zeta=(z-z0)/L
 u=np.zeros((model.b.N,3));u[:,0]=2e-6*(x/L)*zeta;u[:,1]=-1e-6*(y/L)*zeta
 u[:,2]=-105e-6+2e-6*((x/L)**2+(y/L)**2)+1e-6*zeta
 if broken:u[:,2]+=3e-6*x/L
 return u

def rel(a,b,scale):return abs(a-b)/max(abs(b),scale)

def run(n):
 q,f=build(n);mapping_dist,mapping=cKDTree(f.ref).query(q.ref+np.array([L/2,L/2,0]),k=1)
 if max(mapping_dist)>1e-12:raise RuntimeError('quarter P2 nodes do not match reflected full mesh')
 out={'n':n,'quarter_dofs':q.b.N,'full_dofs':f.b.N,'cases':{}}
 for name in ['freefall','contact']:
  u=np.zeros((q.b.N,3)) if name=='freefall' else state(q,True)
  v=np.zeros((f.b.N,3)) if name=='freefall' else state(f,False)
  eq,rq,cq,ceq,gq,jq=q.evaluate(u);ef,rf,cf,cef,gf,jf=f.evaluate(v)
  aq=acceleration(q,rq,True);af=acceleration(f,rf)
  errors={'energy':rel(4*eq,ef,1e-12),'contact_energy':rel(4*ceq,cef,1e-12),'force':rel(4*cq.sum(),cf.sum(),1e-12),
   'mass':rel(4*q.massload.sum(),f.massload.sum(),1e-12),'gap_abs_m':abs(gq-gf),'jacobian_abs':abs(jq-jf),
   'acceleration_max_abs_m_s2':float(np.max(np.abs(aq-af[mapping]))),
   'acceleration_scale_m_s2':float(max(np.max(np.abs(af[mapping])),9.81)),
   'mirror_normal_max_abs_m_s2':float(max(np.max(abs(aq[abs(q.ref[:,0])<1e-13,0])),np.max(abs(aq[abs(q.ref[:,1])<1e-13,1]))))}
  errors['acceleration_relative']=errors['acceleration_max_abs_m_s2']/errors['acceleration_scale_m_s2']
  if errors['energy']>1e-10 or errors['contact_energy']>1e-10 or errors['force']>1e-10 or errors['mass']>1e-10 or errors['gap_abs_m']>1e-12 or errors['jacobian_abs']>1e-11 or errors['acceleration_relative']>2e-8 or errors['mirror_normal_max_abs_m_s2']>1e-12:raise RuntimeError(f'{name}: {errors}')
  out['cases'][name]=errors
 # An odd scalar displacement has opposite signs at reflected physical points.
 u=state(f,False,True);reflect=f.ref.copy();reflect[:,0]=L-reflect[:,0]
 dist,mir=cKDTree(f.ref).query(reflect,k=1);assert max(dist)<1e-12
 asym=float(np.max(abs(u[:,2]-u[mir,2])));assert asym>1e-7
 out['asymmetric_control_max_m']=asym
 return out
if __name__=='__main__':
 result=[run(n) for n in (2,4,8)]
 print(json.dumps({'results':result,'status':'pass'},indent=2))
