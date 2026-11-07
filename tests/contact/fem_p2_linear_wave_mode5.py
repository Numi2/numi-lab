"""Discrete linearized bulk acceleration versus exact continuum mode."""
import json,numpy as np
from scipy.sparse.linalg import cg
from skfem import MeshTet,Basis,ElementTetP1,ElementTetP2,BilinearForm,asm
from skfem.helpers import grad
L=.02;mu=2000.;lam=5000.;rho=1100.;mode=5
omega=mode*np.pi/L*np.sqrt((lam+2*mu)/rho);w2=omega**2
@BilinearForm
def K(u,v,w):
 return mu*(grad(u)[0]*grad(v)[0]+grad(u)[1]*grad(v)[1])+(lam+2*mu)*grad(u)[2]*grad(v)[2]
@BilinearForm
def M(u,v,w):return rho*u*v
out={'continuum_omega_rad_s':float(omega),'mode':mode,'results':[]}
for n in (8,16):
 grid=np.linspace(0,L,n+1);mesh=MeshTet.init_tensor(grid,grid,grid)
 for degree,elem in ((1,ElementTetP1()),(2,ElementTetP2())):
  basis=Basis(mesh,elem,intorder=8);mat=asm(M,basis).tocsr();stiff=asm(K,basis).tocsr()
  u=np.cos(mode*np.pi*basis.doflocs[2]/L)
  for kind in (('consistent','lumped') if degree==1 else ('consistent',)):
   if kind=='lumped':
    diag=np.asarray(mat.sum(axis=1)).ravel();massu=diag*u
    r=stiff@u-w2*massu;sol=r/diag;solve_rel=0.
   else:
    massu=mat@u;r=stiff@u-w2*massu
    sol,info=cg(mat,r,rtol=1e-11,atol=0,maxiter=1000)
    if info!=0:raise RuntimeError('mass solve did not converge')
    solve_rel=float(np.linalg.norm(mat@sol-r)/max(np.linalg.norm(r),1e-30))
   norm=float(u@massu);ray=float(u@(stiff@u)/norm)
   if norm<=0 or np.any(mat.diagonal()<=0):raise RuntimeError('mass not positive')
   acc=float(np.sqrt(max(0.,float(r@sol))/(w2*w2*norm)))
   out['results'].append({'n':n,'degree':degree,'mass':kind,'dofs':int(basis.N),'omega_rayleigh_rad_s':float(np.sqrt(max(0.,ray))),
    'frequency_relative_error':float(abs(np.sqrt(max(0.,ray))/omega-1)),
    'initial_acceleration_mass_relative_error':acc,'mass_solve_relative_residual':solve_rel})
print(json.dumps(out))
