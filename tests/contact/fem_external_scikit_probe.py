"""Independent scikit-fem P1 assembly checks on retained native states."""
import json, sys
import numpy as np
from skfem import MeshTet, Basis, FacetBasis, ElementTetP1, LinearForm, asm
from skfem.helpers import grad

native=json.load(open(sys.argv[1]))
step=int(sys.argv[2])
ref=np.array([[.01*x,.01*y,.0002+.01*z]
              for z in range(3) for y in range(3) for x in range(3)])
tet=np.array(native['compiled_tetrahedra'],dtype=float)[:,:4].astype(int)
mesh=MeshTet(ref.T,tet.T,sort_t=False)
basis=Basis(mesh,ElementTetP1())
facets=mesh.facets_satisfying(lambda x: np.abs(x[2]-.0002)<1e-12, boundaries_only=True)
fb=FacetBasis(mesh,ElementTetP1(),facets=facets)
@LinearForm
def constant(v,w):return v
area=asm(constant,fb)
mass=1100*asm(constant,basis)
u=np.array(native['traces'][step]['node_state'])[:,:3]
rng=np.random.default_rng(218)
direction=rng.normal(size=u.shape)
direction/=np.max(np.abs(direction))
mu,lam,k,d=2000.,5000.,1e16,1e-4

def energy_residual_tangent(pos,vec):
 disp=pos-ref
 fields=[basis.interpolate(disp[:,j]) for j in range(3)]
 fields_d=[basis.interpolate(vec[:,j]) for j in range(3)]
 F=np.broadcast_to(np.eye(3),(basis.nelems,basis.dx.shape[1],3,3)).copy()
 H=np.zeros_like(F)
 for i in range(3):
  for j in range(3):
   F[:,:,i,j]+=fields[i].grad[j]
   H[:,:,i,j]=fields_d[i].grad[j]
 J=np.linalg.det(F)
 if np.any(J<=0):raise ValueError('inversion')
 invt=np.swapaxes(np.linalg.inv(F),-1,-2)
 logj=np.log(J)
 P=mu*F+(lam*logj-mu)[...,None,None]*invt
 dlogj=np.einsum('...ij,...ij->...',invt,H)
 dp=mu*H+lam*dlogj[...,None,None]*invt
 dp+=(mu-lam*logj)[...,None,None]*(invt@np.swapaxes(H,-1,-2)@invt)
 W=mu*(np.sum(F*F,axis=(-1,-2))-3)/2-mu*logj+lam*logj**2/2
 energy=np.sum(W*basis.dx)+np.dot(mass,9.81*pos[:,2])
 residual=np.zeros_like(pos);tangent=np.zeros_like(pos)
 @LinearForm
 def row(v,w):return w.p0*grad(v)[0]+w.p1*grad(v)[1]+w.p2*grad(v)[2]
 for i in range(3):
  residual[:,i]=asm(row,basis,p0=P[:,:,i,0],p1=P[:,:,i,1],p2=P[:,:,i,2])
  tangent[:,i]=asm(row,basis,p0=dp[:,:,i,0],p1=dp[:,:,i,1],p2=dp[:,:,i,2])
 residual[:,2]+=mass*9.81
 g=pos[:,2]; active=(area>0)&(g<d)
 if np.any(g[active]<=0):raise ValueError('penetration')
 q=g[active]**2-d*d;lg=np.log(g[active]**2/d**2)
 energy+=np.sum(-area[active]*k*q*q*lg)
 pressure=2*g[active]*k*(2*q*lg+q*q/g[active]**2)
 term=2*q*lg+q*q/g[active]**2
 deriv=4*g[active]*lg+8*q/g[active]-2*q*q/g[active]**3
 dpdg=2*k*(term+g[active]*deriv)
 residual[active,2]-=area[active]*pressure
 tangent[active,2]-=area[active]*dpdg*vec[active,2]
 return energy,residual,tangent

e,r,t=energy_residual_tangent(u,direction)
eps=1e-8
ep,rp,_=energy_residual_tangent(u+eps*direction,direction)
em,rm,_=energy_residual_tangent(u-eps*direction,direction)
state=np.column_stack((u,direction))
np.savetxt(sys.argv[3],state,fmt='%.17g')
out={'step':step,'active_nodes':int(np.sum((area>0)&(u[:,2]<d))),
 'reference_area_m2':float(area.sum()),'mass_kg':float(mass.sum()),
 'energy_J':float(e),'residual':r.tolist(),'tangent':t.tolist(),
 'energy_directional_derivative':float((ep-em)/(2*eps)),
 'analytic_energy_directional_derivative':float(np.sum(r*direction)),
 'fd_tangent_l2_relative':float(np.linalg.norm((rp-rm)/(2*eps)-t)/max(np.linalg.norm(t),1e-30))}
# Independently defined affine exact integral and uniform active-plane case.
F0=np.array([[1.03,.02,0.],[0.,.98,.01],[0.,0.,1.02]])
xa=ref@F0.T+np.array([0.,0.,-0.000004])
ea,ra,ta=energy_residual_tangent(xa,direction)
j=np.linalg.det(F0);l=np.log(j)
wa=mu*(np.sum(F0*F0)-3)/2-mu*l+lam*l*l/2
expected=wa*.02**3+np.dot(mass,9.81*xa[:,2])
xc=ref+np.array([0.,0.,-.000105])
ec,rc,tc=energy_residual_tangent(xc,np.tile([0.,0.,1.],(len(ref),1)))
g=.000095;q=g*g-d*d;lg=np.log(g*g/d**2)
contact_energy=-k*q*q*lg*.02**2
contact_force=2*g*k*(2*q*lg+q*q/g**2)*.02**2
out['analytic_controls']={
 'affine_energy_absolute_error_J':float(abs(ea-expected)),
 'uniform_contact_energy_absolute_error_J':float(abs(ec-(contact_energy+np.dot(mass,9.81*xc[:,2])))),
 'uniform_contact_total_residual_error_N':float(abs(rc[:,2].sum()-(mass.sum()*9.81-contact_force))),
 'uniform_contact_total_force_N':float(contact_force)}
print(json.dumps(out))
