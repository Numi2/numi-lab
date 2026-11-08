import json,sys,numpy as np
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss
from skfem import FacetBasis,ElementTetP2
from fem_p2_dynamics import Model
nmesh=int(sys.argv[1]);variant=sys.argv[2]
if nmesh not in (8,16):raise ValueError('bad mesh')
if variant=='coarse':rtol,atol,step=1e-8,1e-12,16e-6
elif variant=='fine':rtol,atol,step=1e-10,1e-13,8e-6
else:raise ValueError('bad variant')
m=Model(nmesh,18);n=int(m.b.N);t0=.0045;t1=.01
facets=m.mesh.facets_satisfying(lambda x:np.abs(x[2]-m.z0)<1e-12,boundaries_only=True)
def duffy(n):
 x,w=leggauss(n);x=(x+1)/2;w=w/2
 u,v=np.meshgrid(x,x,indexing='ij');ww=w[:,None]*w[None,:]*(1-u)
 return np.array([u.ravel(),((1-u)*v).ravel()]),ww.ravel()
reference={i:FacetBasis(m.mesh,ElementTetP2(),facets=facets,quadrature=duffy(i)) for i in (48,64)}
def surface(f,u):
 g=f.global_coordinates()[2]+f.interpolate(u[:,2]).value
 if np.any(g<=0):raise ValueError('nonpositive Duffy quadrature gap')
 _,p=m.plane(g,m.d,m.k);return float(np.sum(p*f.dx)),float(np.min(g))
def rhs(t,y):
 u=y[:3*n].reshape(n,3);v=y[3*n:6*n].reshape(n,3)
 _,r,cf,_,_,_=m.evaluate(u)
 dy=np.empty_like(y);dy[:3*n]=v.ravel()
 dy[3*n:6*n]=np.column_stack([m.factor.solve(-r[:,i]) for i in range(3)]).ravel()
 dy[-2]=cf.sum();dy[-1]=np.dot(cf,v[:,2]);return dy
y0=np.zeros(6*n+2);y0[:3*n].reshape(n,3)[:,2]=-.5*9.81*t0*t0
y0[3*n:6*n].reshape(n,3)[:,2]=-9.81*t0
ats=np.full_like(y0,atol);ats[3*n:6*n]*=1000
times=np.linspace(t0,t1,56)
sol=solve_ivp(rhs,(t0,t1),y0,method='DOP853',rtol=rtol,atol=ats,max_step=step,t_eval=times)
if not sol.success:raise RuntimeError(sol.message)
trace=[]
for j,t in enumerate(times):
 y=sol.y[:,j];u=y[:3*n].reshape(n,3);v=y[3*n:6*n].reshape(n,3)
 energy,r,cf,ce,gap,jac=m.evaluate(u);f48,g48=surface(reference[48],u);f64,g64=surface(reference[64],u)
 acc=np.column_stack([m.factor.solve(-r[:,i]) for i in range(3)])
 local_res=m.M@acc+r
 local_relative=float(np.max(np.abs(local_res))/max(np.max(np.abs(r)),np.max(np.abs(m.M@acc)),1e-30))
 kinetic=sum(float(np.dot(v[:,i],m.M@v[:,i])) for i in range(3))/2
 pz=float(np.dot(m.massload,v[:,2]))
 trace.append([float(t),float(cf.sum()),f48,f64,gap,g48,g64,jac,pz,float(y[-2]),float(y[-1]),ce,energy+kinetic,local_relative])
print(json.dumps({'n':nmesh,'variant':variant,'dofs':n,'rtol':rtol,'atol':atol,'max_step_s':step,
 'evaluations':sol.nfev,'bin4_N':float(sol.y[-2,5]/.001),
 'force_bins_N':[0.,0.,0.,0.,float(sol.y[-2,5]/.001)]+(np.diff(sol.y[-2,5::10])/.001).tolist(),
 'impulse_Ns':float(sol.y[-2,-1]),'contact_work_J':float(sol.y[-1,-1]),
 'trace_columns':['time_s','force18_N','force48_N','force64_N','gap18_m','gap48_m','gap64_m','minJ','momentum_Ns','impulse_Ns','contact_work_J','contact_energy_J','total_energy_J','local_mass_solve_relative'],
 'trace':trace}))
