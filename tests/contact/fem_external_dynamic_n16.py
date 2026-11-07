"""Integrate the independently assembled exact P1 contact model with DOP853."""
import contextlib,io,json,runpy,sys
import numpy as np
from scipy.integrate import solve_ivp
probe,native,variant=sys.argv[1:]
sys.argv=[probe,native,'150','/tmp/numi-external-dynamic-probe-state.txt']
with contextlib.redirect_stdout(io.StringIO()):
 p=runpy.run_path(probe)
ref,mass,area=p['ref'],p['mass'],p['area']
equations=p['energy_residual_tangent'];zero=np.zeros_like(ref)
n=len(ref)
def rhs(t,y):
 u=y[:3*n].reshape(n,3);v=y[3*n:6*n].reshape(n,3);pos=ref+u
 _,r,_=equations(pos,zero,compute_tangent=False)
 g=pos[:,2];mask=(area>0)&(g<p['d'])
 if np.any(g[mask]<=0):raise ValueError('nonpositive gap')
 q=g[mask]**2-p['d']**2
 force=2*g[mask]*p['k']*(2*q*np.log(g[mask]**2/p['d']**2)+q*q/g[mask]**2)*area[mask]
 dy=np.zeros_like(y)
 dy[:3*n]=v.ravel();dy[3*n:6*n]=(-r/mass[:,None]).ravel()
 dy[-2]=force.sum();dy[-1]=np.dot(force,v[mask,2])
 return dy
y0=np.zeros(6*n+2)
if variant=='coarse':rtol,atol,step=1e-8,1e-12,16e-6
elif variant=='fine':rtol,atol,step=1e-10,1e-13,8e-6
else:raise ValueError('invalid variant')
ats=np.full_like(y0,atol);ats[3*n:6*n]*=1000
t=np.linspace(0,.01,101)
sol=solve_ivp(rhs,(0,.01),y0,method='DOP853',rtol=rtol,atol=ats,max_step=step,t_eval=t)
if not sol.success:raise RuntimeError(sol.message)
gaps=np.min((ref[None,:,:]+sol.y[:3*n,:].T.reshape(-1,n,3))[:,area>0,2],axis=1)
positions=ref[None,:,:]+sol.y[:3*n,:].T.reshape(-1,n,3)
velocities=sol.y[3*n:6*n,:].T.reshape(-1,n,3)
momentum=np.einsum('i,tij->tj',mass,velocities)[:,2]
imp=sol.y[-2,:];work=sol.y[-1,:]
bins=np.diff(imp[::10])/.001
print(json.dumps({'variant':variant,'rtol':rtol,'position_atol':atol,'velocity_atol':atol*1000,
 'max_step_s':step,'success':sol.success,'function_evaluations':sol.nfev,
 'min_sampled_gap_m':float(gaps.min()),'force_bins_N':bins.tolist(),
 'impulse_Ns':float(imp[-1]),'contact_work_J':float(work[-1]),
 'momentum_error_Ns':float(np.max(np.abs(momentum-(imp-mass.sum()*9.81*t)))),
 'trace_columns':['time_s','gap_m','momentum_Ns','impulse_Ns','contact_work_J'],
 'trace':np.column_stack((t,gaps,momentum,imp,work)).tolist()}))
