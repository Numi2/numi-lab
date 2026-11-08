"""Full-versus-quarter reflected FEM P2 episode; see DYNAMICS-PLAN.md."""
import json,sys,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu
from symmetry import build,require_symmetric

N=8;t0=.0045;t1=.01
q,f=build(N)
def simulate(model,quarter):
 n=int(model.b.N);free=[];factors=[]
 for axis in range(3):
  ids=np.flatnonzero(np.abs(model.ref[:,axis])>1e-13) if quarter and axis<2 else np.arange(n)
  free.append(ids);factors.append(splu(model.M[ids,:][:,ids].tocsc()))
 scale=4. if quarter else 1.
 def acceleration(r):
  a=np.zeros_like(r)
  for axis in range(3):a[free[axis],axis]=factors[axis].solve(-r[free[axis],axis])
  return a
 def rhs(t,y):
  u=y[:3*n].reshape(n,3);v=y[3*n:6*n].reshape(n,3)
  _,r,cf,_,_,_=model.evaluate(u)
  dy=np.empty_like(y);dy[:3*n]=v.ravel();dy[3*n:6*n]=acceleration(r).ravel()
  dy[-2]=scale*cf.sum();dy[-1]=scale*np.dot(cf,v[:,2]);return dy
 y0=np.zeros(6*n+2);y0[:3*n].reshape(n,3)[:,2]=-.5*9.81*t0*t0
 y0[3*n:6*n].reshape(n,3)[:,2]=-9.81*t0
 if not quarter:require_symmetric(model,y0[:3*n].reshape(n,3))
 ats=np.full_like(y0,1e-12);ats[3*n:6*n]*=1000
 times=np.linspace(t0,t1,56);started=time.monotonic()
 sol=solve_ivp(rhs,(t0,t1),y0,method='DOP853',rtol=1e-8,atol=ats,max_step=16e-6,t_eval=times)
 if not sol.success:raise RuntimeError(sol.message)
 trace=[]
 for j,t in enumerate(times):
  y=sol.y[:,j];u=y[:3*n].reshape(n,3);v=y[3*n:6*n].reshape(n,3)
  energy,r,cf,ce,gap,jac=model.evaluate(u)
  pz=scale*float(np.dot(model.massload,v[:,2]))
  trace.append([float(t),scale*float(cf.sum()),float(gap),float(jac),pz,float(y[-2]),float(y[-1])])
 impulses=sol.y[-2];bins=[float(impulses[5]/.001)]+(np.diff(impulses[5::10])/.001).tolist()
 return {'dofs':n,'nfev':sol.nfev,'elapsed_s':time.monotonic()-started,'force_bins_N':bins,'trace':trace}
full=simulate(f,False);quarter=simulate(q,True)
a=np.array(full['trace']);b=np.array(quarter['trace']);peak=max(abs(a[:,1]));fullimp=a[-1,5]
metrics={'binned_force_error_relative_peak':float(np.max(abs(np.array(full['force_bins_N'])-np.array(quarter['force_bins_N'])))/peak),
 'impulse_error_relative':float(abs(fullimp-b[-1,5])/abs(fullimp)),
 'force_trace_error_relative_peak':float(np.max(abs(a[:,1]-b[:,1]))/peak),
 'momentum_max_abs_error_Ns':float(np.max(abs(a[:,4]-b[:,4]))),
 'work_abs_error_J':float(abs(a[-1,6]-b[-1,6])),
 'full_min_gap_m':float(np.min(a[:,2])),'quarter_min_gap_m':float(np.min(b[:,2])),
 'full_min_jacobian':float(np.min(a[:,3])),'quarter_min_jacobian':float(np.min(b[:,3]))}
passed=metrics['binned_force_error_relative_peak']<=.0025 and metrics['impulse_error_relative']<=.0025 and min(metrics['full_min_gap_m'],metrics['quarter_min_gap_m'],metrics['full_min_jacobian'],metrics['quarter_min_jacobian'])>0
print(json.dumps({'status':'pass' if passed else 'fail','metrics':metrics,'full':full,'quarter':quarter},indent=2),flush=True)
if not passed:sys.exit(1)
