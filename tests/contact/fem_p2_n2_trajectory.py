import json,sys,numpy as np
from scipy.integrate import solve_ivp
from fem_p2_dynamics import Model
variant=sys.argv[1]
if variant=='coarse':rtol,atol,step=1e-8,1e-12,8e-6
elif variant=='fine':rtol,atol,step=1e-10,1e-13,4e-6
else:raise ValueError('bad variant')
m=Model(2);m12=Model(2,12);n=int(m.b.N);zero=np.zeros((n,3))
def rhs(t,y):
 u=y[:3*n].reshape(n,3);v=y[3*n:6*n].reshape(n,3)
 _,r,cf,_,_,_=m.evaluate(u)
 dy=np.empty_like(y);dy[:3*n]=v.ravel()
 dy[3*n:6*n]=np.column_stack([m.factor.solve(-r[:,i]) for i in range(3)]).ravel()
 dy[-2]=cf.sum();dy[-1]=np.dot(cf,v[:,2]);return dy
y0=np.zeros(6*n+2);ats=np.full_like(y0,atol);ats[3*n:6*n]*=1000
times=np.linspace(0,.01,101)
sol=solve_ivp(rhs,(0,.01),y0,method='DOP853',rtol=rtol,atol=ats,max_step=step,t_eval=times)
if not sol.success:raise RuntimeError(sol.message)
trace=[];states=[]
for j,t in enumerate(times):
 y=sol.y[:,j];u=y[:3*n].reshape(n,3);v=y[3*n:6*n].reshape(n,3)
 energy,r,cf,ce,gap,jac=m.evaluate(u);_,_,cf12,ce12,gap12,jac12=m12.evaluate(u)
 kinetic=sum(float(np.dot(v[:,i],m.M@v[:,i])) for i in range(3))/2
 pz=float(np.dot(m.massload,v[:,2]))
 trace.append([float(t),float(cf.sum()),float(cf12.sum()),gap,gap12,jac,jac12,pz,float(y[-2]),float(y[-1]),ce,energy+kinetic])
 states.append(u.tolist())
imp=sol.y[-2];bins=np.diff(imp[::10])/.001
print(json.dumps({'variant':variant,'method':'DOP853','rtol':rtol,'atol':atol,'max_step_s':step,
 'n':2,'dofs':n,'evaluations':sol.nfev,'force_bins_N':bins.tolist(),'impulse_Ns':float(imp[-1]),
 'contact_work_J':float(sol.y[-1,-1]),
 'trace_columns':['time_s','force8_N','force12_N','gap8_m','gap12_m','minJ8','minJ12','momentum_Ns','impulse_Ns','contact_work_J','contact_energy_J','total_energy_J'],
 'trace':trace,'displacements':states}))
