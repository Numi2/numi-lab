from itertools import product
import numpy as np

def soft(x,f):return np.sign(x)*np.maximum(np.abs(x)-f,0)
def residual(W,free,b,c,cap,friction,tau):return np.clip(soft(b-c*(free+W@tau),friction),-cap,cap)-tau

def solve(W,free,b,c,cap,friction):
    tau=np.zeros_like(b);n=len(b)
    for sweep in range(20000):
        a=free+W@tau
        for i in list(range(n))+list(range(n-1,-1,-1)):
            rhs=b[i]-c[i]*(a[i]-W[i,i]*tau[i])
            val=float(np.clip(soft(rhs,friction[i])/(1+c[i]*W[i,i]),-cap[i],cap[i]))
            a+=W[:,i]*(val-tau[i]);tau[i]=val
        e=float(np.max(np.abs(residual(W,free,b,c,cap,friction,tau))))
        if e<=1e-13:return tau,sweep+1,e
    raise RuntimeError('Independent FP64 reference did not converge: '+str(e))

def enumerate_regions(W,free,b,c,cap,friction):
    n=len(b);best=None
    for regions in product((-2,-1,0,1,2),repeat=n):
        fixed=np.array([r in (-2,0,2) for r in regions]);tau=np.array([np.sign(r)*cap[i] if abs(r)==2 else 0 for i,r in enumerate(regions)],float)
        u=np.flatnonzero(~fixed)
        if len(u):
            rhs=b[u]-np.array([np.sign(regions[i])*friction[i] for i in u])-c[u]*(free[u]+W[np.ix_(u,np.flatnonzero(fixed))]@tau[fixed])
            tau[u]=np.linalg.solve(np.eye(len(u))+c[u,None]*W[np.ix_(u,u)],rhs)
        e=np.max(np.abs(residual(W,free,b,c,cap,friction,tau)))
        if best is None or e<best[0]:best=(e,tau)
    if best[0]>1e-11:raise RuntimeError('Region enumeration invalid: '+str(best[0]))
    return best[1]

def calibrate():
    cases=0;max_error=0;max_residual=0
    for coupling in (0,.6):
      M=np.array([[1,coupling],[coupling,2]],float);W=np.linalg.inv(M)
      for stiffness in (.01,10,1000):
       for z in range(8):
        free=np.array([(-1)**z*.4,(-1)**(z//2)*.15]);b=np.array([(z-3)*.7,(3-z)*.2]);c=np.array([stiffness,stiffness*.7]);cap=np.array([.1,.3]);friction=np.array([.01 if z&1 else 0,.02 if z&2 else 0])
        tau,sweeps,e=solve(W,free,b,c,cap,friction);reference=enumerate_regions(W,free,b,c,cap,friction)
        max_error=max(max_error,float(np.max(np.abs(tau-reference))));max_residual=max(max_residual,e);cases+=1
    assert max_error<=1e-11 and max_residual<=1e-13
    return dict(calibration='passed',cases=cases,maximum_torque_error_Nm=max_error,maximum_absolute_fixed_point_residual_Nm=max_residual)

if __name__=='__main__':
 import json
 print(json.dumps(calibrate(),sort_keys=True))
