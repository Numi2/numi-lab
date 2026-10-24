"""FP64 whole-cylinder support bound on minimum separating translation.

For any unit direction n, max(0, hA(n)+hB(-n)-abs(dot(cA-cB,n)))
is an upper bound on the smallest translation needed to separate the pair.
Taking the minimum over candidate directions improves that bound. This does
not require a sampled surface, nor does it assert those axes are a complete
cylinder SAT. A small positive bound is also possible for separated bodies.
"""
from pathlib import Path
import json,sys,itertools
import numpy as np
d=json.loads(Path(sys.argv[1]).read_text());fs=d['frames'];R=.015305;h=.001075
p=np.array([f['coins'] for f in fs]);q=np.array([f['coin_orientations'] for f in fs]);q/=np.linalg.norm(q,axis=-1,keepdims=True)
axis=np.stack([2*(q[...,0]*q[...,1]-q[...,3]*q[...,2]),1-2*(q[...,0]**2+q[...,2]**2),2*(q[...,1]*q[...,2]+q[...,3]*q[...,0])],axis=-1)
axis/=np.linalg.norm(axis,axis=-1,keepdims=True)
pairs=np.array(list(itertools.combinations(range(p.shape[1]),2)));ia,ib=pairs.T
a=axis[:,ia];b=axis[:,ib];delta=p[:,ia]-p[:,ib]
axes=np.stack([a,b,delta,np.cross(a,b),delta-np.sum(delta*a,axis=-1,keepdims=True)*a,delta-np.sum(delta*b,axis=-1,keepdims=True)*b,np.cross(a,delta),np.cross(b,delta)],axis=-2)
length=np.linalg.norm(axes,axis=-1);valid=length>1e-12;n=axes/np.maximum(length[...,None],1e-300)
support=R*(np.linalg.norm(np.cross(a[...,None,:],n),axis=-1)+np.linalg.norm(np.cross(b[...,None,:],n),axis=-1))+h*(abs(np.sum(a[...,None,:]*n,axis=-1))+abs(np.sum(b[...,None,:]*n,axis=-1)))
overlap=support-abs(np.sum(delta[...,None,:]*n,axis=-1));overlap[~valid]=np.inf
upper=np.maximum(0,overlap.min(axis=-1))
# Refine large bounds by searching additional directions. Each returned
# direction is independently re-evaluated with the analytic support function;
# the guarantee does not depend on optimizer success or finding a global min.
from scipy.optimize import minimize
refined=0
for fi,pi in np.argwhere(upper>1.e-5):
 aa=a[fi,pi];bb=b[fi,pi];dd=delta[fi,pi]
 def objective(x):
  nn=x/max(np.linalg.norm(x),1.e-30)
  return R*(np.linalg.norm(np.cross(aa,nn))+np.linalg.norm(np.cross(bb,nn)))+h*(abs(aa@nn)+abs(bb@nn))-abs(dd@nn)
 order=np.argsort(overlap[fi,pi])
 best=float(upper[fi,pi])
 for j in order[:4]:
  z=minimize(objective,n[fi,pi,j],method='SLSQP',constraints=[{'type':'eq','fun':lambda x:x@x-1}],options={'maxiter':120,'ftol':1.e-13})
  best=min(best,max(0.,float(objective(z.x))))
  if best<1.e-6:break
 upper[fi,pi]=best;refined+=1
fi,pi=np.unravel_index(upper.argmax(),upper.shape)
worst=float(upper[fi,pi]);tolerance=1e-4
result={'states':len(fs),'through_s':fs[-1]['time_s'],'pairs_per_state':len(pairs),'queries':int(upper.size),'maximum_minimum_separating_translation_upper_bound_m':worst,'upper_bound_mm':worst*1000,'worst_event':{'step':fs[fi]['step'],'time_s':fs[fi]['time_s'],'pair':pairs[pi].tolist()},'tolerance_m':tolerance,'within_tolerance':bool(worst<=tolerance),'refined_pair_state_bounds':refined,'method':'FP64 analytic finite-cylinder support functions with additional numerically searched directions. Every searched direction is re-evaluated analytically, so optimization failure cannot create a false clearance claim. Every pair and saved state. Candidate support directions bound minimum separating translation from above; small positive values do not prove intersection. Substep tunneling and physical coin rim geometry are outside this audit.'}
print(json.dumps(result,indent=2))
