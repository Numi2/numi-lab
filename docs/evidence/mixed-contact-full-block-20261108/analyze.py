import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
control=json.loads((root/"control.json").read_text())
prior=json.loads((root.parent/"mixed-contact-sweep-history-20261108/control.json").read_text())
def exposed(x):
 if isinstance(x,dict): return {k:exposed(v) for k,v in x.items() if k not in ("row_trace","scratch")}
 if isinstance(x,list): return [exposed(v) for v in x]
 return x
assert exposed(control)==exposed(prior)
assert control["success"]==12 and control["failed"]==8 and control["steps"][12]["code"]==11
W=np.empty((9,9),dtype=np.float64)
for axis in range(3):
 data=json.loads((root/("axis%d.json"%axis)).read_text())
 assert exposed(data)==exposed(control) and all(data[k] for k in ("repeat","capture_noninvasive","exact_trace"))
 rows=data["steps"][12]["rows"]
 assert len(rows)==3
 for target,row in enumerate(rows):
  z=row["scratch"]; ids=[int(x) for x in z[9:12]]
  assert sorted(ids)==[0,1,2] and ids==z[9:12]
  for slot,src in enumerate(ids): W[3*target+axis,3*src:3*src+3]=z[3*slot:3*slot+3]
assert np.isfinite(W).all()
eig=np.linalg.eigvalsh((W+W.T)/2)
asym=np.max(np.abs(W-W.T))/np.max(np.abs(W))
assert asym<=1e-4 and eig[0]>=-1e-5*np.max(np.abs(eig))
rows=json.loads((root.parent/"mixed-contact-sweep-history-20261108/sweep-128.json").read_text())["steps"][12]["rows"]
j0=np.array([x for row in rows for x in row["scratch"][:3]],dtype=np.float64)
r0=np.array([x for row in rows for x in row["scratch"][4:7]],dtype=np.float64)
st=np.array([row["scratch"][7] for row in rows],dtype=np.float64)
sn=1/np.diag(W)[::3]
assert (st>0).all() and (sn>0).all()
b=r0+W@j0
jscale=np.array([.01,.01,.01,1e-5,1e-5,1e-5,1e-5,1e-5,1e-5])
vscale=np.maximum(np.abs(np.diag(W)*jscale),.001)
def phys(j):
 rhs=b-W@j
 g=np.empty(9)
 for i in range(3):
  k=3*i
  g[k]=(j[k]-max(0.,j[k]+sn[i]*rhs[k]))/sn[i]
  u=j[k+1:k+3]+st[i]*rhs[k+1:k+3]
  radius=.5*max(0.,j[k]);norm=np.linalg.norm(u)
  proj=u if norm<=radius else u*(radius/norm) if norm>0 else np.zeros(2)
  g[k+1:k+3]=(j[k+1:k+3]-proj)/st[i]
 return g
def f(y): return phys(y*jscale)/vscale
def solve(seed):
 y=seed/jscale
 history=[]
 for it in range(100):
  f0=f(y); err=np.linalg.norm(f0); raw=np.max(np.abs(phys(y*jscale)))
  history.append(float(raw))
  if raw<1e-9: break
  J=np.empty((9,9));h=1e-6
  for c in range(9):
   d=np.zeros(9);d[c]=h
   J[:,c]=(f(y+d)-f(y-d))/(2*h)
  delta=np.linalg.lstsq(J,-f0,rcond=1e-12)[0]
  accepted=False
  for k in range(26):
   candidate=y+delta*(.5**k)
   if np.linalg.norm(f(candidate)) < err*(1-1e-5*.5**k):
    y=candidate;accepted=True;break
  if not accepted: break
 return y*jscale,history
def certificate(j):
 rhs=b-W@j
 normal=[];tangent=[];disk=[]
 for i in range(3):
  k=3*i; n=j[k]
  normal.append(abs(rhs[k]) if n>1e-12 else max(rhs[k],0.))
  disk.append(max(float(np.linalg.norm(j[k+1:k+3])-.5*n),0.))
  tangent.append(float(np.linalg.norm(phys(j)[k+1:k+3])))
 finite=bool(np.isfinite(j).all())
 ok=finite and np.min(j[::3])>=-1e-12 and max(disk)<=1e-12 and max(normal+tangent)<=1e-6
 return {"ok":bool(ok),"finite":finite,"normal_impulses":j[::3].tolist(),"normal_residual_m_s":normal,"tangent_residual_m_s":tangent,"disk_excess_Ns":disk,"max_projected_residual_m_s":float(np.max(np.abs(phys(j))))}
initial=certificate(j0)
previous=json.loads((root.parent/"mixed-pair-tangent-diagnostic-20261108/analysis.json").read_text())
assert abs(max(initial["tangent_residual_m_s"])-previous["cpu_max_tangent_m_s"])<2e-6
solves=[]
for name,seed in (("captured",j0),("half",.5*j0),("zero",np.zeros(9))):
 j,history=solve(seed)
 negative=j.copy();negative[3]+=1e-4
 result={"seed":name,"iterations":len(history)-1,"history_m_s":history,"solution":j.tolist(),"certificate":certificate(j),"perturbed_certificate":certificate(negative)}
 assert not result["perturbed_certificate"]["ok"]
 solves.append(result)
out={"non_scratch_traces_match":True,"asymmetry_relative":float(asym),"symmetric_eigenvalues":eig.tolist(),"matrix":W.tolist(),"affine_rhs":b.tolist(),"initial":initial,"solves":solves}
(root/"analysis.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
for s in solves:
 print(s["seed"],"iterations",s["iterations"],"certificate",s["certificate"],"perturb rejected",not s["perturbed_certificate"]["ok"])
