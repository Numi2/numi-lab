import json
from pathlib import Path
import numpy as np
p=Path(__file__).parent
items=[json.loads(s) for s in (p/"results-v3.jsonl").read_text().splitlines()];assert len(items)==144
out=[]
for a,b in zip(items[::2],items[1::2]):
 assert a["repeat"]==0 and b["repeat"]==1 and a["exit"]==b["exit"]==0 and not a["stderr"] and not b["stderr"] and a["stdout"]==b["stdout"]
 d=json.loads(a["stdout"]);n=d["n"];h=d["h"];mass=np.array(d["mass"]);L=np.array(d["lengths"]);k=np.array(d["ea"])/L
 x0=np.array(d["x0"]);v0=np.array(d["v0"]);pin=np.array(d["pinned"],dtype=bool);target=np.array(d["targets"])
 # Assemble gradient of implicit energy, independent of GPU tridiagonal implementation.
 A=np.diag(mass/(h*h));rhs=mass*(x0+h*v0)/(h*h)
 for i,ki in enumerate(k):
  A[i,i]+=ki;A[i+1,i+1]+=ki;A[i,i+1]-=ki;A[i+1,i]-=ki
  rhs[i]-=ki*L[i];rhs[i+1]+=ki*L[i]
 x=target.copy();free=np.flatnonzero(~pin);fixed=np.flatnonzero(pin)
 x[free]=np.linalg.solve(A[np.ix_(free,free)],rhs[free]-A[np.ix_(free,fixed)]@target[fixed])
 v=(x-x0)/h;force=[]
 for i in fixed:
  fi=0.
  if i>0: fi-=k[i-1]*(x[i]-x[i-1]-L[i-1])
  if i<n-1:fi+=k[i]*(x[i+1]-x[i]-L[i])
  force.append(fi+mass[i]*(v0[i]-v[i])/h)
 force=np.array(force)
 # Independent global momentum balance is separate from local equation checks.
 assert abs(float(np.sum(force)+np.sum(mass*(v-v0)/h))) < 1e-8
 assert np.linalg.norm((A@x-rhs)[free],ord=np.inf) < 1e-8
 axis=np.array(d["axis_v"]);origin=np.array(d["origin"])
 entry={"arm":a["arm"],"n":n,"axis":d["axis"],"offset":d["offset"],"kind":d["kind"],"accepted":d["accepted"],"gpu":d["gpu"]}
 if d["accepted"]:
  pos=np.array(d["positions"]);vel=np.array(d["velocities"]);fs=np.array(d["forces"])
  pd=pos-origin-x[:,None]*axis;vd=vel-v[:,None]*axis;fd=fs-force[:,None]*axis
  pe=np.linalg.norm(pd,axis=1);ve=np.linalg.norm(vd,axis=1);fe=np.linalg.norm(fd,axis=1)
  we=np.abs(np.sum(fs*(target[fixed]-x0[fixed])[:,None]*axis,axis=1)-force*(target[fixed]-x0[fixed]))
  limits=np.maximum(1e-6,.01*np.abs(force));wlimits=np.maximum(1e-9,.01*np.abs(force*(target[fixed]-x0[fixed])))
  passed=bool(np.isfinite(np.r_[pe,ve,fe,we]).all() and (pe<=1e-6).all() and (ve<=.001).all() and (fe<=limits).all() and (we<=wlimits).all())
  entry.update(passed=passed,max_position_error=float(max(pe)),max_velocity_error=float(max(ve)),max_force_error=float(max(fe)),max_work_error=float(max(we)),reference_force=force.tolist())
 else:entry["passed"]=False
 out.append(entry)
(p/"analysis-v3.json").write_text(json.dumps(out,indent=2)+"\n")
for arm in ("production","vector","guarded"):
 r=[x for x in out if x["arm"]==arm];print(arm,"pass",sum(x["passed"] for x in r),"accepted_failed",sum(x["accepted"] and not x["passed"] for x in r),"rejected",sum(not x["accepted"] for x in r))
 for x in r:
  if not x["passed"]:print(" ",x)
