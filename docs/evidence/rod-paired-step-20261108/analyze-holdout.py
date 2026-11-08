import json,numpy as np
from pathlib import Path
p=Path(__file__).parent
items=[json.loads(s) for s in (p/"holdout-corrected.jsonl").read_text().splitlines()];assert len(items)==18
priorp=Path("/Users/numi/numi-lab-tracker-step-rollback-20261108/docs/evidence/rod-local-momentum-20261108/raw-v3.jsonl")
prior={}
for line in priorp.read_text().splitlines():
 a=json.loads(line)
 if a["arm"]=="guarded" and a["repeat"]==0:
  d=json.loads(a["stdout"]);prior[(d["n"],d["axis"],str(d["offset"]),d["kind"])]=d
out=[]
for a,b in zip(items[::2],items[1::2]):
 assert a["repeat"]==0 and b["repeat"]==1 and a["exit"]==b["exit"]==0 and not a["stderr"] and not b["stderr"]
 d=json.loads(a["stdout"]); repeat_equal=(a["stdout"]==b["stdout"])
 n=d["n"];h=d["h"];mass=np.array(d["mass"]);L=np.array(d["lengths"]);k=np.array(d["ea"])/L
 x0=np.array(d["x0"]);v0=np.array(d["v0"]);pin=np.array(d["pinned"],dtype=bool);target=np.array(d["targets"])
 A=np.diag(mass/(h*h));rhs=mass*(x0+h*v0)/(h*h)
 for i,ki in enumerate(k):
  A[i,i]+=ki;A[i+1,i+1]+=ki;A[i,i+1]-=ki;A[i+1,i]-=ki;rhs[i]-=ki*L[i];rhs[i+1]+=ki*L[i]
 x=target.copy();free=np.flatnonzero(~pin);fixed=np.flatnonzero(pin)
 x[free]=np.linalg.solve(A[np.ix_(free,free)],rhs[free]-A[np.ix_(free,fixed)]@target[fixed])
 v=(x-x0)/h;force=[]
 for i in fixed:
  fi=0.
  if i>0:fi-=k[i-1]*(x[i]-x[i-1]-L[i-1])
  if i<n-1:fi+=k[i]*(x[i+1]-x[i]-L[i])
  force.append(fi+mass[i]*(v0[i]-v[i])/h)
 force=np.array(force)
 assert abs(float(np.sum(force)+np.sum(mass*(v-v0)/h)))<1e-8
 assert np.linalg.norm((A@x-rhs)[free],ord=np.inf)<1e-8
 axis=np.array(d["axis_v"]);origin=np.array(d["origin"])
 entry={key:d[key] for key in ("n","axis","offset","kind","accepted","gpu")};entry["repeat_equal"]=repeat_equal
 entry["same_as_prior_guarded"]=False
 if d["accepted"]:
  pos=np.array(d["positions"]);vel=np.array(d["velocities"]);fs=np.array(d["forces"])
  pe=np.linalg.norm(pos-origin-x[:,None]*axis,axis=1);ve=np.linalg.norm(vel-v[:,None]*axis,axis=1);fe=np.linalg.norm(fs-force[:,None]*axis,axis=1)
  we=np.abs(np.sum(fs*(target[fixed]-x0[fixed])[:,None]*axis,axis=1)-force*(target[fixed]-x0[fixed]))
  local=np.linalg.norm(np.array(d["localres5"])[free],axis=1)
  disagreement=np.linalg.norm((np.array(d["localres5"])-np.array(d["localres10"]))[free],axis=1)
  vals=np.r_[pe,ve,fe,we,local,disagreement]
  passed=bool(np.isfinite(vals).all() and (pe<=1e-6).all() and (ve<=.001).all() and (fe<=np.maximum(1e-6,.01*np.abs(force))).all() and (we<=1e-8).all() and (local<=.001).all() and (disagreement<=1e-4).all())
  entry.update(passed=passed,max_position_error_m=float(max(pe)),max_velocity_error_m_s=float(max(ve)),max_force_error_N=float(max(fe)),max_work_error_J=float(max(we)),max_local_residual_m_s=float(max(local)),max_gradient_disagreement_m_s=float(max(disagreement)))
 else:entry["passed"]=False
 out.append(entry)
summary={}
for axis in ("x","oblique3"):
 z=[x for x in out if x["axis"]==axis]
 summary[axis]={"cases":len(z),"accepted":sum(x["accepted"] for x in z),"passed_all_gates":sum(x["passed"] for x in z),"rejected":sum(not x["accepted"] for x in z),"repeat_equal":sum(x["repeat_equal"] for x in z),"identical_prior":sum(x["same_as_prior_guarded"] for x in z)}
 for key in ("max_position_error_m","max_velocity_error_m_s","max_force_error_N","max_work_error_J","max_local_residual_m_s","max_gradient_disagreement_m_s"):
  summary[axis][key]=max((x[key] for x in z if key in x),default=None)
(p/"analysis-holdout.json").write_text(json.dumps({"summary":summary,"cases":out},indent=2)+"\n")
print(json.dumps(summary,indent=2));print("Failures:",json.dumps([x for x in out if not x["passed"]],indent=2))
