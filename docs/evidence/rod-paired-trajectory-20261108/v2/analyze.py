import json,math
from pathlib import Path
import numpy as np
p=Path(__file__).parent
manifest=[json.loads(s) for s in (p/"manifest.jsonl").read_text().splitlines()]
assert len(manifest)==12
cases={}; summaries=[]
for m in manifest:
 path=p/(m["case"]+".jsonl"); stderr=(p/(m["case"]+".stderr")).read_text()
 rows=[json.loads(s) for s in path.read_text().splitlines()]
 steps=[r for r in rows if r["type"]=="step"];rejects=[r for r in rows if r["type"]=="rejection"]
 errors=[]
 if m["exit"]!=0 or m["error"] or stderr:errors.append("process_or_stderr")
 if len(rejects)!=1:errors.append("rejection_count")
 else:
  z=rejects[0]
  if not(z["step"]==16 and z["dispatched"] and not z["published"] and not z["succeeded"] and z["gpu"]!=0 and z["input_unchanged"] and z["output_unchanged"]):errors.append("rejection_gate")
 arms={i:sorted((x for x in steps if x["arm"]==i),key=lambda x:x["step"]) for i in (0,1)}
 for arm,aa in arms.items():
  if len(aa)!=32 or [x["step"] for x in aa]!=list(range(32)) or not all(x["accepted"] for x in aa):errors.append(f"incomplete_{arm}")
  if not all(x.get("output_equal",False) for x in aa):errors.append(f"full_output_mismatch_{arm}")
 replay=len(arms[0])==len(arms[1]) and all({k:v for k,v in a.items() if k!="arm"}=={k:v for k,v in b.items() if k!="arm"} for a,b in zip(arms[0],arms[1]))
 if not replay:errors.append("replay_difference")
 metrics={"position_m":0.,"velocity_m_s":0.,"force_N":0.,"work_J":0.,"local_momentum_m_s":0.,"gradient_disagreement_m_s":0.,"reference_free_row_N":0.}
 gate_failures=[]
 for arm,aa in arms.items():
  if not aa:continue
  refx=np.array(aa[0]["x0"],dtype=np.float64);refv=np.array(aa[0]["v0"],dtype=np.float64)
  for d in aa:
   if not d["accepted"]:continue
   n=d["n"];h=d["h"];mass=np.array(d["mass"]);L=np.array(d["lengths"]);k=np.array(d["ea"])/L
   pin=np.array(d["pinned"],dtype=bool);free=np.flatnonzero(~pin);fixed=np.flatnonzero(pin);target=np.array(d["targets"]);targetv=np.array(d["targetv"])
   A=np.diag(mass/(h*h));rhs=mass*(refx+h*refv)/(h*h)
   for i,ki in enumerate(k):
    A[i,i]+=ki;A[i+1,i+1]+=ki;A[i,i+1]-=ki;A[i+1,i]-=ki;rhs[i]-=ki*L[i];rhs[i+1]+=ki*L[i]
   x=target.copy();x[free]=np.linalg.solve(A[np.ix_(free,free)],rhs[free]-A[np.ix_(free,fixed)]@target[fixed]);v=(x-refx)/h;v[fixed]=targetv[fixed]
   force=[]
   for i in fixed:
    fi=0.
    if i>0:fi-=k[i-1]*(x[i]-x[i-1]-L[i-1])
    if i<n-1:fi+=k[i]*(x[i+1]-x[i]-L[i])
    force.append(fi+mass[i]*(refv[i]-v[i])/h)
   force=np.array(force);pos=np.array(d["positions"]);vel=np.array(d["velocities"]);fs=np.array(d["forces"])
   pe=np.linalg.norm(pos-np.column_stack((x,np.full(n,d["offset"]*.5),np.full(n,-d["offset"]*.25))),axis=1)
   ve=np.linalg.norm(vel-np.column_stack((v,np.zeros(n),np.zeros(n))),axis=1)
   fe=np.linalg.norm(fs-np.column_stack((force,np.zeros(len(force)),np.zeros(len(force)))),axis=1)
   we=np.abs(fs[:,0]*(target[fixed]-refx[fixed])-force*(target[fixed]-refx[fixed]))
   local=np.linalg.norm(np.array(d["localres5"])[free],axis=1);diff=np.linalg.norm((np.array(d["localres5"])-np.array(d["localres10"]))[free],axis=1)
   rr=np.max(np.abs((A@x-rhs)[free])); vals={"position_m":max(pe),"velocity_m_s":max(ve),"force_N":max(fe),"work_J":max(we),"local_momentum_m_s":max(local),"gradient_disagreement_m_s":max(diff),"reference_free_row_N":rr}
   for name,value in vals.items():metrics[name]=max(metrics[name],float(value))
   bad=[]
   if not np.isfinite(np.r_[pe,ve,fe,we,local,diff]).all():bad.append("nonfinite")
   if (pe>1e-6).any():bad.append("position")
   if (ve>.001).any():bad.append("velocity")
   if (fe>np.maximum(1e-6,.01*np.abs(force))).any():bad.append("force")
   if (we>1e-8).any():bad.append("work")
   if (local>.001).any():bad.append("local_momentum")
   if (diff>1e-4).any():bad.append("gradient_disagreement")
   if rr>1e-8:bad.append("reference_residual")
   if bad:gate_failures.append({"arm":arm,"step":d["step"],"failed":bad,"metrics":{k:float(v) for k,v in vals.items()}})
   refx=x;refv=v
 summaries.append({"case":m["case"],"accepted_steps":len([x for x in steps if x["accepted"]]),"rejections":rejects,"replay_equal":replay,"errors":errors,"gate_failures":gate_failures,"maxima":metrics})
 key=(steps[0]["offset"],steps[0]["kind"]) if steps else m["case"]
 canonical=[{k:v for k,v in row.items() if k!="run"} for row in rows]
 cases.setdefault(str(key),[]).append(canonical)
repeat={k:len(v)==2 and v[0]==v[1] for k,v in cases.items()}
summary={"executions":len(summaries),"fully_passing":sum(not x["errors"] and not x["gate_failures"] for x in summaries),"repeated_cases_equal":sum(repeat.values()),"case_count":len(repeat),"accepted_steps":sum(x["accepted_steps"] for x in summaries),"maxima":{k:max(x["maxima"][k] for x in summaries) for k in summaries[0]["maxima"]}}
(p/"analysis.json").write_text(json.dumps({"summary":summary,"repeat_equal":repeat,"cases":summaries},indent=2)+"\n")
print(json.dumps(summary,indent=2));print("failures",json.dumps([x for x in summaries if x["errors"] or x["gate_failures"]],indent=2))
