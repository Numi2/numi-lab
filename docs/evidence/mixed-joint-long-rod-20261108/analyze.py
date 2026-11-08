import ast,json,math
from pathlib import Path
root=Path(__file__).resolve().parent;p=root/"runs"
tree=ast.parse((root.parent/"mixed-sustained-contact-20261108/analyze.py").read_text())
funcs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
ns={"DT":.001,"math":math};exec(compile(funcs,"sustained-analyze","exec"),ns);analyze=ns["analyze"]
def exposed(x):
 if isinstance(x,dict):return {k:exposed(v) for k,v in x.items() if k not in ("scratch","row_trace")}
 if isinstance(x,list):return [exposed(v) for v in x]
 return x
summary={}
for n in (5,9,17):
 name="rod%d"%n
 d={arm:json.loads((p/(name+"-"+arm+".stdout")).read_text()) for arm in ("reference","old","long","trace")}
 ref=analyze(d["reference"]);cand=analyze(d["long"])
 match=exposed(d["long"])==exposed(d["trace"])
 rs,cs=ref["steps"],cand["steps"]
 peak=max(abs(r["ground_normal_impulse_Ns"]/.001) for r in rs)
 force=max(abs(a["ground_normal_impulse_Ns"]-b["ground_normal_impulse_Ns"])/.001 for a,b in zip(rs,cs))
 if peak:force/=peak
 rj=sum(s["ground_normal_impulse_Ns"] for s in rs);j=sum(s["ground_normal_impulse_Ns"] for s in cs)
 imp=abs(j-rj)/abs(rj) if rj else abs(j-rj)
 maxn=maxt=maxcone=0.;bad=[];marked=unmarked=0
 for step,s in enumerate(d["trace"]["steps"]):
  for rowno,row in enumerate(s["rows"]):
   z=row["scratch"];values=z+row["impulse"]
   if not all(math.isfinite(x) for x in values):bad.append([step,rowno,"nonfinite"]);continue
   jn,ju,jv=row["impulse"];eq,reported,scale,marker,u,v,mu,mv,*_=z
   marked+=marker>0;unmarked+=marker<=0
   if scale<=0 or jn<0 or mu<0 or mv<0:bad.append([step,rowno,"invalid input"]);continue
   lu,lv=mu*jn,mv*jn;u1,u2=ju+scale*u,jv+scale*v
   if lu>0 and lv>0:
    ratio=math.hypot(u1/lu,u2/lv)
    if ratio>1:u1/=ratio;u2/=ratio
    cone=max(math.hypot(ju/lu,jv/lv)-1,0.)
   else:u1=u2=0.;cone=math.hypot(ju,jv)
   tang=math.hypot(u1-ju,u2-jv)/scale
   normal=abs(eq) if jn>0 else max(-eq,0.)
   maxn=max(maxn,normal);maxt=max(maxt,tang);maxcone=max(maxcone,cone)
   if abs(tang-reported)>2e-6 or tang>2e-5 or normal>2e-5 or cone>2e-5:bad.append([step,rowno,"residual",normal,tang,cone])
 summary[name]={"reference_accepted":ref["success_count"],"old_accepted":d["old"]["success"],"long_accepted":cand["success_count"],
  "exposed_trace_equal":match,"repeat_capture_checks":all(d[x][k] for x in d for k in ("repeat","exact_trace","capture_noninvasive")),
  "force_error_over_peak_or_absolute_N":force,"force_gate":ref["success_count"]==20 and cand["success_count"]==20 and (force<=.0025 if peak else force<=1e-10),
  "impulse_error_relative_or_absolute_Ns":imp,"impulse_gate":ref["success_count"]==20 and cand["success_count"]==20 and (imp<=.0025 if rj else imp<=1e-12),
  "max_normal_residual":maxn,"max_tangent_residual":maxt,"max_cone_excess":maxcone,"invalid_rows":bad,
  "joint_marked_rows":marked,"fallback_rows":unmarked,"max_momentum_balance_Ns":cand["max_momentum_balance_Ns"],
  "normal_work_estimate_J":cand["total_ground_normal_work_est_J"],"tangent_work_estimate_J":cand["total_ground_tangent_work_est_J"]}
(p/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
