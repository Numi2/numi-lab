from pathlib import Path
import importlib.util,json,math
s=importlib.util.spec_from_file_location("a","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
A=Path("artifacts/crow-v15-delayed-ankle-gait-screen-20261111");old=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text());comp=json.loads((A/"comparison.json").read_text());out=[]
def longest(x):
 best=cur=0
 for b in x:cur=cur+1 if b else 0;best=max(best,cur)
 return best
for seed in (20261126,20261127):
 ref=next(x for x in old["rows"] if x["seed"]==seed);path=A/f"autonomous-{seed}";sup,sub,body,com=a.load(path)
 for x in [x for x in comp["rows"] if x["seed"]==seed and not x["pass_except_phase"]]:
  e=x["lane"]
  for lo,hi in ref["baseline"][e]["support_qualified_swing_bouts"][1][1:]:
   events=[]
   for t in range(max(201,lo-2),min(300,hi+2)+1):
    gaps=[]
    for name in a.names("right"):
     v=body[e,t,name];qx,qy,qz,qw=v[3:7];r=(2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy));h=(.010,.01725,.0092) if name.endswith("_foot") else (.009,.0025,.0092);gaps.append(v[2]-ref["ground_z_m"]-sum(abs(z)*j for z,j in zip(r,h)))
    own=sum(sub[e,t,k][3] for k in range(16));other=sum(sub[e,t,k][2] for k in range(16));non=sum(sub[e,t,k][4] for k in range(16));events.append({"step":t,"all_parts_clear":min(gaps)>.001,"own_unloaded":own<=1e-7,"other_half_weight":other>=a.SUPPORT,"nonfoot_absent":non<=1e-7,"sole_clear_toe_not":gaps[0]>.001 and min(gaps[1:])<=.001})
   out.append({"seed":seed,"lane":e,"baseline_bout":[lo,hi],"candidate_max_clear_and_unloaded_run":longest([z["all_parts_clear"] and z["own_unloaded"] for z in events]),"candidate_max_fully_qualified_run":longest([z["all_parts_clear"] and z["own_unloaded"] and z["other_half_weight"] and z["nonfoot_absent"] for z in events]),"clear_and_unloaded_but_support_below_threshold":sum(z["all_parts_clear"] and z["own_unloaded"] and not z["other_half_weight"] for z in events),"sole_clear_toe_below_threshold":sum(z["sole_clear_toe_not"] for z in events),"events":events})
(A/"failure-diagnosis.json").write_text(json.dumps(out,indent=2)+"\n")
for r in out:print({k:v for k,v in r.items() if k!="events"})
