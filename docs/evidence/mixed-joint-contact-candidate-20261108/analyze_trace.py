import json,math
from pathlib import Path
root=Path(__file__).resolve().parent
a=json.loads((root/"joint.json").read_text());b=json.loads((root/"trace.json").read_text())
def exposed(x):
 if isinstance(x,dict):return {k:exposed(v) for k,v in x.items() if k not in ("scratch","row_trace")}
 if isinstance(x,list):return [exposed(v) for v in x]
 return x
assert exposed(a)==exposed(b)
assert b["success"]==20 and b["failed"]==0 and all(b[k] for k in ("capture_noninvasive","repeat","exact_trace"))
records=[]
for step,s in enumerate(b["steps"]):
 assert s["code"]==0 and len(s["rows"])==3
 for index,row in enumerate(s["rows"]):
  z=row["scratch"];jn,ju,jv=row["impulse"]
  eq,reported,scale,marker,u,v,mu,mv,vu,vv,ru,rv=z
  assert all(math.isfinite(x) for x in z+row["impulse"])
  assert marker>0 and marker<=s["sweeps"] and jn>=0 and scale>0 and mu>=0 and mv>=0
  lu,lv=mu*jn,mv*jn
  a1,a2=ju+scale*u,jv+scale*v
  if lu>0 and lv>0:
   ratio=math.hypot(a1/lu,a2/lv)
   if ratio>1:a1/=ratio;a2/=ratio
   cone=max(math.hypot(ju/lu,jv/lv)-1,0.)
  else:a1=a2=0.;cone=math.hypot(ju,jv)
  natural=math.hypot(a1-ju,a2-jv)/scale
  normal=abs(eq) if jn>0 else max(-eq,0.)
  assert abs(natural-reported)<2e-6 and natural<=2e-5 and normal<=2e-5 and cone<2e-5
  records.append({"step":step,"row":index,"joint_marker":marker,"normal_residual":normal,"tangent_residual":natural,"cone_excess":cone})
out={"exposed_traces_equal":True,"rows":len(records),"all_rows_marked_joint_update":True,
 "max_normal_m_s":max(x["normal_residual"] for x in records),"max_tangent_m_s":max(x["tangent_residual"] for x in records),
 "max_cone_excess":max(x["cone_excess"] for x in records),"records":records}
(root/"trace-analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print({k:v for k,v in out.items() if k!="records"})
