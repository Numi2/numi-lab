from pathlib import Path
import gzip,json,math,sys
A=Path(sys.argv[1]);name=sys.argv[2];xs=[];status=[]
with gzip.open(A/(name+".stderr.gz"),"rt") as f:
 for l in f:
  if l.startswith("CROW_GAIT "):
   x=json.loads(l.split(" ",1)[1])
   if x["lane"]==0 and 100<=x["step"]<=600:xs.append(x)
  elif l.startswith("CROW_TASK_STATE "):
   v=l.split();status.append((int(v[2]),int(v[3])))
def pct(a):
 a=sorted(a);return a[math.ceil(.95*len(a))-1] if a else None
def count(a):
 n=out=0
 for v in a+[False]:
  if v:n+=1
  else:
   out+=n>=3;n=0
 return out
valid=[x["step"] for x in xs]==list(range(100,601));sides=[]
for side in (0,1):
 stance=[x["sides"][side]["mean_slip_m_s"] for x in xs if x["sides"][side]["impulse_Ns"]>.001]
 swing=[x["sides"][side]["impulse_Ns"]<=.001 and x["sides"][1-side]["impulse_Ns"]>.001 and x["sides"][side]["clearance_m"]>=.002 for x in xs]
 sides.append({"slip_p95_m_s":pct(stance),"clear_swing_runs":count(swing)})
r={"valid":valid,"forward_m":xs[-1]["root"][0]-xs[0]["root"][0] if xs else None,"height_range_m":[min(x["root"][2] for x in xs),max(x["root"][2] for x in xs)] if xs else None,"max_tilt_rad":max((math.acos(max(-1,min(1,1-2*(x["quat"][0]**2+x["quat"][1]**2)))) for x in xs),default=None),"nonfoot_endpoints":sum(x["nonfoot_impulse"]>1e-7 for x in xs),"unsupported_endpoints":sum(all(y["impulse_Ns"]<=.001 for y in x["sides"]) for x in xs),"sides":sides,"early_latch_step":next((t for t,p in status if t<670 and p&4096),None)}
(A/"gait-analysis.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
