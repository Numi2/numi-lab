from pathlib import Path
import gzip,json,math,sys
A=Path("artifacts/crow-v15-descent-retry-20261110"); rows=[]
def runs(v):
 best=cur=0
 for x in v:
  cur=cur+1 if x else 0;best=max(best,cur)
 return best
for name in sys.argv[1:]:
 d=Path(name)
 for p in sorted(d.glob("diagnostic-*.stdout")):
  if not p.stat().st_size:continue
  x=json.loads(p.read_text());first=x["first_episode_navigation"];end=first["steps_by_environment"][0];samples={}
  with gzip.open(d/(p.stem+".stderr.gz"),"rt") as f:
   for l in f:
    if l.startswith("CROW_SUPPORT_SAMPLE "):
     z=json.loads(l.split(" ",1)[1]);
     if z["lane"]==0 and end-24<=z["step"]<=end:samples[z["step"]]=z
  good=[];grip=[];measure=[]
  for t in range(end-24,end+1):
   z=samples.get(t)
   if z is None:good.append(False);grip.append(False);continue
   q=z["q"];v=z["v"];sup={"left":0.,"right":0.};op={s:{"front":0.,"rear":0.} for s in sup};up=0.;other=0.
   for c in z["contacts"]:
    an,bn=c["a_name"],c["b_name"]
    if "crow_course_perch" not in (an,bn):continue
    body=an if bn=="crow_course_perch" else bn;sgn=-1 if body==an else 1;imp=c["impulses"][0];side=next((s for s in sup if "_"+s+"_" in body),None)
    if side and ("_foot" in body or "_toe" in body):
     sup[side]+=imp;up+=sgn*c["normal"][2]*imp
     nx=-sgn*c["normal"][0]
     if "front_toe" in body and nx<-.5:op[side]["front"]+=imp
     if "rear_toe" in body and nx>.5:op[side]["rear"]+=imp
    else:other+=imp
   tilt=math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(a*a for a in v[:3]));ok=min(sup.values())>.001 and up>=.5*.45004*9.81*.005 and other<=1e-7 and tilt<=.35 and speed<=.12
   g=all(op[s][d]>.00005 for s in op for d in ("front","rear"));good.append(ok);grip.append(g);measure.append({"step":t,"upward_normal_impulse_Ns":up,"support_impulses_Ns":sup,"nonfoot_impulse_Ns":other,"tilt_rad":tilt,"speed_m_s":speed,"opposing_grip":g})
  passed=bool(end==4500 and not first["physical_failure_by_environment"][0] and first["maximum_waypoints_by_environment"][0]>=5 and x["failed_environment_steps"]==0 and all(good) and runs(grip)>=3)
  row={"trial":str(p),"end":end,"terminal_samples":len(samples),"consecutive_supported":runs(good),"consecutive_opposing_grip":runs(grip),"passed":passed,"terminal":measure};rows.append(row);print(p, "end",end,"support run",runs(good),"grip run",runs(grip),"passed",passed)
(A/"results.json").write_text(json.dumps(rows,indent=2)+"\n")
