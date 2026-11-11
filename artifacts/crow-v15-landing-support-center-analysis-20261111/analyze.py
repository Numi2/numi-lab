from pathlib import Path
import collections,gzip,json,math,os,subprocess
R=Path.cwd();O=R/"artifacts/crow-v15-landing-support-center-analysis-20261111"
trials={"reference":("crow-v15-walk-hold-localized-route-20261111","crow-v15-full-landing-failure-analysis-20261111"),"damping":("crow-v15-descent-damping-screen-20261111","crow-v15-descent-damping-screen-20261111")}
results={};check={}
for label,(run,analysis) in trials.items():
 p=R/"artifacts"/run/"autonomous-20261126"; a=json.loads((R/"artifacts"/analysis/"analysis.json").read_text());rows=a["rows"]; assert not a["errors"] and not a["overflows"] and all(z["missing_substeps"]==0 for z in rows)
 start={z["lane"]:next((t for t,v in z["phase_transitions"] if v>=1.5),None) for z in rows};end={z["lane"]:z["end"] for z in rows};qs={};contacts=collections.defaultdict(list)
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    z=json.loads(line.split(" ",1)[1]);e,t=z["lane"],z["step"]
    if start[e] is not None and start[e]<=t<=end[e]:qs[e,t]=z
   elif line.startswith("CROW_SCENE_CONTACT "):
    z=line.split();t,k,e,ba,bb=map(int,z[1:6])
    if start[e] is None or not(start[e]<=t<=end[e]) or 40 not in (ba,bb):continue
    body=bb if ba==40 else ba
    if body not in (11,15) and not 19<=body<=34:continue
    sgn=1 if ba==40 else -1;imp=list(map(float,z[6:9]));point=list(map(float,z[9:12]));n=list(map(float,z[13:16]));u=list(map(float,z[16:19]));v=[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]]
    force=[sgn*(imp[0]*n[j]+imp[1]*u[j]+imp[2]*v[j]) for j in range(3)];contacts[e,t].append((point,force))
 inp="".join(str(e)+" "+str(t)+" "+" ".join(map(str,z["q"]+z["v"]))+"\n" for (e,t),z in sorted(qs.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")};z=subprocess.run([str(R/"build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics")],input=inp,text=True,capture_output=True,check=True,env=env)
 centers={};bodies={}
 for line in z.stdout.splitlines():
  t=line.split();key=(int(t[1]),int(t[2]))
  if t[0]=="com":centers[key]=[float(x) for x in t[4:7]]
  elif t[0]=="body" and key[0]==5:bodies[key,t[3]]=[float(x) for x in t[4:7]]
 assert len(centers)==len(qs)
 if label=="reference":
  with gzip.open(p/"replay.json.gz","rt") as f:rep=json.load(f)["payload"]
  assert rep["environment"]==5
  bystep={v["step"]:v for v in rep["frames"]};names=rep["body_names"];errs=[]
  for t in (3729,3750,3800):
   frame=bystep[t-1]
   for name in ("crow_head","crow_left_foot","crow_right_foot"):
    i=names.index(name)*13;errs.append(math.dist(bodies[(5,t),name],frame["body_states"][i:i+3]))
  check={"lane":5,"steps":[3729,3750,3800],"max_position_error_m":max(errs),"body_names":["crow_head","crow_left_foot","crow_right_foot"]};assert max(errs)<1e-5
 series=collections.defaultdict(list)
 for key,z in sorted(qs.items()):
  e,t=key;center=centers[key];cs=contacts[key];up=sum(max(f[2],0.) for p,f in cs)
  if up<.5*.45006*9.81*.02:continue
  sx=sum(max(f[2],0.)*p[0] for p,f in cs)/up;sy=sum(max(f[2],0.)*p[1] for p,f in cs)/up
  torque=sum((p[2]-center[2])*f[0]-(p[0]-center[0])*f[2] for p,f in cs)
  q=z["q"];pitch=math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3]))))
  series[e].append({"step":t,"pitch":pitch,"pitch_rate":z["v"][4],"upward_positive_impulse":up,"support_minus_com_x":sx-center[0],"support_minus_com_y":sy-center[1],"contact_pitch_angular_impulse_approx":torque})
 results[label]={"source":str(p),"rows":{str(e):v for e,v in series.items()}}
 for e,v in series.items():
  x=v[-min(40,len(v)):];print(label,e,"samples",len(v),"last",v[-1]["step"],"pitch range",round(min(w["pitch"] for w in x),3),round(max(w["pitch"] for w in x),3),"support-COM x",round(min(w["support_minus_com_x"] for w in x),4),round(max(w["support_minus_com_x"] for w in x),4),"torque",round(min(w["contact_pitch_angular_impulse_approx"] for w in x),5),round(max(w["contact_pitch_angular_impulse_approx"] for w in x),5))
(O/"analysis.json").write_text(json.dumps({"results":results,"replay_body_position_check":check,"limitations":"Full substep impulse vectors, positive vertical weights; end-of-control-interval COM, not matched substep COM. Sampled first episodes, development layouts. Net angular impulse omits aerodynamic, gravity, and internal dynamics. No causal or biological claim."},indent=2)+"\n")
