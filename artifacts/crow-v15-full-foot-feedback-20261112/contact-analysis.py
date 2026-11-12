from pathlib import Path
import collections,gzip,json,math,os,subprocess,importlib.util
A=Path("artifacts/crow-v15-full-foot-feedback-20261112")
P=Path("artifacts/crow-v15-stance-continuity-screen-20261112")
import sys
seed=int(sys.argv[1]);assert seed in (20261126,20261127)
base=Path('artifacts/crow-v15-knee-phase-advance-screen-20261112')/f'candidate-{seed}-rebound-baseline'
paths={'baseline':base,'phase':A/f'candidate-{seed}-rebound-phase'}

def load(path):
 st={};sub={};vcount=collections.Counter();caps=collections.Counter();errors=[]
 result=json.loads((path/"run.stdout").read_text())["first_episode_navigation"]
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if 1<=t<=300:assert (e,t) not in st;st[e,t]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 1<=t<=300:
     assert (e,t,k) not in sub;sub[e,t,k]=z[4:]
     if int(z[4])>int(z[5]):errors.append("overflow")
   elif line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300:
     vcount[e,t,k]+=1
     if max(abs(float(x)) for x in z[4:])>=11.99:caps[e]+=1
 assert len(st)==2400 and len(sub)==38400 and len(vcount)==12800
 assert all(v==1 for v in vcount.values())
 return result,st,sub,caps,errors
arms={n:load(paths[n]) for n in paths}
assert all(not arms[n][4] for n in paths)
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def stat(v):
 v=sorted(v)
 if not v:return {"count":0}
 return {"count":len(v),"median_m_s":v[(len(v)-1)//2],"p95_m_s":v[int(.95*(len(v)-1))],"max_m_s":v[-1],"fraction_over_0_01m_s":sum(x>.01 for x in v)/len(v)}
def slip(name):
 path=paths[name];st=arms[name][1];window={(e,t):st[e,t] for e in range(8) for t in range(201,301)};inp="".join(str(e)+" "+str(t)+" "+" ".join(map(str,x["q"]+x["v"]))+"\n" for (e,t),x in sorted(window.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())};p=subprocess.run(["build/crow-v15-gait-slip-observer-20261112/kinematics"],input=inp,text=True,capture_output=True,check=True,env=env);bodies={}
 for line in p.stdout.splitlines():
  z=line.split()
  if z[0]=="body":
   v=list(map(float,z[4:]));assert len(v)==13 and all(math.isfinite(x) for x in v);bodies[int(z[1]),int(z[2]),z[3]]=v
 with gzip.open(path/"replay.json.gz","rt") as f:replay=json.load(f)["payload"]
 assert replay["environment"]==5 and replay["body_state_stride"]==13
 frames={x["step"]+1:x for x in replay["frames"] if 200<=x["step"]<=299};assert len(frames)==100
 bn=replay["body_names"];foot=[n for n in bn if n.startswith(("crow_left_","crow_right_")) and ("foot" in n or "toe" in n)];qerr=poserr=velerr=angerr=groundspeed=0.
 for t,f in frames.items():
  qerr=max(qerr,max(abs(a-b) for a,b in zip(f["q"],window[5,t]["q"])));g=bn.index("locomotion_ground")*13;groundspeed=max(groundspeed,*map(abs,f["body_states"][g+7:g+13]))
  for n in foot:
   i=bn.index(n)*13;v=f["body_states"][i:i+13];k=bodies[5,t,n];poserr=max(poserr,math.dist(v[:3],k[:3]));velerr=max(velerr,math.dist(v[7:10],k[7:10]));angerr=max(angerr,math.dist(v[10:13],k[10:13]))
 check={"q":qerr,"position_m":poserr,"linear_m_s":velerr,"angular_rad_s":angerr,"ground_speed":groundspeed};assert qerr<1e-6 and poserr<1e-5 and velerr<1e-4 and angerr<1e-4 and groundspeed<1e-7,check
 values=collections.defaultdict(list);excluded=0
 for (e,t),x in window.items():
  for c in x["contacts"]:
   if c["impulses"][0]<=1e-7:continue
   n=c["a_name"] if c["b_name"]=="locomotion_ground" else c["b_name"] if c["a_name"]=="locomotion_ground" else ""
   if n not in foot:continue
   assert "point" in c and len(c["point"])==3
   if abs(c["normal"][2])<.99:excluded+=1;continue
   b=bodies[e,t,n];r=[c["point"][i]-b[i] for i in range(3)];w=cross(b[10:13],r);v=[b[7+i]+w[i] for i in range(3)];values[e,"left" if n.startswith("crow_left_") else "right"].append(math.hypot(v[0],v[1]))
 return {"kinematics_replay_validation":check,"nonvertical_excluded":excluded,"all":stat([v for x in values.values() for v in x]),"rows":[{"lane":e,"side":side,**stat(values[e,side])} for e in range(8) for side in ("left","right")]}



assert all(not arms[n][4] for n in paths)
baseline,phase=arms['baseline'],arms['phase']
parity={str(e):{'endpoints':sum(baseline[1][e,t]!=phase[1][e,t] for t in range(1,301)), 'substeps':sum(baseline[2][e,t,k]!=phase[2][e,t,k] for t in range(1,301) for k in range(16))} for e in range(8)}
rows=[]
for e in range(8):
 row={'lane':e,'feedback_gain':0}
 for name in paths:
  _,st,sub,caps,errors=arms[name]
  row[name]={'both_unloaded_substeps':sum(float(sub[e,t,k][4])<=1e-7 and float(sub[e,t,k][5])<=1e-7 for t in range(201,301) for k in range(16)), 'toe_cap_substeps':caps[e], 'nonfoot_ground_impulse':sum(float(sub[e,t,k][6]) for t in range(201,301) for k in range(16))}
 rows.append(row)
out={'seed':seed,'lane_parity':parity,'contact_speed':{n:slip(n) for n in paths},'rows':rows}
(A/f'contact-analysis-{seed}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
