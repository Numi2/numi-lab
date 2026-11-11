from pathlib import Path
import gzip,json,math,collections,statistics
D=Path("artifacts/crow-v15-localized-landing-contact-moments-20261111"); R=Path("artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126"); first=json.loads((R/"run.stdout").read_text())["first_episode_navigation"]; ends=first["steps_by_environment"]; states={}; hits=collections.defaultdict(list); cov=collections.Counter()
with gzip.open(R/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
   if 3000<=t<=ends[e]:states[e,t]=x
  elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=line.split(); t,k,e=map(int,z[1:4])
   if 3000<=t<=ends[e]:cov[e,t,k]+=1
  elif line.startswith("CROW_SCENE_CONTACT "):
   z=line.split();t,k,e,ba,bb=map(int,z[1:6])
   if 3000<=t<=ends[e] and 40 in (ba,bb):hits[e,t].append((ba,bb,list(map(float,z[6:]))))
rows=[]
for e in (0,5,6):
 bystep={}
 for t in range(3000,ends[e]+1):
  state=states.get((e,t))
  if state is None:continue
  q=state["q"]; root=q[:3]; groups=collections.defaultdict(lambda:[0.,0.,0.])
  for ba,bb,z in hits[e,t]:
   body=bb if ba==40 else ba; sign=1. if ba==40 else -1.; n=z[7:10]; tangent=z[10:13]; second=[n[1]*tangent[2]-n[2]*tangent[1],n[2]*tangent[0]-n[0]*tangent[2],n[0]*tangent[1]-n[1]*tangent[0]]
   j=[sign*(n[i]*z[0]+tangent[i]*z[1]+second[i]*z[2]) for i in range(3)]; p=z[3:6]
   group="rear_toes" if body in (25,26,33,34) else "front_toes" if 19<=body<=34 else "feet" if body in (11,15) else "other"
   g=groups[group]; g[0]+=j[0];g[1]+=j[2];g[2]+=(p[2]-root[2])*j[0]-(p[0]-root[0])*j[2]
  bystep[t]=groups
 windows=[]
 for lo in range(3150,ends[e],50):
  ts=[t for t in range(lo,min(lo+50,ends[e]+1)) if (e,t) in states];
  if not ts:continue
  g={name:[sum(bystep[t][name][i] for t in ts)/len(ts) for i in range(3)] for name in ("rear_toes","front_toes","feet","other")}
  sample=states[e,ts[0]];q=sample["q"];a,b,c,w=q[3:7];pitch=math.asin(max(-1.,min(1.,2*(w*b-c*a))))
  windows.append({"step_range":[ts[0],ts[-1]],"start_pitch_rad":pitch,"start_root_velocity":sample["v"][:6],"mean_contact_Jx_Jz_proxy_pitch_about_root":g})
 missing=sum(cov[e,t,k]!=1 for t in range(3000,ends[e]+1) for k in range(16))
 rows.append({"lane":e,"end":ends[e],"missing_or_duplicate_substeps":missing,"windows":windows})
out={"rows":rows,"limitations":"Contact-point moments are about endpoint root position, not whole-body center of mass; root moves during each control interval. Torsional impulses and gravity and wing moments are omitted. The contact-normal conventions are those used by the retained perch analysis. Values describe associations, not a closed angular momentum balance or unique causes."};(D/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
for r in rows:
 print("lane",r["lane"],"coverage error",r["missing_or_duplicate_substeps"])
 for w in r["windows"]:
  if w["step_range"][0]>=3300:
   print(w["step_range"],"pitch",round(w["start_pitch_rad"],3),"pitchrate",round(w["start_root_velocity"][4],3),"moments",{k:round(v[2],6) for k,v in w["mean_contact_Jx_Jz_proxy_pitch_about_root"].items()})
