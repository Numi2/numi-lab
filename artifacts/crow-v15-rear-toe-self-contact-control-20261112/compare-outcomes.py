from pathlib import Path
import gzip,json,math,collections
p=Path("artifacts/crow-v15-rear-toe-self-contact-control-20261112");out={}
for name in ("control","candidate"):
 d=p/(name+"-20261127");run=json.loads((d/"run.stdout").read_text());ep=run["first_episode_navigation"];ends=ep["steps_by_environment"];st={};sub={};errors=[];pairs=collections.Counter();caps=collections.Counter()
 with gzip.open(d/"run.stderr.gz","rt") as f:
  for line in f:
   if "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"];
    if 1<=t<=300:st[e,t]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=300:
     sub[e,t,k]=list(map(float,z[6:]));
     if int(z[4])>int(z[5]):errors.append("overflow")
   elif line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=min(300,ends[e]) and max(abs(float(x)) for x in z[4:])>=11.99:caps[e]+=1
 assert len(st)==2400 and len(sub)==38400
 rows=[]
 for e,end in enumerate(ends):
  final=st[e,end];q=final["q"];x,y,z,w=q[3:7];pitch=math.asin(max(-1,min(1,2*(w*y-z*x))));selfcontacts=collections.Counter()
  for t in range(1,end+1):
   for c in st[e,t]["contacts"]:
    if c["impulses"][0]<=1e-7:continue
    a,b=c["a_name"],c["b_name"]
    if a.startswith("crow_") and b.startswith("crow_"):selfcontacts[tuple(sorted((a,b)))]+=1
  firstground=next(((t,k,sub[e,t,k][4]) for t in range(1,min(300,end+1)+1) for k in range(16) if sub[e,t,k][4]>1e-7),None)
  firstpair=None
  if end<300:
   later=st[e,min(end+1,300)];firstpair=sorted(set((c["a_name"],c["b_name"]) for c in later["contacts"] if c["impulses"][0]>1e-7 and "locomotion_ground" in (c["a_name"],c["b_name"])))
  rows.append({"lane":e,"steps":end,"physical_failure":ep["physical_failure_by_environment"][e],"reason":ep["termination_reason_by_environment"][e],"final_position_m":q[:3],"final_pitch_rad":pitch,"first_nonfoot_ground_substep":firstground,"following_endpoint_ground_pairs":firstpair,"self_contact_endpoint_pair_counts":[{"pair":k,"count":v} for k,v in selfcontacts.most_common()],"toe_velocity_cap_samples":caps[e]})
 out[name]={"rows":rows,"episode":ep,"errors":errors,"exit_code":json.loads((d/"run-result.json").read_text())["exit_code"]}
out["scope"]="Diagnostic same body, same actor weights and seed, differs only in two named self collision exclusions and fingerprints. Paired whole trajectories can diverge; no-exclusion parity is audited separately. Results are censored at each first episode; endpoint pairs after recorded failure cannot establish the precise first fatal substep. No result of the exclusion arm is physical qualification."
(p/"comparison.json").write_text(json.dumps(out,indent=2)+"\n")
for n in ("control","candidate"):
 print(n,"failures",out[n]["episode"]["physical_failure_count"],"end steps",[x["steps"] for x in out[n]["rows"]],"final_pitch",[round(x["final_pitch_rad"],3) for x in out[n]["rows"]],"errors",out[n]["errors"])
