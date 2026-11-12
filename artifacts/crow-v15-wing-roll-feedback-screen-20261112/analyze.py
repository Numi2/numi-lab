from pathlib import Path
import collections,gzip,json,math,sys
D=Path("artifacts/crow-v15-wing-roll-feedback-screen-20261112"); arm=sys.argv[1];assert arm in ("bank",); data=json.loads((D/(arm+".stdout")).read_text()); first=data["first_episode_navigation"]; ends=first["steps_by_environment"]; samples=collections.defaultdict(list); coverage=collections.Counter(); caps=collections.Counter(); late=collections.Counter(); ground=collections.Counter(); overflow=[]; errors=[]
with gzip.open(D/(arm+".stderr.gz"),"rt") as f:
 for line in f:
  if line.startswith("CROW_FLIGHT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
   if t<=ends[e]:
    q=x["q"]; v=x["v"]; a,b,c,w=q[3:7]; yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c)); pitch=math.asin(max(-1,min(1,2*(w*b-c*a)))); roll=math.atan2(2*(w*a+b*c),1-2*(a*a+b*b)); samples[e].append({"step":t,"position":q[:3],"yaw_rad":yaw,"pitch_rad":pitch,"roll_rad":roll,"velocity":v[:3]})
  elif line.startswith("CROW_TOE_V_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4])
   if t<=ends[e] and max(abs(float(v)) for v in z[4:])>=11.99:caps[e]+=1;late[e]+=int(t>=400)
  elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4])
   if t<=ends[e]:
    coverage[e,t,k]+=1
    if int(z[4])>int(z[5]):overflow.append([e,t,k])
    if t>=400 and sum(float(v) for v in z[8:11])>1e-7:ground[e]+=1
  elif line.startswith("CROW_SUBSTEP_ERROR"):errors.append(line.strip())
rows=[]
for e in range(6):
 vals=samples[e]; target=[math.pi/2,-math.pi/2,math.pi/2,-math.pi/2,math.pi/2,-math.pi/2][e%6]; last=vals[-1] if vals else None
 err=abs(math.atan2(math.sin(target-last["yaw_rad"]),math.cos(target-last["yaw_rad"]))) if last else None
 rows.append({"lane":e,"target_heading_rad":target,"first_episode_steps":ends[e],"physical_failure":first["physical_failure_by_environment"][e],"final_sample_heading_error_rad":err,"toe_caps":caps[e],"toe_caps_from_step400":late[e],"ground_contact_substeps_from_step400":ground[e],"missing_or_duplicate_substeps":sum(coverage[e,t,k]!=1 for t in range(1,ends[e]+1) for k in range(16)),"samples":vals})
out={"rows":rows,"failed_environment_steps":data["failed_environment_steps"],"contact_overflows":overflow,"substep_errors":errors,"scope":"Development seed, obstacle-free diagnostic heading targets; heading, posture, speed and root position sampled only at logged endpoints. No continuously swept or aerodynamic, biological, landing or unseen-layout qualification."};(D/(arm+"-analysis.json")).write_text(json.dumps(out,indent=2)+"\n")
for r in rows:
 print(r["lane"],"target",round(r["target_heading_rad"],2),"end",r["first_episode_steps"],"fail",r["physical_failure"],"final_error",round(r["final_sample_heading_error_rad"],3) if r["final_sample_heading_error_rad"] is not None else None,"toe_caps",r["toe_caps"],"toe_caps_after400",r["toe_caps_from_step400"],"ground_after400",r["ground_contact_substeps_from_step400"],"missing",r["missing_or_duplicate_substeps"])
print("failed steps",data["failed_environment_steps"],"overflows",len(overflow),"errors",errors[-3:])
