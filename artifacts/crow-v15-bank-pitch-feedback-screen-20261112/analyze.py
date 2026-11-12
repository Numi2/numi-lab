from pathlib import Path
import collections,gzip,json,math
A=Path("artifacts/crow-v15-bank-pitch-feedback-screen-20261112");B=Path("artifacts/crow-v15-stronger-bank-feedback-screen-20261112")
def load(path):
 result=json.loads((path/"feedback.stdout").read_text())["first_episode_navigation"];ends=result["steps_by_environment"];samples={};contacts=collections.Counter();errors=[];nonfoot=collections.Counter();first={}
 with gzip.open(path/"feedback.stderr.gz","rt") as f:
  for line in f:
   if "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
   if line.startswith("CROW_FLIGHT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
    if t<=ends[e]:samples[e,t]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=ends[e]:
     contacts[e,t,k]+=1
     if int(z[4])>int(z[5]):errors.append("overflow")
     if t>=450 and abs(float(z[10]))>1e-7:nonfoot[e]+=1;first.setdefault(e,[t,k])
 return result,samples,contacts,errors,nonfoot,first
def angles(q):
 x,y,z,w=q[3:7];return [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
base=load(B);cand=load(A);bad=[];count=0
for e in range(6):
 for t in sorted(t for i,t in base[1] if i==e and t<450):
  count+=1;b=base[1][e,t];c=cand[1].get((e,t))
  if c is None or any(b[k]!=c[k] for k in ("q","v","policy_action")):bad.append([e,t])
rows=[]
for e in range(6):
 end=cand[0]["steps_by_environment"][e];missing=sum(cand[2][e,t,k]!=1 for t in range(1,end+1) for k in range(16));samples=[];target=math.pi/2 if e%2==0 else -math.pi/2
 for t in sorted(t for i,t in cand[1] if i==e and t>=500):
  x=cand[1][e,t];q=x["q"];v=x["v"];a=angles(q);b=base[1].get((e,t));ba=angles(b["q"]) if b else None
  samples.append({"step":t,"height_m":q[2],"roll_pitch_yaw_rad":a,"speed_m_s":math.sqrt(sum(v[i]*v[i] for i in range(3))),"heading_error_rad":abs(math.atan2(math.sin(target-a[2]),math.cos(target-a[2]))),"control_height_m":b["q"][2] if b else None,"control_roll_pitch_yaw_rad":ba,"control_speed_m_s":math.sqrt(sum(b["v"][i]**2 for i in range(3))) if b else None,"control_heading_error_rad":abs(math.atan2(math.sin(target-ba[2]),math.cos(target-ba[2]))) if b else None})
 passed=end>=1600 and not cand[0]["physical_failure_by_environment"][e] and not cand[4][e] and len(samples)==12 and all(.65<=v["height_m"]<=1.15 and abs(v["roll_pitch_yaw_rad"][1])<.35 for v in samples) and all(v["speed_m_s"]>=.5 for v in samples if v["step"]>=800) and samples[-1]["heading_error_rad"]<=.85*math.pi/2
 rows.append({"lane":e,"bank_limit_rad":[.12,.18,.24][e//2],"pitch_gain":[.2,.4,.8][e//2],"candidate_end":end,"candidate_failure":cand[0]["physical_failure_by_environment"][e],"candidate_reason":cand[0]["termination_reason_by_environment"][e],"control_end":base[0]["steps_by_environment"][e],"control_failure":base[0]["physical_failure_by_environment"][e],"missing_substeps":missing,"nonfoot_substeps":cand[4][e],"first_nonfoot":cand[5].get(e),"sampled_eng_pass":passed,"samples":samples})
out={"prefix":{"count":count,"expected":48,"mismatches":bad,"pass":count==48 and not bad},"errors":cand[3],"rows":rows,"paired_passes":[{"lanes":[i,i+1],"pass":rows[i]["sampled_eng_pass"] and rows[i+1]["sampled_eng_pass"]} for i in (0,2,4)],"scope":"One used development seed, sparse endpoint engineering bounds; body moments suppressed only after 9 s, not biological or flight qualification."};(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if not out["prefix"]["pass"] or cand[3] or any(x["missing_substeps"] for x in rows):raise SystemExit("trace or prefix validity failed")
