from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-walk-material-slip-audit-20261112");D=Path("artifacts/crow-v15-continuous-launch-open-diagnostic-20261112/candidate-20261126-rebound-hip")
with gzip.open(D/"replay.json.gz","rt") as f:r=json.load(f)["payload"]
assert r["environment"]==5 and r["body_state_stride"]==13
frames={x["step"]:x for x in r["frames"]};samples=[];match=[];vmatch=[]
with gzip.open(D/"run.stderr.gz","rt") as f:
 for l in f:
  if not l.startswith("CROW_SUPPORT_SAMPLE "):continue
  x=json.loads(l.split(" ",1)[1]);t=x["step"]
  if x["lane"]!=5 or not 201<=t<=501:continue
  frame=frames[t-1];delta=max(abs(a-b) for a,b in zip(x["q"],frame["q"]));match.append(delta);vmatch.append(max(abs(a-b) for a,b in zip(x["v"],frame["v"])))
  if delta>1e-7:raise ValueError((t,"q mismatch",delta))
  bs=frame["body_states"]
  for c in x["contacts"]:
   a,b=c["a"],c["b"];imp=c["impulses"][0]
   if imp<=0 or not ((a<35 and b==35) or (b<35 and a==35)):continue
   i=a if a<35 else b
   side="left" if i==11 or 19<=i<=26 else "right" if i==15 or 27<=i<=34 else None
   if side is None:continue
   body=bs[13*i:13*i+13];ground=bs[13*35:13*36];pos=body[:3];vel=body[7:10];omega=body[10:13];gp=ground[:3];gv=ground[7:10];gw=ground[10:13];p=c["point"];n=c["normal"]
   def at(v,w,o):
    d=[p[j]-o[j] for j in range(3)];cross=[w[1]*d[2]-w[2]*d[1],w[2]*d[0]-w[0]*d[2],w[0]*d[1]-w[1]*d[0]];return [v[j]+cross[j] for j in range(3)]
   u=at(vel,omega,pos);g=at(gv,gw,gp);rel=[u[j]-g[j] for j in range(3)];nm=math.sqrt(sum(v*v for v in n));unit=[v/nm for v in n];dot=sum(rel[j]*unit[j] for j in range(3));tan=math.sqrt(max(0,sum(v*v for v in rel)-dot*dot))
   samples.append({"trace_submission":t,"replay_step":t-1,"body":r["body_names"][i],"side":side,"normal_impulse_ns":imp,"tangential_point_speed_m_s":tan})
def weighted(rows,q):
 rows=sorted(rows,key=lambda x:x["tangential_point_speed_m_s"]);total=sum(x["normal_impulse_ns"] for x in rows);acc=0
 for x in rows:
  acc+=x["normal_impulse_ns"]
  if acc>=q*total:return x["tangential_point_speed_m_s"]
def summary(rows):
 total=sum(x["normal_impulse_ns"] for x in rows)
 return {"contact_samples":len(rows),"summed_sampled_normal_impulse_ns":total,"impulse_weighted_mean_m_s":sum(x["normal_impulse_ns"]*x["tangential_point_speed_m_s"] for x in rows)/total,"impulse_weighted_p50_m_s":weighted(rows,.5),"impulse_weighted_p90_m_s":weighted(rows,.9),"impulse_weighted_p99_m_s":weighted(rows,.99),"max_m_s":max(x["tangential_point_speed_m_s"] for x in rows),"impulse_fraction_above_0_05_m_s":sum(x["normal_impulse_ns"] for x in rows if x["tangential_point_speed_m_s"]>.05)/total}
# Check body-center-of-mass linear velocity against central position differences, not contact slip.
checks={}
for i in (11,15,19,20,27,28):
 errs=[];speeds=[]
 for step in range(201,500):
  before=frames[step-1]["body_states"][13*i:13*i+13];now=frames[step]["body_states"][13*i:13*i+13];after=frames[step+1]["body_states"][13*i:13*i+13];fd=[(after[j]-before[j])/.04 for j in range(3)];errs.append(math.sqrt(sum((fd[j]-now[7+j])**2 for j in range(3))));speeds.append(math.sqrt(sum(now[7+j]**2 for j in range(3))))
 checks[r["body_names"][i]]={"mean_velocity_difference_m_s":sum(errs)/len(errs),"mean_reported_speed_m_s":sum(speeds)/len(speeds)}
out={"source":"retained obstacle-free continuous-launch development run, seed 20261126 lane 5", "accepted_q_match_max_abs":max(match),"accepted_v_match_max_abs":max(vmatch),"matched_steps":len(match),"total":summary(samples),"left":summary([x for x in samples if x["side"]=="left"]),"right":summary([x for x in samples if x["side"]=="right"]),"body_center_of_mass_velocity_consistency":checks,"samples":samples,"limitations":"Final substep contact point and accepted endpoint body velocity, one development lane, not continuous slip distance or a biological target; normal impulse weighting only samples final substeps."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k!="samples"},indent=2))
