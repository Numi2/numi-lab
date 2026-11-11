from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-contact-pitch-feedback-control-20261111/autonomous-20261126")
m=json.loads((A/"perch-analysis.json").read_text());lanes=[x["lane"] for x in m["rows"] if x["last_25_half_weight_bilateral_opposed"] and not x["physical_failure"] and x["first_episode_steps"]==3800];samples={e:[] for e in lanes};failure=[]
with gzip.open(A/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
   if e in samples and 3776<=t<=3800:samples[e].append(x)
out=[]
for e,frames in samples.items():
 assert len(frames)==25 and [x["step"] for x in frames]==list(range(3776,3801))
 def magnitude(v):return math.sqrt(sum(a*a for a in v))
 def pitch(q):
  a,b,c,d=q[3:7];return math.asin(max(-1.,min(1.,2*(d*b-c*a))))
 active=sorted(set((c["a_name"],c["b_name"]) for x in frames for c in x["contacts"] if c["impulses"][0]>1e-6))
 out.append({"lane":e,"control_endpoint_count":len(frames),"body_height_range_m":[min(x["q"][2] for x in frames),max(x["q"][2] for x in frames)],"maximum_linear_speed_m_s":max(magnitude(x["v"][:3]) for x in frames),"maximum_angular_speed_rad_s":max(magnitude(x["v"][3:6]) for x in frames),"pitch_range_rad":[min(pitch(x["q"]) for x in frames),max(pitch(x["q"]) for x in frames)],"max_toe_speed_rad_s":max(abs(v) for x in frames for v in x["v"][22:38]),"positive_endpoint_collider_pairs":active})
o={"final_25_endpoints":out,"limits":"Endpoint stability and collision samples are independent of the substep impulse aggregation. Pair presence alone is not a biological or stability qualification. The test runs without teacher control; toe release and unused-layout success are not established. Termination ordering within substeps remains unresolved."}
(A/"terminal-audit.json").write_text(json.dumps(o,indent=2)+"\n")
print(json.dumps(o["final_25_endpoints"],indent=2))
