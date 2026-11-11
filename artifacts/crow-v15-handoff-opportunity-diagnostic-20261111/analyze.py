from pathlib import Path
import collections,gzip,json,math
R=Path.cwd();A=R/"artifacts/crow-v15-handoff-opportunity-diagnostic-20261111";P=R/"artifacts/crow-v15-landing-handoff-center-control-20261111/autonomous-20261126"
with gzip.open(P/"replay.json.gz","rt") as f:d=json.load(f)["payload"]
assert d["environment"]==7 and d["body_names"][40]=="crow_course_perch"
perch=d["frames"][0]["body_states"][40*13:40*13+3];samples={};imp=collections.defaultdict(lambda:[0.,0.]);coverage=collections.Counter()
with gzip.open(P/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1])
   if x["lane"]==7 and 3000<=x["step"]<=3800:samples[x["step"]]=x
  elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4])
   if e==7 and 3000<=t<=3800:coverage[t,k]+=1
  elif l.startswith("CROW_SCENE_CONTACT "):
   z=l.split();t,k,e,a,b=map(int,z[1:6])
   if e!=7 or not 3000<=t<=3800 or 40 not in (a,b):continue
   body=b if a==40 else a
   side=0 if body==11 or 19<=body<=26 else 1 if body==15 or 27<=body<=34 else -1
   if side>=0:imp[t][side]+=float(z[6])
records=[]
for t,x in sorted(samples.items()):
 q=x["q"];v=x["v"];a,b,c,w=q[3:7];pitch=math.asin(max(-1,min(1,2*(w*b-c*a))));checks={"both_foot_perch_impulse":min(imp[t])>1e-7,"root_center":abs(perch[0]-q[0]-.012)<.010,"lateral":abs(perch[1]+.55-q[1])<.16,"horizontal_speed":math.hypot(*v[:2])<.08,"vertical_speed":abs(v[2])<.12,"pitch":abs(pitch)<.1,"pitch_rate":abs(v[4])<.1}
 records.append({"step":t,"checks":checks,"x_error_m":perch[0]-q[0]-.012,"pitch_rad":pitch,"pitch_rate_rad_s":v[4],"vx_m_s":v[0],"vz_m_s":v[2],"foot_normal_impulse":imp[t]})
both=[r for r in records if r["checks"]["both_foot_perch_impulse"]]
counts=collections.Counter(k for r in both for k,v in r["checks"].items() if not v)
matched=[r for r in both if all(r["checks"].values())]
out={"lane":7,"perch_xyz":perch,"sampled_intervals":len(records),"both_feet_impulse_intervals":len(both),"failed_gate_counts_among_both_foot_intervals":dict(counts),"all_proxy_conditions_intervals":[r["step"] for r in matched],"first_both_foot_intervals":both[:15],"missing_or_duplicate_substeps":sum(coverage[t,k]!=1 for t in range(3000,3801) for k in range(16)),"scope":"Diagnostic reconstruction from lane 7 accepted endpoints and substep normal impulses. Both-foot sum is not the controller compact contact load; joint and solver update indexing can differ by a control step. This does not directly observe internal phase or establish biological realism."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k!="first_both_foot_intervals"},indent=2))
