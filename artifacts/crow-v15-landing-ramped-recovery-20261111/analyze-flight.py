from pathlib import Path
import collections,gzip,json
p=Path("artifacts/crow-v15-landing-ramped-recovery-20261111/autonomous-20261126")
first=json.loads((p/"run.stdout").read_text())["first_episode_navigation"]
ends=first["steps_by_environment"]
coverage=collections.Counter();caps=collections.defaultdict(list);selfhits=collections.defaultdict(list);ground=collections.defaultdict(list);overflow=[];errors=[]
with gzip.open(p/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4])
   if t<=ends[e]:
    coverage[e,t,k]+=1
    if int(z[4])>int(z[5]):overflow.append([e,t,k])
    if t>=400 and sum(float(v) for v in z[8:11])>1e-7:ground[e].append([t,k])
  elif line.startswith("CROW_TOE_V_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4])
   if t<=ends[e] and max(abs(float(v)) for v in z[4:])>=11.99:caps[e].append([t,k])
  elif line.startswith("CROW_SELF_CONTACT "):
   z=line.split();t,k,e,a,b=map(int,z[1:6])
   if t<=ends[e]:selfhits[e].append([t,k,a,b,float(z[6])])
  elif line.startswith("CROW_SUBSTEP_ERROR"):errors.append(line.strip())
rows=[]
for e,end in enumerate(ends):
 h=selfhits[e];c=caps[e]
 rows.append({"lane":e,"first_episode_steps":end,"physical_failure":first["physical_failure_by_environment"][e],"waypoints":first["maximum_waypoints_by_environment"][e],"toe_speed_cap_substeps":len(c),"first_cap":c[:1],"toe_speed_cap_substeps_after_500":sum(t>=500 for t,k in c),"self_contact_samples_steps_380_to_end":len(h),"first_self_contact":h[:1],"distinct_self_contact_step_substeps":len(set((z[0],z[1]) for z in h)),"ground_contact_substeps_after_400":len(ground[e]),"missing_or_duplicate_substeps":sum(coverage[e,t,k]!=1 for t in range(1,end+1) for k in range(16))})
out={"lanes":rows,"contact_capacity_overflows":overflow,"substep_errors":errors[-10:],"scope":"Self-contact observer logs contacts with normal impulse above 1e-7 N s between crow bodies only, steps 380 through 3800. Ground count uses summed normal impulse above 1e-7 N s for both feet and other crow bodies against the ground at covered substeps. Sampled 12 rad/s speed caps do not identify physical cause. No biological calibration or continuous flight qualification."}
(p/"flight-analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
