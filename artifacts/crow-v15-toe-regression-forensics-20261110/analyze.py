from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-toe-regression-forensics-20261110");p=Path("artifacts/crow-v15-guide-midheight-20261110/development-reserved-final-j.stderr.gz");rows={}
with gzip.open(p,"rt") as f:
 for line in f:
  if line.startswith("CROW_TASK_STATE 0 "):
   v=line.split();t=int(v[2]);
   if 4355<=t<=4405:
    s=int(v[3]);rows.setdefault(t,{})["task"]={"front_ramps":[s&63,(s>>6)&63],"rear_ramps":[(s>>13)&63,(s>>19)&63],"foot_loads_N":list(map(float,v[5:7])),"landing_phase":float(v[7])}
  elif line.startswith("CROW_SUPPORT_SAMPLE "):
   z=json.loads(line.split(" ",1)[1]);t=z["step"]
   if z["lane"]==0 and 4355<=t<=4405:
    cs=[c for c in z["contacts"] if any(("toe" in n or "foot" in n) and "crow_" in n for n in (c["a_name"],c["b_name"]))]
    rows.setdefault(t,{})["body"]={"toe_angles_rad":z["q"][23:31],"toe_velocities_rad_s":z["v"][22:30],"root_velocity_m_s":z["v"][:3],"toe_foot_candidate_count":len(cs),"candidate_max_normal_impulse":max([c["impulses"][0] for c in cs],default=0.)}
(A/"analysis.json").write_text(json.dumps(rows,indent=2)+"\n")
for t in (4370,4380,4390,4394,4395,4396,4400):
 z=rows[t];print(t,"front ramp",z["task"]["front_ramps"],"angle",round(z["body"]["toe_angles_rad"][0],4),"toe speed",round(z["body"]["toe_velocities_rad_s"][0],3),"candidate count",z["body"]["toe_foot_candidate_count"],"root vz",round(z["body"]["root_velocity_m_s"][2],3))
