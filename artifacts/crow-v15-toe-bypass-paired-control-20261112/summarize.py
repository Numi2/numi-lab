from pathlib import Path
import json
A=Path("artifacts/crow-v15-toe-bypass-paired-control-20261112");B=Path("artifacts/crow-v15-true-toe-control-replication-20261112");rows=[]
for seed in (20261126,20261127):
 a=json.loads((A/f"analysis-{seed}.json").read_text());b=json.loads((B/f"analysis-{seed}.json").read_text());paired=json.loads((A/f"paired-{seed}.json").read_text());j=json.loads((A/f"joint-speed-{seed}.json").read_text())
 for x,y,p,z in zip(a["rows"],b["rows"],paired,j["rows"]):
  assert x["lane"]==y["lane"]==p["lane"]==z["lane"]
  rows.append({"seed":seed,"lane":x["lane"],"arm":p["previous_arm"],"first_state_difference":p["state_first_difference"],"first_contact_difference":p["contact_first_difference"],"difference_impulse_ns_arm_minus_bypass":y["stats"]["launch_normal_impulse_ns"]-x["stats"]["launch_normal_impulse_ns"],"difference_vz_step556_m_s_arm_minus_bypass":y["z_vz_samples"]["556"][1]-x["z_vz_samples"]["556"][1],"bypass_joint_limit_hits":z["all_joint_limit_hits"],"bypass_toe_limit_hits":len(x["caps"]),"bypass_end":x["end"],"bypass_nonfoot":x["stats"]["nonfoot"]})
(A/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
for r in rows:print(r)
