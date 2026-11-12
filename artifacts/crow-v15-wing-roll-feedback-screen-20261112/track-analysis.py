from pathlib import Path
import json,math
A=Path("artifacts/crow-v15-wing-roll-feedback-screen-20261112");d=json.loads((A/"bank-analysis.json").read_text());rows=[]
for r in d["rows"]:
 vals=[]
 for s in r["samples"]:
  if s["step"]<500:continue
  vx,vy,vz=s["velocity"];speed=math.hypot(vx,vy)
  if speed<.1:continue
  track=math.atan2(vy,vx);diff=math.atan2(math.sin(track-s["yaw_rad"]),math.cos(track-s["yaw_rad"]))
  vals.append({"step":s["step"],"speed_m_s":speed,"track_rad":track,"body_yaw_rad":s["yaw_rad"],"track_minus_body_rad":diff,"roll_rad":s["roll_rad"]})
 rows.append({"lane":r["lane"],"samples":vals})
(A/"track-analysis.json").write_text(json.dumps({"rows":rows,"scope":"Sparse endpoint velocities after interval 500. Body heading need not equal flight-path direction during a turn; the observed divergence motivates longer synchronized kinematic and aerodynamic analysis, not a prespecified crow sideslip limit."},indent=2)+"\n")
for r in rows:print(r["lane"],r["samples"][-1] if r["samples"] else None)
