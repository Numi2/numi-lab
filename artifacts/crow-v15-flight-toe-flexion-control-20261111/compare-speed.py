from pathlib import Path
import json,math
a=Path("artifacts/crow-v15-flight-toe-flexion-control-20261111")
paths={"reference":Path("build/crow-v15-journey-bounded-flight-screen-20261111/preflight.json"),"flexion":Path("build/crow-v15-flight-toe-flexion-control-20261111/flight-replay.json")}
out={}
for name,path in paths.items():
 frames=json.loads(path.read_text())["payload"]["frames"];rows=[]
 for i in (449,999,1099,1199,1299,1399,1499):
  f=frames[i];x,y,z,w=f["q"][3:7];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));v=f["v"];forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1]
  rows.append({"step":i+1,"forward_speed_m_s":forward,"height_m":f["q"][2],"left_leg_q":f["q"][14:17],"left_rear_toe_q":f["q"][29:31]})
 out[name]=rows
out["scope"]="Lane-one accepted-state replay. Matching bounded diagnostic commands, course collision disabled. Actual speed comparison avoids treating raw unadjusted command as the exact tracking target. This compares a single replay lane; visual appearance, approach and physical landing are not qualified."
(a/"speed-comparison.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
