from pathlib import Path
import gzip,json,math,statistics
A=Path("artifacts/crow-v15-sustained-crouch-replication-20261112");out={}
def pitch(q):return math.asin(max(-1,min(1,2*(q[3]*q[1]-q[2]*q[0]))))
for run,p in (("baseline",Path("artifacts/crow-v15-sustained-contact-gait-20261112/candidate-20261126-rebound-hip/replay.json.gz")),("crouch",A/"candidate-20261126-rebound-hip/replay.json.gz")):
 with gzip.open(p,"rt") as f:r=json.load(f)["payload"]
 assert r["environment"]==5 and r["body_state_stride"]==13
 hi=r["body_names"].index("crow_head")*13;bi=r["body_names"].index("crow_body")*13;rows=[]
 for sec in range(4,10):
  f=[x for x in r["frames"] if sec*50<x["step"]<=(sec+1)*50];assert len(f)==50
  b=[x["body_states"][bi:bi+13] for x in f];h=[x["body_states"][hi:hi+13] for x in f];qs=[x["q"] for x in f]
  span=lambda xs:max(xs)-min(xs)
  rows.append({"seconds":[sec,sec+1],"body_dx_m":b[-1][0]-b[0][0],"head_dx_m":h[-1][0]-h[0][0],"body_z_range_m":span([x[2] for x in b]),"head_z_range_m":span([x[2] for x in h]),"body_pitch_range_rad":span([pitch(x[3:7]) for x in b]),"head_pitch_range_rad":span([pitch(x[3:7]) for x in h]),"neck_yaw_range_rad":span([x[20] for x in qs]),"neck_pitch_range_rad":span([x[21] for x in qs])})
 out[run]=rows
res={"lane":5,"seed":20261126,"sample_hz":50,"runs":out,"scope":"Accepted native body positions at 50 Hz for one closely related developmental lane. World-axis displacement, one-second ranges, and neck joint ranges are not stride-synchronous biological calibration or proof of gaze stabilization."};(A/"head-trunk-endpoints.json").write_text(json.dumps(res,indent=2)+"\n")
for k,v in out.items():print(k,*v,sep="\n")
