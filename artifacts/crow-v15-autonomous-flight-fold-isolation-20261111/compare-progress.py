from pathlib import Path
import gzip,json,math
p=Path("artifacts/crow-v15-autonomous-flight-fold-isolation-20261111")
paths=[("reference",Path("artifacts/crow-v15-matched-release-inhibited-20261111/autonomous-20261126/replay.json.gz")),("fold",p/"autonomous-20261126/replay.json.gz")]
samples=[]
for name,path in paths:
 with gzip.open(path,"rt") as f:d=json.load(f)["payload"]
 assert d["environment"]==7
 for t in (400,500,700,1000,1400,1800,2200,2600,3000):
  f=d["frames"][t-1];q=f["q"];v=f["v"];a,b,c,w=q[3:7];yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c))
  samples.append({"trial":name,"step":t,"xyz_m":q[:3],"yaw_rad":yaw,"horizontal_velocity_m_s":v[:2]})
out={"lane":7,"samples":samples,"scope":"Sparse replay endpoints from first episodes on one development seed. Velocity is world-frame, not an airspeed or drag measurement. Does not isolate fold, toe curl or command compensation."}
(p/"autonomous-20261126/progress-comparison.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
