from pathlib import Path
import importlib.util,json,math
p=Path("artifacts/crow-v15-supporting-leg-stroke-20261112");s=importlib.util.spec_from_file_location("a",p/"long-audit.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
paths={"sham":Path("artifacts/crow-v15-upper-body-transition-isolation-20261112/sham-20261127-held"),"stroke":p/"sham-20261127-stance"};old=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text());ground=next(x["ground_z_m"] for x in old["rows"] if x["seed"]==20261127)
def calc(path):
 sup,sub,bodies,com=a.load(path);rows=[]
 for t in range(275,326):
  e=5;x=sup[e,t];entry={"step":t,"q":x["q"],"root_z_m":x["q"][2],"normal_impulse":[],"clearance_m":[],"pitch_rad":2*math.asin(max(-1,min(1,x["q"][4])))}
  for i,side in enumerate(("left","right")):
   names=a.names(side);clear=[]
   for name in names:
    z=bodies[e,t,name];xq,yq,zq,wq=z[3:7];rot=(2*(xq*zq-yq*wq),2*(yq*zq+xq*wq),1-2*(xq*xq+yq*yq));half=(.010,.01725,.0092) if name.endswith("_foot") else (.009,.0025,.0092);clear.append(z[2]-ground-sum(abs(j)*k for j,k in zip(rot,half)))
   entry["normal_impulse"].append(sum(sub[e,t,k][2+i] for k in range(16)));entry["clearance_m"].append({"sole":clear[0],"min_toe":min(clear[1:]),"min_toe_name":names[1+clear[1:].index(min(clear[1:]))]})
  rows.append(entry)
 return rows
out={"arms":{k:calc(v) for k,v in paths.items()},"scope":"Post-hoc lane 5 endpoint forward kinematics in a retained, previously used seed. Contact impulse is summed over each 20 ms interval; q/endpoints are not substep synchronized."};(p/"clearance-diagnostic.json").write_text(json.dumps(out,indent=2)+"\n")
for name,rows in out["arms"].items():
 print(name)
 for r in rows:
  if r["step"]%5==0:print(r["step"],"qz",round(r["root_z_m"],4),"pitch",round(r["pitch_rad"],3),"left/right gap",[[round(y["sole"]*1000,1),round(y["min_toe"]*1000,1)] for y in r["clearance_m"]],"norm",[round(x,3) for x in r["normal_impulse"]],"hip",[round(r["q"][i],2) for i in (14,17)])
