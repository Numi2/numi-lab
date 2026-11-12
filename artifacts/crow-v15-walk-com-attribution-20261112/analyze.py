from pathlib import Path
import importlib.util,json
S=importlib.util.spec_from_file_location("a","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");a=importlib.util.module_from_spec(S);S.loader.exec_module(a)
A=Path("artifacts/crow-v15-walk-com-attribution-20261112");B=Path("artifacts/crow-v15-tiny-swing-toe-screen-20261112");data={}
for arm,d in (("sham","sham-20261127-dev2"),("candidate","candidate-20261127-rebound")):
 sup,sub,bodies,com=a.load(B/d);data[arm]={}
 for e in range(8):
  roots=[sup[e,t]["q"][:3] for t in (201,300)];cs=[com[e,t] for t in (201,300)];data[arm][e]={"root_delta_m":[roots[1][i]-roots[0][i] for i in range(3)],"whole_body_com_delta_m":[cs[1][i]-cs[0][i] for i in range(3)]}
rows=[]
for e in range(8):
 s=data["sham"][e];c=data["candidate"][e];rows.append({"lane":e,"sham":s,"candidate":c,"root_x_delta_difference_m":c["root_delta_m"][0]-s["root_delta_m"][0],"whole_body_com_x_delta_difference_m":c["whole_body_com_delta_m"][0]-s["whole_body_com_delta_m"][0]})
out={"rows":rows,"scope":"Exploratory analysis of retained development samples 201–300. COM is mass-weighted simulated articulated-body position from native forward kinematics, not observed crow mass distribution or joint forces. Model, gait, seed and sampled window remain unqualified."};(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(rows,indent=2))
