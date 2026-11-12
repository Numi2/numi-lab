from pathlib import Path
import importlib.util,json,math
s=importlib.util.spec_from_file_location("a","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
A=Path("artifacts/crow-v15-toe-gait-coupling-audit-20261112");P=Path("artifacts/crow-v15-early-toe-release-screen-20261112");Q=Path("artifacts/crow-v15-short-toe-swing-screen-20261112");paths={"sham":P/"sham-20261126-paired","early":P/"candidate-20261126-rebound","short":Q/"candidate-20261126-rebound"};data={}
for name,path in paths.items():
 sup,sub,bodies,com=a.load(path);data[name]={"sup":sup,"sub":sub,"com":com};data[name]["gait"]=a.analyze(path,0.0)
rows=[]
for e in range(8):
 ref=data["sham"];rr=[]
 for name in ("early","short"):
  cand=data[name];series=[]
  for t in range(201,301):
   rs=ref["sup"][e,t];cs=cand["sup"][e,t];ri=[sum(ref["sub"][e,t,k][2+i] for k in range(16)) for i in range(2)];ci=[sum(cand["sub"][e,t,k][2+i] for k in range(16)) for i in range(2)];series.append({"interval":t,"root_x_delta_m":cs["q"][0]-rs["q"][0],"com_x_delta_m":cand["com"][e,t][0]-ref["com"][e,t][0],"root_x_velocity_delta_m_s":cs["v"][0]-rs["v"][0],"foot_normal_impulse_delta_Ns":[ci[i]-ri[i] for i in range(2)]})
  rr.append({"variant":name,"first_com_difference_over_0_1mm":next((v["interval"] for v in series if abs(v["com_x_delta_m"])>.0001),None),"first_root_velocity_difference_over_0_1mm_s":next((v["interval"] for v in series if abs(v["root_x_velocity_delta_m_s"])>.0001),None),"reference_supported_clearance_bouts":ref["gait"][e]["support_qualified_swing_bouts"],"candidate_supported_clearance_bouts":cand["gait"][e]["support_qualified_swing_bouts"],"samples":series})
 rows.append({"lane":e,"variants":rr})
out={"rows":rows,"scope":"Exploratory reanalysis of accepted sample endpoints and all-substep normal impulse sums in one previously used development seed. Differences are coupled outcomes; no torque or frictional impulse attribution and no biological qualification. Thresholds for earliest differences are diagnostic, not biological."};(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps([{ "lane":r["lane"],"variants":[{k:v for k,v in x.items() if k!="samples"} for x in r["variants"]]} for r in rows],indent=2))
