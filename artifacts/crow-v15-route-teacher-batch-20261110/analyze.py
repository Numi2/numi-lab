from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-route-teacher-batch-20261110");report={}
for c in ("reserved-final-i","reserved-final-j"):
 d=json.loads((A/(c+".stdout")).read_text());f=d["first_episode_navigation"];s={i:[] for i in range(8)};contacts={i:[] for i in range(8)}
 with gzip.open(A/(c+".stderr.gz"),"rt") as z:
  for l in z:
   if l.startswith("CROW_CLEARANCE_SAMPLE "):
    v=json.loads(l.split(" ",1)[1]);s[v["lane"]].append(v)
   if l.startswith("CROW_CONTACT_TRACE "):
    v=json.loads(l.split(" ",1)[1]);contacts[v["lane"]].append(v)
 lanes=[]
 for i in range(8):
  end=f["steps_by_environment"][i];clear=[v for v in s[i] if v["end_step"]<=end+1];cc=[v for v in contacts[i] if v["step"]<=end+1]
  first=f["first_completion_step_by_environment"][i]
  lanes.append({"lane":i,"end":end,"waypoints":f["maximum_waypoints_by_environment"][i],"first_five_step":first,"physical_failure":f["physical_failure_by_environment"][i],"failure_after_five":bool(first is not None and f["physical_failure_by_environment"][i] and end>=first),"minimum_clearance":min(clear,key=lambda x:x["minimum_m"]) if clear else None,"first_fatal_contact":cc[0] if cc else None})
 report[c]={"lanes":lanes,"solver_failures":d["failed_environment_steps"],"screen_pass":all(v["waypoints"]==5 and v["first_five_step"] is not None for v in lanes) and d["failed_environment_steps"]==0}
 print(c,"five_waypoints",sum(v["waypoints"]==5 for v in lanes),"failed_after_five",sum(v["failure_after_five"] for v in lanes),"screen",report[c]["screen_pass"])
(A/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
