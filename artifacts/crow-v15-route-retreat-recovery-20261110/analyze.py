from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-route-retreat-recovery-20261110");B=Path("artifacts/crow-v15-route-teacher-recovery-20261110");report={}
def load(p):
 s={};contacts=[];clear=[]
 with gzip.open(p,"rt") as f:
  for l in f:
   if l.startswith("CROW_FLIGHT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);s[(x["lane"],x["step"])]=(x["q"],x["v"])
   if l.startswith("CROW_CONTACT_TRACE "):contacts.append(json.loads(l.split(" ",1)[1]))
   if l.startswith("CROW_CLEARANCE_SAMPLE "):clear.append(json.loads(l.split(" ",1)[1]))
 return s,contacts,clear
for course in ("reserved-final-i","reserved-final-j"):
 student=json.loads((B/("student-"+course+".stdout")).read_text());recovery=json.loads((A/("recovery-"+course+".stdout")).read_text());s,_,_=load(B/("student-"+course+".stderr.gz"));r,contacts,clear=load(A/("recovery-"+course+".stderr.gz"));fs=student["first_episode_navigation"];fr=recovery["first_episode_navigation"];rows=[]
 for lane in range(8):
  endS=fs["steps_by_environment"][lane];endR=fr["steps_by_environment"][lane];keys=sorted(k for k in s if k[0]==lane and k[1]<2000 and k[1]<=min(endS,endR));bad=[k for k in keys if k not in r or r[k]!=s[k]];eligible=endS>=2000 and endR>=2000
  cc=[x for x in contacts if x["lane"]==lane and x["step"]<=endR+1];cl=[x for x in clear if x["lane"]==lane and x["end_step"]<=endR+1]
  rows.append({"lane":lane,"prefix_samples":len(keys),"prefix_mismatches":bad,"eligible_at_takeover":eligible,"student_end":endS,"student_waypoints":fs["maximum_waypoints_by_environment"][lane],"recovery_end":endR,"recovery_waypoints":fr["maximum_waypoints_by_environment"][lane],"physical_failure":fr["physical_failure_by_environment"][lane],"first_completion_step":fr["first_completion_step_by_environment"][lane],"first_fatal_contact":cc[0] if cc else None,"minimum_clearance":min(cl,key=lambda x:x["minimum_m"]) if cl else None})
 report[course]={"lanes":rows,"solver_failures":recovery["failed_environment_steps"],"prefix_parity":all(x["prefix_samples"]>0 and not x["prefix_mismatches"] for x in rows)}
 print(course,"parity",report[course]["prefix_parity"],"eligible",sum(x["eligible_at_takeover"] for x in rows),"reached five",sum(x["eligible_at_takeover"] and x["recovery_waypoints"]==5 for x in rows),"posttakeover_failures",sum(x["eligible_at_takeover"] and x["physical_failure"] for x in rows))
(A/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
