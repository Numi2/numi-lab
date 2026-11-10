from pathlib import Path
import gzip,json,math,sys
A=Path(sys.argv[1]) if len(sys.argv)>1 else Path("artifacts/crow-v15-integrated-route-baseline-20261110")
for file in sorted(A.glob("diagnostic-*.stdout")):
 if not file.stat().st_size:continue
 name=file.stem;doc=json.loads(file.read_text());first=doc["first_episode_navigation"];end=first["steps_by_environment"][0];samples={};clear=[];coll=[]
 with gzip.open(A/(name+".stderr.gz"),"rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);
    if x["lane"]==0 and x["step"]<=end+1:samples[x["step"]]=x
   elif line.startswith("CROW_CLEARANCE_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);
    if x["lane"]==0 and x["minimum_step"]<=end+1:clear.append(x)
   elif line.startswith("CROW_CONTACT_TRACE "):
    x=json.loads(line.split(" ",1)[1]);
    if x["lane"]==0 and x["step"]<=end+1:coll.append(x)
 def loaded(x):
  per={s:{"front":0.,"rear":0.} for s in ("left","right")};support={"left":0.,"right":0.};other=0.
  for c in x["contacts"]:
   an,bn=c["a_name"],c["b_name"]
   if "crow_course_perch" not in (an,bn):continue
   body=an if bn=="crow_course_perch" else bn;imp=c["impulses"][0];side=next((s for s in per if "_"+s+"_" in body),None)
   if side and ("_foot" in body or "_toe" in body):support[side]+=imp
   elif imp>0:other+=imp
   nx=c["normal"][0]*(1 if body==an else -1)
   if side and "front_toe" in body and nx<-.5:per[side]["front"]+=imp
   if side and "rear_toe" in body and nx>.5:per[side]["rear"]+=imp
  return per,support,other
 holds=[];supports=[]
 for t,x in sorted(samples.items()):
  per,support,other=loaded(x)
  if all(per[s][d]>.00005 for s in per for d in ("front","rear")):holds.append(t)
  if all(v>.001 for v in support.values()) and not other:supports.append(t)
 def longest(ts):
  best=run=0;old=None
  for t in ts:run=run+1 if old is not None and t==old+1 else 1;best=max(best,run);old=t
  return best
 result={"first_episode":first,"teacher_on":doc["birdflow_journey_teacher"],"scheduled_resets":doc["scheduled_resets"],"host_requested_resets":doc["host_requested_resets"],"solver_failures":doc["failed_environment_steps"],"accepted_sample_count":len(samples),"bilateral_support_endpoints":len(supports),"longest_bilateral_support_run":longest(supports),"bilateral_opposed_toe_grip_endpoints":len(holds),"longest_opposed_toe_grip_run":longest(holds),"minimum_clearance":min(clear,key=lambda v:v["minimum_m"]) if clear else None,"first_fatal_contact":coll[0] if coll else None,"final_supported_endpoint":supports[-1] if supports else None,"scope":"development layout; former actor with changed visual inputs; last-substep endpoint contacts only; not unseen-layout qualification"}
 (A/(name+"-analysis.json")).write_text(json.dumps(result,indent=2)+"\n");print(name,"waypoints",first["maximum_waypoints_by_environment"],"end",end,"failure",first["physical_failure_by_environment"],"support endpoints",len(supports),"longest",longest(supports),"grip",len(holds),"longest",longest(holds),"clearance",result["minimum_clearance"]["minimum_m"] if clear else None)
