from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-substep-contact-observer-20261110");O=Path("artifacts/crow-v15-guide-midheight-20261110");name="development-reserved-final-j"
def load(p):
 with gzip.open(p,"rt") as f:return json.load(f)["payload"]
a=load(O/(name+".json.gz"));b=load(A/(name+".json.gz"));d=[]
for fa,fb in zip(a["frames"],b["frames"]):
 for k in ("step","q","v","actor_observation","accepted_actions","termination_reason"):
  if fa[k]!=fb[k]:
   if len(d)<20:d.append({"step":fa["step"],"field":k})
   break
parity=len(a["frames"])==len(b["frames"]) and not d
sub=[];statuses=[];endpoints={}
with gzip.open(A/(name+".stderr.gz"),"rt") as f:
 for line in f:
  if line.startswith("CROW_SUBSTEP_TOE "):
   v=line.split();sub.append({"submission":int(v[1]),"substep":int(v[2]),"lane":int(v[3]),"a":int(v[4]),"b":int(v[5]),"impulse":float(v[6]),"separation":float(v[7]),"normal_z":float(v[8])})
  elif line.startswith("CROW_SUBSTEP_STATUS "):
   v=line.split();statuses.append({"submission":int(v[1]),"substep":int(v[2]),"lane":int(v[3]),"required":int(v[4]),"reported_substep":int(v[5])})
  elif line.startswith("CROW_SUPPORT_SAMPLE "):
   v=json.loads(line.split(" ",1)[1]);t=v["step"]
   if v["lane"]==0 and 4370<=t<=4412:
    endpoints[t]=[{"a":c["a"],"b":c["b"],"impulse":c["impulses"][0]} for c in v["contacts"] if "crow_course_perch" in (c["a_name"],c["b_name"]) and ("toe" in c["a_name"] or "toe" in c["b_name"])]
status_ok=len(statuses)==36*4*8 and all(z["reported_substep"]==z["substep"] for z in statuses)
endpoint_diff=[]
for t in range(4375,4411):
 left=sorted((z["a"],z["b"],round(z["impulse"],8)) for z in sub if z["lane"]==0 and z["substep"]==3 and z["submission"]==t)
 right=sorted((z["a"],z["b"],round(z["impulse"],8)) for z in endpoints.get(t,[]))
 if left!=right:endpoint_diff.append(t)
pos=[z for z in sub if z["lane"]==0 and z["impulse"]>1e-8]
(A/"analysis.json").write_text(json.dumps({"parity":parity,"parity_differences":d,"frame_count":[len(a["frames"]),len(b["frames"])],"status_count":len(statuses),"status_substeps_match":status_ok,"endpoint_parity":not endpoint_diff,"endpoint_differences":endpoint_diff,"toe_contacts":sub,"positive_lane0":pos,"endpoint_records":endpoints},indent=2)+"\n")
print("q/v/observation/action/termination parity",parity,"differences",d,"frames",len(a["frames"]),len(b["frames"]))
print("final substep endpoint parity",not endpoint_diff,"differences",endpoint_diff)
print("status count",len(statuses),"substep indices match",status_ok,"toe records",len(sub),"positive lane 0",len(pos))
for z in pos:
 if 4387<=z["submission"]<=4400:print(z)
