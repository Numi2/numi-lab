from pathlib import Path
import json,sys
base=Path("artifacts")
sys.path.insert(0,str(base/"crow-v15-downstroke-full-sequence-screen-20261112"))
from split_stream import open_text
p=base/"crow-v15-downstroke-full-sequence-screen-20261112/autonomous-20261126"
ends=json.loads((p/"run.stdout").read_text())["first_episode_navigation"]["steps_by_environment"]
lanes=[dict(lane=i, release_samples=0, negative_requested_samples=0, any_floored_samples=0, max_flight_fold=0, example=None) for i in range(8)]
with open_text(p/"run.stderr.gz") as f:
 for line in f:
  if not line.startswith("CROW_PHASE_SAMPLE "):continue
  parts=line.split(); step,lane,status=map(int,parts[1:4]);
  if step>ends[lane]:continue
  seq=((status>>16)&15)|(((status>>28)&15)<<4)
  if not 170<=seq<230:continue
  x=lanes[lane];x["release_samples"]+=1
  ramp=status&63; front=max(ramp-2,0); ext=-.20*min(max((seq-170)/35,0),1); flight=.025*((status>>23)&31)
  values=[min(.03*front,1.05)+ext]*3+[min(.04*ramp,1.15)+ext]
  x["negative_requested_samples"]+=any(v<0 for v in values)
  x["any_floored_samples"]+=any(v<flight for v in values)
  x["max_flight_fold"]=max(x["max_flight_fold"],flight)
  if x["example"] is None and any(v<flight for v in values):x["example"]={"step":step,"sequence":seq,"ramp":ramp,"flight_target":flight,"perch_targets":values,"selected_targets":[max(v,flight) for v in values]}
out={"lanes":lanes,"limitations":"Decodes endpoint packed status using the current downstroke shader expression. The expression is evaluated from a state during the task pass; timing relative to the host sample must be confirmed before asserting an exact applied target. Command targets are not measured toe positions or forces. Used development layout and seed only."}
(base/"crow-v15-autonomous-toe-command-audit-20261112/analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
