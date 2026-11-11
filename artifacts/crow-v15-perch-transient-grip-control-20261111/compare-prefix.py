from pathlib import Path
import gzip,json
base=Path("artifacts/crow-v15-teacher-perch-balanced-approach-control-20261111/teacher-20261126")
cur=Path("artifacts/crow-v15-perch-transient-grip-control-20261111/teacher-20261126")
info=json.loads((base/"perch-analysis.json").read_text())
ends=[x["first_perch_foot_impulse_step"]-1 for x in info["rows"]]
def load(p):
 d={}
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if t<=ends[e]:d[e,t]=x
 return d
x,y=load(base),load(cur);rows=[]
for e,end in enumerate(ends):
 missing=[t for t in range(1,end+1) if (e,t) not in x or (e,t) not in y]
 diff=[t for t in range(1,end+1) if (e,t) in x and (e,t) in y and x[e,t]!=y[e,t]]
 rows.append({"lane":e,"through_interval_before_first_reference_contact":end,"missing_count":len(missing),"different_count":len(diff),"first_difference":diff[:1]})
(Path("artifacts/crow-v15-perch-transient-grip-control-20261111")/"prefix-comparison.json").write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps(rows,indent=2))
