from pathlib import Path
import gzip,json
D=Path("artifacts/crow-v15-landing-staged-long-horizon-20261111");P=D/"autonomous-20261126";R=Path("artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126")
ends=[min(a,b) for a,b in zip(json.loads((P/"run.stdout").read_text())["first_episode_navigation"]["steps_by_environment"],json.loads((R/"run.stdout").read_text())["first_episode_navigation"]["steps_by_environment"])]
def read(path):
 d={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e=x["lane"];t=x["step"]
    if t<=ends[e]:d[e,t]=x
 return d
a=read(P);b=read(R);rows=[]
for e,end in enumerate(ends):
 dif=[t for t in range(1,end+1) if (e,t) not in a or (e,t) not in b or any(a[e,t][k]!=b[e,t][k] for k in ("q","v","policy_action","contacts"))]
 rows.append({"lane":e,"common_first_episode_end":end,"first_difference":dif[:1],"different_endpoint_count":len(dif)})
x={"rows":rows,"scope":"Accepted q, v, policy output and endpoint contacts only through the shorter first episode. A match is not an all-substep or biological claim."};(D/"reference-comparison.json").write_text(json.dumps(x,indent=2)+"\n");print(json.dumps(x,indent=2))
