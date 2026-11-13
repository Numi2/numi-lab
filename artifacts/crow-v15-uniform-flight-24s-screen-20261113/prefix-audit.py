from pathlib import Path
import gzip,json,sys
A=Path("artifacts/crow-v15-uniform-flight-24s-screen-20261113");P=A/sys.argv[1];seed=sys.argv[2];Q=Path("artifacts/crow-v15-uniform-flight-seed-screen-20261113")/("push-wing-012-"+seed+"-rebound-uniform");data=[]
for p in (Q,P):
 d={};data.append(d)
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
   x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
   if e>=2 and t<=900:
    key=(e,t);assert key not in d;d[key]={k:x[k] for k in ("q","v","policy_action","contacts")}
old,new=data;rows=[]
for e in range(2,8):
 missing=[];diff=[]
 for t in range(1,901):
  if (e,t) not in old or (e,t) not in new:missing.append(t)
  elif old[e,t]!=new[e,t]:diff.append(t)
 rows.append({"lane":e,"missing_ticks":missing,"different_ticks":diff,"exact_900_tick_parity":not missing and not diff})
a=json.loads((P/"uniform-audit.json").read_text());out={"seed":seed,"rows":rows,"all_six_exact_prefix":all(x["exact_900_tick_parity"] for x in rows),"validated_six_lane_24s_pass":bool(a["all_six_candidate_lanes_pass"] and all(x["exact_900_tick_parity"] for x in rows)),"scope":"Prefix parity does not establish physical flight qualification."};(P/"prefix-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
