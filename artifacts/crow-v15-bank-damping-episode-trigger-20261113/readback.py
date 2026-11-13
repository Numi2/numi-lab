from pathlib import Path
import gzip,json,sys
P=Path("artifacts/crow-v15-bank-damping-episode-trigger-20261113")/sys.argv[1];meta=json.loads((P/"run.stdout").read_text())["first_episode_navigation"];rows=[];found={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_TORSO_PARAMETERS "):
   z=line.split();t,k,e=map(int,z[1:4]);assert k==0 and (e,t) not in found;found[e,t]=list(map(float,z[4:]))
for e in (6,7):
 end=min(1150,meta["steps_by_environment"][e]);ticks=list(range(650,end+1,10));values=[found[e,t][3] for t in ticks if (e,t) in found];rows.append({"lane":e,"expected":len(ticks),"samples":len(values),"values":sorted(set(values)),"passed":bool(ticks) and len(values)==len(ticks) and all(abs(x-.5)<1e-6 for x in values)})
out={"rows":rows,"passed":all(x["passed"] for x in rows),"scope":"Sampled GPU torso damping parameter readback, not biological calibration."};(P/"readback.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
