from pathlib import Path
import gzip,collections,json
p=Path("artifacts/crow-v15-three-joint-lift-screen-20261112");d=collections.defaultdict(list)
with gzip.open(p/"candidate-20261127-rebound/run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4]);v=list(map(float,z[6:]));
   if 201<=t<=300 and v[2]<=1e-7 and v[3]<=1e-7:d[e].append([t,k])
out={"lane_substeps":dict(d),"scope":"16 substeps per 20 ms step; zero normal threshold is not sufficient to prove a fall."};(p/"unsupported-times.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
