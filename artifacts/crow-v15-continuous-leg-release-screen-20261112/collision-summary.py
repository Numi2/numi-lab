from pathlib import Path
import collections,gzip,json
A=Path("artifacts/crow-v15-continuous-leg-release-screen-20261112");B=Path("artifacts/crow-v15-continuous-leg-release-hold-20261112");rows=[]
for label,root in (("extension",A),("hold",B)):
 for seed in (20261126,20261127):
  c=collections.Counter()
  with gzip.open(root/f"candidate-{seed}-rebound-hip"/"run.stderr.gz","rt") as f:
   for line in f:
    if line.startswith(("CROW_SELF_CONTACT ","CROW_SCENE_CONTACT ")):
     z=line.split();t,k,e=map(int,z[1:4])
     if 501<=t<=625:c[z[0],e]+=1
  rows.append({"arm":label,"seed":seed,"self_by_lane":[c['CROW_SELF_CONTACT',e] for e in range(8)],"scene_by_lane":[c['CROW_SCENE_CONTACT',e] for e in range(8)]})
(A/"collision-summary.json").write_text(json.dumps(rows,indent=2)+"\n")
for r in rows:print(r)
