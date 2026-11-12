import gzip,json
from pathlib import Path
A=Path("artifacts/crow-v15-toe-only-interpolation-20261112");data={}
for arm,d in (("unsmoothed","sham-20261127-paired"),("smoothed","candidate-20261127-rebound")):
 vals=[]
 with gzip.open(A/d/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_TOE_V_SUBSTEP 215 "):
    z=l.split()
    if int(z[3])==0:vals.append(float(z[18]))
 assert len(vals)==16;data[arm]={"minimum_rad_s":min(vals),"maximum_rad_s":max(vals)}
(A/"toe-velocity-interval215.json").write_text(json.dumps(data,indent=2)+"\n");print(json.dumps(data,indent=2))
