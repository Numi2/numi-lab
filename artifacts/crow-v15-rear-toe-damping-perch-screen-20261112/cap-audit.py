from pathlib import Path
import gzip,json,sys
A=Path("artifacts/crow-v15-rear-toe-damping-perch-screen-20261112");out={}
for arm in sys.argv[1:]:
 caps=[]
 with gzip.open(A/arm/"substeps.log.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_TOE_VELOCITY "):continue
   z=line.split();t,k,e=map(int,z[1:4]); v=list(map(float,z[4:]))[-18:-2];assert len(v)==16
   for slot,x in enumerate(v):
    if abs(x)>=11.99:caps.append({"lane":e,"interval":t,"substep":k,"slot":slot,"velocity_rad_s":x})
 out[arm]=caps
(A/"cap-audit.json").write_text(json.dumps({"caps":out,"mapping":"slots 0-7 left foot, 8-15 right; each pair corresponds to proximal/distal, digit 3 is rear; endpoint trace zero-based and substep interval one-based","scope":"sampled velocity cap is not measured drive torque or cause"},indent=2)+"\n");print(json.dumps(out,indent=2))
