from pathlib import Path
import gzip,json
P=Path("artifacts/crow-v15-walk-ankle-phase-screen-20261112"); prior=Path("artifacts/crow-v15-walk-paired-phase-trial-20261112/candidate-20261127-rebound");candidate=P/"candidate-20261127-rebound"
def read(path):
 a={};b={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for s in f:
   if s.startswith("CROW_SUPPORT_SAMPLE "):
    z=json.loads(s.split(" ",1)[1]); a[z["lane"],z["step"]]=z
   elif s.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=s.split();t,k,e=map(int,z[1:4]);b[e,t,k]=z[4:]
 assert len(a)==2408 and len(b)==38528
 return a,b
c=read(candidate);r=read(prior)
out={"all_lane_before_walk_endpoint_mismatches":sum(c[0][x]!=r[0][x] for x in c[0] if x[1]<=200),"all_lane_before_walk_substep_mismatches":sum(c[1][x]!=r[1][x] for x in c[1] if x[1]<200),"lane3_all_endpoint_mismatches":sum(c[0][x]!=r[0][x] for x in c[0] if x[0]==3),"lane3_all_substep_mismatches":sum(c[1][x]!=r[1][x] for x in c[1] if x[0]==3)}
(P/"parity.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));assert not any(out.values())
