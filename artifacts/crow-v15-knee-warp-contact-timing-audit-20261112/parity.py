from pathlib import Path
import gzip,hashlib,json,sys
A=Path("artifacts/crow-v15-knee-warp-contact-timing-audit-20261112");B=Path("artifacts/crow-v15-knee-phase-advance-screen-20261112");arm=sys.argv[1];assert arm in ("baseline","phase")
def read(d, suffix):
 h=hashlib.sha256();counts={"state":0,"contact":0,"toe":0}
 with gzip.open(d/("candidate-20261127-rebound-"+arm+suffix)/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);t=x["step"];kind="state"
   elif line.startswith(("CROW_TWO_TOE_SUBSTEP ","CROW_TOE_V_SUBSTEP ")):
    z=line.split(None,4);t=int(z[1]);kind="contact" if z[0]=="CROW_TWO_TOE_SUBSTEP" else "toe"
   else:continue
   if 1<=t<=300:h.update(line.encode());counts[kind]+=1
 return {"sha256":h.hexdigest(),"counts":counts}
x,y=read(A,"-repaired"),read(B,"");out={"arm":arm,"observer":x,"previous":y,"exact_match":x==y};(A/("parity-"+arm+".json")).write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));assert x==y
