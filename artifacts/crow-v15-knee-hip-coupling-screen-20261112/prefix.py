from pathlib import Path
import gzip,hashlib,json
A=Path("artifacts/crow-v15-knee-hip-coupling-screen-20261112");P=Path("artifacts/crow-v15-knee-phase-advance-screen-20261112");out={}
def read(seed,arm):
 h=hashlib.sha256();counts={"state":0,"contact":0}
 with gzip.open((P if arm=="baseline" else A)/f"candidate-{seed}-rebound-{arm}"/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);t=x["step"];kind="state"
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    t=int(line.split(None,2)[1]);kind="contact"
   else:continue
   if 1<=t<=200:h.update(line.encode());counts[kind]+=1
 return {"sha256":h.hexdigest(),"counts":counts}
for seed in (20261126,20261127):
 a,b=read(seed,"baseline"),read(seed,"phase");assert a==b;out[str(seed)]=a
(A/"prefix-parity.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
