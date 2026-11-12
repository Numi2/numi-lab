from pathlib import Path
import gzip,hashlib,json,sys
r=Path("artifacts");old=r/"crow-v15-negative-ankle-matched-control-20261112";new=r/"crow-v15-full-foot-feedback-20261112";out={}
def get(path):
 hashes={(e,k):hashlib.sha256() for e in range(8) for k in ("state","toe","q","v")}; counts={k:0 for k in hashes}
 with gzip.open(path,"rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t,k=x["lane"],x["step"],"state"
   elif line.startswith(("CROW_TWO_TOE_SUBSTEP ","CROW_Q_SUBSTEP ","CROW_V_SUBSTEP ")):
    z=line.split(None,4);t,e=int(z[1]),int(z[3]);k={"CROW_TWO_TOE_SUBSTEP":"toe","CROW_Q_SUBSTEP":"q","CROW_V_SUBSTEP":"v"}[z[0]]
   else:continue
   if 1<=t<=300: hashes[e,k].update(line.encode());counts[e,k]+=1
 return {(e,k):{"sha256":hashes[e,k].hexdigest(),"count":counts[e,k]} for e in range(8) for k in ("state","toe","q","v")}
for seed in (20261126,20261127):
 a=get(old/f"candidate-{seed}-rebound-phase/run.stderr.gz");b=get(new/f"candidate-{seed}-rebound-phase/run.stderr.gz");out[str(seed)]={}
 for e in range(8):
  out[str(seed)][str(e)]={k:{"old":a[e,k],"new":b[e,k],"equal":a[e,k]==b[e,k]} for k in ("state","toe","q","v")}
(new/"matched-parity.json").write_text(json.dumps(out,indent=2)+"\n")
for seed,lanes in out.items():
 for e in (0,1,2,3):
  for k,v in lanes[str(e)].items():
   if v["old"]["count"]==0 or not v["equal"]:raise SystemExit(f"Unchanged lane mismatch: {seed} {e} {k}: {v}")
print("Unchanged lanes match all recorded state, toe, q and v samples on both seeds.")
