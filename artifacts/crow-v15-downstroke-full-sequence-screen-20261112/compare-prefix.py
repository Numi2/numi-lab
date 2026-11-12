from split_stream import open_text
from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-downstroke-full-sequence-screen-20261112");C=A/"autonomous-20261126";R=Path("artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126")
def read(p):
 st={};sub={};errors=[]
 with open_text(p/"run.stderr.gz") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if 1<=t<=344:st[e,t]=[x[k] for k in ("q","v","policy_action","contacts")]
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if 1<=t<=344:sub[e,t,k]=z[4:]
 assert len(st)==8*344 and len(sub)==8*344*16
 return st,sub
c=read(C);r=read(R)
s=[k for k in c[0] if c[0][k]!=r[0][k]];t=[k for k in c[1] if c[1][k]!=r[1][k]]
o={"states_checked":len(c[0]),"state_mismatches":len(s),"first_state_mismatches":s[:10],"substeps_checked":len(c[1]),"contact_mismatches":len(t),"first_contact_mismatches":t[:10],"pass":not s and not t};(A/"prefix-comparison.json").write_text(json.dumps(o,indent=2)+chr(10));print(json.dumps(o,indent=2))
