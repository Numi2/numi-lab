from pathlib import Path
import gzip,json
p=Path("artifacts/crow-v15-hallux-capsule-contact-screen-20261112");paths={"reference":Path("artifacts/crow-v15-corvid-hallux-amplitude-screen-20261112/candidate-20261127-rebound"),"control":p/"control-20261127"};d={}
for name,path in paths.items():
 st={};sub={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);key=(x["lane"],x["step"]);
    if 1<=x["step"]<=300:assert key not in st;st[key]=[x[k] for k in ("q","v","policy_action","contacts")]
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);key=(e,t,k);
    if 1<=t<=300:assert key not in sub;sub[key]=z[4:]
 assert len(st)==2400 and len(sub)==38400,(name,len(st),len(sub));d[name]=(st,sub)
a,b=d["reference"],d["control"];stbad=[k for k in a[0] if a[0][k]!=b[0][k]];subbad=[k for k in a[1] if a[1][k]!=b[1][k]];out={"state_samples":len(a[0]),"contact_substeps":len(a[1]),"state_mismatches":len(stbad),"contact_mismatches":len(subbad),"first_state_mismatch":stbad[:1],"first_contact_mismatch":subbad[:1],"pass":not stbad and not subbad,"scope":"Exact logged endpoints and substeps, not an independent biological validation."};(p/"control-parity.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));assert out["pass"]
