from pathlib import Path
import collections,gzip,json
R=Path.cwd();A=R/"artifacts/crow-v15-autonomous-gated-phase-walk-20261111";paths={"autonomous":R/"artifacts/crow-v15-autonomous-gated-phase-walk-20261111/autonomous-20261126/run.stderr.gz","gait_control":R/"artifacts/crow-v15-autonomous-curl-release-control-20261111/autonomous-20261126/run.stderr.gz"}
def read(p):
 s={};c={}
 with gzip.open(p,"rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t=x["step"];e=x["lane"]
    if t<=200:s[e,t]=x
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    x=l.split();t,k,e=map(int,x[1:4]);
    if t<=200:c[e,t,k]=x[4:]
 return s,c
s,c=read(paths["autonomous"]);r,v=read(paths["gait_control"]);rows=[]
for e in range(8):
 sd=[t for t in range(1,201) if (e,t) not in s or (e,t) not in r or any(s[e,t][k]!=r[e,t][k] for k in ("q","v","contacts"))]
 cd=[(t,k) for t in range(1,201) for k in range(16) if (e,t,k) not in c or (e,t,k) not in v or c[e,t,k]!=v[e,t,k]]
 rows.append({"lane":e,"first_state_difference":sd[:1],"state_difference_count":len(sd),"first_substep_contact_difference":cd[:1],"substep_contact_difference_count":len(cd)})
out={"rows":rows,"scope":"Compares accepted endpoint q, v and contacts and all recorded substep aggregate contacts for the first 200 intervals of matching development seed. A match does not establish physical realism or other seeds. Both runs use the same teacher-off actor, v15 body and frozen flight law; the extended walking command differs afterward."};(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
