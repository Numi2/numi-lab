from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-downstroke-pushoff-substep-observer-20261112");P=A/"autonomous-20261126";R=Path("artifacts/crow-v15-downstroke-pushoff-screen-20261112/autonomous-20261126")
def parse(p,observe=False):
 states={};contacts={};coords={};errors=[]
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for l in f:
   if "CROW_SUBSTEP_ERROR" in l:errors.append(l.strip())
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if 1<=t<=370:
     if (e,t) in states:raise ValueError("duplicate endpoint")
     states[e,t]=[x[k] for k in ("q","v","policy_action","contacts")]
   if l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if int(z[4])>int(z[5]):errors.append("contact overflow")
    if 1<=t<=370:
     if (e,t,k) in contacts:raise ValueError("duplicate contact")
     contacts[e,t,k]=z[4:]
   if l.startswith("CROW_Q_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);key=e,t,k
    if key in coords:raise ValueError("duplicate coordinate")
    q=list(map(float,z[4:]));
    if len(q)!=41 or not all(math.isfinite(x) for x in q):raise ValueError("bad q")
    coords[key]=q
 return states,contacts,coords,errors
p=parse(P,True);r=parse(R);assert json.loads((P/"run-result.json").read_text())["exit_code"]==0
nstate=ncontact=0;maxq=0.;fail=[]
for e in range(8):
 for t in range(1,371):
  if (e,t) not in p[0] or (e,t) not in r[0] or p[0][e,t]!=r[0][e,t]:nstate+=1
  for k in range(16):
   if (e,t,k) not in p[1] or (e,t,k) not in r[1] or p[1][e,t,k]!=r[1][e,t,k]:ncontact+=1
   if 335<=t<=365 and (e,t,k) not in p[2]:fail.append([e,t,k])
  if 335<=t<=365 and (e,t,15) in p[2] and (e,t) in p[0]:maxq=max(maxq,max(abs(x-y) for x,y in zip(p[2][e,t,15],p[0][e,t][0])))
out={"state_mismatches":nstate,"contact_substep_mismatches":ncontact,"q_substeps_found":len(p[2]),"q_substeps_expected":8*31*16,"missing_q_examples":fail[:10],"max_final_substep_q_vs_endpoint":maxq,"errors":p[3],"reference_errors":r[3]};out["pass"]=not nstate and not ncontact and not fail and len(p[2])==8*31*16 and maxq<1e-6 and not p[3]
(A/"verification.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
