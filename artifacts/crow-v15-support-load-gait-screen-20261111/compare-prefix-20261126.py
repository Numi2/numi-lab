from pathlib import Path
import gzip,json
D=Path("artifacts/crow-v15-support-load-gait-screen-20261111");P=D/"autonomous-20261126";R=Path("artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126")
def read(path):
 end={};sub={};caps={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e=x["lane"];t=x["step"]
    if t<=199:end[e,t]=[x[k] for k in ("q","v","policy_action","contacts")]
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if t<=199:sub[e,t,k]=z[4:]
   elif l.startswith("CROW_TOE_V_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300 and max(abs(float(v)) for v in z[4:])>=11.99:caps[e,t,k]=1
 return end,sub,caps
a=read(P);b=read(R);rows=[]
for e in range(8):
 de=[t for t in range(1,200) if (e,t) not in a[0] or (e,t) not in b[0] or a[0][e,t]!=b[0][e,t]];ds=[(t,k) for t in range(1,200) for k in range(16) if (e,t,k) not in a[1] or (e,t,k) not in b[1] or a[1][e,t,k]!=b[1][e,t,k]]
 rows.append({"lane":e,"prefix_endpoint_differences":len(de),"first_endpoint_difference":de[:1],"prefix_contact_substep_differences":len(ds),"first_substep_difference":ds[:1],"walk_candidate_toe_caps":sum(k[0]==e for k in a[2]),"walk_reference_toe_caps":sum(k[0]==e for k in b[2])})
out={"rows":rows,"scope":"Accepted endpoint q/v, actor output, reported contacts and printed aggregate contact substeps 1–199 before gait adjustment. Equality of logged output does not establish parity of unlogged internals. Toe speed limit counted only in 201–300."};(D/"prefix-comparison-20261126.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
