from pathlib import Path
import collections,gzip,json
D=Path("artifacts/crow-v15-walk-target-observer-20261111"); P=D/"autonomous-20261126";R=Path("artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126")
def read(p):
 endpoints={};substeps={};targets={};phase={}
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e=x["lane"];t=x["step"]
    if t<=300:endpoints[e,t]=[x[k] for k in ("q","v","policy_action","contacts")]
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);
    if t<=300:substeps[e,t,k]=z[4:]
   elif l.startswith("CROW_PHASE_SAMPLE "):
    z=l.split();t,e=map(int,z[1:3]);
    if t<=300:phase[e,t]=z[3:]
   elif l.startswith("CROW_DRIVE_TARGET "):
    z=l.split();t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
 return endpoints,substeps,targets,phase
a=read(P);b=read(R); rows=[]
for e in range(8):
 endpoint=[t for t in range(1,301) if a[0].get((e,t))!=b[0].get((e,t)) or (e,t) not in a[0] or (e,t) not in b[0]];sub=[(t,k) for t in range(1,301) for k in range(16) if a[1].get((e,t,k))!=b[1].get((e,t,k)) or (e,t,k) not in a[1] or (e,t,k) not in b[1]];ph=[t for t in range(1,301) if a[3].get((e,t))!=b[3].get((e,t)) or (e,t) not in a[3] or (e,t) not in b[3]]
 rows.append({"lane":e,"endpoint_differences":len(endpoint),"first_endpoint_difference":endpoint[:1],"substep_differences":len(sub),"first_substep_difference":sub[:1],"phase_differences":len(ph),"first_phase_difference":ph[:1],"target_samples":sum((e,t) in a[2] and len(a[2][e,t])==40 for t in range(200,301))})
out={"rows":rows,"scope":"Endpoint equality covers q, v, actor output and reported contacts; substep equality covers printed aggregate contact fields. This does not prove equality of unlogged solver internals. Target buffer timing and index bindings still require validation."};(D/"comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
