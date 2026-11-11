from pathlib import Path
import gzip,json,collections
A=Path("artifacts/crow-v15-journey-bounded-flight-screen-20261111");base=json.loads((A/"flight-analysis.json").read_text());ends={x["lane"]:x["steps"] for x in base["lanes"]};coverage=collections.Counter();ground=collections.defaultdict(list);caps=collections.defaultdict(list)
with gzip.open(A/"flight.stderr.gz","rt") as f:
 for l in f:
  a=l.split()
  if a and a[0]=="CROW_TWO_TOE_SUBSTEP":
   t,k,e=map(int,a[1:4]);x=list(map(float,a[6:]));
   if t<=ends.get(e,0):
    coverage[e,t,k]+=1
    if t>=400 and sum(x[2:5])>1e-7:ground[e].append([t,k,sum(x[2:5])])
  elif a and a[0]=="CROW_TOE_V_SUBSTEP":
   t,k,e=map(int,a[1:4]);v=list(map(float,a[4:]));
   if t<=ends.get(e,0) and max(map(abs,v))>=11.99:caps[e].append([t,k])
rows=[]
for e in range(5):
 missing=sum(coverage[e,t,k]!=1 for t in range(1,ends[e]+1) for k in range(16));rows.append({"lane":e,"first_episode_steps":ends[e],"flight_ground_impulse_substeps_from_400":len(ground[e]),"first_ground_impulses":ground[e][:5],"toe_cap_substeps":caps[e],"missing_or_duplicate_contact_substeps":missing})
out={"rows":rows,"limits":"Observer records end-of-substep contact and velocity samples. Course colliders disabled, teacher off with diagnostic command substitution; not autonomous, recovery or landing qualification."};(A/"phase-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(rows,indent=2))
