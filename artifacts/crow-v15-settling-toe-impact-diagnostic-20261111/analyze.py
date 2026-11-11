from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-settling-toe-impact-diagnostic-20261111");P=Path("artifacts/crow-v15-autonomous-curl-release-control-20261111/autonomous-20261126/run.stderr.gz")
contact={};hits=[];coverage={};over=[]
with gzip.open(P,"rt") as f:
 for line in f:
  if line.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t>300:continue
   coverage[e,t,k]=coverage.get((e,t,k),0)+1
   if int(z[4])>int(z[5]):over.append([e,t,k])
   v=list(map(float,z[6:]));contact[e,t,k]=v
  elif line.startswith("CROW_TOE_V_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t>300:continue
   v=list(map(float,z[4:]));ids=[i for i,x in enumerate(v) if abs(x)>=11.99]
   if ids:hits.append({"lane":e,"step":t,"substep":k,"indices":ids,"velocities":[v[i] for i in ids]})
for h in hits:
 v=contact[h["lane"],h["step"],h["substep"]];h["left_up_Ns"],h["right_up_Ns"],h["left_normal_Ns"],h["right_normal_Ns"],h["nonfoot_normal_Ns"]=v[:5];h["toe_normal_Ns"]=[v[5+i] for i in h["indices"]]
rows=[]
for lane in range(8):
 x=[h for h in hits if h["lane"]==lane];missing=sum(coverage.get((lane,t,k),0)!=1 for t in range(1,301) for k in range(16))
 rows.append({"lane":lane,"hit_substeps":len(x),"hit_indices":sorted(set(i for h in x for i in h["indices"])),"max_normal_impulse_on_hit_Ns":max((sum(h["toe_normal_Ns"]) for h in x),default=0),"missing_or_duplicate_substeps":missing})
out={"rows":rows,"hits":hits,"contact_capacity_overflows":over,"nominal_weight_impulse_per_substep_Ns":.45006*9.81*.02/16,"scope":"Same used development first episodes and full recorded substeps; toe velocity is post-solver. Impulse and speed coincidence is associative and does not identify the cause of the contact oscillation or prove physical realism."};(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(rows,indent=2))
