from pathlib import Path
import gzip,json
root=Path("artifacts/crow-v15-phase-led-toe-screen-20261112");results=[]
for seed in (20261126,20261127):
 for arm in ("sham","candidate"):
  path=root/(arm+"-"+str(seed)+("-paired" if arm=="sham" else "-rebound"))/"run.stderr.gz";sums=[[0.,0.] for _ in range(8)];counts=[0]*8
  with gzip.open(path,"rt") as f:
   for line in f:
    if not line.startswith("CROW_TWO_TOE_SUBSTEP "):continue
    z=line.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300:
     assert int(z[4])<=int(z[5]);v=list(map(float,z[6:]));assert len(v)==37
     sums[e][0]+=v[2]+v[3];sums[e][1]+=sum(v[5:21]);counts[e]+=1
  assert counts==[1600]*8
  rows=[]
  for lane,(total,toes) in enumerate(sums):
   assert total>0 and -1e-7<=toes<=total+1e-5,(total,toes)
   rows.append({"lane":lane,"foot_plus_toe_normal_impulse":total,"toe_normal_impulse":toes,"toe_fraction":toes/total,"foot_collider_fraction":(total-toes)/total})
  results.append({"seed":seed,"arm":arm,"lanes":rows})
out={"scope":"Post-hoc summed simulated ground-contact normal impulses in steps 201-300; the remainder is assigned to each foot collider. Not vertical force, duty factor or a biological measurement. Anatomical mapping of the foot collider is unresolved.","results":results}
Path("artifacts/crow-v15-corvid-stance-comparison-20261112/contact-share.json").write_text(json.dumps(out,indent=2)+"\n")
print([(x["seed"],x["arm"],round(min(r["toe_fraction"] for r in x["lanes"]),3),round(max(r["toe_fraction"] for r in x["lanes"]),3)) for x in results])
