from pathlib import Path
import collections,gzip,json,sys
A=Path("artifacts/crow-v15-full-foot-feedback-20261112");seed=int(sys.argv[1]);D=A/f"candidate-{seed}-rebound-phase";bits={};normal={}
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=="CROW_SENSOR_BITS":
   t,e=map(int,z[1:3]);assert (e,t) not in bits;bits[e,t]=int(z[3])
  elif z[0]=="CROW_TWO_TOE_SUBSTEP":
   t,k,e=map(int,z[1:4])
   if 201<=t<=300 and k==15:normal[e,t]=list(map(float,z[8:10]))
assert len(bits)==len(normal)==800 and bits.keys()==normal.keys()
rows=[]
for e in range(4,8):
 for side in (0,1):
  bad=[];mismatch=[];modes=collections.Counter()
  for t in range(202,301):
   prev,cur=bits[e,t-1],bits[e,t]
   assert cur&(1<<20)
   before=((prev>>(25+2*side))&3) if prev&(1<<20) else 0
   rear=bool(cur&(1<<(21+2*side)));front=bool(cur&(1<<(22+2*side)))
   own=bool(cur&(1<<(29+side)));opposite=bool(cur&(1<<(29+1-side)))
   if not own and opposite:expected=min(before+(1 if e%2==0 else 2),3);mode="rise"
   elif rear and not front:expected=before;mode="hold"
   else:expected=max(before-1,0);mode="fall"
   actual=(cur>>(25+2*side))&3;modes[mode]+=1
   if actual!=expected:bad.append({"step":t,"before":before,"expected":expected,"actual":actual})
   logged=normal[e,t][side]>1e-7
   if own!=logged:mismatch.append({"step":t,"sensor_full_foot":own,"trace_aggregate_impulse":normal[e,t][side]})
  rows.append({"lane":e,"side":["left","right"][side],"verified_transitions":99,"modes":dict(modes),"transition_violations":bad,"final_substep_disagreements":len(mismatch),"examples":mismatch[:3]})
(A/f"sensor-audit-{seed}.json").write_text(json.dumps({"seed":seed,"rows":rows,"scope":"Sensor considers each qualifying contact individually in its active-contact buffer; trace is summed impulse over required constraints at final substep. Capture timing, threshold aggregation and buffer selection differ. This is not proof of force at action time or at intermediate substeps."},indent=2)+"\n")
assert all(not x["transition_violations"] for x in rows)
for x in rows:print(seed,x["lane"],x["side"],x["modes"],"trace mismatches",x["final_substep_disagreements"])
