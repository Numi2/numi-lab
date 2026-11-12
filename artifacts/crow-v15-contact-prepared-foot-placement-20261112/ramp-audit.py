from pathlib import Path
import collections,gzip,json,sys
A=Path("artifacts/crow-v15-contact-prepared-foot-placement-20261112");seed=int(sys.argv[1]);D=A/f"candidate-{seed}-rebound-phase";bits={}
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  z=line.split()
  if z and z[0]=="CROW_SENSOR_BITS":
   t,e=map(int,z[1:3]);assert (e,t) not in bits;bits[e,t]=int(z[3])
assert len(bits)==800
rows=[]
for e in range(4,8):
 for side in (0,1):
  bad=[];counts=collections.Counter();verified=0
  for t in range(202,301):
   prev=bits[e,t-1];cur=bits[e,t]
   if not (cur&(1<<20)):continue
   before=((prev>>(25+3*side))&7) if prev&(1<<20) else 0
   rear=bool(cur&(1<<(21+2*side)));front=bool(cur&(1<<(22+2*side)))
   opp=1-side;opprear=bool(cur&(1<<(21+2*opp)));oppfront=bool(cur&(1<<(22+2*opp)))
   if not rear and not front and (opprear or oppfront):expected=min(before+(1 if e%2==0 else 2),7);mode="rise"
   elif rear and not front:expected=before;mode="hold"
   else:expected=max(before-1,0);mode="fall"
   actual=(cur>>(25+3*side))&7;verified+=1;counts[mode]+=1
   if actual!=expected:bad.append({"step":t,"before":before,"rear":rear,"front":front,"oppositeRear":opprear,"oppositeFront":oppfront,"expected":expected,"actual":actual})
  rows.append({"lane":e,"side":["left","right"][side],"verified":verified,"mode_counts":dict(counts),"violations":bad})
(A/f"ramp-audit-{seed}.json").write_text(json.dumps({"seed":seed,"rows":rows},indent=2)+"\n")
assert all(x["verified"]==99 and not x["violations"] for x in rows)
for x in rows:print(seed,x["lane"],x["side"],x["mode_counts"])
