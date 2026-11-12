from pathlib import Path
import gzip,json
a=Path("artifacts/crow-v15-preimpact-wide-perch-grip-screen-20261112/high/trace.log.gz");b=Path("artifacts/crow-v15-preimpact-immediate-toe-contact-20261112/high/trace.log.gz")
def read(p):
 d={};first={}
 with gzip.open(p,"rt") as f:
  for l in f:
   z=l.split()
   if not z:continue
   if z[0] in ("step","control"):
    e,t=map(int,z[1:3]);d[z[0],e,t]=z[3:]
   elif z[0]=="contact" and float(z[5])>0 and ("toe" in z[3] or "toe" in z[4] or "foot" in z[3] or "foot" in z[4]):
    e,t=map(int,z[1:3]);first[e]=min(first.get(e,t),t)
 return d,first
x,first=read(a);y,_=read(b);diff=[];counts=[]
for e in range(8):
 last=min(first.get(e,18),18);count=0
 for t in range(last+1):
  for kind in ("step","control"):
   key=(kind,e,t)
   if key not in x or key not in y:continue
   count+=1
   if x[key]!=y[key]:diff.append(list(key))
 counts.append({"lane":e,"first_baseline_foot_contact_step":first.get(e),"matching_records_compared":count})
out={"lanes":counts,"differences":diff,"limitations":"Only retained common prefix through first accepted contact is compared; stopping early reduces coverage and does not establish parity afterward."};Path("artifacts/crow-v15-preimpact-immediate-toe-contact-20261112/prefix-parity.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));assert not diff
