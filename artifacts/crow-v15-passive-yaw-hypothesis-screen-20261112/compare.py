from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-passive-yaw-hypothesis-screen-20261112")
def load(label):
 out={"flight":{},"sub":{},"wrench":{},"toe":{}};errors=[]
 with gzip.open(A/(label+".stderr.gz"),"rt") as f:
  for line in f:
   if "CROW_AERO_ERROR" in line or "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
   if line.startswith("CROW_FLIGHT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);key=x["lane"],x["step"];assert key not in out["flight"];out["flight"][key]=[x["q"],x["v"],x["policy_action"]]
   for tag,name in (("CROW_TWO_TOE_SUBSTEP ","sub"),("CROW_WING_WRENCH ","wrench"),("CROW_TOE_V_SUBSTEP ","toe")):
    if line.startswith(tag):
     z=line.split();t,k,e=map(int,z[1:4]);key=e,t,k;assert key not in out[name];out[name][key]=z[4:]
 assert not errors,errors
 return out
c=load("control");p=load("passive");rows=[]
for e in range(6):
 distances=[0,0,.03,.03,.06,.06];d=distances[e];row={"lane":e,"offset_m":d}
 for field in c:
  keys=sorted(k for k in c[field] if k[0]==e);assert keys and set(keys)=={k for k in p[field] if k[0]==e};miss=[k for k in keys if c[field][k]!=p[field][k]];row[field]={"samples":len(keys),"mismatches":len(miss),"first_mismatch":miss[0] if miss else None}
  if d==0:assert not miss,(e,field,miss[:3])
 rows.append(row)
out={"rows":rows,"scope":"Matched control and intervention. Zero-offset lanes require exact recorded flight, toe, contact and modeled-wrench parity, including post-first-episode records; other lanes first mismatch is limited to sampled fields. Force model remains uncalibrated."};(A/"comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
