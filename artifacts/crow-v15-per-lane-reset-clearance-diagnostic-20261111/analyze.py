from pathlib import Path
import gzip,json,collections
A=Path("artifacts/crow-v15-per-lane-reset-clearance-diagnostic-20261111")
paths={"baseline16":Path("artifacts/crow-v15-aligned-contact-release-control-20261111/autonomous-20261126/run.stderr.gz"),"candidate16":A/"autonomous-20261126/run.stderr.gz"}
result={}
for name,path in paths.items():
 hits=collections.Counter();records=collections.Counter();peak=[0.]*8;err=[]
 with gzip.open(path,"rt") as f:
  for line in f:
   if line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if t>120:continue
    v=max(map(lambda x:abs(float(x)),z[4:]));records[e]+=1;peak[e]=max(peak[e],v);hits[e]+=v>=11.99
   elif line.startswith("CROW_SUBSTEP_ERROR"):err.append(line.strip())
 n=16
 result[name]={"lanes":[{"lane":e,"cap_substeps":hits[e],"peak_toe_speed_rad_s":peak[e],"records":records[e],"expected_records":120*n} for e in range(8)],"errors":err}
result["scope"]="Same used development seed, v15 body and identical policy and time step; seed-specific placement offsets change the reset positions. Counts are end-of-substep velocity observations, not maxima within substeps. This initial-contact diagnostic does not qualify walking, takeoff, flight or landing, and the lane-specific offsets do not generalize."
(A/"analysis.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
