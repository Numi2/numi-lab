from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-pregrip-damping-screen-20261111");B=Path("artifacts/crow-v15-uniform-release-perturbation-20261111/high")
def load(p):
 states={}
 with gzip.open(p,"rt") as f:
  for line in f:
   x=line.split()
   if x and x[0]=="step":
    e,t=int(x[1]),int(x[2]);i=x.index("q");j=x.index("v");q=list(map(float,x[i+1:j]));v=list(map(float,x[j+1:]));states[e,t]=(q,v)
 return states
a=load(B/"trace.log.gz");b=load(A/"high/trace.log.gz");rows=[]
for e in range(8):
 times=sorted(t for lane,t in b if lane==e);dif=[t for t in times if a.get((e,t))!=b[e,t]];bad=[t for t in times if not all(map(math.isfinite,b[e,t][0]+b[e,t][1]))];rows.append({"lane":e,"logged_accepted_endpoints":len(times),"last_endpoint_zero_based":max(times) if times else None,"first_q_v_difference_from_reference":dif[0] if dif else None,"nonfinite_endpoint_steps":bad})
errors=[]
with gzip.open(A/"high/substeps.log.gz","rt") as f:
 for line in f:
  if line.startswith(("failure ","CROW_SUBSTEP_ERROR")):errors.append(line.strip())
out={"lanes":rows,"error_lines":errors,"exit_code":json.loads((A/"high/run-result.json").read_text())["exit_code"],"scope":"Partial native failure. No conclusion about full release or grip, other substep validity, or failure cause."};(A/"partial-analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
