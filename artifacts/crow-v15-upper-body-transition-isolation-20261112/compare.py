from pathlib import Path
import gzip,json,math
p=Path("artifacts/crow-v15-upper-body-transition-isolation-20261112")
paths={"baseline":Path("artifacts/crow-v15-long-gait-screen-20261112/sham-20261127-long"),"held":p/"sham-20261127-held"}
def read(path):
 states={};sub={};err=[];r=json.loads((path/"run.stdout").read_text())["first_episode_navigation"]
 assert json.loads((path/"run-result.json").read_text())["exit_code"]==0 and r["steps_by_environment"]==[500]*8 and not any(r["closed_by_environment"]) and not any(r["physical_failure_by_environment"])
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if "CROW_SUBSTEP_ERROR" in line:err.append(line.strip())
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    z=json.loads(line.split(" ",1)[1]);e,t=z["lane"],z["step"]
    if 1<=t<=500:
     assert (e,t) not in states and all(math.isfinite(x) for x in z["q"]+z["v"]);states[e,t]=[z[k] for k in ("q","v","policy_action","contacts")]
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=500:
     assert (e,t,k) not in sub
     if int(z[4])>int(z[5]):err.append("overflow")
     sub[e,t,k]=list(map(float,z[6:]))
 assert len(states)==4000 and len(sub)==64000
 return states,sub,err
arms={k:read(v) for k,v in paths.items()};b=arms["baseline"];h=arms["held"];ib=json.loads((paths["baseline"]/"inputs.json").read_text());ih=json.loads((paths["held"]/"inputs.json").read_text());assert all(ib[k]==ih[k] for k in ib if k!="metallib_sha256") and ib["metallib_sha256"]!=ih["metallib_sha256"]
rows=[]
for e in range(8):
 def metrics(a):
  q,cs,_=a;flags=[cs[e,t,k][2]<=1e-7 and cs[e,t,k][3]<=1e-7 for t in range(301,401) for k in range(16)];nonfoot=[t for t in range(301,501) if any(cs[e,t,k][4]>1e-7 for k in range(16))];lengths=[];n=0
  for x in flags+[False]:
   if x:n+=1
   elif n:lengths.append(n);n=0
  return {"unsupported_substeps_301_400":sum(flags),"longest_unsupported_seconds_301_400":max(lengths,default=0)*.00125,"nonfoot_intervals_301_500":nonfoot,"root_displacement_sections_m":[q[e,y][0][0]-q[e,x][0][0] for x,y in ((200,300),(300,400),(400,500))],"upper_body_q_300":[round(x,9) for x in q[e,300][0][7:14]],"upper_body_q_305":[round(x,9) for x in q[e,305][0][7:14]]}
 rows.append({"lane":e,"prefix_state_mismatches":sum(b[0][e,t]!=h[0][e,t] for t in range(1,301)),"prefix_substep_mismatches":sum(b[1][e,t,k]!=h[1][e,t,k] for t in range(1,301) for k in range(16)),"baseline":metrics(b),"held":metrics(h)})
base=sum(x["baseline"]["unsupported_substeps_301_400"] for x in rows);held=sum(x["held"]["unsupported_substeps_301_400"] for x in rows)
valid=all(not x["prefix_state_mismatches"] and not x["prefix_substep_mismatches"] and not x["held"]["nonfoot_intervals_301_500"] for x in rows) and not b[2] and not h[2]
out={"valid_matched_comparison":valid,"prediction_supported":valid and base>0 and held<=.2*base,"baseline_unsupported_substeps":base,"held_unsupported_substeps":held,"rows":rows,"errors":{"baseline":b[2],"held":h[2]},"limits":"One previously used seed; observational normal-impulse threshold, held policy commands after 6 seconds, not biological validation or proof of a unique causal mechanism."}
(p/"comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:out[k] for k in ("valid_matched_comparison","prediction_supported","baseline_unsupported_substeps","held_unsupported_substeps")},indent=2));print([(x["lane"],x["baseline"]["unsupported_substeps_301_400"],x["held"]["unsupported_substeps_301_400"],x["prefix_state_mismatches"],x["prefix_substep_mismatches"]) for x in rows])
