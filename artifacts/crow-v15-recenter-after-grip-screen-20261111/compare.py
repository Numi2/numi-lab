from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-recenter-after-grip-screen-20261111");B=Path("artifacts/crow-v15-contact-friction-screen-20261111/high");C=A/"high"
a=json.loads((B/"analysis.json").read_text());b=json.loads((C/"analysis.json").read_text());keys=("pass","first_root_below_0_1m","nonfoot_impulse_intervals","release_trigger_step","return_step","terminal_100_bilateral_supported_grip","longest_qualified_release_intervals","peak_minimum_left_clearance_m")
rows=[]
for x,y in zip(a["lanes"],b["lanes"]):
 assert x["lane"]==y["lane"];rows.append({"lane":x["lane"],"reference":{k:x[k] for k in keys},"candidate":{k:y[k] for k in keys}})
def states(p):
 out={};finished={}
 with gzip.open(p,"rt") as f:
  for l in f:
   x=l.split()
   if x and x[0]=="step":out[int(x[1]),int(x[2])]=x[x.index("q"):]
   if x and x[0]=="event" and int(x[11]) and int(x[1]) not in finished:finished[int(x[1])]=int(x[2])
 return out,finished
s1,f1=states(B/"trace.log.gz");s2,f2=states(C/"trace.log.gz");prefix=[]
for e in range(8):
 stop=min(f1.get(e,550),f2.get(e,550));diff=[t for t in range(stop) if s1.get((e,t))!=s2.get((e,t))];prefix.append({"lane":e,"reference_finished_step":f1.get(e),"candidate_finished_step":f2.get(e),"compared_before_first_finish":stop,"different_logged_q_v_states":len(diff),"first_difference":diff[:1]})
out={"rows":rows,"reference_passes":sum(x["reference"]["pass"] for x in rows),"candidate_passes":sum(x["candidate"]["pass"] for x in rows),"reference_falls":sum(x["reference"]["first_root_below_0_1m"] is not None for x in rows),"candidate_falls":sum(x["candidate"]["first_root_below_0_1m"] is not None for x in rows),"prefix":prefix,"candidate_missing_substeps":b["missing_substeps"],"candidate_bad_substeps":b["bad_substeps"],"candidate_errors":b["errors"],"scope":"Constructed stationary-perch development starts, not whole flight, landing, object manipulation or biology."}
(A/"comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k not in ("rows","scope")},indent=2))
