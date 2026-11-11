from pathlib import Path
import collections,gzip,json,math
p=Path("artifacts/crow-v15-release-hysteresis-control-20261111/autonomous-20261126")
states={};samples={};dups=[]
with gzip.open(p/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_PHASE_SAMPLE "):
   z=line.split();t,e,status=map(int,z[1:4]);v=list(map(float,z[4:]))
   if (e,t) in states:dups.append([e,t])
   states[e,t]={"step":t,"lane":e,"status":status,"phase":v[0],"left_load":v[1],"right_load":v[2],"goal_x_error":v[3],"goal_y_error":v[4],"vertical_speed":v[5],"pitch":v[6],"pitch_rate":v[7]}
  elif line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1])
   if x["lane"]==5:samples[x["step"]]=x
r=[states[5,t] for t in range(3000,3801) if (5,t) in states];transitions=[];prev=None
for x in r:
 phase=x["phase"]
 x["release_sequence"] = ((x["status"] >> 16) & 15) | (((x["status"] >> 28) & 15) << 4)
 x["toe_flight_counter"] = (x["status"] >> 23) & 31
 if prev is None or phase!=prev:transitions.append({"step":x["step"],"phase":phase});prev=phase
counts=collections.Counter();loaded=[]
for x in r:
 if x["left_load"]>.05 and x["right_load"]>.05:
  loaded.append(x)
  if abs(x["goal_x_error"]-.012)>=.020:counts["x"]+=1
  if abs(x["goal_y_error"])>=.16:counts["y"]+=1
  if abs(x["vertical_speed"])>=.12:counts["vertical_speed"]+=1
  if abs(x["pitch"])>=.10:counts["pitch"]+=1
  if abs(x["pitch_rate"])>=.10:counts["pitch_rate"]+=1
out={"lane":5,"observed_states":len(r),"phase_transitions":transitions,"sequence_samples_every_25":[x for x in r if x["step"]%25==0],"both_loads_above_threshold":len(loaded),"condition_failure_counts":dict(counts),"both_load_samples":loaded[:20],"duplicate_phase_samples":dups,"scope":"Task-state snapshot after command-buffer completion, not assumed aligned with individual substep contact or replay until parity and indexing are separately checked. Horizontal speed is not in this state snapshot; it remains an additional gate. Observation does not itself qualify landing."}
(p/"phase-analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k!="both_load_samples"},indent=2))
