from pathlib import Path
import gzip,json,collections
p=Path("artifacts/crow-v15-earlier-hallux-release-screen-20261112");events=json.loads((p/"target-timing.json").read_text())["earlier_phase_candidate"]["events"];states={};targets={}
with gzip.open(p/"candidate-20261127-rebound/run.stderr.gz","rt") as f:
 for l in f:
  z=l.split()
  if z and z[0]=="CROW_PHASE_SAMPLE":states[int(z[2]),int(z[1])]=int(z[3])
  elif z and z[0]=="CROW_DRIVE_TARGET":targets[int(z[2]),int(z[1])]=list(map(float,z[3:]))
out=[];counts=collections.Counter()
for v in events:
 e,side=v["lane"],v["side"];first=v["stance_first"][0];last=v["last_front"][0]; own=20+side;opp=21-side
 # The prior step flag is the nearest recorded state before this control interval.
 vals=[(t,bool(states[e,t-1]&(1<<opp)),targets[e,t][28+8*side]) for t in range(first,last+1) if (e,t-1) in states]
 eligible=[t for t,s,target in vals if s]; commanded=[t for t,s,target in vals if target<-.001]
 flags_before=sum(1 for t,s,target in vals if t<last and s)
 counts["events"]+=1;counts["no_previous_opposite_support_before_last_interval" if not flags_before else "previous_opposite_support_before_last_interval"]+=1
 counts["prior_opposite_support_at_last"]+=int(bool(states[e,last-1]&(1<<opp)))
 counts["prior_opposite_support_1_before_last"]+=int(bool(states[e,last-2]&(1<<opp)))
 counts["prior_opposite_support_2_before_last"]+=int(bool(states[e,last-3]&(1<<opp)))
 counts["command_without_previous_opposite_support"]+=sum(1 for t,s,target in vals if target<-.001 and not s)
 out.append({"lane":e,"side":side,"front_last_step":last,"first_prior_opposite_flag_step":min(eligible) if eligible else None,"first_captured_release_target_step":min(commanded) if commanded else None,"prior_opposite_support_intervals_before_last":flags_before})
result={"counts":dict(counts),"events":out,"limits":"Flags are copied after the preceding submitted control interval; command buffers and interpolated targets have their own timing. The comparison is observational, not a proof of the exact gate evaluation or of why release is late."};(p/"gate-timing.json").write_text(json.dumps(result,indent=2)+"\n");print(result["counts"])
