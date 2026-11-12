from pathlib import Path
import gzip,json,collections,statistics
P=Path("artifacts/crow-v15-corvid-front-support-screen-20261112")
base=Path("artifacts/crow-v15-own-support-hallux-release-screen-20261112")
paths={"opposite_support_sham":base/"sham-20261127-paired", "own_or_opposite_candidate":P/"candidate-20261127-rebound"}
result={}
for label,path in paths.items():
 data={};targets={};state={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   z=line.split();
   if not z:continue
   if z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4]);v=list(map(float,z[6:]));
    if 201<=t<=300:data[e,t,k]=v
   elif z[0]=="CROW_DRIVE_TARGET":
    t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
   elif z[0]=="CROW_PHASE_SAMPLE":
    t,e=map(int,z[1:3]);state[e,t]=int(z[3])
 assert len(data)==12800 and len(targets)==800 and len(state)>=800
 counts=collections.Counter(); events=[]
 for e in range(8):
  for side in range(2):
   samples=[]
   for t in range(201,301):
    for k in range(16):
     v=data[e,t,k];toes=v[5+8*side:13+8*side]
     samples.append((t,k,v[2+side]>1e-7,sum(toes[:6])>1e-7,sum(toes[6:])>1e-7))
   active=[i for i,x in enumerate(samples) if x[2]];groups=[]
   for i in active:
    if not groups or i-groups[-1][-1]>3:groups.append([i])
    else:groups[-1].append(i)
   for g in groups:
    first,last=g[0],g[-1]
    if first==0 or last==len(samples)-1:continue
    front=[i for i in range(first,last+1) if samples[i][3]]; rear=[i for i in range(first,last+1) if samples[i][4]]
    if not front or not rear:continue
    ft,rt=front[-1],rear[-1];counts["complete_stances"]+=1
    # A proximal motor target below -0.001 rad is a directly observed release command.
    qualifying=[i for i in range(first,ft+1) if targets[e,samples[i][0]][28+side*8] < -.001]
    onset=qualifying[0] if qualifying else None
    counts["target_before_or_at_front_last" if onset is not None else "no_target_before_front_last"]+=1
    counts["rear_last_after_front_last" if rt>ft else "rear_last_before_front_last" if rt<ft else "last_same_substep"]+=1
    st=samples[ft][0];tgt=targets[e,st][28+side*8];flags=state[e,st]
    events.append({"lane":e,"side":side,"stance_first":samples[first][:2],"last_front":samples[ft][:2],"last_rear":samples[rt][:2],"rear_minus_front_ms":(rt-ft)*1.25,"first_command_before_front":None if onset is None else samples[onset][:2],"command_lead_ms":None if onset is None else (ft-onset)*1.25,"target_at_front_last_rad":tgt,"post_step_own_support_flag":bool(flags & (1<<(20+side))),"post_step_opposite_support_flag":bool(flags & (1<<(21-side)))})
 result[label]={"counts":dict(counts),"events":events}
result["method"]="Actual substep-zero captured proximal targets for rear toe (dof 28 left, 36 right), threshold -0.001 rad, joined to measured contact groups at 1.25 ms. Foot-normal contact >1e-7; bridge at most two silent substeps; only complete stances with both front/rear toe contact. Target is held/interpolated by the actual actuator system; target does not prove motor followed. Commands on same control interval as last contact are not necessarily applied before that substep. Status flags are sampled after the interval and must not be treated as causal pre-step gate values. Stance detection and contact thresholds are engineering choices; toe anatomical identity remains unverified."
(P/"target-timing.json").write_text(json.dumps(result,indent=2)+"\n")
for k in paths:
 ev=result[k]["events"];lead=[x["command_lead_ms"] for x in ev if x["command_lead_ms"] is not None];print(k,result[k]["counts"],"lead median/range",(statistics.median(lead),min(lead),max(lead)) if lead else None)
