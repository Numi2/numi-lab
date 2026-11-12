from pathlib import Path
import gzip,json,collections
p=Path("artifacts/crow-v15-walk-hip-phase-screen-20261112");src=Path("artifacts/crow-v15-own-support-hallux-release-screen-20261112")
paths={"held":Path("artifacts/crow-v15-small-support-stroke-20261112/sham-20261127-paired"),"opposite_support_sham":src/"sham-20261127-paired","own_or_opposite_candidate":p/"candidate-20261127-rebound"};eps=1e-7
out={}
for label,path in paths.items():
 data={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300:
     v=list(map(float,z[6:]));assert len(v)==37;assert (e,t,k) not in data;data[e,t,k]=v
 assert len(data)==12800
 counts=collections.Counter();events=[];missing=[]
 for e in range(8):
  for side in range(2):
   samples=[]
   for t in range(201,301):
    for k in range(16):
     z=data[e,t,k];toes=z[5+side*8:5+side*8+8];front=sum(toes[:6]);rear=sum(toes[6:]);total=z[2+side];assert front>=0 and rear>=0 and total>=0
     samples.append({"t":t,"k":k,"foot":total>eps,"front":front>eps,"rear":rear>eps})
   # A contact stance is a maximal foot-contact run allowing up to two silent substeps.
   active=[i for i,x in enumerate(samples) if x["foot"]];groups=[]
   for i in active:
    if not groups or i-groups[-1][-1]>3:groups.append([i])
    else:groups[-1].append(i)
   for g in groups:
    start,end=g[0],g[-1]
    if start==0 or end==len(samples)-1:counts["partial_runs"]+=1;continue
    front=[i for i in range(start,end+1) if samples[i]["front"]];rear=[i for i in range(start,end+1) if samples[i]["rear"]]
    if not front or not rear:
     counts["runs_without_both_toe_groups"]+=1;missing.append({"lane":e,"side":side,"missing_front":not bool(front),"missing_rear":not bool(rear)});continue
    first=rear[0]-front[0];last=rear[-1]-front[-1]
    touchdown="rear_first" if first<0 else "front_first" if first>0 else "same_substep"
    toeoff="rear_first" if last<0 else "front_first" if last>0 else "same_substep"
    counts["complete_runs_with_both_toe_groups"]+=1;counts["touchdown_"+touchdown]+=1;counts["toeoff_"+toeoff]+=1
    events.append({"lane":e,"side":"left" if side==0 else "right","first_step":samples[start]["t"],"first_substep":samples[start]["k"],"last_step":samples[end]["t"],"last_substep":samples[end]["k"],"rear_minus_front_first_substeps":first,"rear_minus_front_last_substeps":last,"touchdown":touchdown,"toeoff":toeoff})
 out[label]={"counts":dict(counts),"events":events,"missing":missing}
out["method"]="Foot normal >1e-7 at 1.25 ms substeps, foot-contact runs bridge at most two silent substeps. In complete runs, compare first and last positive normal impulses for rear-toe two segments and front-toe six segments. Equal substep means unresolved order, not simultaneous biological contact. Group sums can mix toe segments and do not establish plantar identity or biological timing; thresholds and gap merging are engineering choices. Used development seed and 4–6 s only."
out["corvid_reference"]={"species":"black-billed magpie, Pica pica", "doi":"10.1242/jeb.203.14.2159", "observation":"Hallux contact precedes front toe contact; hallux leaves before the third digit in filmed steps on thin cork; distal toe flexes after leaving the ground and extends later in recovery. Surface and species limits apply. No American-crow angle or timing transfer is asserted."}
(p/"contact-order.json").write_text(json.dumps(out,indent=2)+"\n")
for k in paths:print(k,out[k]["counts"])
