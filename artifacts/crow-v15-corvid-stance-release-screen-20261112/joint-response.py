from pathlib import Path
import gzip,json,statistics
p=Path("artifacts/crow-v15-corvid-stance-release-screen-20261112");ev=json.loads((p/"target-timing.json").read_text())["own_or_opposite_candidate"]["events"]
def load(name):
 q={};target={}
 with gzip.open((p/name if name.startswith("candidate") else Path("artifacts/crow-v15-own-support-hallux-release-screen-20261112")/"candidate-20261127-rebound")/"run.stderr.gz","rt") as f:
  for l in f:
   z=l.split()
   if not z:continue
   if z[0]=="CROW_Q_SUBSTEP":
    t,k,e=map(int,z[1:4]);q[e,t,k]=list(map(float,z[4:]));assert len(q[e,t,k])==41
   elif z[0]=="CROW_DRIVE_TARGET":
    t,e=map(int,z[1:3]);target[e,t]=list(map(float,z[3:]))
 return q,target
cq,ct=load("candidate-20261127-rebound");sq,st=load("sham-20261127-paired");rows=[]
for x in ev:
 e,side=x["lane"],x["side"];t,k=x["last_front"];qi=29+side*8;di=28+side*8
 rows.append({"lane":e,"side":side,"step":t,"substep":k,"candidate_proximal_q_rad":cq[e,t,k][qi],"sham_proximal_q_rad":sq[e,t,k][qi],"candidate_minus_sham_q_rad":cq[e,t,k][qi]-sq[e,t,k][qi],"candidate_distal_q_rad":cq[e,t,k][qi+1],"sham_distal_q_rad":sq[e,t,k][qi+1],"candidate_target_at_interval_start_rad":ct[e,t][di],"sham_target_at_interval_start_rad":st[e,t][di]})
def stats(k):
 x=[r[k] for r in rows];return {"min":min(x),"median":statistics.median(x),"max":max(x)}
o={"count":len(rows),"candidate_q":stats("candidate_proximal_q_rad"),"sham_q":stats("sham_proximal_q_rad"),"candidate_minus_sham_q":stats("candidate_minus_sham_q_rad"),"candidate_target":stats("candidate_target_at_interval_start_rad"),"rows":rows,"limits":"Substep joint angles sampled at candidate last-front-contact, matched by lane and simulated time to the previous -0.02-rad own-support run; trajectories may have diverged. q index equals DOF index plus one because free-root position has seven coordinates. Target captured at substep zero may interpolate; not a force or tracking-error isolation. Toe identity remains unverified."};(p/"joint-response.json").write_text(json.dumps(o,indent=2)+"\n");print({k:o[k] for k in ("count","candidate_q","sham_q","candidate_minus_sham_q","candidate_target")})
