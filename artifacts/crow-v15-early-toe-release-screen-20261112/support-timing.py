from pathlib import Path
import gzip,json,collections
A=Path("artifacts/crow-v15-early-toe-release-screen-20261112");rows=[]
for seed in (20261126,20261127):
 d=A/("candidate-"+str(seed)+"-rebound");targets={};sub=collections.defaultdict(lambda:[0.,0.])
 with gzip.open(d/"run.stderr.gz","rt") as f:
  for line in f:
   z=line.split()
   if z[0]=="CROW_DRIVE_TARGET":
    t,e=map(int,z[1:3])
    if 201<=t<=300:targets[e,t]=[float(z[3+28]),float(z[3+36])]
   elif z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4])
    if 201<=t<=300:
     sub[e,t][0]+=float(z[8]);sub[e,t][1]+=float(z[9])
 assert len(targets)==800 and len(sub)==800
 for e in range(8):
  x={"seed":seed,"lane":e,"sides":[]}
  for side in range(2):
   active=[t for t in range(201,301) if targets[e,t][side]<-1e-8];loaded=[t for t in active if sub[e,t][side]>1e-7];half=[t for t in active if sub[e,t][side]>.5*.45006*9.81*.02];x["sides"].append({"active_intervals":len(active),"intervals_with_any_summed_contact":len(loaded),"intervals_with_half_weight_impulse":len(half),"loaded_intervals":loaded})
  rows.append(x)
out={"rows":rows,"scope":"Descriptive overlap between recorded control-interval toe target and summed foot normal impulse from all 16 substeps. Any-overlap threshold is a numerical instrumentation threshold, not true support classification. Does not expose the prior accepted status bit, actual torque, continuous kinematics or real-crow data."};(A/"support-timing.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(rows,indent=2))
