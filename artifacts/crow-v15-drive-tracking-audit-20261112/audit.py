from pathlib import Path
import gzip,json,statistics,sys
root=Path("/Users/n/Developer/numi-crow-rl-20261107/artifacts")
seed=int(sys.argv[1]); out=root/"crow-v15-drive-tracking-audit-20261112"
def load(path):
 targets={};q={};bits={}
 with gzip.open(path,"rt") as f:
  for line in f:
   z=line.split()
   if not z: continue
   if z[0]=="CROW_DRIVE_TARGET":
    t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
   elif z[0]=="CROW_Q_SUBSTEP":
    t,k,e=map(int,z[1:4])
    if k in (0,15):q[e,t,k]=list(map(float,z[4:]))
   elif z[0]=="CROW_SENSOR_BITS":
    t,e=map(int,z[1:3]);bits[e,t]=int(z[3])
 assert len(targets)==800 and len(q)==1600
 return targets,q,bits
base=load(root/f"crow-v15-negative-ankle-matched-control-20261112/candidate-{seed}-rebound-phase/run.stderr.gz")
trial=load(root/f"crow-v15-asymmetric-foot-settling-20261112/candidate-{seed}-rebound-phase/run.stderr.gz")
def quantile(x,p):
 x=sorted(x); i=(len(x)-1)*p; a=int(i); return x[a]+(x[min(a+1,len(x)-1)]-x[a])*(i-a)
rows=[]
for e in range(4,8):
 for side,(knee,ankle) in enumerate(((14,15),(17,18))):
  for joint,j in (("knee",knee),("ankle",ankle)):
   for label,(targets,q,bits) in (("baseline",base),("candidate",trial)):
    errors=[q[e,t,15][j+1]-targets[e,t][j] for t in range(201,301)]
    begin=[q[e,t,0][j+1]-targets[e,t][j] for t in range(201,301)]
    rows.append(dict(lane=e,side=["left","right"][side],joint=joint,trial=label,median_abs_error_rad=statistics.median(map(abs,errors)),p95_abs_error_rad=quantile(list(map(abs,errors)),.95),median_signed_error_rad=statistics.median(errors),median_start_abs_error_rad=statistics.median(map(abs,begin))))
(out/f"tracking-{seed}.json").write_text(json.dumps({"seed":seed,"rows":rows,"scope":"Recorded q at final substep minus target logged at first substep of the same control step; target constancy within that interval is unverified. q index is target velocity index + 1 for free-root 7-coordinate vs 6-velocity layout. One hundred samples per lane; short correlated development trial, not independent biological validation or proof of torque saturation."},indent=2)+"\n")
for r in rows: print(seed,r["lane"],r["side"],r["joint"],r["trial"],"median",round(r["median_abs_error_rad"],4),"p95",round(r["p95_abs_error_rad"],4))
