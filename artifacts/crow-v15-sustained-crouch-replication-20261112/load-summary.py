from pathlib import Path
import collections,gzip,json,statistics
A=Path("artifacts/crow-v15-sustained-crouch-replication-20261112");res={}
for seed in (20261126,20261127):
 d=A/("candidate-"+str(seed)+"-rebound-hip");v=collections.defaultdict(lambda:[0.,0.,0]);sub=collections.Counter()
 with gzip.open(d/"run.stderr.gz","rt") as f:
  for line in f:
   z=line.split()
   if z and z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4])
    if 201<=t<=500:
     x=v[e,t];x[0]+=float(z[8]);x[1]+=float(z[9]);x[2]+=int(float(z[8])+float(z[9])>1e-7);sub[e,t]+=1
 rows=[];weight=.45006*9.81*.02
 for e in range(8):
  assert all(sub[e,t]==16 for t in range(201,501)); loads=[sum(v[e,t][:2]) for t in range(201,501)];left=[v[e,t][0] for t in range(201,501)];right=[v[e,t][1] for t in range(201,501)]
  rows.append({"lane":e,"intervals":len(loads),"weight_impulse_ns":weight,"total_normal_impulse_ns":sum(loads),"mean_force_over_bodyweight":sum(loads)/(300*weight),"intervals_below_half_weight_impulse":sum(x<.5*weight for x in loads),"intervals_below_tenth_weight_impulse":sum(x<.1*weight for x in loads),"both_feet_loaded_intervals":sum(l>1e-7 and r>1e-7 for l,r in zip(left,right)),"left_only_loaded_intervals":sum(l>1e-7 and r<=1e-7 for l,r in zip(left,right)),"right_only_loaded_intervals":sum(r>1e-7 and l<=1e-7 for l,r in zip(left,right)),"minimum_interval_impulse_ns":min(loads)})
 res[str(seed)]=rows
r={"seeds":res,"scope":"Ground contact normal impulse integrated across 20 ms; support thresholds are engineering diagnostics and do not measure material-point slip or biomechanical realism."};(A/"load-summary.json").write_text(json.dumps(r,indent=2)+"\n")
for seed,rows in res.items():
 print(seed)
 for x in rows:print(x)
