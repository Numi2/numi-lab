from pathlib import Path
import collections,gzip,json,math,statistics
A=Path("artifacts/crow-v15-jackdaw-whole-gait-audit-20261112"); B=Path("artifacts/crow-v15-sustained-crouch-replication-20261112")
weight=.088301772; threshold=.05*weight
out={}
for seed in (20261126,20261127):
 D=B/(f"candidate-{seed}-rebound-hip"); states={}; impulses=collections.defaultdict(lambda:[0.,0.,0])
 with gzip.open(D/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]); t=x["step"]; e=x["lane"]
    if 201<=t<=500:states[e,t]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 201<=t<=500:
     impulses[e,t][0]+=float(z[8]);impulses[e,t][1]+=float(z[9]);impulses[e,t][2]+=1
 rows=[]
 for e in range(8):
  groups=collections.defaultdict(list)
  for t in range(201,501):
   assert (e,t) in states and impulses[e,t][2]==16
   x=states[e,t];q=x["q"];l,r,_=impulses[e,t];support="both" if l>=threshold and r>=threshold else "left" if l>=threshold else "right" if r>=threshold else "neither"
   a,b,c,d=q[3:7];roll=math.atan2(2*(d*a+b*c),1-2*(a*a+b*b));pitch=math.asin(max(-1,min(1,2*(d*b-c*a))));yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c))
   groups[support].append([q[39],q[40],roll,pitch,yaw])
  result={}
  for key,values in groups.items():
   result[key]={"count":len(values),"median_rad":[statistics.median(x[i] for x in values) for i in range(5)],"range_rad":[max(x[i] for x in values)-min(x[i] for x in values) for i in range(5)]}
  rows.append({"lane":e,"support":result})
 out[str(seed)]=rows
result={"columns":["left_hip_roll","right_hip_roll","root_roll","root_pitch","root_yaw"],"interval_s":.02,"foot_threshold_Ns":threshold,"seeds":out,"limitations":"Descriptive sampled endpoint angles only; left/right support classes can include changing contact within an interval. No numeric jackdaw or crow target, whole-body COM, physical slip, held-out replication, or biological validation."}
(A/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
for seed,rows in out.items():
 for r in rows:
  print(seed,r["lane"],{k:{"n":v["count"],"median": [round(n,4) for n in v["median_rad"]],"range": [round(n,4) for n in v["range_rad"]]} for k,v in r["support"].items()})
