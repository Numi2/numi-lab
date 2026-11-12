from pathlib import Path
import collections,gzip,json,math,os,statistics,subprocess,sys
A=Path("artifacts/crow-v15-knee-warp-contact-timing-audit-20261112");arm=sys.argv[1];assert arm in ("baseline","phase");D=A/("candidate-20261127-rebound-"+arm+"-repaired");Q={};C=collections.defaultdict(list)
with gzip.open(D/"replay.json.gz","rt") as f:names=json.load(f)["payload"]["body_names"]
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=="CROW_SYNC_KIN":
   t,k,e=map(int,z[1:4]);assert e==4 and len(z)==4+41+40
   key=(t-201)*16+k;assert key not in Q;Q[key]=list(map(float,z[4:]))
  elif z[0]=="CROW_SYNC_CONTACT":
   t,k,e,a,b=map(int,z[1:6]);key=(t-201)*16+k;C[key].append(dict(body=a if a<35 else b,impulse=float(z[6]),point=list(map(float,z[7:10])),normal=list(map(float,z[10:13]))))
assert sorted(Q)==list(range(100*16)),(len(Q),min(Q),max(Q))
payload="".join("4 "+str(k)+" "+" ".join(map(str,v))+"\n" for k,v in sorted(Q.items()))
env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())}
r=subprocess.run(["build/crow-v15-gait-slip-observer-20261112/kinematics"],input=payload,text=True,capture_output=True,check=True,env=env)
bodies={}
for l in r.stdout.splitlines():
 z=l.split()
 if z and z[0]=="body":bodies[int(z[2]),z[3]]=list(map(float,z[4:]))
assert len(bodies)==len(Q)*35,(len(bodies),len(Q))
checks={}
for i in (0,11,15,19,20,27,28):
 body=names[i];err=[];speed=[]
 for k in range(1,len(Q)-1):
  a=bodies[k-1,body];b=bodies[k,body];c=bodies[k+1,body];fd=[(c[j]-a[j])/.0025 for j in range(3)];err.append(math.sqrt(sum((fd[j]-b[7+j])**2 for j in range(3))));speed.append(math.sqrt(sum(b[7+j]**2 for j in range(3))))
 checks[body]={"mean_difference_m_s":statistics.mean(err),"median_difference_m_s":statistics.median(err),"p90_difference_m_s":sorted(err)[int(.9*(len(err)-1))],"mean_reported_speed_m_s":statistics.mean(speed)}
samples=[]
for k,contacts in sorted(C.items()):
 for c in contacts:
  i=c["body"];side="left" if i==11 or 19<=i<=26 else "right" if i==15 or 27<=i<=34 else None
  if not side:continue
  b=bodies[k,names[i]];p=c["point"];d=[p[j]-b[j] for j in range(3)];w=b[10:13];u=[b[7]+w[1]*d[2]-w[2]*d[1],b[8]+w[2]*d[0]-w[0]*d[2],b[9]+w[0]*d[1]-w[1]*d[0]];n=c["normal"];nn=math.sqrt(sum(x*x for x in n));dot=sum(u[j]*n[j]/nn for j in range(3));tan=math.sqrt(max(0,sum(x*x for x in u)-dot*dot));samples.append({"index":k,"side":side,"body":names[i],"normal_impulse_ns":c["impulse"],"tangential_speed_m_s":tan})
def summary(xs):
 total=sum(x["normal_impulse_ns"] for x in xs);ys=sorted(xs,key=lambda x:x["tangential_speed_m_s"])
 def percentile(q):
  a=0
  for x in ys:
   a+=x["normal_impulse_ns"]
   if a>=q*total:return x["tangential_speed_m_s"]
 return {"samples":len(xs),"normal_impulse_ns":total,"mean_m_s":sum(x["normal_impulse_ns"]*x["tangential_speed_m_s"] for x in xs)/total,"median_m_s":percentile(.5),"p90_m_s":percentile(.9),"p99_m_s":percentile(.99),"impulse_fraction_above_0_05_m_s":sum(x["normal_impulse_ns"] for x in xs if x["tangential_speed_m_s"]>.05)/total}
out={"synchronized_samples":len(Q),"substep_seconds":.00125,"kinematic_velocity_validation":checks,"total":summary(samples),"left":summary([x for x in samples if x["side"]=="left"]),"right":summary([x for x in samples if x["side"]=="right"]),"samples":samples,"scope":"Two-second development walking window in lane 4. Candidate point and post-substep body state have solver/integration timing differences. Contact-normal impulse weights. No biological or independent-start qualification."}
(A/("analysis-"+arm+".json")).write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k!="samples"},indent=2))
