from pathlib import Path
import gzip,json,math,collections
A=Path('artifacts/crow-v15-gait-target-observer-20261112');N=A/'autonomous-20261126'/'run.stderr.gz';R=Path('artifacts/crow-v15-substep-gait-observer-20261111/autonomous-20261126/run.stderr.gz')
def load(path):
 q={};c={};d={}
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_Q_SUBSTEP '):
    x=l.split();t,k,e=map(int,x[1:4]);key=e,t,k;assert key not in q;q[key]=x[4:]
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    x=l.split();t,k,e=map(int,x[1:4])
    if 201<=t<=300:
     key=e,t,k;assert key not in c;c[key]=x[4:]
   elif l.startswith('CROW_DRIVE_TARGET '):
    x=l.split();t,e=map(int,x[1:3]);key=e,t;assert key not in d;d[key]=list(map(float,x[3:]));assert len(d[key])==40
 return q,c,d
q,c,d=load(N);rq,rc,_=load(R);assert len(q)==len(c)==len(rq)==len(rc)==12800;assert len(d)==800
q_mismatch=[k for k in q if q[k]!=rq.get(k)];c_mismatch=[k for k in c if c[k]!=rc.get(k)];parity={'q_samples':len(q),'contact_samples':len(c),'target_samples':len(d),'q_mismatch_count':len(q_mismatch),'contact_mismatch_count':len(c_mismatch),'q_examples':q_mismatch[:8],'contact_examples':c_mismatch[:8]}
rows=[]
def correlation(x,y):
 xm=sum(x)/len(x);ym=sum(y)/len(y);v=sum((a-xm)**2 for a in x)*sum((b-ym)**2 for b in y);return sum((a-xm)*(b-ym) for a,b in zip(x,y))/math.sqrt(v) if v>0 else None
for side,qi,vi,joint in [('left',15,14,'knee'),('left',16,15,'ankle'),('right',18,17,'knee'),('right',19,18,'ankle')]:
 for e in range(8):
  xs=[];ys=[]
  for t in range(202,301):
   delta=d[e,t][vi]-d[e,t-1][vi];a,b,z=[float(q[e,u,k][qi]) for u,k in ((t,0),(t-1,15),(t-1,14))];acc=(a-2*b+z)/(.00125**2);xs.append(delta);ys.append(acc)
  rows.append({'side':side,'joint':joint,'lane':e,'n':len(xs),'signed_pearson':correlation(xs,ys),'mean_abs_target_step_rad':sum(map(abs,xs))/len(xs),'mean_abs_q_second_difference_rad_s2':sum(map(abs,ys))/len(ys)})
out={'parity':parity,'rows':rows,'limitations':'One used development seed and 2 s window. Logged target is sampled after substep zero; time alignment and active solver input path require separate verification. Correlation does not prove causality. Q second difference is a motion proxy, not torque or biological stiffness.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('parity',parity)
for r in rows:print(r)
if q_mismatch or c_mismatch:raise SystemExit('observer parity failed')
