from pathlib import Path
import collections,gzip,json,numpy as np
A=Path('artifacts/crow-v15-v1-load-relieved-landing-20261111');B=Path('artifacts/crow-v15-v1-deeper-touchdown-20261111');out=json.loads((A/'analysis.json').read_text());first=out['first_episode'];episodeEnd=min(first['steps_by_environment'][0]+1,3000) if first['physical_failure_by_environment'][0] else 3000;states={};ref={};sub=collections.defaultdict(lambda:[0.0]*21)
def samples(p):
 d={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);d[x['step']]=x
 return d
states=samples(A/'route.stderr.gz');ref=samples(B/'route.stderr.gz');diff=[t for t in range(1,2420) if t not in states or t not in ref or any(states[t][k]!=ref[t][k] for k in ('q','v'))]
with gzip.open(A/'route.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_PERCH_SUBSTEP '):
   x=l.split();t=int(x[1]);lane=int(x[3])
   if lane==0 and t<=episodeEnd:
    for i,v in enumerate(map(float,x[6:])):sub[t][i]+=v
opposed=[]
for t,v in sub.items():
 support=min(v[0],v[1])>.0002 and v[0]+v[1]>=.5*.45004*9.81*.02 and v[4]<=1e-7
 if support and all(sum(v[13+b+k] for k in (0,1,2))>=.0002 and v[13+b+3]<=-.0002 for b in (0,4)):opposed.append(t)
with gzip.open(A/'route.json.gz','rt') as f:r=json.load(f)['payload']; frames={x['step']:x for x in r['frames']}
names=r['body_names'];toe=[n for n in names if '_toe' in n];assert len(toe)==8;stride=r['body_state_stride'];perch=names.index('crow_course_perch')
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(a,b):
 P=rot(a[3:7]);Q=rot(b[3:7]);delta=np.array(b[:3])-np.array(a[:3]);axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];g=[]
 for axis in axes:
  n=np.linalg.norm(axis)
  if n>1e-8:
   u=axis/n;g.append(abs(delta@u)-sum(abs(P.T@u)*[.018,.0025,.0092])-sum(abs(Q.T@u)*[.025,.72,.055]))
 return float(max(g))
measure=[];maxq=0.;missing=[]
for t in sorted(opposed):
 pre=frames.get(t-1);end=frames.get(t);sample=states.get(t)
 if pre is None or end is None or sample is None:missing.append(t);continue
 maxq=max(maxq,max(abs(float(x)-float(y)) for x,y in zip(pre['q'],sample['q'])));b=end['body_states'];pp=b[perch*stride:perch*stride+7];gaps={n:gap(b[names.index(n)*stride:names.index(n)*stride+7],pp) for n in toe};measure.append({'step':t,'worst_gap_m':min(gaps.values()),'toe_gaps_m':gaps})
worst=min((v['worst_gap_m'] for v in measure),default=None);res={'pre_first_foot_contact_q_v_difference_count':len(diff),'first_q_v_difference':diff[:1],'opposed_interval_count':len(opposed),'missing_replay_intervals':missing,'max_prestate_q_error':maxq,'worst_opposed_endpoint_gap_m':worst,'opposed_endpoint_overlap_within_0_5mm':None if worst is None else worst>=-.0005,'per_step':measure,'limitations':'Endpoint oriented-box geometry only; no continuous collision separation, contact-point slip, or complete release and regrip evidence.'};(A/'geometry.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='per_step'},indent=2))
