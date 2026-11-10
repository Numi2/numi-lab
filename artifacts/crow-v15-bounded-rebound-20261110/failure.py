from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-bounded-rebound-20261110');D={'first_opposed':None,'longest_opposed':0,'opposed_samples':0,'nonfoot':[],'samples':{},'first_phase2':None,'flare_samples':0};run=0;previous=0
with gzip.open(A/'development-j-bounded-rebound.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_TASK_STATE '):
   p=line.split();t=int(p[2]);
   if t<previous:break
   previous=t;phase=float(p[7]);D['flare_samples']+=1.2<=phase<1.5
   if phase>=1.5 and D['first_phase2'] is None:D['first_phase2']=t
  elif line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);t=x['step'];op={s:{k:0 for k in ('front','rear')} for s in ('left','right')}
   for c in x['contacts']:
    a,b=c['a_name'],c['b_name'];n=a if b=='crow_course_perch' else b if a=='crow_course_perch' else None
    if not n or c['impulses'][0]<=1e-8:continue
    side=next((s for s in op if '_'+s+'_' in n),None);sgn=-1 if n==a else 1;nx=-sgn*c['normal'][0]
    if side and 'front_toe' in n and nx<-.5:op[side]['front']+=c['impulses'][0]
    if side and 'rear_toe' in n and nx>.5:op[side]['rear']+=c['impulses'][0]
    if 'toe' not in n and 'foot' not in n:D['nonfoot'].append({'step':t,'body':n,'impulse_Ns':c['impulses'][0]})
   ok=all(op[s][k]>.00005 for s in op for k in ('front','rear'));run=run+1 if ok else 0;D['opposed_samples']+=ok;D['longest_opposed']=max(run,D['longest_opposed'])
   if ok and D['first_opposed'] is None:D['first_opposed']=t
   if t in (3800,3900,4000,4040,4050,4054,4055):D['samples'][str(t)]={'position':x['q'][:3],'orientation_xyzw':x['q'][3:7],'velocity':x['v'][:6]}
(A/'failure-analysis.json').write_text(json.dumps(D,indent=2)+'\n');print(json.dumps(D,indent=2))
