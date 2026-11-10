from pathlib import Path
import gzip,json,math,statistics
A=Path('artifacts/crow-v15-front-grip-floor-20261110');results={}
for side,t in [('j',4433),('i',5565)]:
 good=[];bad=[]
 with gzip.open('artifacts/crow-v15-student-landing-eval-20261110/'+side+'-student-landing.stderr.gz','rt') as f:
  for l in f:
   if not l.startswith('CROW_SUPPORT_SAMPLE '):continue
   z=json.loads(l.split(' ',1)[1]);s=z['step']
   if not(t-100<=s<t):continue
   d={k:{'front':0.,'rear':0.} for k in ('left','right')}
   for c in z['contacts']:
    a,b=c['a_name'],c['b_name'];body=a if b=='crow_course_perch' else b if a=='crow_course_perch' else None;who=next((v for v in d if body and '_'+v+'_' in body),None);nx=c['normal'][0]*(1 if body==a else -1)
    if who and 'front_toe' in body and nx<-.5:d[who]['front']+=c['impulses'][0]
    if who and 'rear_toe' in body and nx>.5:d[who]['rear']+=c['impulses'][0]
   x,y,q,w=z['q'][3:7];pitch=math.asin(max(-1,min(1,2*(w*y-q*x))));v=z['v'];tar=1.12-.8*pitch-.8*v[4]-2*v[0];(good if all(d[k][h]>.00005 for k in d for h in ('front','rear')) else bad).append(tar)
 results[side]={'opposed_samples':len(good),'opposed_mean_unclamped_target':statistics.mean(good),'not_opposed_samples':len(bad),'not_opposed_mean_unclamped_target':statistics.mean(bad) if bad else None}
(A/'correlation.json').write_text(json.dumps({'samples':results,'limitations':'Endpoint correlations using world-frame velocity approximation; not causal and not biological validation.'},indent=2)+'\n');print(json.dumps(results,indent=2))
