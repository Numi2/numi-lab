from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-clearance-corrective-eval-20261110');name=sys.argv[1];release=json.loads((A/(name+'-release-audit.json')).read_text())['first_release_step'];run=longest=count=0;seen=[]
with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
 for line in f:
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(line.split(' ',1)[1]);t=x['step']
  if t<release-100 or t>=release:continue
  d={s:{'front':0.,'rear':0.} for s in ('left','right')}
  for c in x['contacts']:
   a,b=c['a_name'],c['b_name'];body=a if b=='crow_course_perch' else b if a=='crow_course_perch' else None
   side=next((z for z in d if body and '_'+z+'_' in body),None);sgn=-1 if body==a else 1;nx=-sgn*c['normal'][0]
   if side and 'front_toe' in body and nx<-.5:d[side]['front']+=c['impulses'][0]
   if side and 'rear_toe' in body and nx>.5:d[side]['rear']+=c['impulses'][0]
  ok=all(d[z][k]>.00005 for z in d for k in ('front','rear'));count+=ok;run=run+1 if ok and (not seen or seen[-1]==t-1) else int(ok);longest=max(longest,run);seen.append(t)
out={'name':name,'samples':len(seen),'qualifying_opposed_samples':count,'longest_consecutive':longest,'reference_gate':9,'limits':'Accepted endpoint impulses, not continuous substep force, biological qualification, or perturbation robustness. Nine-sample gate is the previous isolated toe grip criterion.'}
(A/(name+'-grip-runs.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
