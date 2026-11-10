from pathlib import Path
import gzip,json,statistics,sys
A=Path('artifacts/crow-v15-alternating-foot-release-20261110');name=sys.argv[1];p=A/(name+'.stderr.gz');state={};support={};last=-1;release=None
with gzip.open(p,'rt') as f:
 for l in f:
  if l.startswith('CROW_TASK_STATE '):
   v=l.split();t=int(v[2]);
   if t<last:break
   last=t;state[t]={'nav':float(v[4]),'left_load_n':float(v[5]),'right_load_n':float(v[6]),'phase':float(v[7])}
   if float(v[7])>=2.5 and release is None:release=t
  elif l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);support[x['step']]=x
if release is None:
 out={'first_release_step':None,'result':'no contact-conditioned release observed'}
else:
 def summarize(lo,hi):
  values=[(t,x) for t,x in support.items() if lo<=t<=hi];front=rear=op=0;maxangle=[]
  for t,x in values:
   d={side:{'front':0.,'rear':0.} for side in ('left','right')}
   for c in x['contacts']:
    a,b=c['a_name'],c['b_name'];body=a if b=='crow_course_perch' else b if a=='crow_course_perch' else None
    side=next((z for z in d if body and '_'+z+'_' in body),None);sgn=-1 if body==a else 1;nx=-sgn*c['normal'][0]
    if side and 'front_toe' in body and nx<-.5:d[side]['front']+=c['impulses'][0]
    if side and 'rear_toe' in body and nx>.5:d[side]['rear']+=c['impulses'][0]
   front+=all(d[z]['front']>.00005 for z in d);rear+=all(d[z]['rear']>.00005 for z in d);op+=all(d[z][k]>.00005 for z in d for k in ('front','rear'));maxangle.append(max(abs(q) for q in x['q'][23:31]))
  return {'samples':len(values),'bilateral_front':front,'bilateral_rear':rear,'bilateral_opposed':op,'mean_max_toe_abs_rad':statistics.mean(maxangle) if maxangle else None,'maximum_toe_abs_rad':max(maxangle) if maxangle else None,'last_toe_angles':values[-1][1]['q'][23:31] if values else None}
 def load_count(lo,hi):
  xs=[v for t,v in state.items() if lo<=t<=hi];return {'samples':len(xs),'bilateral_loaded':sum(x['left_load_n']>.1 and x['right_load_n']>.1 and x['left_load_n']+x['right_load_n']>2.2 for x in xs)}
 out={'first_release_step':release,'before_100':summarize(release-100,release-1),'after_early':summarize(release+20,release+119),'after_late':summarize(release+150,min(last,release+249)),'early_load':load_count(release+20,release+119),'late_load':load_count(release+150,min(last,release+249)),'last_state':state[last], 'limitations':'Automatic event on accepted bilateral support, not learned departure; endpoint contact sampling does not prove toe clearance or uninterrupted substeps.'}
(A/(name+'-release-audit.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
