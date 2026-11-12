from pathlib import Path
import gzip,json,math,sys
base=Path('artifacts')
sys.path.insert(0,str(base/'crow-v15-downstroke-full-sequence-screen-20261112'))
from split_stream import open_text
runs=[('early',base/'crow-v15-early-wing-sequence-screen-20261112'/'autonomous-20261126'),('downstroke',base/'crow-v15-downstroke-full-sequence-screen-20261112'/'autonomous-20261126')]
results={}
for name,p in runs:
 d=json.loads((p/'run.stdout').read_text())['first_episode_navigation'];ends=d['steps_by_environment'];q={};ph={}
 with open_text(p/'run.stderr.gz') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
    if 3000<=t<=ends[e]:
     a,b,c,w=x['q'][3:7];q[e,t]=math.asin(max(-1,min(1,2*(w*b-c*a))))
   elif line.startswith('CROW_PHASE_SAMPLE '):
    x=line.split();t,e,status=int(x[1]),int(x[2]),int(x[3])
    if 3000<=t<=ends[e]:ph[e,t]=((status>>16)&15)|(((status>>28)&15)<<4)
 rows=[]
 for e,end in enumerate(ends):
  events={}
  for v in (.1,.2,.4,.8):
   t=next((t for t in range(3000,end+1) if (e,t) in q and abs(q[e,t])>=v),None)
   events['pitch_'+str(v)]={'step':t,'pitch':q.get((e,t)),'sequence':ph.get((e,t))} if t is not None else None
  for label,fn in [('weight_shift',lambda s:50<=s<150),('release_request',lambda s:150<=s<230),('release_complete_or_abort',lambda s:s>=230)]:
   t=next((t for t in range(3000,end+1) if (e,t) in ph and fn(ph[e,t])),None)
   events[label]={'step':t,'pitch':q.get((e,t)),'sequence':ph.get((e,t))} if t is not None else None
  rows.append({'lane':e,'end':end,'physical_failure':d['physical_failure_by_environment'][e],'events':events})
 results[name]=rows
 print(name)
 for x in rows:
  print(x['lane'],x['physical_failure'],[(k,v and (v['step'],round(v['pitch'],3) if v['pitch'] is not None else None,v['sequence'])) for k,v in x['events'].items()])
out={'runs':results,'limitations':'Used development seed; endpoint and phase samples may differ within each interval. Sequence range indicates a controller request, not proof that toes opened or slipped. Timing is descriptive, not causation.'}
(base/'crow-v15-release-pitch-timing-audit-20261112'/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
