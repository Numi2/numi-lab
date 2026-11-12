from pathlib import Path
import gzip,json,math
R=Path('artifacts/crow-v15-early-wing-sequence-screen-20261112')
P=R/'autonomous-20261126'
A=Path('artifacts/crow-v15-landing-pitch-drift-audit-20261112')
d=json.loads((P/'run.stdout').read_text())['first_episode_navigation']; ends=d['steps_by_environment']
q={};ph={}
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
   if 3000<=t<=ends[e]:
    a,b,c,w=x['q'][3:7];p=math.asin(max(-1,min(1,2*(w*b-c*a))))
    q[e,t]={'pitch':p,'pitch_rate_world_y':x['v'][4],'x':x['q'][0],'z':x['q'][2],'vx':x['v'][0]}
  elif line.startswith('CROW_PHASE_SAMPLE '):
   z=line.split();t,e,status=int(z[1]),int(z[2]),int(z[3])
   if 3000<=t<=ends[e]:ph[e,t]={'status':status,'sequence':((status>>16)&15)|(((status>>28)&15)<<4),'landing_phase':float(z[4])}
rows=[]
for e,end in enumerate(ends):
 events=[]
 for threshold in (.1,.2,.4,.8,1.1):
  t=next((t for t in range(3000,end+1) if (e,t) in q and abs(q[e,t]['pitch'])>=threshold),None)
  events.append({'absolute_pitch_threshold':threshold,'step':t,'state':q.get((e,t)),'phase':ph.get((e,t))})
 rows.append({'lane':e,'end':end,'terminated':d['physical_failure_by_environment'][e],'events':events})
out={'rows':rows,'limitations':'Existing used-seed early-wing run; endpoint state, not torque measurement. Phase/status decoding follows the experimental feedback implementation. Thresholds describe divergence and do not establish causation; the action field was not the applied joint target.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 print(r['lane'],r['end'],[(x['absolute_pitch_threshold'],x['step'],round(x['state']['pitch'],3) if x['state'] else None,x['phase']['sequence'] if x['phase'] else None) for x in r['events']])
