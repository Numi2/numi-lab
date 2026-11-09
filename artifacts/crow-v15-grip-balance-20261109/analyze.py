from pathlib import Path
from collections import defaultdict
import gzip,json,math
A=Path('artifacts/crow-v15-grip-balance-20261109');rows=[]
for course in ('reserved-final-j',):
 for arm in ('positive','negative'):
  n='layout-j-'+arm;p=A/(n+'.stdout')
  if not p.exists():continue
  d=json.loads(p.read_text());first=d['first_episode_navigation'];last=first['steps_by_environment'][0]+1;contacts=defaultdict(float);samples={};clear=None
  for line in gzip.open(A/(n+'.stderr.gz'),'rt'):
   if line.startswith('CROW_CLEARANCE_SAMPLE '):
    s=json.loads(line.split(' ',1)[1]);
    if s['lane']==0 and s['end_step']<=last:clear=s
   if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
   s=json.loads(line.split(' ',1)[1]);t=s['step']
   if s['lane']!=0 or t>last:continue
   samples[t]=s
   for c in s['contacts']:
    a,b=c['a_name'],c['b_name'];names=(a,b)
    if 'crow_course_perch' not in names or c['impulses'][0]<=0:continue
    toe=next((x for x in names if '_toe' in x and x.startswith('crow_')),None)
    if not toe:continue
    nx=c['normal'][0]*(1 if a==toe else -1);side='left' if '_left_' in toe else 'right';digit='rear' if '_rear_' in toe else 'front';direction='positive' if nx>.5 else 'negative' if nx<-.5 else 'other';contacts[t,side,digit,direction]+=c['impulses'][0]
  both=[t for t in sorted(samples) if t<2750 and all(contacts[t,side,'front','negative']>.00005 and contacts[t,side,'rear','positive']>.00005 for side in ('left','right'))]
  run=0;maxrun=0;prior=None
  for t in both:run=run+1 if prior is not None and t==prior+1 else 1;maxrun=max(maxrun,run);prior=t
  terminal=0
  for t in range(2977,3002):
   if t not in samples:continue
   s=samples[t];up=0.;foot=False;other=False
   for c in s['contacts']:
    a,b=c['a_name'],c['b_name'];names=(a,b)
    if 'crow_course_perch' not in names or c['impulses'][0]<=0:continue
    obj=b if a=='crow_course_perch' else a
    if obj in ('crow_left_foot','crow_right_foot') or '_toe' in obj:
     foot=True;up+=(-1 if a==obj else 1)*c['normal'][2]*c['impulses'][0]
    else:other=True
   v=s['v'];terminal+=bool(foot and not other and up>0 and math.hypot(v[0],v[1])<=.12 and abs(v[2])<=.12)
  release=2750;loss={side:next((t for t in sorted(samples) if t>=release and (contacts[t,side,'front','negative']<=.00005 or contacts[t,side,'rear','positive']<=.00005)),None) for side in ('left','right')}
  row={'course':course,'arm':arm,'first_episode':first,'failed_environment_steps':d['failed_environment_steps'],'sampled_both_feet_opposing_steps':both,'max_consecutive_both_feet_opposing':maxrun,'post_release_first_load_loss':loss,'terminal_supported_endpoints':terminal,'minimum_clearance':clear,'end_sample':{k:samples[last][k] for k in ('step','q','v','contacts')} if last in samples else None,'complete_gate':first['steps_by_environment'][0]==3000 and first['maximum_waypoints_by_environment'][0]==5 and not first['physical_failure_by_environment'][0] and d['failed_environment_steps']==0 and terminal==25 and maxrun>=3 and all(x is not None for x in loss.values())};rows.append(row);print(course,arm,'waypoints',first['maximum_waypoints_by_environment'],'physical_failure',first['physical_failure_by_environment'],'last_step',last,'opposing_max',maxrun,'terminal_support',terminal,'gate',row['complete_gate'])
(A/'full-analysis.json').write_text(json.dumps({'results':rows,'scope':'one lane each, exposed layouts, estimated toes; no real-bird validation'},indent=2)+'\n')
