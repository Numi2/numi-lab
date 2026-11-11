from pathlib import Path
import gzip,json
a=Path('artifacts/crow-v15-teacher-launch-parity-candidate-20261111')
def support(p):
 d={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);d[x['lane'],x['step']]=x
 return d
off=a/'off-20261126';teacher=a/'teacher-20261126';ref=Path('artifacts/crow-v15-hip-roll-autonomous-route-dev-20261111/autonomous-20261126')
x,y,z=support(off),support(teacher),support(ref)
rows=[]
for e in range(8):
 ts=range(1,400)
 dif=[t for t in ts if x.get((e,t))!=y.get((e,t))]
 refdif=[t for t in ts if x.get((e,t))!=z.get((e,t))]
 missing=[t for t in ts if (e,t) not in x or (e,t) not in y or (e,t) not in z]
 rows.append({'lane':e,'missing':missing,'teacher_off_first_difference':dif[0] if dif else None,'teacher_off_difference_count':len(dif),'candidate_off_reference_first_difference':refdif[0] if refdif else None,'candidate_off_reference_difference_count':len(refdif)})
def replay(p):
 with gzip.open(p/'replay.json.gz','rt') as f:return json.load(f)['payload']['frames']
rx,ry=replay(off),replay(teacher)
rd=[]
for i,(u,v) in enumerate(zip(rx,ry)):
 if i>=399:break
 if any(u[k]!=v[k] for k in ('q','v','actor_observation','accepted_actions')):rd.append(i)
r={'lanes':rows,'replay_lane':1,'replay_q_v_observation_action_first_difference':rd[0] if rd else None,'replay_difference_count':len(rd),'scope':'Pre-handoff samples only; policy outputs are not low-level actuator targets. Replay comparison lane 1 only. No claim for later teacher behavior or landing.'}
(a/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
