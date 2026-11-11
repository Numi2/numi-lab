from pathlib import Path
import gzip,json
a=Path('artifacts/crow-v15-dagger-isolated-labels-control-20261111')
def support(p):
 d={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);k=(x['lane'],x['step'])
    if k in d: raise ValueError('duplicate support sample '+str(k))
    d[k]=x
 return d
off=a/'off-20261126';teacher=a/'label-20261126'
x,y=support(off),support(teacher)
out0=json.loads((off/'run.stdout').read_text())['first_episode_navigation'];out1=json.loads((teacher/'run.stdout').read_text())['first_episode_navigation']
rows=[]
for e in range(8):
 end=min(out0['steps_by_environment'][e],out1['steps_by_environment'][e])
 missing=[t for t in range(1,end+1) if (e,t) not in x or (e,t) not in y]
 dif=[t for t in range(1,end+1) if (e,t) in x and (e,t) in y and x[e,t]!=y[e,t]]
 rows.append({'lane':e,'first_episode_steps_off':out0['steps_by_environment'][e],'first_episode_steps_labels':out1['steps_by_environment'][e],'missing_samples':len(missing),'first_missing':missing[:1],'different_samples':len(dif),'first_difference':dif[:1]})
def replay(p):
 with gzip.open(p/'replay.json.gz','rt') as f:return json.load(f)['payload']['frames']
rx,ry=replay(off),replay(teacher);end=min(out0['steps_by_environment'][1],out1['steps_by_environment'][1],len(rx),len(ry))
rd=[i+1 for i,(u,v) in enumerate(zip(rx[:end],ry[:end])) if any(u[k]!=v[k] for k in ('q','v','actor_observation','accepted_actions'))]
r={'lanes':rows,'replay_lane':1,'replay_compared_frames':end,'replay_q_v_observation_action_first_difference':rd[:1],'replay_difference_count':len(rd),'first_episode_outcomes_equal':out0==out1,'scope':'Accepted q/v, policy action, and endpoint contact samples for each first episode; observations and replay action in lane 1. Does not prove low-level actuator parity or correctness of the labels. Teacher still cannot be used for corrective training until its full flight and landing recoveries qualify.'}
(a/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
