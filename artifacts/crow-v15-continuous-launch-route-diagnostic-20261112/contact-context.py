from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-continuous-launch-route-diagnostic-20261112");D=A/"candidate-20261126-rebound-hip";r=json.loads((A/"analysis-20261126.json").read_text());names=json.load(gzip.open(D/"replay.json.gz","rt"))["payload"]["body_names"];states={};rows=[]
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);states[x['lane'],x['step']]=x
def pose(x):
 q=x['q'];a,b,c,d=q[3:7];return {'step':x['step'],'position_m':q[:3],'roll_pitch_yaw_rad':[math.atan2(2*(d*a+b*c),1-2*(a*a+b*b)),math.asin(max(-1,min(1,2*(d*b-c*a)))),math.atan2(2*(d*c+a*b),1-2*(b*b+c*c))],'linear_angular_velocity':x['v'][:6],'policy_action':x['policy_action']}
for z in r['rows']:
 e=z['lane'];event=z['first_scene_contact'];t=event['step'] if event else z['first_episode_steps'];context=[pose(states[e,k]) for k in range(max(1,t-3),t+1) if (e,k) in states];record={'lane':e,'end':z['first_episode_steps'],'reason':z['termination_reason'],'first_scene_contact':event,'preceding_states':context}
 if event:record['collider_names']=[names[event['bodyA']],names[event['bodyB']]]
 rows.append(record)
(A/"contact-context.json").write_text(json.dumps(rows,indent=2)+"\n")
for x in rows:print(x['lane'],'end',x['end'],'reason',x['reason'],'contact',x['first_scene_contact'],'names',x.get('collider_names'),'context_steps',[t['step'] for t in x['preceding_states']])
