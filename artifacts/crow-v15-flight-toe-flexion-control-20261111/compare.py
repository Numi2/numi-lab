from pathlib import Path
import json,hashlib
a=Path('artifacts/crow-v15-flight-toe-flexion-control-20261111');ref=Path('build/crow-v15-journey-bounded-flight-screen-20261111/preflight.json');cur=Path('build/crow-v15-flight-toe-flexion-control-20261111/flight-replay.json')
with ref.open() as f:r=json.load(f)['payload']
with cur.open() as f:c=json.load(f)['payload']
assert r['environment']==c['environment']==1
x=r['frames'];y=c['frames'];assert len(x)==len(y)==1600
pre=[i for i in range(370) if any(x[i][k]!=y[i][k] for k in ('q','v','accepted_actions','actor_observation'))]
samples=[]
for i in (350,369,380,390,400,450,800,1200,1599):
 q=y[i]['q'];v=y[i]['v'];samples.append({'frame':i,'candidate_toe_q_rad':q[23:39],'candidate_toe_speed_rad_s':v[22:38],'reference_toe_q_rad':x[i]['q'][23:39]})
out={'pre_curl_frames_compared':370,'pre_curl_first_difference':pre[0] if pre else None,'pre_curl_difference_count':len(pre),'replay_environment':1,'samples':samples,'scope':'Recorded native replay frames at control endpoints. The toe ramp is triggered by a physical liftoff latch and released on final approach, but this obstacle-free run does not exercise approach release, grip, landing or obstacle navigation. The end-of-substep velocity audit is separate.'}
(a/'toe-pose-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print('pre differences',len(pre),'at frame 450 toe q',samples[5]['candidate_toe_q_rad'])
