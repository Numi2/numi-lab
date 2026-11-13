from pathlib import Path
import gzip,json
A=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-pronation-phase-screen-20261113')
path=A/'push-wing-20261126-rebound-phase/run.stderr.gz';targets={};states={}
with gzip.open(path,'rt') as f:
 for l in f:
  z=l.split()
  if not z:continue
  if z[0]=='CROW_DRIVE_TARGET':
   t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
  elif z[0]=='CROW_SUPPORT_SAMPLE':
   x=json.loads(l.split(' ',1)[1]);states[x['lane'],x['step']]=x
out=[]
for e in range(2,8):
 values=[targets[e,t][10] for t in range(354,371)];qs=[states[e,t]['q'][11] for t in range(354,371)]
 out.append({'lane':e,'left_pronation_target_min_rad':min(values),'left_pronation_target_max_rad':max(values),'left_pronation_q_min_rad':min(qs),'left_pronation_q_max_rad':max(qs)})
(A/'pronation-effect.json').write_text(json.dumps({'scope':'Posthoc min/max demonstrates executed joint target and recorded endpoint response only; it does not validate aerodynamic effect or realism.','rows':out},indent=2)+'\n')
for x in out:print(x)
