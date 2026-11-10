import gzip,json,math
from pathlib import Path
A=Path('artifacts/crow-v15-wing-aero-alignment-20261110');d=json.loads((A/'flight-analysis.json').read_text());ends=d['first_episode']['steps_by_environment']; contacts={};samples={}
for l in gzip.open(A/'flight.stderr.gz','rt'):
 if l.startswith('CROW_CONTACT_TRACE '):
  x=json.loads(l.split(' ',1)[1]); lane=x['lane'];
  if lane not in contacts and x['step']<=ends[lane]+1:
   c=[z for z in x['contacts'] if 'beak' in z['a_name'] or 'beak' in z['b_name']];q=x['pre_q'];v=x['pre_v'];pitch=math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3]))));contacts[lane]={'step':x['step'],'q_z':q[2],'pitch_rad':pitch,'velocity_mps':v[:3],'pair':[c[0]['a_name'],c[0]['b_name']] if c else None,'impulse_ns':sum(z['normal_impulse'] for z in c),'flap_q':q[9:11],'flap_rates_rad_s':v[8:10]}
 elif l.startswith('CROW_FLIGHT_SAMPLE '):
  x=json.loads(l.split(' ',1)[1]);
  if x['step'] in [300,400,500] and x['step']<=ends[x['lane']]:
   q=x['q'];v=x['v'];samples.setdefault(x['lane'],[]).append({'step':x['step'],'height_m':q[2],'forward_speed_mps':v[0],'vertical_speed_mps':v[2],'pitch_rad':math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3]))))})
r={'first_episode_failure_by_lane':contacts,'preceding_samples':samples,'caveats':['The trace records raw policy outputs, which are zero and are not the low-level controller actions.','Contact trace is logged at the next control-step boundary; use the recorded pre-state.','Aerodynamic closure and wing shape remain uncalibrated.']};(A/'failure-analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2)[:4500])
