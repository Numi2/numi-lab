from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-approach-damping-20261110');R={}
for c in ['i','j']:
 name='development-reserved-final-'+c;e=json.loads((A/(name+'.stdout')).read_text())['first_episode_navigation']; samples=[];nav={}
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_TASK_STATE '):
    p=l.split();lane=int(p[1]);step=int(p[2]);nav[(lane,step)]=float(p[4])
   elif l.startswith('CROW_FLIGHT_SAMPLE '): samples.append(json.loads(l.split(' ',1)[1]))
 lanes=[]
 for lane in range(8):
  ss=[x for x in samples if x['lane']==lane and x['step']>=800 and x['step']<=e['steps_by_environment'][lane] and nav.get((lane,x['step']),5)<5]
  aa=[abs(q) for x in ss for q in x['q'][23:31]];vv=[abs(v) for x in ss for v in x['v'][22:30]]
  lanes.append({'lane':lane,'samples':len(ss),'max_abs_toe_angle_rad':max(aa,default=None),'max_abs_toe_speed_rad_s':max(vv,default=None),'sample_steps':[x['step'] for x in ss]})
 R[c]=lanes
(A/'toe-flight-analysis.json').write_text(json.dumps({'method':'100-step samples, first episode only; time >= 16s and before fifth waypoint. Simulated toe joint positions/velocities only; does not prove continuous flight release.','lanes':R},indent=2)+'\n')
for c,lanes in R.items():print(c,'max angles',max(x['max_abs_toe_angle_rad'] for x in lanes if x['max_abs_toe_angle_rad'] is not None),'max speeds',max(x['max_abs_toe_speed_rad_s'] for x in lanes if x['max_abs_toe_speed_rad_s'] is not None))
