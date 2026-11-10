from pathlib import Path
import gzip,json
R=Path.cwd();A=R/'artifacts/crow-v15-foot-sensor-20261110';B=R/'build/crow-v15-foot-sensor-20261110';C=R/'build/crow-v15-causal-route-20261110';rows=[]
for name in ('layout-i','layout-j'):
 old=json.loads((C/(name+'.json')).read_text())['payload'];new=json.loads((B/(name+'.json')).read_text())['payload']; oldframes=old['frames'];frames=new['frames'];end=json.loads((A/(name+'.stdout')).read_text())['first_episode_navigation']['steps_by_environment'][0]+1
 parity=all(a['q']==b['q'] and a['v']==b['v'] for a,b in zip(oldframes[:end],frames[:end]));by_step={x['step']:x for x in frames[:end]};sensor=[]
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for line in f:
   if not line.startswith('CROW_TASK_STATE '):continue
   _,lane,step,packed,nav,left,right=line.split();step=int(step);packed=int(packed)
   if int(lane)!=0 or step>end or int(nav)<5 or step not in by_step:continue
   x=by_step[step];b=x['body_states'];names=new['body_names'];stride=new['body_state_stride'];pos=lambda s:b[names.index(s)*stride];expected=max(-.5,min(.5,pos('crow_course_perch')-(pos('crow_left_foot')+pos('crow_right_foot'))*.5));actual=(((packed>>13)&1023)-512)/1000
   sensor.append(abs(actual-expected))
 row={'layout':name,'first_episode_steps':end,'q_v_bitwise_equal':parity,'sensor_samples':len(sensor),'max_absolute_error_m':max(sensor) if sensor else None,'within_0_00051_m':bool(sensor) and max(sensor)<=0.00051};rows.append(row);print(row,flush=True)
(A/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
