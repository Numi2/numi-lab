from pathlib import Path
import gzip,json
R=Path.cwd();A=R/'artifacts/crow-v15-planar-sensor-20261110';B=R/'build/crow-v15-planar-sensor-20261110';C=R/'build/crow-v15-causal-route-20261110';rows=[]
for name in ('layout-i','layout-j'):
 old=json.loads((C/(name+'.json')).read_text())['payload'];new=json.loads((B/(name+'.json')).read_text())['payload'];end=json.loads((A/(name+'.stdout')).read_text())['first_episode_navigation']['steps_by_environment'][0]+1;frames=new['frames'];parity=all(a['q']==b['q'] and a['v']==b['v'] for a,b in zip(old['frames'][:end],frames[:end]));by={x['step']:x for x in frames[:end]};errors=[]
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if not l.startswith('CROW_TASK_STATE '):continue
   _,lane,t,packed,nav,_,_=l.split();t=int(t);packed=int(packed)
   if int(lane)!=0 or t>end or int(nav)<5 or t not in by:continue
   b=by[t]['body_states'];n=new['body_names'];stride=new['body_state_stride'];y=lambda s:b[n.index(s)*stride+1];expected=max(-1.,min(1.,y('crow_course_perch')+.55-.5*(y('crow_left_foot')+y('crow_right_foot'))));actual=(((packed>>23)&511)-256)/250.;errors.append(abs(actual-expected))
 row={'layout':name,'first_episode_steps':end,'q_v_bitwise_equal':parity,'samples':len(errors),'max_y_error_m':max(errors) if errors else None,'within_6mm':bool(errors) and max(errors)<=.006};rows.append(row);print(row,flush=True)
(A/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
