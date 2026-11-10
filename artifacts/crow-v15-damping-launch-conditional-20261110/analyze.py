from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-damping-launch-conditional-20261110'); result={}
for seed in (20261126,18647371,18647393):
 arms={}
 for arm in ('push-trigger','push-timer','control-trigger'):
  name=str(seed)+'-'+arm; lanes=[{} for _ in range(8)];status=[[] for _ in range(8)]
  with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
   for line in f:
    if line.startswith('CROW_SUPPORT_SAMPLE '):
     d=json.loads(line.split(' ',1)[1]);lanes[d['lane']][d['step']]=d
    elif line.startswith('CROW_TASK_STATE '):
     _,lane,step,packed,nav,l,r=line.split();status[int(lane)].append({'step':int(step),'packed':int(packed),'left_support_N':float(l),'right_support_N':float(r)})
  doc=json.loads((A/(name+'.stdout')).read_text());arms[arm]=(lanes,status,doc)
 parity=[]
 for i in range(8):
  ds=[]
  for t in range(1,247):
   for arm in ('push-timer','control-trigger'):
    left=arms['push-trigger'][0][i].get(t);right=arms[arm][0][i].get(t)
    if left is None or right is None: ds.append(None);continue
    ds.append(max(abs(x-y) for k in ('q','v') for x,y in zip(left[k],right[k])))
  parity.append(None if any(x is None for x in ds) else max(ds))
 rr={}
 for arm,(lanes,status,doc) in arms.items():
  nav=doc['first_episode_navigation'];rows=[]
  for i,lane in enumerate(lanes):
   end=min(400,nav['steps_by_environment'][i]+1);latches=[s for s in status[i] if s['step']<270 and s['packed']&4096];first=latches[0] if latches else None
   nofoot={};bad=[]
   for t,s in lane.items():
    if t>end:continue
    ground=[]
    for c in s['contacts']:
     a,b=c['a_name'],c['b_name'];imp=c['impulses'][0]
     if 'locomotion_ground' not in (a,b) or imp<=1e-7:continue
     bird=b if a=='locomotion_ground' else a;ground.append(bird)
     if bird not in ('crow_left_foot','crow_right_foot') and '_toe' not in bird:bad.append({'step':t,'body':bird,'impulse_Ns':imp})
    nofoot[t]=not ground
   initial=lane.get(1);last=lane.get(end);flight=bool(initial and last and end==400 and all(nofoot.get(t,False) for t in range(350,401)) and last['q'][2]>initial['q'][2]+.15 and not nav['physical_failure_by_environment'][i]);rows.append({'lane':i,'first_latch':first,'end':end,'physical_failure':nav['physical_failure_by_environment'][i],'nonfoot_ground_contacts':bad,'initial_z_m':initial['q'][2] if initial else None,'final_z_m':last['q'][2] if last else None,'final_second_unloaded_and_15cm_higher':flight,'continuous_ground_clear_after_latch':bool(first and end==400 and all(nofoot.get(t,False) for t in range(first['step']+1,401)))})
  rr[arm]={'lanes':rows,'early_latches':sum(x['first_latch'] is not None for x in rows),'final_second_airborne':sum(x['final_second_unloaded_and_15cm_higher'] for x in rows),'physical_failures':sum(x['physical_failure'] for x in rows),'incomplete':sum(x['end']!=400 for x in rows),'invalid_latch':sum(x['first_latch'] is not None and not (x['first_latch']['left_support_N']<0.05 and x['first_latch']['right_support_N']<0.05) for x in rows),'failed_environment_steps':doc['failed_environment_steps']}
 push=rr['push-trigger'];control=rr['control-trigger'];valid=all(x is not None and x<1e-6 for x in parity);passed=valid and push['early_latches']>=6 and push['final_second_airborne']>=6 and control['early_latches']<=2 and all(rr[a]['physical_failures']==0 and rr[a]['incomplete']==0 and rr[a]['invalid_latch']==0 and rr[a]['failed_environment_steps']==0 for a in rr)
 result[str(seed)]={'preextension_max_abs_state_difference_by_lane':parity,'arms':rr,'predeclared_pass':passed}
 print(seed,'valid',valid,'pass',passed,[(a,rr[a]['early_latches'],rr[a]['final_second_airborne'],rr[a]['physical_failures']) for a in rr])
(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
