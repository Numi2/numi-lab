from pathlib import Path
import json,gzip,math
A=Path('artifacts/crow-v15-approach-j-recovery-20261110');name='development-j-approach-recovery';d=json.loads((A/(name+'.stdout')).read_text());data={'nav5_step':None,'bilateral_load_samples':0,'longest_bilateral_load':0,'phase2_samples':0,'perch_foot_contact_samples':0,'opposed_bilateral_samples':0,'nonfoot_contact_samples':0,'max_flight_toe_abs_after_25_impulse_free':0.,'flight_toe_samples_after_25_impulse_free':0,'last':None};lastnav=0;run=0;free=[0,0];state={};support={};previous_step=0
with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
 for line in f:
  if line.startswith('CROW_TASK_STATE '):
   v=line.split();step=int(v[2]);nav=float(v[4]);
   if step<previous_step:break
   previous_step=step;lastnav=nav
   if nav>=5:
    if data['nav5_step'] is None:data['nav5_step']=step
    a,b=float(v[5]),float(v[6]);ok=a>.1 and b>.1 and a+b>2.2;run=run+1 if ok else 0;data['bilateral_load_samples']+=ok;data['longest_bilateral_load']=max(run,data['longest_bilateral_load']);data['phase2_samples']+=float(v[7])>=1.5;state[step+1]=(a,b)
   data['last']={'step':step,'nav':nav,'left_load_n':float(v[5]),'right_load_n':float(v[6]),'phase':float(v[7]),'height_target':float(v[8]),'forward_velocity':float(v[9])}
  elif line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);t=x['step'];support[t]=x;op={s:{'front':0.,'rear':0.} for s in ('left','right')};imp=other=0.
   for c in x['contacts']:
    an,bn=c['a_name'],c['b_name'];body=an if bn=='crow_course_perch' else bn if an=='crow_course_perch' else None
    if body is None:continue
    side=next((z for z in op if '_'+z+'_' in body),None);force=c['impulses'][0];sgn=-1 if body==an else 1;nx=-sgn*c['normal'][0]
    if side and ('_foot' in body or '_toe' in body):
     imp+=force
     if 'front_toe' in body and nx<-.5:op[side]['front']+=force
     if 'rear_toe' in body and nx>.5:op[side]['rear']+=force
    else:other+=force
   if lastnav>=5:
    data['perch_foot_contact_samples']+=imp>1e-8;data['nonfoot_contact_samples']+=other>1e-8;data['opposed_bilateral_samples']+=all(op[s][j]>.00005 for s in op for j in ('front','rear'))
   if 800<=t and lastnav<5:
    for side in range(2):
     label='crow_left_' if side==0 else 'crow_right_';active=any(any(label in n and ('toe' in n or 'foot' in n) for n in (c['a_name'],c['b_name'])) and c['impulses'][0]>1e-8 for c in x['contacts']);free[side]=0 if active else free[side]+1
     if free[side]>=25:data['flight_toe_samples_after_25_impulse_free']+=1;data['max_flight_toe_abs_after_25_impulse_free']=max(data['max_flight_toe_abs_after_25_impulse_free'],*map(abs,x['q'][23+4*side:27+4*side]))
chosen=sorted(set(state)&set(support) | {t for t in support if 2800<=t<=4501 and t%100==0});(A/'samples.txt').write_text(''.join(str(t)+' '+' '.join(map(str,support[t]['q']+support[t]['v']))+'\n' for t in chosen));data['first_episode']=d['first_episode_navigation'];data['failed_environment_steps']=d['failed_environment_steps'];data['clearance_floor_m']=None
(A/'analysis.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
