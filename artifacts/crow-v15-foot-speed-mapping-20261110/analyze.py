from pathlib import Path
from collections import defaultdict
import gzip,json,math
A=Path('artifacts/crow-v15-foot-speed-mapping-20261110');rows=[]
for course in ('i','j'):
 n='layout-'+course;p=A/(n+'.stdout')
 if not p.exists():continue
 d=json.loads(p.read_text());first=d['first_episode_navigation'];end=first['steps_by_environment'][0]+1;samples={};loads=defaultdict(float);clear=[];latches=[];first_bad=None
 for line in gzip.open(A/(n+'.stderr.gz'),'rt'):
  if line.startswith('CROW_TASK_STATE '):
   _,lane,step,packed,nav,l,r=line.split();
   if int(lane)==0 and int(step)<270 and int(packed)&4096:latches.append({'step':int(step),'left_support_N':float(l),'right_support_N':float(r)})
  if line.startswith('CROW_CLEARANCE_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);
   if x['lane']==0 and x['end_step']<=end:clear.append(x)
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(line.split(' ',1)[1]);t=x['step']
  if x['lane']!=0 or t>end:continue
  samples[t]=x
  for c in x['contacts']:
   a,b=c['a_name'],c['b_name'];imp=c['impulses'][0];names=(a,b)
   if imp<=1e-5:continue
   bird=next((z for z in names if z.startswith('crow_') and not z.startswith('crow_course_')),None);surface=next((z for z in names if z=='locomotion_ground' or z.startswith('crow_course_')),None)
   if bird and surface and bird not in ('crow_left_foot','crow_right_foot') and '_toe' not in bird and first_bad is None:
    first_bad={'step':t,'bird':bird,'surface':surface,'normal_impulse_Ns':imp,'root_position_m':x['q'][:3],'root_velocity_m_s':x['v'][:6]}
   if 'crow_course_perch' not in names:continue
   toe=next((z for z in names if '_toe' in z and z.startswith('crow_')),None)
   if not toe:continue
   nx=c['normal'][0]*(1 if a==toe else -1);side='left' if '_left_' in toe else 'right';digit='rear' if '_rear_' in toe else 'front';direction='positive' if nx>.5 else 'negative' if nx<-.5 else 'other';loads[t,side,digit,direction]+=imp
 both=[t for t in sorted(samples) if t<2250 and all(loads[t,s,'front','negative']>.00005 and loads[t,s,'rear','positive']>.00005 for s in ('left','right'))];run=0;maxrun=0;prev=None
 for t in both:run=run+1 if prev is not None and t==prev+1 else 1;maxrun=max(maxrun,run);prev=t
 loss={s:next((t for t in sorted(samples) if t>=2250 and (loads[t,s,'front','negative']<=.00005 or loads[t,s,'rear','positive']<=.00005)),None) for s in ('left','right')};terminal=0
 for t in range(2977,3002):
  if t not in samples or t>end:continue
  x=samples[t];up=0.;foot=False;other=False
  for c in x['contacts']:
   a,b=c['a_name'],c['b_name'];names=(a,b)
   if 'crow_course_perch' not in names or c['impulses'][0]<=0:continue
   obj=b if a=='crow_course_perch' else a
   if obj in ('crow_left_foot','crow_right_foot') or '_toe' in obj:foot=True;up+=(-1 if a==obj else 1)*c['normal'][2]*c['impulses'][0]
   else:other=True
  v=x['v'];terminal+=bool(foot and not other and up>0 and math.hypot(v[0],v[1])<=.12 and abs(v[2])<=.12)
 end_sample=samples.get(end);gate=end==3001 and first['maximum_waypoints_by_environment']==[5] and not first['physical_failure_by_environment'][0] and d['failed_environment_steps']==0 and first_bad is None and maxrun>=3 and all(v is not None for v in loss.values()) and terminal==25
 row={'layout':course,'first_episode':first,'failed_environment_steps':d['failed_environment_steps'],'first_latch':latches[0] if latches else None,'first_nonfoot_contact':first_bad,'minimum_obstacle_clearance':clear[-1] if clear else None,'max_consecutive_bilateral_opposing_toe_load_endpoints':maxrun,'last_pre_release_opposing_step':both[-1] if both else None,'post_release_first_load_loss':loss,'terminal_supported_endpoints':terminal,'end_state':{'step':end,'q':end_sample['q'],'v':end_sample['v']} if end_sample else None,'complete_gate':gate};rows.append(row);print(course,'waypoints',first['maximum_waypoints_by_environment'],'end',end,'native_failure',first['physical_failure_by_environment'],'grip',maxrun,'terminal',terminal,'first_bad',first_bad,'gate',gate)
(A/'analysis.json').write_text(json.dumps({'results':rows,'scope':'one developmental full start per layout, estimated body and contact physics'},indent=2)+'\n')
