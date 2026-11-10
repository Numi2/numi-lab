from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-sustained-gait-20261110')
def pct(x,p):
 x=sorted(x);return x[min(len(x)-1,math.ceil(p*len(x))-1)] if x else None
def runs(x):
 r=[];n=0
 for v in x+[False]:
  if v:n+=1
  elif n:r.append(n);n=0
 return r
out={}
for seed in (74419361,74419417):
 arms={}
 for arm in ('push-trigger','control-trigger'):
  name=str(seed)+'-'+arm;gait=[[] for _ in range(8)];support=[{} for _ in range(8)];status=[[] for _ in range(8)]
  with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
   for l in f:
    if l.startswith('CROW_GAIT '):
     x=json.loads(l.split(' ',1)[1]);gait[x['lane']].append(x)
    elif l.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(l.split(' ',1)[1]);support[x['lane']][x['step']]=x
    elif l.startswith('CROW_TASK_STATE '):
     _,lane,step,packed,nav,left,right=l.split();status[int(lane)].append({'step':int(step),'packed':int(packed),'left_support_N':float(left),'right_support_N':float(right)})
  doc=json.loads((A/(name+'.stdout')).read_text());first=doc['first_episode_navigation'];rows=[]
  for lane in range(8):
   end=min(900,first['steps_by_environment'][lane]+1);xs=sorted((x for x in gait[lane] if 100<=x['step']<=600 and x['step']<=end),key=lambda x:x['step']);valid=len(xs)==501 and [x['step'] for x in xs]==list(range(100,601));sides=[]
   for side in (0,1):
    stance=[x['sides'][side] for x in xs if x['sides'][side]['impulse_Ns']>.001];swing=[x['sides'][side]['impulse_Ns']<=.001 and x['sides'][1-side]['impulse_Ns']>.001 and x['sides'][side]['clearance_m']>=.002 for x in xs];r=runs(swing);sides.append({'loaded_endpoints':len(stance),'slip_p95_m_s':pct([x['mean_slip_m_s'] for x in stance],.95),'max_contact_slip_m_s':max((x['max_slip_m_s'] for x in stance),default=None),'clear_swing_runs_at_least_3':sum(y>=3 for y in r),'longest_clear_swing':max(r,default=0)})
   forward=xs[-1]['root'][0]-xs[0]['root'][0] if xs else None;minz=min((x['root'][2] for x in xs),default=None);maxz=max((x['root'][2] for x in xs),default=None);maxtilt=max((math.acos(max(-1,min(1,1-2*(x['quat'][0]**2+x['quat'][1]**2)))) for x in xs),default=None);nonfoot=sum(x['nonfoot_impulse']>1e-7 for x in xs);unsupported=sum(all(y['impulse_Ns']<=.001 for y in x['sides']) for x in xs)
   walk=bool(valid and not first['physical_failure_by_environment'][lane] and end==900 and doc['failed_environment_steps']==0 and forward>=.10 and minz>=.13 and maxz<=.25 and maxtilt<=.35 and nonfoot==0 and unsupported==0 and all(y['slip_p95_m_s'] is not None and y['slip_p95_m_s']<=.05 and y['clear_swing_runs_at_least_3']>=8 for y in sides))
   latch=next((x for x in status[lane] if x['step']<670 and x['packed']&4096),None); initial=support[lane].get(1);last=support[lane].get(end);clear=True
   for t in range(850,901):
    s=support[lane].get(t)
    if s is None:clear=False;break
    if any('locomotion_ground' in (c['a_name'],c['b_name']) and c['impulses'][0]>1e-7 for c in s['contacts']):clear=False;break
   flight=bool(end==900 and initial and last and clear and last['q'][2]>initial['q'][2]+.15 and not first['physical_failure_by_environment'][lane]);rows.append({'lane':lane,'valid_gait_data':valid,'forward_m':forward,'height_m':[minz,maxz],'max_tilt_rad':maxtilt,'nonfoot_endpoints':nonfoot,'unsupported_endpoints':unsupported,'sides':sides,'walk_pass':walk,'early_latch':latch,'final_airborne':flight,'episode_end':end,'physical_failure':first['physical_failure_by_environment'][lane]})
  arms[arm]={'lanes':rows,'walk_passes':sum(x['walk_pass'] for x in rows),'early_latches':sum(x['early_latch'] is not None for x in rows),'final_airborne':sum(x['final_airborne'] for x in rows),'physical_failures':sum(x['physical_failure'] for x in rows),'solver_failures':doc['failed_environment_steps']}
 push=arms['push-trigger'];control=arms['control-trigger'];passed=push['walk_passes']==8 and push['early_latches']>=6 and push['final_airborne']>=6 and control['early_latches']<=2 and all(x['physical_failures']==0 and x['solver_failures']==0 for x in arms.values());out[str(seed)]={'arms':arms,'predeclared_pass':passed};print(seed,'pass',passed,[(a,v['walk_passes'],v['early_latches'],v['final_airborne'],v['physical_failures']) for a,v in arms.items()]);print('lane0',push['lanes'][0])
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
