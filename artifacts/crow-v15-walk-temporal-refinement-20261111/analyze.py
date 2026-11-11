from pathlib import Path
from collections import defaultdict
import gzip,json,math
A=Path('artifacts/crow-v15-walk-temporal-refinement-20261111');O=Path('artifacts/crow-v15-toe-launch-control-20261110/control.stderr.gz')
def read(p):
 s={};sub={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);s[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();sub[int(v[3]),int(v[1]),int(v[2])]=list(map(float,v[6:]))
 return s,sub
ref,_=read(O);arms={}
def longest(x):
 best=cur=0
 for z in x:cur=cur+1 if z else 0;best=max(best,cur)
 return best
def bouts(x):
 out=[];start=None
 for i,z in enumerate(x+[False]):
  if z and start is None:start=i
  if not z and start is not None:
   if i-start>=3:out.append({'first_step':100+start,'last_step':99+i,'samples':i-start})
   start=None
 return out
for arm in ('zero32','zero64'):
 steps=int(arm[4:])
 sup,sub=read(A/(arm+'.stderr.gz'));assert len(sub)==202*8*steps and len(sup)==202*8
 if arm=='old':
  diffs=[]
  for (lane,t),d in sup.items():
   if t>201:continue
   r=ref[lane,t];diffs += [abs(x-y) for f in ('q','v') for x,y in zip(d[f],r[f])]
  assert len(diffs)>0 and max(diffs)==0
 err=[]
 for (lane,t),d in sup.items():
  n=[0.,0.]
  for c in d['contacts']:
   a,b=c['a_name'],c['b_name'];body=b if a=='locomotion_ground' else a if b=='locomotion_ground' else ''
   for i,side in enumerate(('left','right')):
    if body.startswith('crow_'+side+'_') and ('foot' in body or '_toe' in body):n[i]+=max(0.,c['impulses'][0])
  err.extend(abs(sub[lane,t,steps-1][2+i]-n[i]) for i in (0,1))
 assert max(err)<1e-6
 intervals={}
 for lane in range(8):
  for t in range(101,202):
   z=[sub[lane,t,k] for k in range(steps)];intervals[lane,t]=[sum(q[i] for q in z) for i in range(7)]
 with gzip.open(A/(arm+'.replay.json.gz'),'rt') as f:p=json.load(f)['payload']
 assert p['environment']==1;frames={f['step']:f for f in p['frames'] if 100<=f['step']<=200};assert len(frames)==101
 maxq=max(abs(x-y) for t,f in frames.items() for x,y in zip(f['q'],sup[1,t+1]['q']));assert maxq<1e-6
 ground=p['body_names'].index('locomotion_ground');names=p['body_names'];records={s:[] for s in ('left','right')};root=[]
 for t,f in sorted(frames.items()):
  st=f['body_states'];z0=st[ground*13+2];interval=intervals[1,t+1];root.append({'speed':math.sqrt(sum(x*x for x in f['v'][:3])),'tilt':f['tilt']})
  for side,index in (('left',0),('right',1)):
   gaps=[]
   for name,h in [('crow_'+side+'_foot',(.010,.01725,.0092))]+[('crow_'+side+'_'+n,(.018,.0025,.0092)) for n in ('front_toe_1','front_toe_2','front_toe_3','rear_toe')]:
    i=names.index(name)*13;x,y,z,w=st[i+3:i+7];row=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));gaps.append(st[i+2]-z0-sum(abs(a)*b for a,b in zip(row,h)))
   records[side].append({'step':t,'min_gap_m':min(gaps),'own_all_substep_normal_Ns':interval[2+index],'other_foot_up_Ns':interval[1-index],'other_body_normal_Ns':interval[6]})
 sides={}
 for side,rows in records.items():
  flags=[r['min_gap_m']>.001 and r['own_all_substep_normal_Ns']<=1e-7 and r['other_foot_up_Ns']>=.5*.45004*9.81*.02 and r['other_body_normal_Ns']<=1e-7 for r in rows]
  sides[side]={'qualified_samples':sum(flags),'longest_consecutive':longest(flags),'bouts_3_or_more':bouts(flags),'max_minimum_gap_m':max(r['min_gap_m'] for r in rows)}
 lane=[]
 for i in range(8):
  ints=[intervals[i,t] for t in range(101,202)]
  lane.append({'lane':i,'nonfoot_ground_contact_intervals':sum(v[6]>1e-7 for v in ints),'no_foot_normal_intervals':sum(v[2]+v[3]<=1e-7 for v in ints),'min_total_foot_up_Ns':min(v[0]+v[1] for v in ints)})
 data=json.loads((A/(arm+'.stdout')).read_text());arms[arm]={'failed_environment_steps':data['failed_environment_steps'],'termination_count':data['termination_count'],'substep_count':len(sub),'last_substep_contact_max_error_Ns':max(err),'replay_q_mismatch':maxq,'lane_1_swing':sides,'lane_1_root_x_displacement_m':frames[200]['q'][0]-frames[100]['q'][0],'max_root_speed_m_s':max(x['speed'] for x in root),'max_tilt_rad':max(x['tilt'] for x in root),'lanes':lane}
out={'old_reference_qv_exact':None,'replay_environment':1,'observer_lane':1,'replay_to_observer_step_offset':1,'substeps_per_interval':{'zero32':32,'zero64':64},'swing_requires_other_foot_upward_impulse_Ns':.5*.45004*9.81*.02,'arms':arms,'limitations':'one replayed lane for geometry, 2-second window, endpoint gap and complete-interval impulse; no contact-point slip, 32/64-substep convergence, takeoff or course qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
