from pathlib import Path
import gzip,json,math,subprocess,os,sys
D=Path('artifacts/crow-v15-walk-hold-wing-off-20261111');seed=20261126
def read(mode):
 P=Path('artifacts/crow-v15-autonomous-walk-hold-journey-20261111/autonomous-20261126') if mode=='on' else D/'autonomous-20261126';sup={};sub={};toe={e:[] for e in range(8)}
 for p in sorted(P.glob('run.stderr.gz')):
  with gzip.open(p,'rt') as f:
   for l in f:
    if l.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(l.split(' ',1)[1]);e=x['lane'];t=x['step']
     if t<=500:sup[e,t]=x
    elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
     a=l.split();t,k,e=map(int,a[1:4])
     if t<=500:sub[e,t,k]=list(map(float,a[6:]))
    elif l.startswith('CROW_TOE_V_SUBSTEP '):
     a=l.split();t,k,e=map(int,a[1:4]);v=list(map(float,a[4:]));assert len(v)==16
     if t<=500 and any(abs(z)>=11.999 for z in v):toe[e].append([t,k,[i for i,z in enumerate(v) if abs(z)>=11.999]])
 meta=json.loads((P/'run.stdout').read_text());return sup,sub,toe,meta
on,ons,ont,om=read('on');off,offs,offt,fm=read('off');ends={z:m['first_episode_navigation']['steps_by_environment'] for z,m in [('on',om),('off',fm)]}
def ints(s,e,t):
 v=[s.get((e,t,k)) for k in range(16)]
 return None if any(z is None for z in v) else [sum(x[i] for x in v) for i in range(37)]
def eq(e,t):return all(on[e,t][k]==off[e,t][k] for k in ('q','v')) and all(ons[e,t,k]==offs[e,t,k] for k in range(16))
first={}
for e in range(8):
 first[e]=next((t for t in range(348,400) if t+1<=ends['off'][e] and (e,t) in off and (e,t+1) in off and all(ints(offs,e,u) is not None and sum(ints(offs,e,u)[2:4])<=1e-7 and off[e,u]['v'][2]>0 for u in (t,t+1))),None)
records=[(e,t,off[e,t]) for e,t0 in first.items() if t0 is not None for t in (t0,t0+1)]
text=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for e,t,x in records)
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics'],input=text,text=True,capture_output=True,check=True,env=env);pose={}
for l in p.stdout.splitlines():
 a=l.split()
 if a[0]=='body':pose[int(a[1]),int(a[2]),a[3]]=list(map(float,a[4:]))
with gzip.open(D/'autonomous-20261126'/'replay.json.gz','rt') as f:r=json.load(f)['payload']
assert r['environment']==7;frames={x['step']:x for x in r['frames']};maxpos=maxrot=maxq=0.;validated=0
for e,t,x in records:
 if e!=7:continue
 frame=frames[t-1];maxq=max(maxq,max(abs(a-b) for a,b in zip(frame['q'],x['q'])))
 for side in ('left','right'):
  for name in ['crow_'+side+'_foot']+['crow_'+side+'_'+d+'_'+seg for d in ('front_toe_1','front_toe_2','front_toe_3','rear_toe') for seg in ('proximal','distal')]:
   z=pose[e,t,name];i=r['body_names'].index(name)*13;v=frame['body_states'][i:i+13];maxpos=max(maxpos,math.sqrt(sum((a-b)**2 for a,b in zip(z[:3],v[:3]))));maxrot=max(maxrot,min(math.sqrt(sum((a-b)**2 for a,b in zip(z[3:7],v[3:7]))),math.sqrt(sum((a+b)**2 for a,b in zip(z[3:7],v[3:7])))))
 validated+=1
geomvalid=validated==2 and maxq<1e-6 and maxpos<1e-5 and maxrot<1e-5
rows=[]
for e in range(8):
 t0=first[e];paired=[];gaps=[];recontact=None;rise=None;flightground=None;legup=None;nonfoot=None
 if t0 is not None:
  legup=[sum(ints(offs,e,t)[side] for t in range(347,t0+1) if ints(offs,e,t) is not None) for side in range(2)]
  nonfoot=max((ints(offs,e,t)[4] for t in range(347,t0+2) if ints(offs,e,t) is not None),default=float('inf'))
  paired=[t for t in range(1,t0+2) if (e,t) not in on or (e,t) not in off or ints(ons,e,t) is None or ints(offs,e,t) is None or not eq(e,t)]
  for t in (t0,t0+1):
   for side in ('left','right'):
    for name in ['crow_'+side+'_foot']+['crow_'+side+'_'+d+'_'+seg for d in ('front_toe_1','front_toe_2','front_toe_3','rear_toe') for seg in ('proximal','distal')]:
     z=pose[e,t,name];x,y,q,w=z[3:7];axis=(2*(x*q-y*w),2*(y*q+x*w),1-2*(x*x+y*y));h=(.010,.01725,.0092) if name.endswith('_foot') else (.009,.0025,.0092);gaps.append(z[2]-sum(abs(a)*b for a,b in zip(axis,h)))
  recontact=next((t for t in range(t0+2,min(ends['off'][e],500)+1) if ints(offs,e,t) is not None and sum(ints(offs,e,t)[2:5])>1e-7),None)
  horizon=(recontact-1) if recontact is not None else min(ends['off'][e],500)
  rise=max((off[e,t]['q'][2]-off[e,347]['q'][2] for t in range(t0,horizon+1) if (e,t) in off),default=None)
  flightground=next((t for t in range(t0,min(ends['on'][e],500)+1) if ints(ons,e,t) is not None and sum(ints(ons,e,t)[2:5])>1e-7),None)
 met={}
 for t in (400,500):
  if (e,t) in on and (e,347) in on and t<=ends['on'][e]:
   q=on[e,t]['q'];v=on[e,t]['v'];met[str(t)]={'height_gain_m':q[2]-on[e,347]['q'][2],'forward_speed_m_s':v[0],'tilt_rad':math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2))))}
 missing={z:sum(ints(sub,e,t) is None for t in range(1,min(500,ends[z][e])+1)) for z,sub in [('on',ons),('off',offs)]}
 legpass=t0 is not None and not paired and geomvalid and len(gaps)==36 and min(gaps)>.001 and rise is not None and rise>.02 and legup is not None and min(legup)>.0002 and nonfoot is not None and nonfoot<=1e-7 and not missing['on'] and not missing['off']
 a=met.get('500',{});flightpass=legpass and ends['on'][e]>=500 and flightground is None and a.get('height_gain_m',-1)>.15 and a.get('forward_speed_m_s',-1)>.2 and a.get('tilt_rad',9)<.35
 rows.append({'lane':e,'first_unloaded_rising_interval_off':t0,'first_off_recontact':recontact,'wing_off_max_rise_before_recontact_m':rise,'foot_upward_impulse_until_unload_Ns':legup,'max_nonfoot_interval_impulse_Ns':nonfoot,'paired_qv_or_contact_differences':paired[:20],'minimum_sole_toe_gap_m':min(gaps) if gaps else None,'missing_intervals':missing,'first_on_ground_recontact':flightground,'flight_metrics':met,'toe_limit_hits_on':ont[e],'toe_limit_hits_off':offt[e],'on_first_episode_end':ends['on'][e],'off_first_episode_end':ends['off'][e],'leg_control_pass':legpass,'flight_handoff_pass':flightpass})
out={'seed':seed,'replay_geometry_validation':{'passed':geomvalid,'lane':7,'samples':validated,'q_error':maxq,'max_position_error_m':maxpos,'max_quaternion_error':maxrot},'on_errors':{'failed_environment_steps':om['failed_environment_steps'],'termination_count':om['termination_count']},'off_errors':{'failed_environment_steps':fm['failed_environment_steps'],'termination_count':fm['termination_count']},'rows':rows,'limitations':'Teacher-off selected development starts. Wing-suppressed control does not eliminate passive aerodynamic forces. Ground clearance is sampled at control endpoints. This is not landing, release, held-out route, full ground-sequence qualification, or biological validation.'}
(D/'wing-off-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('geometry',out['replay_geometry_validation']);print([(x['lane'],x['first_unloaded_rising_interval_off'],x['wing_off_max_rise_before_recontact_m'],x['leg_control_pass'],x['flight_handoff_pass'],len(x['toe_limit_hits_on'])) for x in rows])
