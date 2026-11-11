from pathlib import Path
import gzip,json,math,subprocess
A=Path('artifacts/crow-v15-perturbed-leg-handoff-20261111');B=Path('artifacts/crow-v15-delayed-flight-handoff-20261111')
def read(path,maxstep=500):
 sup={};sub={}
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);
    if d['step']<=maxstep:sup[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=maxstep:sub[lane,t,k]=list(map(float,v[6:]))
 return sup,sub
def eq(s,b,l,t):return (l,t) in s and (l,t) in b and all(s[l,t][f]==b[l,t][f] for f in ('q','v'))
def ints(sub,l,t):
 v=[sub.get((l,t,k)) for k in range(16)]
 return None if any(x is None for x in v) else [sum(x[i] for x in v) for i in range(7)]
old,osub=read(B/'hn50-ap18.stderr.gz');control,csub=read(A/'parity.stderr.gz');pd=[t for t in range(1,501) if not eq(old,control,0,t) or any(osub.get((0,t,k))!=csub.get((0,t,k)) for k in range(16))];out={'control_lane0_exact':not pd,'control_first_difference':min(pd) if pd else None,'seeds':{}}
for seed in ('63','91'):
 f,fs=read(A/('seed'+seed+'-flight.stderr.gz'));g,gs=read(A/('seed'+seed+'-leg.stderr.gz'));lanes=[];geom=[]
 for l in range(8):
  first=next((t for t in range(249,271) if ints(gs,l,t) is not None and sum(ints(gs,l,t)[2:4])<=1e-7),None)
  rising=first is not None and first<270 and all((l,t) in g and g[l,t]['v'][2]>.05 and ints(gs,l,t) is not None and sum(ints(gs,l,t)[2:4])<=1e-7 for t in (first,first+1))
  re=next((t for t in range(first+1,271) if ints(gs,l,t) is not None and sum(ints(gs,l,t)[2:4])+ints(gs,l,t)[6]>1e-7),None) if first is not None else None
  end=re-1 if re else 270;leg_rise=max((g[l,t]['q'][2]-g[l,247]['q'][2] for t in range(249,end+1) if (l,t) in g),default=-1);other=max((ints(gs,l,t)[6] for t in range(249,271) if ints(gs,l,t) is not None),default=float('inf'));legpass=rising and leg_rise>.02 and other<=1e-7
  if first is not None and first<270:
   for t in (first,first+1):
    if (l,t) in g:geom.append((l,t,g[l,t]))
  diffs=[t for t in range(1,271) if not eq(f,g,l,t) or any(fs.get((l,t,k))!=gs.get((l,t,k)) for k in range(16))];match=first is not None and not any(t<=first+1 for t in diffs)
  missing=[t for t in range(1,501) if ints(fs,l,t) is None];ffirst=next((t for t in range(249,501) if ints(fs,l,t) is not None and sum(ints(fs,l,t)[2:4])<=1e-7),None);ground=None
  if ffirst:
   for t in range(ffirst,501):
    z=ints(fs,l,t)
    if z is not None and z[2]+z[3]+z[6]>1e-7:
     ground={'step':t,'foot_normal_impulse_Ns':z[2]+z[3],'nonfoot_impulse_Ns':z[6],'root_position_m':f[l,t]['q'][:3] if (l,t) in f else None};break
  metrics={}
  for t in (353,500):
   if (l,t) in f and (l,247) in f:
    q=f[l,t]['q'];v=f[l,t]['v'];metrics[str(t)]={'height_gain_m':q[2]-f[l,247]['q'][2],'forward_speed_m_s':v[0],'tilt_rad':math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2)))}
  seen=[(t,f[l,t]['q'][0]-f[l,247]['q'][0]) for t in range(247,501) if (l,t) in f and (l,247) in f];m=metrics.get('353',{});z=metrics.get('500',{});ok=not pd and legpass and match and not missing and ffirst==first and ground is None and m.get('height_gain_m',-1)>=.15 and m.get('forward_speed_m_s',-1)>.2 and m.get('tilt_rad',9)<=.35 and z.get('forward_speed_m_s',-1)>.2 and z.get('tilt_rad',9)<=.35
  lanes.append({'lane':l,'initial_observed_q':f.get((l,1),{}).get('q'),'initial_observed_v':f.get((l,1),{}).get('v'),'prepush_q':f.get((l,247),{}).get('q'),'prepush_v':f.get((l,247),{}).get('v'),'leg_first_unload':first,'leg_two_rising_intervals':rising,'leg_precontact_height_gain_m':leg_rise,'leg_max_nonfoot_impulse_Ns':other,'first_paired_difference':min(diffs) if diffs else None,'leg_jump_exact_in_flight':match,'first_flight_unload':ffirst,'first_flight_ground_recontact':ground,'min_relative_x_m':min(seen,key=lambda x:x[1]) if seen else None,'metrics':metrics,'pass_before_geometry':ok})
 text=''.join(str(l)+' '+str(t)+' '+' '.join(map(str,d['q']+d['v']))+'\n' for l,t,d in geom);p=subprocess.run(['build/crow-v15-walk-all-lanes-geometry-20261111/kinematics'],input=text,text=True,capture_output=True,check=True);poses={}
 for line in p.stdout.splitlines():
  v=line.split();poses[int(v[0]),int(v[1]),v[2]]=list(map(float,v[3:]))
 with gzip.open(A/('seed'+seed+'-leg.replay.json.gz'),'rt') as handle:replay=json.load(handle)['payload']
 frames={x['step']:x for x in replay['frames']};maxpos=0.;maxq=0.
 for l,t,d in geom:
  if l!=1:continue
  frame=frames[t-1];maxq=max(maxq,max(abs(a-b) for a,b in zip(frame['q'],d['q'])))
  for name in replay['body_names']:
   if (l,t,name) not in poses:continue
   z=poses[l,t,name];idx=replay['body_names'].index(name)*13;v=frame['body_states'][idx:idx+13];maxpos=max(maxpos,math.sqrt(sum((a-b)**2 for a,b in zip(z[:3],v[:3]))))
 geomvalid=maxq<1e-6 and maxpos<1e-5
 for row in lanes:
  l=row['lane'];first=row['leg_first_unload'];gaps=[]
  if first is not None:
   for t in (first,first+1):
    for side in ('left','right'):
     for name,h in [('crow_'+side+'_foot',(.010,.01725,.0092))]+[('crow_'+side+'_'+x,(.018,.0025,.0092)) for x in ('front_toe_1','front_toe_2','front_toe_3','rear_toe')]:
      z=poses.get((l,t,name))
      if z:
       x,y,q,w=z[3:7];axis=(2*(x*q-y*w),2*(y*q+x*w),1-2*(x*x+y*y));gaps.append(z[2]-sum(abs(a)*b for a,b in zip(axis,h)))
  row['minimum_two_pose_foot_gap_m']=min(gaps) if len(gaps)==20 else None;row['geometry_pass']=geomvalid and len(gaps)==20 and min(gaps)>.001;row['pass']=row['pass_before_geometry'] and row['geometry_pass']
 data=json.loads((A/('seed'+seed+'-flight.stdout')).read_text());control_data=json.loads((A/('seed'+seed+'-leg.stdout')).read_text());out['seeds'][seed]={'lanes':lanes,'replay_max_q_error':maxq,'replay_max_position_error_m':maxpos,'failed_environment_steps':data['failed_environment_steps'],'termination_count':data['termination_count'],'leg_failed_environment_steps':control_data['failed_environment_steps'],'leg_termination_count':control_data['termination_count'],'pass':all(r['pass'] for r in lanes) and not data['failed_environment_steps'] and not data['termination_count'] and not control_data['failed_environment_steps'] and not control_data['termination_count']}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('control exact',out['control_lane0_exact'],'first difference',out['control_first_difference'])
for seed,o in out['seeds'].items():
 print('seed',seed,'pass',sum(x['pass'] for x in o['lanes']),'/8','geometry',sum(x['geometry_pass'] for x in o['lanes']),'leg',sum(bool(x['leg_two_rising_intervals']) for x in o['lanes']),'paired',sum(x['leg_jump_exact_in_flight'] for x in o['lanes']),'recontacts',sum(x['first_flight_ground_recontact'] is not None for x in o['lanes']))
 for x in o['lanes']:print(' ',x['lane'],'pass',x['pass'],'unload',x['leg_first_unload'],'firstdiff',x['first_paired_difference'],'clear',x['minimum_two_pose_foot_gap_m'],'vx353',x['metrics'].get('353',{}).get('forward_speed_m_s'),'ground',x['first_flight_ground_recontact'])
