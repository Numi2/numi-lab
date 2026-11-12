from pathlib import Path
import collections,gzip,hashlib,json,math,os,statistics,subprocess
R=Path('.');A=R/'artifacts/crow-v15-foot-settling-geometry-audit-20261112';prior=R/'artifacts/crow-v15-negative-ankle-matched-control-20261112';trial=R/'artifacts/crow-v15-front-toe-settling-screen-20261112';seq0=json.loads((R/'artifacts/crow-v15-toe-corvid-sequence-audit-20261112/analysis.json').read_text());seq1=json.loads((trial/'toe-sequence.json').read_text());ground=next(x['ground_z_m'] for x in json.loads((R/'artifacts/crow-v15-whole-gait-audit-20261111/audit.json').read_text())['rows'] if x['seed']==20261127);kin=R/'build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics'
def sha(p):
 z=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):z.update(b)
 return z.hexdigest()
def box_clearance(v,half):
 x,y,z,w=v[3:7];normal=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))
 return v[2]-ground-sum(abs(a)*b for a,b in zip(normal,half))
assert abs(box_clearance([0,0,ground+0.01,0,0,0,1],(.009,.0025,.0092))-.0008)<1e-12
allout={}
for seed in (20261126,20261127):
 for arm,base,seq in (('previous',prior,seq0),('trial',trial,seq1)):
  D=base/f'candidate-{seed}-rebound-phase';source=D/'run.stderr.gz';replay=D/'replay.json.gz';selected=[];keys=set()
  for row in seq[f'{seed}-candidate']['rows']:
   for x in row['events']:
    if x['type']!='touchdown' or x['order']!='hallux_first':continue
    first=x['to_sample']-round(x['gap_ms']/1.25);last=x['to_sample'];mid=(first+last)//2
    assert 0<=first<=mid<=last<1600
    ev={'lane':row['lane'],'side':row['side'],'first_sample':first,'both_sample':last,'gap_ms':x['gap_ms'],'samples':{}}
    for label,i in (('first',first),('middle',mid),('both',last)):
     ev['samples'][label]=i;keys.add((row['lane'],i))
    selected.append(ev)
  # Validate endpoint forward-kinematics against the independently written native replay.
  validation=[(5,(t-201)*16+15) for t in (201,220,240,260,280,300)];keys.update(validation);q={};v={}
  with gzip.open(source,'rt') as f:
   for line in f:
    if not line.startswith(('CROW_Q_SUBSTEP ','CROW_V_SUBSTEP ')):continue
    z=line.split();t,k,e=map(int,z[1:4]);i=(t-201)*16+k
    if (e,i) in keys:(q if z[0]=='CROW_Q_SUBSTEP' else v)[e,i]=list(map(float,z[4:]))
  assert q.keys()==v.keys()==keys,(seed,arm,len(q),len(v),len(keys))
  ordered=sorted(keys);request=''.join(f'{e} {i} '+' '.join(map(str,q[e,i]+v[e,i]))+'\n' for e,i in ordered)
  env={**os.environ,'DYLD_LIBRARY_PATH':str((R/'build/crow-v15-hip-roll-journey-integration-20261111').resolve())};p=subprocess.run([str(kin)],input=request,text=True,capture_output=True,env=env,check=True)
  names={f'crow_{side}_{name}' for side in ('left','right') for name in ('foot','front_toe_2_proximal','front_toe_2_distal','rear_toe_proximal','rear_toe_distal')};b={}
  for line in p.stdout.splitlines():
   z=line.split()
   if z and z[0]=='body' and z[3] in names:b[int(z[1]),int(z[2]),z[3]]=list(map(float,z[4:]))
  assert len(b)==len(keys)*len(names),(seed,arm,len(b),len(keys))
  with gzip.open(replay,'rt') as f:rp=json.load(f)['payload']
  error=0.;rotation_error=0.;frame_by_step={fr['step']+1:fr for fr in rp['frames']};rnames=rp['body_names']
  for e,i in validation:
   t=201+i//16;fr=frame_by_step[t]
   for name in names:
    idx=rnames.index(name)*13;raw=fr['body_states'][idx:idx+7];body=b[e,i,name];error=max(error,math.dist(raw[:3],body[:3]));rotation_error=max(rotation_error,min(math.dist(raw[3:7],body[3:7]),math.dist(raw[3:7],[-v for v in body[3:7]])))
  assert error<1e-5 and rotation_error<1e-5,(seed,arm,error,rotation_error)
  for ev in selected:
   e=ev['lane'];side=ev['side'];xout={}
   for label,i in ev['samples'].items():
    get=lambda name:b[e,i,f'crow_{side}_{name}']
    front=min(box_clearance(get('front_toe_2_'+part),(.009,.0025,.0092)) for part in ('proximal','distal'))
    rear=min(box_clearance(get('rear_toe_'+part),(.009,.0025,.0092)) for part in ('proximal','distal'))
    foot=box_clearance(get('foot'),(.010,.01725,.0092));raw=q[e,i];ki,ai=(15,16) if side=='left' else (18,19)
    xout[label]={'front_group_min_clearance_m':front,'rear_group_min_clearance_m':rear,'foot_sole_clearance_m':foot,'knee_coordinate_rad':raw[ki],'ankle_coordinate_rad':raw[ai]}
   ev['samples']=xout
  allout[f'{seed}-{arm}']={'source_sha256':sha(source),'replay_sha256':sha(replay),'kinematics_sha256':sha(kin),'ground_z_m':ground,'max_replay_body_position_error_m':error,'max_replay_quaternion_component_distance':rotation_error,'events':selected}
(A/'analysis.json').write_text(json.dumps(allout,indent=2)+'\n')
for key,x in allout.items():
 print(key,'n',len(x['events']),'front group clearance mm at first/mid/both',*[round(1000*statistics.median(e['samples'][k]['front_group_min_clearance_m'] for e in x['events']),3) for k in ('first','middle','both')],'maximum position error',x['max_replay_body_position_error_m'])
