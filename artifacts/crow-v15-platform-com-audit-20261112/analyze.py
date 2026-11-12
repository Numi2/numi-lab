from pathlib import Path
import subprocess,gzip,json,collections,math,os
A=Path('artifacts/crow-v15-platform-com-audit-20261112');D=Path('artifacts/crow-v15-platform-pitch-balance-screen-20261112/autonomous-20261126-posture');meta=json.loads((D/'run.stdout').read_text())['first_episode_navigation'];end=meta['steps_by_environment'][7];states={};contacts=collections.defaultdict(list)
p=subprocess.Popen(['cat',*[str(x) for x in sorted(D.glob('run.stderr.gz.part[0-9][0-9][0-9]'))]],stdout=subprocess.PIPE)
with gzip.open(p.stdout,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);t=x['step']
   if x['lane']==7 and 3750<=t<=end and t%25==0:states[t]=x
  elif line.startswith('CROW_SCENE_CONTACT '):
   x=line.split();t,k,e,a,b=map(int,x[1:6])
   if e!=7 or not 3750<=t<=end or t%25 or 40 not in (a,b):continue
   body=b if a==40 else a
   if body not in (11,15) and not 19<=body<=34:continue
   nimp,t1,t2=map(float,x[6:9]);point=list(map(float,x[9:12]));n=list(map(float,x[13:16]));u=list(map(float,x[16:19]));v=[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]];sign=1 if a==40 else -1
   force=[sign*(n[i]*nimp+u[i]*t1+v[i]*t2) for i in range(3)];contacts[t].append((point,force))
assert p.wait()==0
with gzip.open(D/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
assert replay['environment']==7 and replay['body_state_stride']==13
frames={fr['step']+1:fr for fr in replay['frames']};names=replay['body_names'];assert all(t in frames for t in states)
inp=''.join('7 '+str(t)+' '+' '.join(map(str,states[t]['q']+states[t]['v']))+'\n' for t in sorted(states));env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};out=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env).stdout
com={};maxerr=0.;nbody=0
for line in out.splitlines():
 z=line.split()
 if z[0]=='com':com[int(z[2])]=list(map(float,z[3:]))
 elif z[0]=='body':
  t=int(z[2]);name=z[3]
  if name not in names:continue
  idx=names.index(name)*13;actual=frames[t]['body_states'][idx:idx+3];calc=list(map(float,z[4:7]));maxerr=max(maxerr,math.dist(actual,calc));nbody+=1
assert maxerr<1e-5,(maxerr,nbody)
assert set(com)==set(states)
rows=[]
for t in sorted(states):
 q=states[t]['q'];mass,cx,cy,cz=com[t]; J=[0.,0.,0.];moment=0.;weighted=0.
 for point,f in contacts[t]:
  J=[J[i]+f[i] for i in range(3)];moment+=(point[2]-cz)*f[0]-(point[0]-cx)*f[2];weighted+=point[0]*f[2]
 rows.append({'step':t,'pitch_rad':math.asin(max(-1.,min(1.,2*(q[6]*q[4]-q[5]*q[3])))),'mass_kg':mass,'com_x_m':cx,'com_z_m':cz,'root_x_m':q[0],'vertical_foot_impulse_ns':J[2],'contact_x_relative_com_m':weighted/J[2]-cx if abs(J[2])>1e-7 else None,'foot_contact_pitch_angular_impulse_about_com_nms':moment})
r={'scope':'Lane 7 only on a previously used development course and older unqualified controller; first episode only. Full-body center of mass reconstructed from the v15 articulated model and independently checked against stored body positions. The vertical-impulse-weighted location is diagnostic; summed contact impulse is not a full momentum budget, as gravity, aero, motor transfers, and other contacts must also be considered.', 'validation':{'body_position_max_error_m':maxerr,'body_states_checked':nbody,'samples':len(states)},'rows':rows};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print('validation',r['validation'])
for row in rows:
 if row['step']%100==0:print(row)
