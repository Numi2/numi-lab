from pathlib import Path
import collections,gzip,json,math,os,subprocess
A=Path('artifacts/crow-v15-gait-slip-observer-20261112')
C=A/'sham-20261126'
R=Path('artifacts/crow-v15-substep-target-interpolation-screen-20261112/sham-20261126')
def load(p):
 sup={};sub={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);e,t=x['lane'],x['step'];assert (e,t) not in sup;sup[e,t]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in sub;sub[e,t,k]=z[4:]
 return sup,sub
ref,rc=load(R);cur,cc=load(C)
assert len(ref)==len(cur)==2400 and len(rc)==len(cc)==38400
def strip(x):
 x=dict(x);x['contacts']=[{k:v for k,v in c.items() if k!='point'} for c in x['contacts']];return x
sm=[k for k in ref if strip(cur[k])!=ref[k]]
cm=[k for k in rc if cc[k]!=rc[k]]
par={'state_count':len(ref),'substep_contact_count':len(rc),'state_mismatches':len(sm),'contact_mismatches':len(cm),'examples':{'states':sm[:3],'contacts':cm[:3]}}
assert not sm and not cm,par
window={(e,t):cur[e,t] for e in range(8) for t in range(201,301)}
inp=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(window.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
bodies={}
for l in p.stdout.splitlines():
 z=l.split()
 if z[0]=='body':
  v=list(map(float,z[4:]));assert len(v)==13 and all(math.isfinite(x) for x in v)
  bodies[int(z[1]),int(z[2]),z[3]]=v
with gzip.open(C/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
assert replay['environment']==5 and replay['body_state_stride']==13
frames={x['step']+1:x for x in replay['frames'] if 200<=x['step']<=299}
assert len(frames)==100
footnames=[n for n in replay['body_names'] if n.startswith(('crow_left_','crow_right_')) and ('foot' in n or 'toe' in n)]
poserr=velerr=angerr=qerr=groundspeed=0.
for t,f in frames.items():
 qerr=max(qerr,max(abs(a-b) for a,b in zip(f['q'],window[5,t]['q'])))
 bn=replay['body_names'];g=bn.index('locomotion_ground')*13;groundspeed=max(groundspeed,*map(abs,f['body_states'][g+7:g+13]))
 for name in footnames:
  i=bn.index(name)*13;v=f['body_states'][i:i+13];k=bodies[5,t,name]
  poserr=max(poserr,math.dist(v[:3],k[:3]));velerr=max(velerr,math.dist(v[7:10],k[7:10]));angerr=max(angerr,math.dist(v[10:13],k[10:13]))
check={'max_q_error':qerr,'max_position_error_m':poserr,'max_linear_velocity_error_m_s':velerr,'max_angular_velocity_error_rad_s':angerr,'max_ground_speed':groundspeed,'replay_samples':len(frames),'foot_bodies_per_sample':len(footnames)}
(A/'parity.json').write_text(json.dumps({'trace_parity':par,'kinematics_replay_validation':check},indent=2)+'\n')
assert qerr<1e-6 and poserr<1e-5 and velerr<1e-4 and angerr<1e-4 and groundspeed<1e-7,check
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
values=collections.defaultdict(list);nonvertical=0;points=0
for (e,t),x in window.items():
 for c in x['contacts']:
  if c['impulses'][0]<=1e-7:continue
  name=c['a_name'] if c['b_name']=='locomotion_ground' else c['b_name'] if c['a_name']=='locomotion_ground' else ''
  if name not in footnames:continue
  assert 'point' in c and len(c['point'])==3 and all(math.isfinite(z) for z in c['point'])
  if abs(c['normal'][2])<.99:nonvertical+=1;continue
  b=bodies[e,t,name];r=[c['point'][i]-b[i] for i in range(3)];omega=cross(b[10:13],r);v=[b[7+i]+omega[i] for i in range(3)]
  side='left' if name.startswith('crow_left_') else 'right'
  values[e,side].append(math.hypot(v[0],v[1]));points+=1
def stat(v):
 v=sorted(v)
 if not v:return {'count':0}
 return {'count':len(v),'median_m_s':v[(len(v)-1)//2],'p95_m_s':v[int(.95*(len(v)-1))],'max_m_s':v[-1],'fraction_over_0_01m_s':sum(x>.01 for x in v)/len(v)}
rows=[{'lane':e,'side':side,**stat(values[e,side])} for e in range(8) for side in ('left','right')]
out={'trace_parity':par,'kinematics_replay_validation':check,'contact_point_samples':points,'nonvertical_contacts_excluded':nonvertical,'rows':rows,'scope':'Instantaneous accepted endpoint contact-point velocity over 2 s; not integrated slip distance, friction, or biological gait validation. Contacts are correlated samples.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
