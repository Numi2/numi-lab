from pathlib import Path
import gzip,json,math,subprocess,os,sys
seed=20261126
S=Path('artifacts/crow-v15-autonomous-gated-phase-walk-20261111/autonomous-20261126');cmd=json.loads((S/'command.json').read_text());nsub=int(cmd[cmd.index('--physics-substeps')+1]);sup={};sub={}
from contextlib import ExitStack
from itertools import chain
with ExitStack() as stack:
 f=chain.from_iterable(stack.enter_context(gzip.open(p,'rt')) for p in sorted(S.glob('run.stderr.gz')))
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);
   if 201<=x['step']<=600:sup[x['lane'],x['step']]=x
  elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
   x=l.split();t,k,e=map(int,x[1:4]);
   if 201<=t<=600:sub[e,t,k]=list(map(float,x[6:]))
assert len(sup)==8*400 and len(sub)==8*400*nsub
text=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(sup.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics'],input=text,text=True,capture_output=True,check=True,env=env);body={}
for l in p.stdout.splitlines():
 x=l.split()
 if x[0]=='body':body[int(x[1]),int(x[2]),x[3]]=list(map(float,x[4:]))
with gzip.open(S/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
assert replay['environment']==5
frames={x['step']:x for x in replay['frames'] if 200<=x['step']<=599};assert len(frames)==400
qerr=max(abs(a-b) for t,f in frames.items() for a,b in zip(sup[5,t+1]['q'],f['q']));assert qerr<1e-6
names=replay['body_names'];ground=names.index('locomotion_ground');groundz=[f['body_states'][ground*13+2] for f in frames.values()];assert max(groundz)-min(groundz)<1e-6
z0=sum(groundz)/len(groundz);toe_names=lambda side:['crow_'+side+'_foot']+['crow_'+side+'_'+d+'_'+seg for d in ('front_toe_1','front_toe_2','front_toe_3','rear_toe') for seg in ('proximal','distal')]
maxpos=maxrot=0.
for t,f in frames.items():
 for name in toe_names('left')+toe_names('right'):
  y=body[5,t+1,name];i=names.index(name)*13;z=f['body_states'][i:i+13]
  maxpos=max(maxpos,math.sqrt(sum((a-b)**2 for a,b in zip(y[:3],z[:3]))));maxrot=max(maxrot,min(math.sqrt(sum((a-b)**2 for a,b in zip(y[3:7],z[3:7]))),math.sqrt(sum((a+b)**2 for a,b in zip(y[3:7],z[3:7])))))
assert maxpos<1e-5 and maxrot<1e-5,(maxpos,maxrot)
def bouts(flags):
 out=[];start=None
 for i,x in enumerate(flags+[False]):
  if x and start is None:start=i
  if not x and start is not None:
   if i-start>=3:out.append([201+start,200+i])
   start=None
 return out
rows=[]
for e in range(8):
 flags=[[],[]];minload=[];other=[];nofoot=[];bothclear=[];tilt=[];speed=[]
 for t in range(201,601):
  x=sup[e,t];q=x['q'];v=x['v'];tilt.append(math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2)))));speed.append(math.sqrt(sum(a*a for a in v[:3])));i=[sum(sub[e,t,k][j] for k in range(nsub)) for j in range(37)];minload.append(i[0]+i[1]);other.append(i[4]);nofoot.append(i[2]+i[3])
  stepgaps=[]
  for side,index in [('left',0),('right',1)]:
   gaps=[]
   for name in toe_names(side):
    z=body[e,t,name];qx,qy,qz,qw=z[3:7];rot=(2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy));h=(.010,.01725,.0092) if name.endswith('_foot') else (.009,.0025,.0092);gaps.append(z[2]-z0-sum(abs(a)*b for a,b in zip(rot,h)))
   stepgaps.append(min(gaps))
   flags[index].append(min(gaps)>.001 and i[2+index]<=1e-7 and i[1-index]>=.5*.45006*9.81*.02 and i[4]<=1e-7)
  if min(stepgaps)>.001:bothclear.append(t)
 rows.append({'lane':e,'left_bouts':bouts(flags[0]),'right_bouts':bouts(flags[1]),'forward_displacement_m':sup[e,600]['q'][0]-sup[e,201]['q'][0],'min_foot_upward_impulse_Ns':min(minload),'intervals_below_half_nominal_weight':[201+i for i,x in enumerate(minload) if x < .5*.45006*9.81*.02],'both_feet_clear_over_1mm_endpoint_steps':bothclear,'no_foot_normal_intervals':[201+i for i,x in enumerate(nofoot) if x<=1e-7],'no_foot_normal_substeps':[[t,k] for t in range(201,601) for k in range(nsub) if sub[e,t,k][2]+sub[e,t,k][3]<=1e-7],'nonfoot_contact_intervals':sum(x>1e-7 for x in other),'max_tilt_rad':max(tilt),'max_speed_m_s':max(speed)})
r={'source':str(S),'replay_validation':{'lane':5,'observer_to_replay_offset':1,'q_max_error':qerr,'foot_toe_position_max_error_m':maxpos,'foot_toe_quaternion_max_error':maxrot,'replay_ground_z_m':z0},'rows':rows,'limitations':'Eight-second teacher-off development gait. Shape clearance sampled at control endpoints; all contact impulses summed over all configured substeps. The common ground height is taken from lane 1 replay and the same authored task, not independently replayed in every lane. Alternating bouts, slip, continuous clearance, toe speed, takeoff, landing, autonomous navigation, unseen starts and biological realism require separate tests.'}
(Path('artifacts/crow-v15-autonomous-gated-phase-walk-20261111')/'autonomous-walk-analysis.json').write_text(json.dumps(r,indent=2)+'\n');print('validation',r['replay_validation']);print([(x['lane'],len(x['left_bouts']),len(x['right_bouts']),round(x['forward_displacement_m'],4),x['nonfoot_contact_intervals']) for x in rows])
