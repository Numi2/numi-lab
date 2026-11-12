from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-strong-bank-early-contact-observer-20261112');R=Path('artifacts/crow-v15-stronger-bank-feedback-screen-20261112')
def load(root):
 result=json.loads((root/'feedback.stdout').read_text())['first_episode_navigation'];ends=result['steps_by_environment'];samples={};contacts={};pairs=[];targets={};qsub={}
 with gzip.open(root/'feedback.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e=x['lane'];t=x['step']
    if t<=ends[e]: samples[e,t]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=ends[e]:contacts[e,t,k]=z[4:]
   elif line.startswith('CROW_FAILURE_PAIR '):
    z=line.split();t,k,e,a,b=map(int,z[1:6])
    if t<=ends[e]:pairs.append({'step':t,'substep':k,'lane':e,'bodyA':a,'bodyB':b,'pair_key':z[6],'impulse':list(map(float,z[7:10])),'point':list(map(float,z[10:13])),'separation':float(z[13]),'normal':list(map(float,z[14:17])),'pre_solve_normal_velocity':float(z[17])})
   elif line.startswith('CROW_DRIVE_TARGET '):
    z=line.split();t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
   elif line.startswith('CROW_Q_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);qsub[e,t,k]=list(map(float,z[4:]))
 return result,samples,contacts,pairs,targets,qsub
old=load(R);new=load(A)
bad_sample=[k for k,v in old[1].items() if new[1].get(k)!=v];bad_contact=[k for k,v in old[2].items() if new[2].get(k)!=v]
coverage={'reference_state_samples':len(old[1]),'state_mismatches':len(bad_sample),'reference_contact_samples':len(old[2]),'contact_mismatches':len(bad_contact),'same_first_episodes':old[0]==new[0],'examples':{'states':bad_sample[:5],'contacts':bad_contact[:5]}}
valid=not bad_sample and not bad_contact and old[0]==new[0] and len(old[2])>100000 and len(old[1])>=90
first=sorted(new[3],key=lambda x:(x['step'],x['substep']))[0] if new[3] else None
with gzip.open(R/'feedback-replay.json.gz','rt') as f: payload=json.load(f)['payload'];names=payload['body_names']
if first:
 for field in ('bodyA','bodyB'):
  n=first[field];first[field+'_name']=names[n] if n<len(names) else None
 t=first['step'];prev=new[1].get((4,t-1));q=prev['q'] if prev else None
 if q:
  x,y,z,w=q[3:7];first['prior_control_state']={'step':t-1,'position_m':q[:3],'roll_rad':math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),'pitch_rad':math.asin(max(-1,min(1,2*(w*y-z*x)))),'yaw_rad':math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),'linear_velocity_m_s':prev['v'][:3],'angular_velocity_rad_s':prev['v'][3:6],'policy_action':prev['policy_action'],'policy_action_caveat':'Raw policy output, not necessarily the executed fixed-controller command.'}
 first['position_drive_targets_at_contact_control_step']=new[4].get((4,t))
out={'valid_matched_replay':valid,'parity':coverage,'first_observed_positive_impulse_pair_in_logged_window':first,'pair_count_in_window':len(new[3]),'q_substep_count':len(new[5]),'target_sample_count':len(new[4]),'scope':'Observer only covers a prespecified window near a failure in one development lane. Body pair and positive impulse do not establish fatal cause or uniquely name collider shapes. No biological or flight qualification.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if not valid:raise SystemExit('Observer parity failed')
