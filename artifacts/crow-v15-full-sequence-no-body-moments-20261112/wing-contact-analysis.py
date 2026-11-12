from pathlib import Path
import gzip,importlib.util,json,math,os,subprocess
R=Path.cwd();A=Path('artifacts/crow-v15-full-sequence-no-body-moments-20261112');P=A/'candidate-20261127-rebound-flexed';states={};contacts={};coords={};caps={};errors=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];states[e,t]=[x[k] for k in ('q','v','policy_action','contacts')]
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in contacts;contacts[e,t,k]=z[4:]
   if int(z[4])>int(z[5]):errors.append('contact overflow')
  elif line.startswith('CROW_Q_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);q=list(map(float,z[4:]));assert (e,t,k) not in coords and len(q)==41 and all(math.isfinite(x) for x in q);coords[e,t,k]=q
  elif line.startswith('CROW_TOE_V_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);caps[e,t,k]=max(abs(float(v)) for v in z[4:])
assert not errors and len(coords)==8*31*16,(errors[:3],len(coords))
err=max(abs(x-y) for (e,t,k),q in coords.items() if k==15 for x,y in zip(q,states[e,t][0]));assert err<1e-6,err
inp=''.join(f'{e} {t} {k} '+' '.join(map(str,v))+'\n' for (e,t,k),v in sorted(coords.items()));env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/crow-v15-hip-roll-journey-integration-20261111')};proc=subprocess.run(['build/crow-v15-takeoff-substep-wing-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,env=env,check=True);body={}
for line in proc.stdout.splitlines():
 z=line.split();body[int(z[1]),int(z[2]),int(z[3]),z[4]]=list(map(float,z[5:]))
assert len(body)==len(coords)*3
with gzip.open(P/'replay.json.gz','rt') as f:rp=json.load(f)['payload']
assert rp['environment']==5;frames={x['step']+1:x for x in rp['frames'] if 334<=x['step']<=364};assert len(frames)==31
poserr=roterr=0.
for t,f in frames.items():
 for name in ('crow_body','crow_left_wing','crow_right_wing'):
  x=body[5,t,15,name];i=rp['body_names'].index(name)*13;y=f['body_states'][i:i+7];poserr=max(poserr,math.dist(x[:3],y[:3]));roterr=max(roterr,min(math.dist(x[3:7],y[3:7]),math.dist(x[3:7],[-v for v in y[3:7]])))
assert poserr<1e-5 and roterr<1e-5,(poserr,roterr)
spec=importlib.util.spec_from_file_location('audit','artifacts/crow-v15-whole-gait-audit-20261111/audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def wingz(e,t,k,side):
 q=coords[e,t,k];p=body[e,t,k,'crow_'+side+'_wing'];return a.rotate_inv(q[3:7],[p[i]-q[i] for i in range(3)])[2]
rows=[]
for e in range(8):
 seq=[(t,k,sum(float(x) for x in contacts[e,t,k][4:6])>1e-7) for t in range(340,366) for k in range(16)];trans=[];last=seq[0][2]
 for t,k,loaded in seq[1:]:
  if loaded!=last:trans.append([t,k,'loaded' if loaded else 'unloaded']);last=loaded
 first=next(((t,k) for t,k,s in trans if s=='unloaded'),None)
 if first:
  idx=next(i for i,(t,k,_) in enumerate(seq) if (t,k)==first);before=seq[idx-1][:2];window=seq[:idx];imp=sum(sum(float(x) for x in contacts[e,t,k][2:4]) for t,k,_ in window); weight=.45006*9.81*.00125*len(window);wing={side:wingz(e,*before,side)-wingz(e,339,15,side) for side in ('left','right')}
 else: before=None;imp=weight=None;wing=None
 rows.append({'lane':e,'initially_loaded':seq[0][2],'contact_transitions':trans,'first_unloaded_substep':first,'last_substep_before_first_unload':before,'wing_center_delta_m_before_first_unload':wing,'foot_upward_impulse_to_first_unload_Ns':imp,'model_weight_impulse_same_window_Ns':weight,'toe_velocity_caps_340_370':sum(v>=11.99 for (lane,t,k),v in caps.items() if lane==e and 340<=t<=370)})
out={'rows':rows,'endpoint_substep_q_max_error':err,'replay_fk_position_error_m':poserr,'replay_fk_orientation_error':roterr,'scope':'Used development seed; first-unload transitions are explicit and may be followed by recontact. Wing-center movement is relative to body, not feather or force; within-substep ordering cannot be resolved. Contact impulse includes body weight and non-muscle forces.'}
(A/'wing-contact-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
