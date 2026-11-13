from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-low-wing-com-audit-20261113';exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib'
arms={'push':R/'artifacts/crow-v15-low-wing-effort-screen-20261113/push-wing-20261126-rebound-low','hold':R/'artifacts/crow-v15-low-wing-effort-screen-20261113/hold-wing-20261126-rebound-low'}
data={}; inputs=[];counter=0;keys={}
for arm,p in arms.items():
 state={};contact={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step'];
    if 346<=key[1]<=380:assert key not in state;state[key]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 348<=t<=380:assert (e,t,k) not in contact;contact[e,t,k]=list(map(float,z[6:11]))
 assert len(state)==8*35 and len(contact)==8*33*16,(arm,len(state),len(contact))
 with gzip.open(p/'trace.tsv.gz','rt') as f:head=f.readline().strip()
 assert 'timestep=0.02' in head,head
 data[arm]=(state,contact)
 for (e,t),x in sorted(state.items()):keys[counter]=(arm,e,t);inputs.append(f'{counter} 0 '+' '.join(map(str,x['q']+x['v'])));counter+=1
q=data['push'][0][5,347]['q'];zero=[0.]*40;cal=zero.copy();cal[:3]=[.3,-.2,.1]
for idx,v in ((counter,zero),(counter+1,cal)):inputs.append(f'{idx} 0 '+' '.join(map(str,q+v)))
r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});out={}
for l in r.stdout.splitlines():
 z=l.split();assert z[0]=='C';i=int(z[1]);x=list(map(float,z[3:]));assert len(x)==7 and all(map(math.isfinite,x));out[i]=x
assert len(out)==counter+2
translation_error=max(abs((out[counter+1][4+j]-out[counter][4+j])-cal[j]) for j in range(3));assert translation_error<1e-8,translation_error
com={keys[i]:v for i,v in out.items() if i<counter};rows=[]
for arm,(state,contact) in data.items():
 for e in range(8):
  before=com[arm,e,347];after=com[arm,e,365];mass=before[0];assert abs(mass-.45006)<1e-5 and abs(after[0]-mass)<1e-10
  jfoot=sum(sum(contact[e,t,k][:2]) for t in range(348,366) for k in range(16));dp=mass*(after[6]-before[6]);jg=-mass*9.81*.02*18;residual=dp-jfoot-jg
  unloaded=[t for t in range(351,381) if all(sum(contact[e,t,k][2:4])<=1e-7 for k in range(16))];release=next((t for t in unloaded if t+1 in unloaded and t+2 in unloaded),None);peak=max(com[arm,e,t][3]-com[arm,e,350][3] for t in range(351,381));release_vz=com[arm,e,release][6] if release is not None else None;rows.append({'first_three_unloaded_control_intervals':release,'com_vz_at_release_m_s':release_vz,'com_peak_rise_from_350_m':peak,'arm':arm,'lane':e,'modeled_mass_kg':mass,'com_z_before_m':before[3],'com_z_after_m':after[3],'com_vz_before_m_s':before[6],'com_vz_after_m_s':after[6],'delta_vertical_momentum_ns':dp,'upward_foot_impulse_ns':jfoot,'gravity_impulse_assuming_g_9_81_ns':jg,'unattributed_momentum_residual_ns':residual})
intervals=[]
for arm,(state,contact) in data.items():
 for e in range(2,8):
  for lo,hi in ((348,355),(356,365)):
   before=com[arm,e,lo-1];after=com[arm,e,hi];mass=before[0]
   jfoot=sum(sum(contact[e,t,k][:2]) for t in range(lo,hi+1) for k in range(16))
   dp=mass*(after[6]-before[6]);jg=-mass*9.81*.02*(hi-lo+1)
   intervals.append({'arm':arm,'lane':e,'steps':[lo,hi],'delta_com_momentum_ns':dp,'foot_impulse_ns':jfoot,'assumed_gravity_impulse_ns':jg,'residual_ns':dp-jfoot-jg,'com_vz_before':before[6],'com_vz_after':after[6]})
(A/'interval-momentum.json').write_text(json.dumps({'scope':'Posthoc COM endpoint and foot impulse comparison. Gravity assumed; aerodynamic, body wrench and other residual forces not isolated.','rows':intervals},indent=2)+'\n')
for row in intervals:print(row['arm'],row['lane'],row['steps'],'dp',round(row['delta_com_momentum_ns'],4),'foot',round(row['foot_impulse_ns'],4),'residual',round(row['residual_ns'],4))
