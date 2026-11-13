from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-push-ankle-alignment-20261113';exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib'
arms={'push':R/'artifacts/crow-v15-push-ankle-alignment-20261113/push-wing-20261126-rebound-ankle','hold':R/'artifacts/crow-v15-supported-slow-push-screen-20261113/hold-wing-20261126-rebound-slowpush'}
data={}; inputs=[];counter=0;keys={}
for arm,p in arms.items():
 state={};contact={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step'];
    if 346<=key[1]<=400:assert key not in state;state[key]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 348<=t<=400:assert (e,t,k) not in contact;contact[e,t,k]=list(map(float,z[6:11]))
 assert len(state)==8*55 and len(contact)==8*53*16,(arm,len(state),len(contact))
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
  unloaded=[t for t in range(351,401) if all(sum(contact[e,t,k][2:4])<=1e-7 for k in range(16))];release=next((t for t in unloaded if t+1 in unloaded and t+2 in unloaded),None);peak=max(com[arm,e,t][3]-com[arm,e,350][3] for t in range(351,401));release_vz=com[arm,e,release][6] if release is not None else None;rows.append({'first_three_unloaded_control_intervals':release,'com_vz_at_release_m_s':release_vz,'com_peak_rise_from_350_m':peak,'arm':arm,'lane':e,'modeled_mass_kg':mass,'com_z_before_m':before[3],'com_z_after_m':after[3],'com_vz_before_m_s':before[6],'com_vz_after_m_s':after[6],'delta_vertical_momentum_ns':dp,'upward_foot_impulse_ns':jfoot,'gravity_impulse_assuming_g_9_81_ns':jg,'unattributed_momentum_residual_ns':residual})
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result={'translation_velocity_calibration_max_error_m_s':translation_error,'window_steps':[348,365],'control_dt_s':.02,'rows':rows,'source_hashes':{'executable':h(exe),'dylib':h(lib)},'scope':'Posthoc whole-v15 COM momentum accounting for a supported wing transition screen, one used seed and correlated lanes. The two logged foot impulses, assumed uniform -9.81 m/s^2 gravity, and FK endpoint momentum do not isolate aerodynamic, body wrench, numerical or unlogged forces. The residual is not independently measured aero. Timing and model identity require independent validation before causal interpretation.'};(A/'com-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('translation test max error',translation_error)
for row in rows:
 if row['lane']>=2:print(row['arm'],row['lane'],'dp',round(row['delta_vertical_momentum_ns'],4),'feet',round(row['upward_foot_impulse_ns'],4),'gravity',round(row['gravity_impulse_assuming_g_9_81_ns'],4),'residual',round(row['unattributed_momentum_residual_ns'],4))
