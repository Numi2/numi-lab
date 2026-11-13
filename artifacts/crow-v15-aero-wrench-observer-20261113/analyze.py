from pathlib import Path
import collections,gzip,json,math,subprocess,os
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-aero-wrench-observer-20261113';E=R/'artifacts/crow-v15-ankle-push-reach-20261113';O=R/'artifacts/crow-v15-leg-work-observer-20261113';S=R/'artifacts/crow-v15-slow-leg-push-20261113';arms={'observer':A/'push-wing-20261126-rebound-aero','reference':R/'artifacts/crow-v15-smooth-release-wing-20261113/push-wing-20261126-rebound-smoothrelease'}
def load(p):
 d={'state':{},'contact':{},'work':{},'toe':{},'target':{},'phase':{},'errors':[]};d['run_result']=json.loads((p/'run-result.json').read_text());d['meta']=json.loads((p/'run.stdout').read_text())['first_episode_navigation'];
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];
    if 1<=t<=400:assert (e,t) not in d['state'];d['state'][(e,t)]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in d['contact'];d['contact'][(e,t,k)]=list(map(float,z[6:11]));
    if int(z[4])>int(z[5]):d['errors'].append(['overflow',e,t,k])
   elif line.startswith('CROW_LEG_EFFORT '):
    z=line.split();t,k,e=map(int,z[1:4]);assert tuple(map(int,z[4:7]))==(1,0,0);v=list(map(float,z[7:]));assert len(v)==30;d['work'][(e,t,k)]=[v[i:i+5] for i in range(0,30,5)]
   elif line.startswith('CROW_PHASE_SAMPLE '):
    z=line.split();t,e,st=map(int,z[1:4]);assert (e,t) not in d['phase'];d['phase'][e,t]=st
   elif line.startswith('CROW_DRIVE_TARGET '):
    z=line.split();t,e=map(int,z[1:3]);v=list(map(float,z[3:]));assert len(v)==40 and (e,t) not in d['target'];d['target'][(e,t)]=v
   elif line.startswith('CROW_TOE_V_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);d['toe'][(e,t,k)]=max(abs(float(x)) for x in z[4:])
   elif line.startswith('CROW_SUBSTEP_ERROR'):d['errors'].append(line.strip())
 if d['run_result']['exit_code']!=0:d['errors'].append('nonzero exit')
 for e in range(8):
  if d['meta']['steps_by_environment'][e]<400:d['errors'].append(['early first-episode end',e])
  if sum((e,t) in d['state'] for t in range(1,401))!=400:d['errors'].append(['missing state',e])
  if sum((e,t,k) in d['contact'] for t in range(335,401) for k in range(16))!=66*16:d['errors'].append(['missing contact',e])
  if sum((e,t,k) in d['work'] for t in range(335,401) for k in range(16))!=66*16:d['errors'].append(['missing effort',e])
  if sum((e,t,k) in d['toe'] for t in range(335,401) for k in range(16))!=66*16:d['errors'].append(['missing toe',e])
  if sum((e,t) in d['target'] for t in range(335,401))!=66:d['errors'].append(['missing target',e])
  if sum((e,t) in d['phase'] for t in range(335,401))!=66:d['errors'].append(['missing phase',e])
 return d
D={n:load(p) for n,p in arms.items()}; parity=[]
for e in range(8):
 bad=[t for t in range(1,401) if any(D['observer']['state'][e,t][k]!=D['reference']['state'][e,t][k] for k in ('q','v','policy_action','contacts'))]
 bc=[(t,k) for t in range(335,401) for k in range(16) if D['observer']['contact'][e,t,k]!=D['reference']['contact'][e,t,k]]
 parity.append({'lane':e,'state_mismatches':len(bad),'first_state_mismatch':bad[:1],'contact_mismatches':len(bc),'first_contact_mismatch':bc[:1]})
aero={};kin={};errs=[];strides=set()
with gzip.open(arms['observer']/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_KIN_STATE '):
   z=line.split();t,k,e,nq,nv=map(int,z[1:6]);v=list(map(float,z[6:]));assert len(v)==nq+nv and (e,t,k) not in kin;kin[e,t,k]=(v[:nq],v[nq:])
  elif line.startswith('CROW_AERO_OBSERVER_ERROR'):errs.append(line.strip())
  elif line.startswith('CROW_AERO_FORCE '):
   z=line.split();t,k,e,stride=map(int,z[1:5]);v=list(map(float,z[5:]));assert len(v)==12 and all(map(math.isfinite,v)),(z,len(v));assert (e,t,k) not in aero;assert v[-1]==4.0,('status',v[-1]);aero[e,t,k]=v;strides.add(stride)
if len(aero)!=8*28*16 or len(kin)!=8*28*16 or len(strides)!=1 or min(strides)<35 or errs or any(x['errors'] for x in D.values()) or any(x['state_mismatches'] or x['contact_mismatches'] for x in parity):
 (A/'integrity-failed.json').write_text(json.dumps({'parity':parity,'aero_records':len(aero),'strides':list(strides),'aero_errors':errs,'run_errors':{n:d['errors'] for n,d in D.items()}},indent=2)+'\n');raise SystemExit('Observer invalid; see integrity-failed.json')
inputs=[];keys={}
for e in range(8):
 for t in (347,357,365,375):
  x=D['observer']['state'][e,t];i=len(inputs);keys[i]=(e,t);inputs.append(str(i)+' 0 '+' '.join(map(str,x['q']+x['v'])))
exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113';z=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib)});cm={}
for line in z.stdout.splitlines():
 x=line.split();assert x[0]=='C';v=list(map(float,x[3:]));assert len(v)==7;cm[keys[int(x[1])]]=v
assert len(cm)==32
rows=[]
for e in range(8):
 for end in (357,365,375):
  keys2=[(e,t,k) for t in range(348,end+1) for k in range(16)];ja=sum(aero[x][2] for x in keys2)*.02/16;parts=[sum(aero[x][j] for x in keys2)*.02/16 for j in range(3,7)];jf=sum(sum(D['observer']['contact'][x][:2]) for x in keys2);mass=cm[e,347][0];dp=mass*(cm[e,end][6]-cm[e,347][6]);jg=-mass*9.81*.02*(end-347);rows.append({'lane':e,'end_step':end,'aero_impulse_z_ns':ja,'aero_root_left_right_tail_z_ns':parts,'foot_impulse_z_ns':jf,'assumed_gravity_impulse_z_ns':jg,'com_delta_momentum_z_ns':dp,'unattributed_residual_ns':dp-jf-ja-jg})
phase=[]
for e in range(2,8):
 a=next(x for x in rows if x['lane']==e and x['end_step']==357);b=next(x for x in rows if x['lane']==e and x['end_step']==365)
 phase.append({'lane':e,'steps':[358,365],'aero_impulse_z_ns':b['aero_impulse_z_ns']-a['aero_impulse_z_ns'],'foot_impulse_z_ns':b['foot_impulse_z_ns']-a['foot_impulse_z_ns'],'assumed_gravity_impulse_z_ns':b['assumed_gravity_impulse_z_ns']-a['assumed_gravity_impulse_z_ns'],'com_delta_momentum_z_ns':b['com_delta_momentum_z_ns']-a['com_delta_momentum_z_ns'],'residual_ns':b['unattributed_residual_ns']-a['unattributed_residual_ns']})
wing_transients=[]
for e in range(2,8):
 key=max((x for x in kin if x[0]==e),key=lambda x:max(abs(kin[x][1][8]),abs(kin[x][1][9])))
 vmax=max(abs(kin[key][1][8]),abs(kin[key][1][9]));wing_transients.append({'lane':e,'max_substep_wing_speed_rad_s':vmax,'step':key[1],'substep':key[2],'aero_vertical_force_n':aero[key][2],'foot_normal_impulse_ns':sum(D['observer']['contact'][key][2:4]),'root_vz_m_s':kin[key][1][2]})
force_profile=[]
for t in range(348,376):
 v=[aero[e,t,k][2] for e in range(2,8) for k in range(16)]
 force_profile.append({'step':t,'mean_aero_vertical_force_n':sum(v)/len(v),'min_aero_vertical_force_n':min(v),'max_aero_vertical_force_n':max(v)})
out={'scope':'Bracketed simulated aerodynamic kernel force; no biological validation. Delta of all body force slots before/after kernel, integrated using stated control dt and substeps. Gravity is assumed, not independently read from the runtime dispatch. Remaining residual can include other applied forces, dynamics and sampling/rounding.','parity':parity,'aero_records':len(aero),'body_strides':list(strides),'rows':rows,'phase_358_365':phase,'force_profile_by_step':force_profile,'wing_substep_speed_peaks':wing_transients};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:
 if x['lane']>=2 and x['end_step'] in (357,365,375):print(x['lane'],x['end_step'],'aero',round(x['aero_impulse_z_ns'],4),'feet',round(x['foot_impulse_z_ns'],4),'dp',round(x['com_delta_momentum_z_ns'],4),'residual',round(x['unattributed_residual_ns'],4))
