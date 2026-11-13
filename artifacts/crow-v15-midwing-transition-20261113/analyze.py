from pathlib import Path
import collections,gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-midwing-transition-20261113';E=R/'artifacts/crow-v15-ankle-push-reach-20261113';O=R/'artifacts/crow-v15-leg-work-observer-20261113';S=R/'artifacts/crow-v15-slow-leg-push-20261113';arms={'push':A/'push-wing-20261127-rebound-midwing','hold':A/'hold-wing-20261127-rebound-midwing'}
def load(p):
 d={'state':{},'contact':{},'work':{},'toe':{},'target':{},'errors':[]};d['run_result']=json.loads((p/'run-result.json').read_text());d['meta']=json.loads((p/'run.stdout').read_text())['first_episode_navigation'];
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
 return d
D={n:load(p) for n,p in arms.items()}
parity=[]
for e in range(8):
 bad=[t for t in range(1,346) if any(D['push']['state'][e,t][k]!=D['hold']['state'][e,t][k] for k in ('q','v','policy_action','contacts'))]
 bc=[(t,k) for t in range(335,346) for k in range(16) if D['push']['contact'][e,t,k]!=D['hold']['contact'][e,t,k]]
 parity.append({'lane':e,'state_mismatches':len(bad),'contact_mismatches':len(bc)})
reference_state={};reference_contact={}
with gzip.open(R/'artifacts/crow-v15-leg-wing-transition-20261113/push-wing-20261127-rebound-transition/run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
   if t<=345:reference_state[e,t]=x
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4])
   if 335<=t<=345:reference_contact[e,t,k]=list(map(float,z[6:11]))
assert len(reference_state)==8*345 and len(reference_contact)==8*11*16
reference_parity=[]
for e in range(8):
 mismatch=sum(any(D['push']['state'][e,t][k]!=reference_state[e,t][k] for k in ('q','v','policy_action','contacts')) for t in range(1,346))
 contact=sum(D['push']['contact'][e,t,k]!=reference_contact[e,t,k] for t in range(335,346) for k in range(16))
 reference_parity.append({'lane':e,'state_mismatches':mismatch,'contact_mismatches':contact})
if any(x['errors'] for x in D.values()) or any(x['state_mismatches'] or x['contact_mismatches'] for x in parity+reference_parity):
 (A/'integrity-failed.json').write_text(json.dumps({'parity':parity,'reference_parity':reference_parity,'errors':{n:d['errors'] for n,d in D.items()}},indent=2)+'\n');raise SystemExit('Invalid paired screen; see integrity-failed.json')
rows=[]
for e in range(8):
 row={'lane':e}
 for name,d in D.items():
  foot=lambda lo,hi:sum(sum(d['contact'][e,t,k][:2]) for t in range(lo,hi+1) for k in range(16))
  support=lambda t:any(sum(d['contact'][e,t,k][2:4])>1e-7 for k in range(16))
  release=next((t for t in range(351,399) if not any(support(u) for u in range(t,t+3))),None)
  recontacts=[] if release is None else [u for u in range(release+3,401) if support(u)]
  unloaded400=None if release is None else not any(support(u) for u in range(release,401))
  rise=max(d['state'][e,t]['q'][2]-d['state'][e,350]['q'][2] for t in range(351,401))
  toe=sum(d['toe'][e,t,k]>=11.99 for t in range(348,401) for k in range(16))
  nonfoot=sum(d['contact'][e,t,k][4] for t in range(348,401) for k in range(16))
  targets=[[t,*d['target'][e,t][8:10]] for t in range(335,401)]
  row[name]={'impulse_348_357_ns':foot(348,357),'impulse_348_365_ns':foot(348,365),'impulse_348_375_ns':foot(348,375),'first_three_unloaded':release,'root_rise_max_m':rise,'root_z400_m':d['state'][e,400]['q'][2],'root_vz400_m_s':d['state'][e,400]['v'][2],'unloaded_through_400':unloaded400,'later_supported_control_intervals':recontacts,'nonfoot_impulse_348_400_ns':nonfoot,'toe_speed_threshold_substeps':toe,'flap_drive_target_rad_steps_335_400':targets}
 row['early_impulse_delta_ns']=row['push']['impulse_348_357_ns']-row['hold']['impulse_348_357_ns']
 rows.append(row)
out={'parity':parity,'reference_parity':reference_parity,'errors':{n:d['errors'] for n,d in D.items()},'rows':rows,'scope':'One used development seed, correlated lanes. Applied flap-joint targets from DOF indices 8/9 and sampled endpoints; high-resolution contact records extend through 400. No qualification.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print('lane',r['lane'],'early delta',round(r['early_impulse_delta_ns'],4),'push release',r['push']['first_three_unloaded'],'hold release',r['hold']['first_three_unloaded'],'push rise',round(r['push']['root_rise_max_m'],4),'hold rise',round(r['hold']['root_rise_max_m'],4),'nonfoot through400',r['push']['nonfoot_impulse_348_400_ns'],'toe caps',r['push']['toe_speed_threshold_substeps'])
