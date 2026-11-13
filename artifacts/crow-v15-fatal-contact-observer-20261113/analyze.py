from pathlib import Path
import gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-fatal-contact-observer-20261113';P={'observer':A/'push-wing-20261126-rebound-fatal','reference':R/'crow-v15-crow-period-probe-20261113/push-wing-20261126-rebound-crowperiod'}
def read(p):
 d={'state':{},'foot':{},'kin':{},'pre':{},'wing':{},'ground':{},'errors':[]};d['meta']=json.loads((p/'run.stdout').read_text())['first_episode_navigation'];d['result']=json.loads((p/'run-result.json').read_text())
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   z=l.split()
   if not z:continue
   if z[0]=='CROW_SUPPORT_SAMPLE':
    x=json.loads(l.split(' ',1)[1]);key=(x['lane'],x['step']);assert key not in d['state'];d['state'][key]=x
   elif z[0]=='CROW_TWO_TOE_SUBSTEP':
    t,k,e=map(int,z[1:4]);key=(e,t,k);assert key not in d['foot'];d['foot'][key]=list(map(float,z[6:11]));
    if int(z[4])>int(z[5]):d['errors'].append('overflow')
   elif z[0]=='CROW_KIN_STATE':
    t,k,e,nq,nv=map(int,z[1:6]);v=list(map(float,z[6:]));assert len(v)==nq+nv;d['kin'][e,t,k]={'q':v[:nq],'v':v[nq:]}
   elif z[0]=='CROW_PRESOLVE_V':
    t,k,e=map(int,z[1:4]);d['pre'][e,t,k]=list(map(float,z[5:]))
   elif z[0]=='CROW_WING_EFFORT':
    t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==14;d['wing'][e,t,k]=[v[:7],v[7:]]
   elif z[0]=='CROW_GROUND_CONTACT':
    t,k,e=map(int,z[1:4]);a,b=map(int,z[4:6]);d['ground'].setdefault((e,t,k),[]).append({'body_a':a,'body_b':b,'raw':z[6:]})
   elif z[0] in ('CROW_AERO_OBSERVER_ERROR','CROW_SUBSTEP_ERROR'):d['errors'].append(l.strip())
 return d
d={n:read(p) for n,p in P.items()};a,b=d['observer'],d['reference'];errors=[]
for name,x in d.items():
 if x['result']['exit_code'] or x['errors']:errors.append([name,x['result'],x['errors']])
for k in ('steps_by_environment','termination_reason_by_environment','physical_failure_by_environment'):
 if a['meta'][k]!=b['meta'][k]:errors.append(['meta',k])
for e in range(8):
 for t in range(1,401):
  if any(a['state'][e,t][k]!=b['state'][e,t][k] for k in ('q','v','policy_action','contacts')):errors.append(['state',e,t])
 for t in range(335,401):
  for k in range(16):
   if a['foot'][e,t,k]!=b['foot'][e,t,k]:errors.append(['foot',e,t,k])
for e in (2,3):
 for t in range(348,386):
  for k in range(16):
   if (e,t,k) not in a['kin'] or (e,t,k) not in a['pre'] or (e,t,k) not in a['wing']:errors.append(['missing',e,t,k])
if errors:(A/'integrity-failed.json').write_text(json.dumps(errors,indent=2)+'\n');raise SystemExit('Observer parity invalid')
rows=[]
for e in (2,3):
 hits=[(t,k,v) for (lane,t,k),v in a['ground'].items() if lane==e and 348<=t<=385 and any(x['body_a'] in (5,6) or x['body_b'] in (5,6) for x in v)]
 assert hits,(e,'no wing ground pairs');t,k,v=min(hits,key=lambda x:(x[0],x[1]));q=a['kin'][e,t,k];s=a['state'][e,t-1];rows.append({'lane':e,'first_logged_wing_ground_step':t,'substep':k,'pairs':v,'start_q':q['q'],'start_v':q['v'],'precontact_v':a['pre'][e,t,k],'wing_buffer_target_and_state':a['wing'][e,t,k],'previous_control_q':s['q'],'previous_control_v':s['v'],'previous_policy_actions':s['policy_action'],'first_episode_step_count':a['meta']['steps_by_environment'][e],'termination_reason':a['meta']['termination_reason_by_environment'][e]})
(A/'analysis.json').write_text(json.dumps({'scope':'parity-matched failure replay; positive-impulse pairs logged 348–385; contact-pair occurrence and task termination are distinct observations','rows':rows},indent=2)+'\n')
for x in rows:print('lane',x['lane'],'first wing-ground',x['first_logged_wing_ground_step'],x['substep'],'pairs',[(p['body_a'],p['body_b']) for p in x['pairs']],'root_z',x['start_q'][2],'root_vz',x['start_v'][2],'episode_steps',x['first_episode_step_count'])
