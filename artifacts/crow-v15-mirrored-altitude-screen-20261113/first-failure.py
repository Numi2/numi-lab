from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-mirrored-altitude-screen-20261113');P=A/'signed-turn-20261126-rebound-altitude';meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];ends={e:meta['steps_by_environment'][e] for e in (6,7)};states={};targets={};contacts={};out=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
   if e in ends and ends[e]-5<=t<=ends[e]:states[e,t]=x
   continue
  z=line.split()
  if not z:continue
  if z[0]=='CROW_DRIVE_TARGET':
   t,e=map(int,z[1:3]);
   if e in ends and ends[e]-5<=t<=ends[e]:targets[e,t]=list(map(float,z[3:]))
  elif z[0] in ('CROW_GROUND_CONTACT','CROW_SCENE_CONTACT','CROW_SELF_CONTACT'):
   t,sub,e=map(int,z[1:4]);
   if e in ends and 600<=t<=ends[e] and (e,z[0]) not in contacts:contacts[e,z[0]]=line.strip()
for e,end in ends.items():
 x=states.get((e,end-1));assert x and (e,end-1) in targets
 out.append({'lane':e,'termination_tick':end,'termination_reason_raw':meta['termination_reason_by_environment'][e],'physical_failure':meta['physical_failure_by_environment'][e],'first_recorded_contacts_after_tick600':{k[1]:v for k,v in contacts.items() if k[0]==e},'state_one_control_tick_before_end':{'step':end-1,'q':x['q'],'v':x['v'],'policy_action':x['policy_action'],'drive_target':targets[e,end-1]},'scope':'Contact records limited to configured ground, scene and self-contact observers; absence of a logged pair does not establish absence of a physical contact.'})
(P/'first-failure.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
