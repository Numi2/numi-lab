from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-overlap-long-flight-20261113');P=A/'push-wing-20261126-rebound-long';m=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];sc=collections.defaultdict(list);term={}
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_CONTACT_TRACE '):
   x=json.loads(line.split(' ',1)[1]);e=x['lane'];
   if e>=2 and e not in term and x['step'] in (m['steps_by_environment'][e],m['steps_by_environment'][e]+1):term[e]=x
  elif line.startswith('CROW_SCENE_CONTACT '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if e>=2 and t<=m['steps_by_environment'][e]+1:sc[e].append({'tick':t,'substep':k,'body_a':int(z[4]),'body_b':int(z[5]),'record':z[6:]})
rows=[]
for e in range(2,8):
 t=term.get(e);rows.append({'lane':e,'first_episode_steps':m['steps_by_environment'][e],'termination_reason':m['termination_reason_by_environment'][e],'first_scene_contact':sc[e][0] if sc[e] else None,'contact_termination_trace':t})
out={'rows':rows,'scope':'CONTACT termination is code 3. Contact-trace step uses the host trace submission convention and can be one greater than first-episode step count. Earlier ground/toe contact is reported separately in analysis.json. These are modeled collider events, not evidence of solver failure.'};(A/'fatal-contact.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 t=r['contact_termination_trace'];print(r['lane'],'end',r['first_episode_steps'],'trace',None if t is None else t['step'],'scene',None if t is None else [(x['a_name'],x['b_name'],x['normal_impulse']) for x in t['contacts'] if 'crow_course_' in x['a_name'] or 'crow_course_' in x['b_name']])
