from pathlib import Path
import gzip,json,math,os,subprocess
from collections import defaultdict
A=Path('artifacts/crow-v15-perch-com-moment-audit-20261112');R=Path('artifacts/crow-v15-early-wing-sequence-screen-20261112/autonomous-20261126');previous=json.loads(Path('artifacts/crow-v15-perch-contact-moment-audit-20261112/analysis.json').read_text())
wanted={(r['lane'],s['step']) for r in previous['rows'] for ev in r['events'] for s in ev['samples']};q={};contacts=defaultdict(list)
with gzip.open(R/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step']
   if key in wanted:q[key]=x['q']
  elif line.startswith('CROW_SCENE_CONTACT '):
   z=line.split();t,k,e,ba,bb=map(int,z[1:6]);key=e,t
   if key in wanted and 40 in (ba,bb):
    n,t1,t2=map(float,z[6:9]);pt=list(map(float,z[9:12]));normal=list(map(float,z[13:16]));tan=list(map(float,z[16:19]));cross=[normal[1]*tan[2]-normal[2]*tan[1],normal[2]*tan[0]-normal[0]*tan[2],normal[0]*tan[1]-normal[1]*tan[0]];sign=1 if ba==40 else -1
    impulse=[sign*(n*normal[i]+t1*tan[i]+t2*cross[i]) for i in range(3)];contacts[key].append((pt,impulse))
assert set(q)==wanted
input_data=''.join(f'{e} {t} '+' '.join(map(str,q[e,t]))+'\n' for e,t in sorted(wanted))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path.cwd()/'build/crow-v15-hip-roll-journey-integration-20261111')}
p=subprocess.run(['build/crow-v15-perch-com-moment-audit-20261112/kinematics'],input=input_data,capture_output=True,text=True,env=env,check=True);com={};poses={};masses=[]
for l in p.stdout.splitlines():
 z=l.split();key=int(z[1]),int(z[2])
 if z[0]=='C':masses.append(float(z[3]));com[key]=list(map(float,z[4:7]))
 elif z[0]=='B':poses[key+(z[3],)]=list(map(float,z[4:11]))
assert set(com)==wanted and max(masses)-min(masses)<1e-10
with gzip.open(R/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
assert replay['environment']==5
frames={r['step']+1:r for r in replay['frames'] if r['step']+1 in (4084,4259)}
assert set(frames)=={4084,4259}
poserr=roterr=qerr=0.
for t,frame in frames.items():
 qerr=max(qerr,max(abs(a-b) for a,b in zip(q[5,t],frame['q'])))
 for i,name in enumerate(replay['body_names'][:35]):
  p=poses[5,t,name];ref=frame['body_states'][i*13:i*13+7]
  poserr=max(poserr,math.dist(p[:3],ref[:3]));roterr=max(roterr,min(math.dist(p[3:7],ref[3:7]),math.dist(p[3:7],[-x for x in ref[3:7]])))
assert poserr<1e-5 and roterr<1e-5 and qerr<1e-6
rows=[]
for r in previous['rows']:
 evs=[]
 for ev in r['events']:
  vals=[]
  for s in ev['samples']:
   key=r['lane'],s['step'];c=com[key];moment=0.
   for pt,j in contacts[key]:moment+=(pt[2]-c[2])*j[0]-(pt[0]-c[0])*j[2]
   vals.append({'step':s['step'],'pitch_rad':s['pitch_rad'],'pitch_rate_world_y':s['pitch_rate_world_y'],'center_of_mass_m':c,'perch_pitch_impulse_about_com_Nms':moment,'perch_upward_impulse_Ns':s['perch_impulse_Ns'][2]})
  evs.append({'threshold_rad':ev['threshold_rad'],'crossing_step':ev['crossing_step'],'sum_perch_pitch_impulse_about_com_Nms':sum(v['perch_pitch_impulse_about_com_Nms'] for v in vals),'samples':vals})
 rows.append({'lane':r['lane'],'events':evs})
out={'fk_validation':{'lane':5,'steps':sorted(frames),'bodies_compared_per_step':35,'max_position_error_m':poserr,'max_quaternion_error':roterr,'max_q_error':qerr,'model_mass_kg':masses[0]},'rows':rows,'limits':'Native FK model validated against 35 replay body poses in two lane-5 frames, not all states or independent anatomy. COM is computed from the estimated v15 model and sampled at each interval endpoint, while contacts are substep impulses. This is perch angular impulse only; direct body moments, aerodynamic effects and temporal COM movement are not included. These are used-seed development failures.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('fk',out['fk_validation'])
for r in rows:print(r['lane'],[(e['threshold_rad'],round(e['sum_perch_pitch_impulse_about_com_Nms'],6)) for e in r['events']])
