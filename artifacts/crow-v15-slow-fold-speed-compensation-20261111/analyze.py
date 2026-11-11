from pathlib import Path
import collections,gzip,json,math,statistics
R=Path.cwd();A=R/'artifacts/crow-v15-slow-fold-speed-compensation-20261111';d=A/'full'
with gzip.open(d/'replay.json.gz','rt') as f:cur=json.load(f)['payload']
with gzip.open(R/'artifacts/crow-v15-slow-flight-fold-control-20261111/slow/replay.json.gz','rt') as f:prior=json.load(f)['payload']
assert cur['environment']==prior['environment']==1 and len(prior['frames'])==600
mismatch=[i+1 for i,(x,y) in enumerate(zip(cur['frames'][:370],prior['frames'][:370])) if any(x[k]!=y[k] for k in ('q','v','actor_observation','accepted_actions'))]
stdout=json.loads((d/'run.stdout').read_text());first=stdout['first_episode_navigation'];samples=[];contacts=collections.Counter();contact_steps=collections.defaultdict(set);caps=[0]*5;caps_after500=[0]*5;coverage=[0]*5;ground_after400=[0]*5;errors=[]
with gzip.open(d/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_FLIGHT_SAMPLE '):samples.append(json.loads(line.split(' ',1)[1]))
  elif line.startswith('CROW_SELF_CONTACT '):
   z=line.split();t,k,e,a,b=map(int,z[1:6]);contacts[e,a,b]+=1;contact_steps[e,a,b].add((t,k))
  elif line.startswith('CROW_TOE_V_SUBSTEP '):
   z=line.split();t=int(z[1]);e=int(z[3]);v=max(abs(float(x)) for x in z[4:]);coverage[e]+=1;caps[e]+=v>=11.99;caps_after500[e]+=(t>=500 and v>=11.99)
  elif line.startswith('CROW_SCENE_CONTACT '):
   z=line.split();t,k,e,a,b=map(int,z[1:6]);imp=float(z[6]);ground_after400[e]+=(t>=400 and 35 in (a,b) and imp>1e-7)
  elif line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
rows=[]
for e in range(5):
 rate=[0.,.03,-.03,0.,0.][e];target=[.85,.85,.85,1.05,.65][e];ss=sorted((x for x in samples if x['lane']==e and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][e]),key=lambda x:x['step']);vals=[]
 for x in ss:
  q=x['q'];v=x['v'];a,b,c,w=q[3:7];yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c));desired=rate*(x['step']*.02-7.2);head=abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw)));forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];vals.append((head,abs(lateral),abs(forward-.35),abs(q[2]-target)))
 maxima=[max((x[i] for x in vals),default=None) for i in range(4)];passed=(not first['physical_failure_by_environment'][e] and first['steps_by_environment'][e]==1600 and [x['step'] for x in ss]==list(range(1000,1501,100)) and all(maxima[i] is not None and maxima[i]<=v for i,v in enumerate([.15,.20,.20,.15])) and caps[e]==0 and ground_after400[e]==0 and not any(key[0]==e for key in contacts))
 rows.append({'lane':e,'steps':first['steps_by_environment'][e],'physical_failure':first['physical_failure_by_environment'][e],'sample_steps':[x['step'] for x in ss],'max_heading_error_rad':maxima[0],'max_lateral_m_s':maxima[1],'max_speed_error_m_s':maxima[2],'max_height_error_m':maxima[3],'toe_cap_substeps':caps[e],'toe_cap_substeps_after500':caps_after500[e],'toe_substep_records':coverage[e],'ground_contact_samples_after400':ground_after400[e],'pass_screen':passed})
self_hits=[{'lane':e,'bodyA':cur['body_names'][a],'bodyB':cur['body_names'][b],'samples':n,'distinct_step_substeps':len(contact_steps[e,a,b])} for (e,a,b),n in sorted(contacts.items())]
out={'first370_lane1_first_difference':mismatch[:1],'failed_environment_steps':stdout['failed_environment_steps'],'self_contacts':self_hits,'lanes':rows,'substep_errors':errors[-10:],'intent_speed_m_s':.35,'applied_command_m_s_during_full_fold':.40,'scope':'Five-lane obstacle-free bounded-command diagnostic; self contacts observed only steps 380–1600. Heading, speed and altitude are six sampled time points per lane. Does not qualify perturbed recovery, route, supported landing or biological realism.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
