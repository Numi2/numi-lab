from pathlib import Path
import gzip,json,collections
R=Path.cwd();A=R/'artifacts/crow-v15-slow-flight-fold-control-20261111'
with gzip.open(A/'slow/replay.json.gz','rt') as f:cur=json.load(f)['payload']
with gzip.open(R/'artifacts/crow-v15-flight-fold-self-contact-observer-20261111/reference/replay.json.gz','rt') as f:ref=json.load(f)['payload']
assert cur['environment']==ref['environment']==1 and len(cur['frames'])==len(ref['frames'])==600
mismatch=[i+1 for i,(a,b) in enumerate(zip(cur['frames'][:370],ref['frames'][:370])) if any(a[k]!=b[k] for k in ('q','v','actor_observation','accepted_actions'))]
counts=collections.Counter();pairs=collections.defaultdict(set);maximum={}
with gzip.open(A/'slow/run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SELF_CONTACT '):
   z=line.split();t,k,e,a,b=map(int,z[1:6]);imp=float(z[6]);key=(e,a,b);counts[key]+=1;pairs[key].add((t,k));maximum[key]=max(imp,maximum.get(key,0))
contacts=[{'lane':e,'bodyA':cur['body_names'][a],'bodyB':cur['body_names'][b],'contact_samples':n,'distinct_step_substeps':len(pairs[e,a,b]),'max_normal_impulse':maximum[e,a,b]} for (e,a,b),n in sorted(counts.items())]
samples=[]
for i in (370,400,450,500,550,600):
 x=cur['frames'][i-1];y=ref['frames'][i-1];samples.append({'frame':i,'slow_fold_root_velocity_m_s':x['v'][:3],'reference_root_velocity_m_s':y['v'][:3],'slow_fold_root_position_m':x['q'][:3],'reference_root_position_m':y['q'][:3]})
out={'pre_intervention_lane1_first_difference':mismatch[:1],'contacts':contacts,'samples':samples,'scope':'Five-lane obstacle-free screen, self-contact log only steps 380–600; velocity/position comparison lane 1 only. Does not qualify route, landing, toe release, or biological accuracy.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
