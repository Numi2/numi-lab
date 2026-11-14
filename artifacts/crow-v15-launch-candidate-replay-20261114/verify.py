from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-launch-candidate-replay-20261114');P=A/'push-20261126-rebound-lane7';O=Path('artifacts/crow-v15-near-wing-transition-20261113/push-20261126-rebound-near')
def states(folder):
 d={}
 with gzip.open(folder/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=(x['lane'],x['step']);assert key not in d;d[key]=x
 return d
r,s=states(O),states(P);par=[]
for e in range(8):
 mismatches=[t for t in range(1,501) if (e,t) not in r or (e,t) not in s or any(r[e,t][k]!=s[e,t][k] for k in ('q','v','policy_action','contacts'))]
 par.append({'lane':e,'mismatch_count':len(mismatches),'first_mismatches':mismatches[:5]})
old=json.loads((O/'inputs.json').read_text());new=json.loads((P/'inputs.json').read_text());oldout=json.loads((O/'run.stdout').read_text())['first_episode_navigation'];newout=json.loads((P/'run.stdout').read_text())['first_episode_navigation']
payload=json.load(gzip.open(P/'replay.json.gz','rt'))['payload'];frames=payload['frames'];errors=[];maxd=0
for f in frames:
 x=s.get((7,f['step']))
 if x is None or len(f['q'])!=len(x['q']) or len(f['v'])!=len(x['v']):errors.append(f['step']);continue
 maxd=max(maxd,max(abs(a-b) for a,b in zip(f['q']+f['v'],x['q']+x['v'])))
checks={'all_lanes_exact':all(x['mismatch_count']==0 for x in par),'same_inputs':old==new,'same_outcomes':oldout==newout,'candidate_replay_identity':payload['environment']==7 and len(frames)==500 and payload['frame_count']==500,'replay_state_match':not errors and maxd<1e-6}
o={'checks':checks,'pass':all(checks.values()),'parity':par,'replay_state_max_abs_error':maxd,'missing_replay_steps':errors[:10],'scope':'candidate development takeoff replay, not a mission or landing'};(A/'verification.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
