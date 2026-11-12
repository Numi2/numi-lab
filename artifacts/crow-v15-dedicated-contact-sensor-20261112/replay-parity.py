from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-dedicated-contact-sensor-20261112');B=Path('artifacts/crow-v15-negative-ankle-matched-control-20261112');seed=int(sys.argv[1]);assert seed in (20261126,20261127)
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)['payload']
a=read(A/f'candidate-{seed}-rebound-phase'/'replay.json.gz');b=read(B/f'candidate-{seed}-rebound-phase'/'replay.json.gz');assert a['environment']==b['environment']==5;assert len(a['frames'])==len(b['frames'])
keys=('q','v','reward','actor_observation','accepted_actions','body_states','done','termination_reason','timeout','tracking_score','outcomes');out={'seed':seed,'environment':5,'frames':len(a['frames']),'differences':{}}
for k in keys:
 dif=[i for i,(x,y) in enumerate(zip(a['frames'],b['frames'])) if x[k]!=y[k]];out['differences'][k]={'count':len(dif),'first_frame':dif[0] if dif else None}
(A/f'replay-parity-{seed}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
