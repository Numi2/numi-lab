from pathlib import Path
import gzip,json,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-carrion-biofit-20261112/expanded';paths=['crow-v15-sweep-pitch-feedback-screen-20261112','crow-v15-bank-pitch-feedback-screen-20261112','crow-v15-flight-sideslip-feedback-screen-20261112','crow-v15-forward-pitch-feedback-screen-20261112'];out={}
for name in paths:
 p=R/'artifacts'/name/'feedback-replay.json.gz';
 with gzip.open(p,'rt') as f:d=json.load(f)['payload']
 assert d['timestep_seconds']==.02 and len(d['frames'])==1600
 rows=[]
 for x in d['frames']:
  v=x['v'];rows.append({'step':x['step'],'left_flap_rad_s':v[8],'right_flap_rad_s':v[9]})
 out[name]={'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'flap_v_indices':[8,9],'rows':rows}
(A/'sim-flap-velocity.json').write_text(json.dumps(out,separators=(',',':'))+'\n');print('screens',len(out))
