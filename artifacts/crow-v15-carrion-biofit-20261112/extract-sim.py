from pathlib import Path
import gzip,json,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-carrion-biofit-20261112';out={}
names=['crow-v15-sweep-pitch-feedback-screen-20261112','crow-v15-bank-pitch-feedback-screen-20261112','crow-v15-flight-sideslip-feedback-screen-20261112','crow-v15-forward-pitch-feedback-screen-20261112']
for name in names:
 p=R/'artifacts'/name/'feedback-replay.json.gz';h=hashlib.sha256(p.read_bytes()).hexdigest()
 with gzip.open(p,'rt') as f:doc=json.load(f)
 x=doc['payload'];assert x['timestep_seconds']==.02 and x['body_state_stride']==13 and len(x['frames'])==1600 and not x['scheduled_resets'];j=x['body_names'].index('crow_tail');states=[]
 for f in x['frames']:
  b=f['body_states'][j*13:(j+1)*13];assert len(b)==13;states.append({'step':f['step'],'orientation_xyzw':b[3:7],'velocity_world_m_s':b[7:10],'done':f['done']})
 out[name]={'input_path':str(p.relative_to(R)),'input_sha256':h,'payload_sha256':doc['payload_sha256'],'environment':x['environment'],'timestep_seconds':x['timestep_seconds'],'tail_index':j,'states':states}
(A/'sim-tail-states.json').write_text(json.dumps(out,separators=(',',':'))+'\n');print('screens',len(out))
