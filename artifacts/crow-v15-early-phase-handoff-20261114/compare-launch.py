from pathlib import Path
import gzip,json,math,hashlib
root=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');paths=[root/'crow-v15-common-flight-role-20261114/straight-20261126-rebound-commonrole/replay.json.gz',root/'crow-v15-phase-aligned-handoff-20261114/push-20261126-rebound-phase/replay.json.gz'];runs=[]
for p in paths:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 x=json.load(gzip.open(p,'rt'))['payload'];assert x['environment']==5 and x['frame_count']>=500;runs.append((h.hexdigest(),x))
rows=[]
for t in (375,400,410,420,430,440,450,475,500):
 pair=[]
 for sha,x in runs:
  a=x['frames'][t-1];assert a['step']==t;q=a['q'];v=a['v'];pitch=math.asin(max(-1.,min(1.,2*(q[6]*q[4]-q[5]*q[3]))));pair.append({'height_m':q[2],'pitch_rad':pitch,'vx_m_s':v[0],'vz_m_s':v[2],'pitch_rate_rad_s':v[4]})
 rows.append({'step':t,'older_role6':pair[0],'phase_aligned':pair[1]})
out={'source_compressed_sha256':[r[0] for r in runs],'environment':5,'rows':rows,'limitation':'Different prior runs, not causal pairs; both used development seed and training layout.'};print(json.dumps(out,indent=2))
