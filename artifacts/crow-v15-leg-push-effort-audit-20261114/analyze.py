from pathlib import Path
import collections,gzip,hashlib,json
A=Path('artifacts/crow-v15-leg-push-effort-audit-20261114');p=Path('artifacts/crow-v15-two-low-wing-20261114/push-window.json.gz');d=json.load(gzip.open(p,'rt'));limits=[.75,.75,.600000024,.75,.75,.600000024];names=['left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle'];rows=[]
for e in (6,7):
 for lo,hi in ((350,368),(369,374),(375,380)):
  a=[x for x in d['leg'] if x[0]==e and lo<=x[1]<=hi];keys=[tuple(x[:3]) for x in a]
  assert len(keys)==16*(hi-lo+1)==len(set(keys));joints=[]
  for j,(name,lim) in enumerate(zip(names,limits)):
   effort=[x[3+5*j] for x in a];speed=[x[7+5*j] for x in a]
   joints.append({'joint':name,'min_effort_Nm':min(effort),'max_effort_Nm':max(effort),'abs_max_effort_Nm':max(map(abs,effort)),'limit_Nm':lim,'fraction_at_effort_limit':sum(abs(v)>=lim-1e-5 for v in effort)/len(effort),'max_abs_observed_velocity_rad_s':max(map(abs,speed))})
  rows.append({'lane':e,'intervals':[lo,hi],'unique_substeps':len(keys),'joints':joints})
out={'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_compressed_sha256':d['source_sha256'],'rows':rows,'scope':'Retrospective used-seed observer audit; effort and accepted/presolve velocities are modeled; q/v and contact epochs inferred for these lanes.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
