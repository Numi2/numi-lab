from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-toe-soft-drive-probe-20261110');rows=[]
for l in (A/'fk-check-output.txt').read_text().splitlines():
 v=l.split();rows.append((int(v[0]),v[1],list(map(float,v[2:]))))
with gzip.open('artifacts/crow-v15-full-substep-sensitivity-20261110/j-original-8-rebound.json.gz','rt') as f:d=json.load(f)['payload']
frames={x['step']:x for x in d['frames']};names=d['body_names'];out={}
for off in (0,-1,1):
 pe=[];qe=[]
 for t,name,pose in rows:
  if t+off not in frames:continue
  i=names.index(name)*13;b=frames[t+off]['body_states'][i:i+7];pe.append(math.dist(pose[:3],b[:3]));qe.append(min(math.dist(pose[3:],b[3:]),math.dist(pose[3:],[-x for x in b[3:]])))
 out[str(off)]={'comparisons':len(pe),'max_position_error_m':max(pe,default=None),'max_quaternion_euclidean_error':max(qe,default=None)}
(A/'fk-check-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
