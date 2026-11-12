from pathlib import Path
import collections,gzip,json,sys
A=Path('artifacts/crow-v15-matched-gait-second-seed-20261112');name='candidate'
files={'reference':A/'sham-20261127-dev/run.stderr.gz','candidate':A/(name+'-20261127-rebound/run.stderr.gz')}
def load(path):
 q={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_Q_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);q[e,t,k]=list(map(float,z[4:]))
 assert len(q)==12800
 return q
def stats(q,j):
 values=[]
 for e in range(8):
  for t in range(202,301):
   for k in range(16):
    def get(n):return q[e,t+n//16,n%16][j]
    values.append(abs(get(k)-2*get(k-1)+get(k-2))/(.00125**2))
 values.sort();return {'count':len(values),'mean':sum(values)/len(values),'p95':values[int(.95*(len(values)-1))],'max':values[-1]}
q={name:load(p) for name,p in files.items()}
out={'scope':'Post-hoc descriptive whole-substep finite-difference motion proxies across 8 dependent lanes from one used seed. These are not joint torque or biological acceleration validation.','rows':[]}
for name,j in (('left_knee',15),('left_ankle',16),('right_knee',18),('right_ankle',19)):
 out['rows'].append({'joint':name,'reference':stats(q['reference'],j),'candidate':stats(q['candidate'],j)})
(A/'candidate-posthoc-motion.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
