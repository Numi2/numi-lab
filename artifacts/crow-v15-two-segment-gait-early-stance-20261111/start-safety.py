from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-two-segment-gait-early-stance-20261111');B=Path('artifacts/crow-v15-two-segment-gait-uniform-adaptive-20261111')
def read(path):
 state={};coverage=collections.Counter()
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);state[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);coverage[e,t,k]+=1
 return state,coverage
s,c=read(A/'stderr.gz');ref,_=read(B/'stderr.gz');gait=json.loads((A/'analysis.json').read_text());rows=[]
for e in range(8):
 samples=[s.get((e,t)) for t in range(1,302)];base=[ref.get((e,t)) for t in range(1,302)];missing=sum(c[e,t,k]!=1 for t in range(1,302) for k in range(16));hits=[t for t,x in enumerate(samples,1) if x and any(abs(v)>=11.999 for v in x['v'][22:38])];oldhits=[t for t,x in enumerate(base,1) if x and any(abs(v)>=11.999 for v in x['v'][22:38])];rows.append({'lane':e,'complete':all(samples),'missing_substeps':missing,'baseline_toe_speed_cap_steps':oldhits,'candidate_toe_speed_cap_steps':hits,'walking_pass':gait['lanes'][e]['pass'],'pass':bool(all(samples) and not missing and not hits and gait['lanes'][e]['pass'])})
(A/'start-safety.json').write_text(json.dumps({'lanes':rows,'scope':'toe joint speeds at accepted control endpoints, not all physics substeps'},indent=2)+'\n')
