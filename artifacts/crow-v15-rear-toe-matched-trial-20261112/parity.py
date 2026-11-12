from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-rear-toe-matched-trial-20261112');old=Path('artifacts/crow-v15-rear-toe-touchdown-screen-20261112/candidate-20261127-rebound')
def load(path):
 st={};sub={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    z=json.loads(line.split(' ',1)[1]);st[z['lane'],z['step']]=z
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);sub[e,t,k]=z[4:]
 assert len(st)==2408 and len(sub)==38528
 return st,sub
c=load(A/'candidate-20261127-rebound-control');v=load(A/'candidate-20261127-rebound-preload');o=load(old)
out={'matched_prefix_endpoints':sum(c[0][x]!=v[0][x] for x in c[0] if x[1]<=200),'matched_prefix_contact_substeps':sum(c[1][x]!=v[1][x] for x in c[1] if x[1]<=200),'control_lane4_endpoints':sum(c[0][x]!=o[0][x] for x in c[0] if x[0]==4),'control_lane4_substeps':sum(c[1][x]!=o[1][x] for x in c[1] if x[0]==4),'preload_lane3_endpoints':sum(v[0][x]!=o[0][x] for x in v[0] if x[0]==3),'preload_lane3_substeps':sum(v[1][x]!=o[1][x] for x in v[1] if x[0]==3)}
(A/'parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));assert not any(out.values())
