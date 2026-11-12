from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-slow-push-uniform-dev-20261112');P=A/'candidate-20261127-rebound-flexed';R=Path('artifacts/crow-v15-slow-push-speed-screen-20261112/candidate-20261127-rebound-flexed')
def read(p):
 states={};contacts={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step']
    if k[0] in (4,5) and k[1]<=500:assert k not in states;states[k]=[x[z] for z in ('q','v','policy_action','contacts')]
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if e in (4,5) and t<=500:assert (e,t,k) not in contacts;contacts[e,t,k]=z[4:]
 return states,contacts
p=read(P);r=read(R);assert len(r[0])==1000 and len(r[1])==16000
m1=[k for k in r[0] if k not in p[0] or p[0][k]!=r[0][k]];m2=[k for k in r[1] if k not in p[1] or p[1][k]!=r[1][k]]
out={'seed':20261127,'lanes':[4,5],'expected_endpoints':1000,'expected_substeps':16000,'endpoint_mismatch_count':len(m1),'contact_mismatch_count':len(m2),'first_mismatches':m1[:3]+m2[:3]};(A/'parity-20261127.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
