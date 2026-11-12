from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-drive-work-observer-20261112');arm=sys.argv[1];assert arm in ('extension','noextension')
P=A/('candidate-20261127-rebound-'+arm);R=Path('artifacts/crow-v15-unassisted-push-wing-screen-20261112/candidate-20261127-rebound-flexed' if arm=='extension' else 'artifacts/crow-v15-wing-launch-without-leg-extension-20261112/candidate-20261127-rebound-flexed')
def read(p):
 states={};contacts={};work={};errors=[]
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
   elif line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step']
    if k[1]<=370:assert k not in states;states[k]=[x[z] for z in ('q','v','policy_action','contacts')]
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=370:assert (e,t,k) not in contacts;contacts[e,t,k]=z[4:]
   elif line.startswith('CROW_DRIVE_WORK '):
    z=line.split();t,k,e,j,flags,qi=map(int,z[1:7]);v=list(map(float,z[7:]));assert len(v)==8 and (e,t,k,j) not in work;work[e,t,k,j]=(flags,qi,v)
 return states,contacts,work,errors
x=read(P);r=read(R);assert not x[3];assert len(x[0])==len(r[0])==8*370 and len(x[1])==len(r[1])==8*370*16; assert len(x[2])==8*26*16*40, len(x[2]);m1=[k for k in x[0] if x[0][k]!=r[0][k]];m2=[k for k in x[1] if x[1][k]!=r[1][k]]
out={'arm':arm,'endpoint_count':len(x[0]),'endpoint_mismatches':len(m1),'contact_substep_count':len(x[1]),'contact_substep_mismatches':len(m2),'drive_records':len(x[2]),'first_mismatches':(m1[:2]+m2[:2]),'errors':x[3]};(A/(arm+'-parity.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));assert not m1 and not m2
