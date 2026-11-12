from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-substep-target-interpolation-screen-20261112')
B=Path('artifacts/crow-v15-gait-target-observer-20261112/autonomous-20261126/run.stderr.gz')
C=A/'sham-20261126/run.stderr.gz'
def read(p):
 out={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith(('CROW_Q_SUBSTEP ','CROW_TWO_TOE_SUBSTEP ','CROW_DRIVE_TARGET ')):
    z=line.split()
    if z[0]=='CROW_DRIVE_TARGET': key=(z[0],int(z[1]),int(z[2])); val=z[3:]
    else:
     t,k,e=map(int,z[1:4])
     if not 201<=t<=300: continue
     key=(z[0],t,k,e);val=z[4:]
    assert key not in out
    out[key]=val
 return out
b,c=read(B),read(C); mismatch=[k for k in b.keys()|c.keys() if b.get(k)!=c.get(k)]
res={'baseline_samples':len(b),'sham_samples':len(c),'mismatch_count':len(mismatch),'first_mismatches':mismatch[:8],'pass':len(b)==26400 and len(c)==26400 and not mismatch}
(A/'sham-parity.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res))
if not res['pass']:raise SystemExit('Sham parity failed: candidate cannot be interpreted.')
