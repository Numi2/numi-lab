from pathlib import Path
import gzip,json,hashlib
A=Path('artifacts/crow-v15-stance-toe-workspace-20261114');P=Path('artifacts/crow-v15-pitch-normal-support-20261114/trace.log.gz')
for line in gzip.open(P,'rt'):
 z=line.split()
 if z and z[0]=='step' and z[1]=='0' and z[2]=='304':
  i=z.index('q');j=z.index('v');q=list(map(float,z[i+1:j]));v=list(map(float,z[j+1:]));break
else:raise RuntimeError('snapshot missing')
records=[('original',0,q.copy(),None)]
for group in ('front','rear'):
 for i in range(16):
  for j in range(16):
   qq=q.copy();a=-.3+.1*i;b=-.3+.1*j
   if group=='front':
    for k in (31,33,35):qq[k]=a;qq[k+1]=b
   else:qq[37]=a;qq[38]=b
   records.append((group,len(records),qq,[round(a,6),round(b,6)]))
with (A/'input.tsv').open('w') as f:
 for group,n,qq,pair in records:f.write('0 '+str(n)+' '+' '.join(map(str,qq+v))+'\n')
(A/'inputs.json').write_text(json.dumps({'source_path':str(P),'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'lane':0,'source_zero_based_interval':304,'records':[{'kind':g,'id':n,'angles':pair} for g,n,qq,pair in records]},indent=2)+'\n')
