from pathlib import Path
import gzip,hashlib,json
A=Path('artifacts/crow-v15-return-stance-geometry-20261114');p=Path('artifacts/crow-v15-pitch-normal-support-20261114/trace.log.gz');steps={295,300,302,304,306};found={}
with gzip.open(p,'rt') as f:
 for l in f:
  z=l.split()
  if len(z)>4 and z[0]=='step' and z[1]=='0' and int(z[2]) in steps:
   t=int(z[2]);i=z.index('q');j=z.index('v');found[t]='0 '+str(t)+' '+' '.join(z[i+1:j]+z[j+1:])+'\n'
assert set(found)==steps
(A/'input.tsv').write_text(''.join(found[t] for t in sorted(found)))
(A/'inputs.json').write_text(json.dumps({'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lane':0,'steps':sorted(steps)},indent=2)+'\n')
