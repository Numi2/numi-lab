from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-contact-gated-hip-screen-20261112');rows=collections.defaultdict(list)
with gzip.open(A/'mixed/trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='gate':
   e,t=int(z[1]),int(z[2]);rows[e].append((t,*map(float,z[3:])))
out=[]
for e in range(8):
 samples=rows[e];bad=[];prev=0.
 for t,g,left,right in samples:
  expected=max(0.,min(1.,prev+(.04 if left<=.000001 or right<=.000001 else -.04))) if t>=10 else prev
  if abs(g-expected)>2e-5:bad.append(t)
  prev=g
 out.append({'lane':e,'sample_count':len(samples),'first_step':samples[0][0] if samples else None,'last_step':samples[-1][0] if samples else None,'gate_nonzero_intervals':sum(x[1]>0 for x in samples),'gate_mean':sum(x[1] for x in samples)/len(samples) if samples else None,'reconstruction_mismatches':bad})
(A/'gate-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if any(x['sample_count']!=550 or x['reconstruction_mismatches'] for x in out):raise SystemExit('incomplete gate evidence')
