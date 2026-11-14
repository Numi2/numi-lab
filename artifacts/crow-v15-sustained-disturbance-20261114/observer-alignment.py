from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-sustained-disturbance-20261114');arm=sys.argv[1];P=A/(arm+'-20261127-rebound-observer-'+arm);payload=json.load(gzip.open(P/'replay.json.gz','rt'))['payload'];frames=payload['frames'];e=payload['environment'];s={}
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);
   if x['lane']==e:s[x['step']]=x
maxerr=0;missing=[];wrong=0
for x in frames[:-1]:
 t=x['step'];a=s.get(t+1);b=s.get(t)
 if not a or not b:missing.append(t);continue
 maxerr=max(maxerr,max(abs(p-q) for p,q in zip(x['q']+x['v'],a['q']+a['v'])));wrong+=max(abs(p-q) for p,q in zip(x['q']+x['v'],b['q']+b['v']))>1e-6
out={'arm':arm,'replay_environment':e,'frames':len(frames),'shifted_qv_max_abs_error':maxerr,'same_number_mismatch_count':wrong,'missing':missing,'pass_shifted':not missing and maxerr<1e-6,'scope':'replay environment only; does not independently verify candidate lanes 6/7'};(A/(arm+'-alignment.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
