from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-ground-balance-offset-screen-20261112')
def read(tag):
 q={};s={}
 with gzip.open(p/('candidate-smoke-20261127-'+tag)/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_DRIVE_TARGET '):
    x=l.split();q[int(x[1]),int(x[2])]=[float(v) for v in x[3:]]
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);s[x['step'],x['lane']]=x
 return q,s
a,sa=read('target');b,sb=read('target-baseline');rows=[]
for t in [1,2]:
 for e in range(8):
  d=[{'index':i,'candidate':x,'baseline':y,'delta':x-y} for i,(x,y) in enumerate(zip(a[t,e],b[t,e])) if x!=y];rows.append({'step':t,'lane':e,'target_differences':d,'state_matches_baseline':sa[t,e]==sb[t,e]})
out={'rows':rows,'scope':'Two-step drive target buffer comparison with identical instrumented host, initial seed and model. Buffer represents motor targets, not measured applied torque; instrumentation parity with the retained host must be checked independently.'};(p/'target-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for row in rows: print(row['step'],row['lane'],row['target_differences'],'state_exact',row['state_matches_baseline'])
