from pathlib import Path
import gzip,json
D=Path('artifacts/crow-v15-perch-hip-feedback-repeat-20261111');other=Path('artifacts/crow-v15-landing-root-balance-latch-screen-20261111/trial')
def read(p):
 out={}
 for f in sorted(p.glob('stderr-*.gz')):
  with gzip.open(f,'rt') as g:
   for l in g:
    if l.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(l.split(' ',1)[1]);
     if x['lane']==3:out[x['step']]=(x['q'],x['v'],x['policy_action'])
 return out
a,b=read(other),read(D/'trial');common=sorted(a.keys()&b.keys());first=next((t for t in common if a[t]!=b[t]),None)
r={'lane':3,'reference':str(other),'samples_compared':len(common),'first_difference_step':first,'reference_only':len(a.keys()-b.keys()),'repeat_only':len(b.keys()-a.keys()),'scope':'State and policy-action endpoint samples only, no contact-substep parity asserted.'}
(D/'lane3-repeat-comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
