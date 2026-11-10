from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-landing-joint-identification-20261110');base=Path('artifacts/crow-v15-grip-latch-20261110/diagnostic-reserved-final-j-latch-original.stderr.gz')
def read(p):
 x={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    v=json.loads(l.split(' ',1)[1]);s=v['step']
    if 2400<=s<=2650:x[s]=v
 return x
def val(z):
 q=z['q'];v=z['v'];x,y,zq,w=q[3:7];return {'pitch':math.asin(max(-1,min(1,2*(w*y-zq*x)))),'forward_v':v[0],'pitch_rate':v[4],'hip_l':q[14],'ankle_l':q[16]}
b=read(base);out={}
for p in sorted(A.glob('*.stderr.gz')):
 x=read(p);prefix=[t for t in x if t in b and t<=2499];bad=[t for t in prefix if (x[t]['q'],x[t]['v'])!=(b[t]['q'],b[t]['v'])];row={'prefix_samples':len(prefix),'prefix_mismatches':len(bad),'measurements':{}}
 for t in (2500,2510,2520,2530,2540,2560,2600):
  if t in x and t in b:
   a=val(x[t]);c=val(b[t]);row['measurements'][t]={k:a[k]-c[k] for k in a}
 out[p.stem]=row
(A/'response.json').write_text(json.dumps(out,indent=2)+'\n')
for name,x in out.items():print(name,'prefix',x['prefix_samples'],'mismatches',x['prefix_mismatches'],'at2540',x['measurements'].get(2540))
