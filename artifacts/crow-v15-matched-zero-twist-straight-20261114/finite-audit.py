from pathlib import Path
import collections,gzip,json,math,sys
p=Path(sys.argv[1]);counts=collections.Counter();bad=[]
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return not isinstance(x,(int,float)) or math.isfinite(x)
with gzip.open(p/'run.stderr.gz','rt') as f:
 for n,line in enumerate(f,1):
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);counts['states']+=1
   if not all(finite(x[k]) for k in ('q','v','policy_action','contacts')):bad.append((n,'state'))
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   a=line.split();counts['contact_substeps']+=1
   if len(a)!=43 or not all(math.isfinite(float(v)) for v in a[1:]):bad.append((n,'contact'))
  elif line.startswith('CROW_DRIVE_TARGET '):
   a=line.split();counts['targets']+=1
   if not all(math.isfinite(float(v)) for v in a[1:]):bad.append((n,'target'))
r={'counts':dict(counts),'invalid_count':len(bad),'examples':bad[:10],'scope':'all logged lanes, finite values and contact record width only'}
(p/'finite-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
