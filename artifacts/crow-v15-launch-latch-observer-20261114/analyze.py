from pathlib import Path
import gzip,json,collections,hashlib
A=Path('artifacts/crow-v15-launch-latch-observer-20261114');P=Path('artifacts/crow-v15-early-retraction-sweep-20261114');d={}
for label,path in [('prior',P/'push-20261126-rebound-nonleg/run.stderr.gz'),('observer',A/'push-20261126-rebound-nonleg/run.stderr.gz')]:
 states={};targets={};latches={};duplicates=[];errors=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step']
    if k in states:duplicates.append(['state',*k])
    states[k]=x
   elif line.startswith('CROW_DRIVE_TARGET '):
    z=line.split();k=(int(z[2]),int(z[1]))
    if k in targets:duplicates.append(['target',*k])
    targets[k]=z[3:]
   elif line.startswith('CROW_LAUNCH_LATCH '):
    z=line.split();t,sub,e,episode,bits=map(int,z[1:6]);k=e,t,sub
    if k in latches:duplicates.append(['latch',*k])
    latches[k]={'episode':episode,'status':bits,'height_feedback':float(z[6]),'vertical_feedback':float(z[7]),'navigation':float(z[8]),'loaded_left':bool(bits&(1<<20)),'loaded_right':bool(bits&(1<<21)),'release':bool(bits&8192),'retract':bool(bits&4096),'low1':bool(bits&16384),'low2':bool(bits&32768),'push_latch':bool(bits&(1<<19))}
   elif 'OBSERVER_ERROR' in line or line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
 d[label]={'states':states,'targets':targets,'latches':latches,'duplicates':duplicates,'errors':errors}
parity={}
for name in ('states','targets'):
 x=d['prior'][name];y=d['observer'][name];missing=sorted(set(x)^set(y));different=[k for k in x.keys()&y.keys() if x[k]!=y[k]]
 parity[name]={'prior_count':len(x),'observer_count':len(y),'missing_count':len(missing),'missing_examples':missing[:10],'different_count':len(different),'different_examples':different[:10]}
L=d['observer']['latches'];missing=[(e,t,s) for e in range(8) for t in range(365,391) for s in range(16) if (e,t,s) not in L];table=[]
for e in (0,1):
 for t in range(368,386):
  if (e,t,0) in L:table.append({'lane':e,'observer_step':t,'substep':0,**L[e,t,0]})
out={'parity':parity,'prior_duplicates':d['prior']['duplicates'],'observer_duplicates':d['observer']['duplicates'],'errors':{k:v['errors'] for k,v in d.items()},'latch_samples':len(L),'expected_missing_count':len(missing),'first_missing':missing[:10],'samples':table,'scope':'Observer samples prior to aero; status update and drive target epochs require alignment. Lanes have different initial positions.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
