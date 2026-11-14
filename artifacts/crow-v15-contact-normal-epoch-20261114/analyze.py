from pathlib import Path
from collections import defaultdict
import gzip, json, math, itertools, hashlib
A=Path('artifacts/crow-v15-contact-normal-epoch-20261114')
P=Path('artifacts/crow-v15-pretransfer-pitch-feedback-20261114')

def stream(path):
 with gzip.open(path,'rt') as f:
  yield from f

parity={'lines':0,'different':0,'examples':[]}
for i,(a,b) in enumerate(itertools.zip_longest(stream(P/'trace.log.gz'),stream(A/'trace.log.gz'))):
 parity['lines']+=1
 if a!=b:
  parity['different']+=1
  if len(parity['examples'])<5:parity['examples'].append({'line':i+1,'prior':a,'observed':b})
events={}; event_duplicate=[]
for line in stream(A/'trace.log.gz'):
 z=line.split()
 if z and z[0]=='event':
  key=(int(z[1]),int(z[2]))
  if key in events:event_duplicate.append(key)
  events[key]=(float(z[5]),float(z[6]))
normal={};full=defaultdict(lambda:[0.,0.]);coverage=defaultdict(set);duplicates=[]
for line in stream(A/'substeps.log.gz'):
 z=line.split()
 if not z:continue
 if z[0]=='CROW_NORMAL_Z_SUBSTEP':
  t,s,e=map(int,z[1:4]);key=(e,t,s)
  if key in normal:duplicates.append(key)
  normal[key]=(float(z[4]),float(z[5]))
 elif z[0]=='CROW_TWO_TOE_SUBSTEP':
  t,s,e=map(int,z[1:4]);key=(e,t)
  if s in coverage[key]:duplicates.append((e,t,s,'full'))
  coverage[key].add(s);full[key][0]+=float(z[6]);full[key][1]+=float(z[7])
comparisons={}
for offset in (0,1):
 missing=[];mismatch=[];maxerr=0.;n=0
 for (e,t),v in events.items():
  other=normal.get((e,t+offset,31))
  if other is None:missing.append((e,t));continue
  err=max(abs(v[i]-other[i]) for i in (0,1));n+=1;maxerr=max(maxerr,err)
  if not math.isfinite(err) or err>1e-8:mismatch.append((e,t,err,v,other))
 comparisons[str(offset)]={'matched_count':n,'missing_count':len(missing),'missing_examples':missing[:10],'mismatch_count':len(mismatch),'mismatch_examples':mismatch[:10],'max_absolute_error_Ns':maxerr}
rows=[]
for e in (0,1):
 for t in range(285,321):
  v=events.get((e,t)); x=full.get((e,t+1));
  if v is None or x is None:continue
  ns=sum(v);fs=sum(x)
  rows.append({'lane':e,'event_interval':t,'observer_interval':t+1,'normal_final_substep_Ns':v,'full_interval_normal_and_friction_Ns':x,'normal_left_share':v[0]/ns if abs(ns)>1e-8 else None,'full_left_share':x[0]/fs if abs(fs)>1e-8 else None,'full_substeps':len(coverage[e,t+1])})
out={'prior_stdout_parity':parity,'event_count':len(events),'event_duplicates':event_duplicate,'normal_substep_count':len(normal),'full_interval_count':len(full),'normal_or_full_duplicates':duplicates,'offset_comparisons':comparisons,'return_intervals':rows,'interpretation':'Event uses normal-only final contact impulse, while full-interval includes tangential vertical impulse. The latter can be negative; shares alone are not a center of pressure or proof of causality.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='return_intervals'},indent=2))
