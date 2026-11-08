#!/usr/bin/env python3
"""Compare the source-pinned prior native trajectory with the reintegrated baseline."""
import json,hashlib,math
from pathlib import Path
p=Path(__file__).resolve().parent
new=p/'native.json'
old=p/'original-h62.5us.json'
a=json.loads(old.read_text());b=json.loads(new.read_text())
skip={'wall_seconds','gpu_seconds','accepted_simulated_seconds_per_wall_second'}
def diff(x,y,path=''):
 if isinstance(x,dict) and isinstance(y,dict):
  for k in sorted(set(x)|set(y)):
   if k not in skip:yield from diff(x.get(k),y.get(k),path+'/'+k)
 elif isinstance(x,list) and isinstance(y,list):
  if len(x)!=len(y):yield [path+'/length',len(x),len(y)]
  for i,(v,w) in enumerate(zip(x,y)):yield from diff(v,w,path+'/'+str(i))
 elif x!=y:yield [path,x,y]
differences=list(diff(a,b));h=a['h'];n=round(.001/h)
def forcebins(x):
 z=x['traces'];assert len(z)%n==0
 return [sum(row['normal_impulse'] for row in z[i:i+n])/.001 for i in range(0,len(z),n)]
A=forcebins(a);B=forcebins(b)
result={'original_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),
 'exposed_exact_excluding_timing':not differences,'differences':differences,'force_bins_old_N':A,'force_bins_new_N':B,
 'bin_max_abs_difference_N':max(abs(x-y) for x,y in zip(A,B)),'integrated_impulse_old_Ns':sum(z['normal_impulse'] for z in a['traces']),
 'integrated_impulse_new_Ns':sum(z['normal_impulse'] for z in b['traces']),
 'complete':b['accepted_steps']==160 and b['rejected_steps']==0,
 'transaction_checks':{k:b[k] for k in ('identity_mismatch_rejected','forced_rollback_exact','restore_replay')}}
(p/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
if not result['exposed_exact_excluding_timing'] or not result['complete'] or not all(result['transaction_checks'].values()):raise SystemExit(1)
