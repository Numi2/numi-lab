from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-two-segment-toe-velocity-substep-observer-20261111');B=Path('artifacts/crow-v15-two-segment-gait-uniform-adaptive-20261111')
def read(path,withv):
 state={};velocity={};count=collections.Counter();errors=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);state[x['lane'],x['step']]=x
   elif withv and line.startswith('CROW_TOE_V_SUBSTEP '):
    a=line.split();t,k,e=map(int,a[1:4]);key=e,t,k;count[key]+=1;velocity[key]=list(map(float,a[4:]))
   elif withv and ('CROW_SUBSTEP_ERROR' in line or 'failed' in line.lower()):errors.append(line.strip())
 return state,velocity,count,errors
s,v,c,errors=read(A/'stderr.gz',True);ref,_,_,_=read(B/'stderr.gz',False);rows=[]
for e in range(8):
 diff=sum(s.get((e,t))!=ref.get((e,t)) for t in range(1,302));missing=sum(c[e,t,k]!=1 or len(v.get((e,t,k),[]))!=16 for t in range(1,302) for k in range(16));deltas=[];hits=[];peak=0.
 for t in range(1,302):
  if (e,t) in s and (e,t,15) in v:deltas.extend(abs(a-b) for a,b in zip(s[e,t]['v'][22:38],v[e,t,15]))
  for k in range(16):
   q=v.get((e,t,k),[]);peak=max([peak]+[abs(x) for x in q]);indices=[i for i,x in enumerate(q) if abs(x)>=11.999]
   if indices:hits.append({'step':t,'substep':k,'toes':indices})
 maxerr=max(deltas) if deltas else None;rows.append({'lane':e,'different_endpoint_samples':diff,'missing_substeps':missing,'compared_endpoint_coordinates':len(deltas),'max_endpoint_error':maxerr,'maximum_abs_toe_speed':peak,'toe_speed_cap_substeps':hits,'observer_pass':bool(diff==0 and missing==0 and len(deltas)==301*16 and maxerr<=1e-5 and not errors)})
try:data=json.loads((A/'stdout').read_text());failed=data['failed_environment_steps'];terms=data['termination_count']
except Exception as ex:failed=terms=None;errors.append(str(ex))
for row in rows:row['observer_pass']=bool(row['observer_pass'] and failed==terms==0 and not errors)
(A/'analysis.json').write_text(json.dumps({'lanes':rows,'failed_environment_steps':failed,'terminations':terms,'errors':errors[-5:],'scope':'actual retained 16-substep generalized velocities; observer validity distinct from physical toe-speed result'},indent=2)+'\n')
