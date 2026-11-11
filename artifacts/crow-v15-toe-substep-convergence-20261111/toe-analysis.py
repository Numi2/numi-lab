from pathlib import Path
import collections,gzip,json,sys
A=Path('artifacts/crow-v15-toe-substep-convergence-20261111');name='32-substeps';O=A/'m32';state={};vel={};count=collections.Counter();errors=[]
with gzip.open(O/'stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);state[x['lane'],x['step']]=x
  elif line.startswith('CROW_TOE_V_SUBSTEP '):
   a=line.split();t,k,e=map(int,a[1:4]);key=e,t,k;count[key]+=1;vel[key]=list(map(float,a[4:]))
  elif 'CROW_SUBSTEP_ERROR' in line or 'failed' in line.lower():errors.append(line.strip())
gait=json.loads((O/'analysis.json').read_text());rows=[]
for e in range(8):
 missing=sum(count[e,t,k]!=1 or len(vel.get((e,t,k),[]))!=16 for t in range(1,302) for k in range(32));deltas=[];hits=[];peak=0.
 for t in range(1,302):
  if (e,t) in state and (e,t,31) in vel:deltas.extend(abs(a-b) for a,b in zip(state[e,t]['v'][22:38],vel[e,t,31]))
  for k in range(32):
   v=vel.get((e,t,k),[]);peak=max([peak]+[abs(x) for x in v]);ii=[i for i,x in enumerate(v) if abs(x)>=11.999]
   if ii:hits.append({'step':t,'substep':k,'toes':ii})
 err=max(deltas) if deltas else None;rows.append({'lane':e,'missing_substeps':missing,'compared_endpoint_coordinates':len(deltas),'max_endpoint_error':err,'maximum_abs_toe_speed':peak,'toe_speed_cap_substeps':hits,'gait_pass':gait['lanes'][e]['pass'],'pass':bool(missing==0 and len(deltas)==301*16 and err<=1e-5 and not hits and gait['lanes'][e]['pass'] and not errors)})
(O/'toe-analysis.json').write_text(json.dumps({'variant':name,'lanes':rows,'errors':errors[-5:],'scope':'uncalibrated development damping sensitivity; substep velocities, not flight or landing'},indent=2)+'\n')
