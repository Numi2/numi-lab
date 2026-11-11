import sys
from pathlib import Path
import gzip,json,math,subprocess
A=Path('artifacts/crow-v15-gait-onset-ramp-20261111');arm=sys.argv[1];P=Path('artifacts/crow-v15-walk-16substep-screen-20261111')
def read(path,lim):
 sup={};sub={}
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);
    if d['step']<=lim:sup[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=lim:sub[lane,t,k]=list(map(float,v[6:]))
 return sup,sub
old,os=read(P/'zero16.stderr.gz',99);sup,sub=read(A/(arm+'.stderr.gz'),500);diff=[]
for l in range(8):
 for t in range(1,100):
  if (l,t) not in old or (l,t) not in sup or any(old[l,t][f]!=sup[l,t][f] for f in ('q','v')) or any(os.get((l,t,k))!=sub.get((l,t,k)) for k in range(16)):diff.append((l,t))
inputs=[(l,t,sup[l,t]) for l in range(8) for t in range(101,501) if (l,t) in sup];text=''.join(str(l)+' '+str(t)+' '+' '.join(map(str,d['q']+d['v']))+'\n' for l,t,d in inputs);p=subprocess.run(['build/crow-v15-walk-all-lanes-geometry-20261111/kinematics'],input=text,text=True,capture_output=True,check=True);pose={}
for line in p.stdout.splitlines():
 v=line.split();pose[int(v[0]),int(v[1]),v[2]]=list(map(float,v[3:]))
with gzip.open(A/(arm+'.replay.json.gz'),'rt') as f:replay=json.load(f)['payload']
frames={x['step']:x for x in replay['frames']};maxq=maxpos=0.
for l,t,d in inputs:
 if l!=1:continue
 frame=frames.get(t-1)
 if not frame:maxq=float('inf');continue
 maxq=max(maxq,max(abs(a-b) for a,b in zip(frame['q'],d['q'])))
 for name in replay['body_names']:
  if (l,t,name) not in pose:continue
  z=pose[l,t,name];i=replay['body_names'].index(name)*13;v=frame['body_states'][i:i+13];maxpos=max(maxpos,math.sqrt(sum((a-b)**2 for a,b in zip(z[:3],v[:3]))))
geomvalid=maxq<1e-6 and maxpos<1e-5
def bouts(flags):
 out=[];a=None
 for i,on in enumerate(flags+[False]):
  if on and a is None:a=i
  if not on and a is not None:
   if i-a>=3:out.append([101+a,100+i,i-a])
   a=None
 return out
rows=[]
for l in range(8):
 missing=[t for t in range(101,501) if (l,t) not in sup or not all((l,t,k) in sub for k in range(16))];flags=[[],[]];loss=[];other=[];tilts=[];speeds=[]
 for t in range(101,501):
  if t in missing:flags[0].append(False);flags[1].append(False);continue
  parts=[sub[l,t,k] for k in range(16)];imp=[sum(x[i] for x in parts) for i in range(7)]
  loss.extend([[t,k] for k,x in enumerate(parts) if x[2]+x[3]<=1e-7]);other.extend([[t,k,x[6]] for k,x in enumerate(parts) if x[6]>1e-7]);q=sup[l,t]['q'];v=sup[l,t]['v'];tilts.append(math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2)))));speeds.append(math.sqrt(sum(x*x for x in v[:3])))
  for side,ind in (('left',0),('right',1)):
   gaps=[]
   for name,h in [('crow_'+side+'_foot',(.010,.01725,.0092))]+[('crow_'+side+'_'+x,(.018,.0025,.0092)) for x in ('front_toe_1','front_toe_2','front_toe_3','rear_toe')]:
    z=pose.get((l,t,name))
    if z:
     x,y,q,w=z[3:7];axis=(2*(x*q-y*w),2*(y*q+x*w),1-2*(x*x+y*y));gaps.append(z[2]-sum(abs(a)*b for a,b in zip(axis,h)))
   flags[ind].append(len(gaps)==5 and min(gaps)>.001 and imp[2+ind]<=1e-7 and imp[1-ind]>=.5*.45004*9.81*.02 and imp[6]<=1e-7)
 bs=[bouts(x) for x in flags];order=sorted([(v[0],i) for i in range(2) for v in bs[i]]);alternating=all(order[i][1]!=order[i-1][1] for i in range(1,len(order)));distance=sup[l,500]['q'][0]-sup[l,101]['q'][0] if (l,500) in sup and (l,101) in sup else None;ok=not diff and geomvalid and not missing and min(map(len,bs))>=12 and alternating and not loss and not other and distance is not None and distance>=.05 and max(tilts,default=9)<=.35 and max(speeds,default=9)<=.5
 rows.append({'lane':l,'missing_intervals':missing,'left_bouts':bs[0],'right_bouts':bs[1],'strictly_alternating':alternating,'unsupported_substeps':len(loss),'first_unsupported_substep':loss[0] if loss else None,'nonfoot_substeps':len(other),'first_nonfoot_substep':other[0] if other else None,'forward_displacement_m':distance,'maximum_tilt_rad':max(tilts,default=None),'maximum_root_speed_m_s':max(speeds,default=None),'pass':ok})
data=json.loads((A/(arm+'.stdout')).read_text());out={'baseline_exact_before_gait':not diff,'first_baseline_difference':diff[0] if diff else None,'replay_max_q_error':maxq,'replay_max_position_error_m':maxpos,'lanes':rows,'failed_environment_steps':data['failed_environment_steps'],'termination_count':data['termination_count'],'pass':all(r['pass'] for r in rows) and not data['failed_environment_steps'] and not data['termination_count']};(A/(arm+'-analysis.json')).write_text(json.dumps(out,indent=2)+'\n');print('parity',not diff,'first diff',diff[:1],'geometry',geomvalid,'failures',data['failed_environment_steps'],'terminations',data['termination_count']);
for x in rows:print('lane',x['lane'],'pass',x['pass'],'bouts',len(x['left_bouts']),len(x['right_bouts']),'alternating',x['strictly_alternating'],'unsupported',x['unsupported_substeps'],'other',x['nonfoot_substeps'],'forward',x['forward_displacement_m'],'tilt',x['maximum_tilt_rad'],'speed',x['maximum_root_speed_m_s'])
