from pathlib import Path
import gzip,json,math,subprocess
A=Path('artifacts/crow-v15-walk-to-flight-20261111');P=Path('artifacts/crow-v15-perturbed-leg-handoff-20261111')
def read(path):
 sup={};sub={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(line.split(' ',1)[1]);
    if d['step']<=500:sup[d['lane'],d['step']]=d
   elif line.startswith('CROW_WALK_SUBSTEP '):
    v=line.split();l,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=500:sub[l,t,k]=list(map(float,v[6:]))
 return sup,sub
def eq(a,b,l,t):return (l,t) in a and (l,t) in b and all(a[l,t][x]==b[l,t][x] for x in ('q','v'))
def ints(sub,l,t):
 x=[sub.get((l,t,k)) for k in range(16)]
 return None if any(y is None for y in x) else [sum(y[i] for y in x) for i in range(7)]
def bouts(flags):
 out=[];start=None
 for i,on in enumerate(flags+[False]):
  if on and start is None:start=i
  if not on and start is not None:
   if i-start>=3:out.append([101+start,100+i,i-start])
   start=None
 return out
out={}
for seed,oldname in [('20261126','parity'),('91827463','seed63-flight'),('91827491','seed91-flight')]:
 f,fs=read(A/(seed+'-flight.stderr.gz'));g,gs=read(A/(seed+'-leg.stderr.gz'));old,os=read(P/(oldname+'.stderr.gz'));diff=[(l,t) for l in range(8) for t in range(1,100) if not eq(f,old,l,t) or any(fs.get((l,t,k))!=os.get((l,t,k)) for k in range(16))];firsts={l:next((t for t in range(249,271) if ints(gs,l,t) is not None and sum(ints(gs,l,t)[2:4])<=1e-7),None) for l in range(8)}
 records=[(l,t,f[l,t]) for l in range(8) for t in range(101,201) if (l,t) in f]
 records += [(l,t,g[l,t]) for l,first in firsts.items() if first is not None and first<270 for t in (first,first+1) if (l,t) in g]
 text=''.join(str(l)+' '+str(t)+' '+' '.join(map(str,d['q']+d['v']))+'\n' for l,t,d in records);p=subprocess.run(['build/crow-v15-walk-all-lanes-geometry-20261111/kinematics'],input=text,text=True,capture_output=True,check=True);pose={}
 for line in p.stdout.splitlines():
  v=line.split();pose[int(v[0]),int(v[1]),v[2]]=list(map(float,v[3:]))
 replay={}
 for kind in ('flight','leg'):
  with gzip.open(A/(seed+'-'+kind+'.replay.json.gz'),'rt') as stream:r=json.load(stream)['payload'];replay[kind]=(r,{x['step']:x for x in r['frames']})
 maxpos=maxq=0.
 for l,t,d in records:
  if l!=1:continue
  kind='flight' if t<=200 else 'leg';r,frames=replay[kind];frame=frames.get(t-1)
  if frame is None:maxq=float('inf');continue
  maxq=max(maxq,max(abs(a-b) for a,b in zip(frame['q'],d['q'])))
  for name in r['body_names']:
   z=pose.get((l,t,name))
   if z:
    i=r['body_names'].index(name)*13;v=frame['body_states'][i:i+13];maxpos=max(maxpos,math.sqrt(sum((a-b)**2 for a,b in zip(z[:3],v[:3]))))
 geomvalid=maxq<1e-6 and maxpos<1e-5;rows=[]
 for l in range(8):
  flags=[[],[]];missing=[];loss=[];other=[];tilts=[];speeds=[]
  for t in range(101,201):
   x=[fs.get((l,t,k)) for k in range(16)]
   if (l,t) not in f or any(v is None for v in x):missing.append(t);flags[0].append(False);flags[1].append(False);continue
   imp=[sum(v[i] for v in x) for i in range(7)];loss.extend([[t,k] for k,v in enumerate(x) if v[2]+v[3]<=1e-7]);other.extend([[t,k,v[6]] for k,v in enumerate(x) if v[6]>1e-7]);q=f[l,t]['q'];v=f[l,t]['v'];tilts.append(math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2)))));speeds.append(math.sqrt(sum(z*z for z in v[:3])))
   for side,ind in (('left',0),('right',1)):
    gaps=[]
    for name,h in [('crow_'+side+'_foot',(.010,.01725,.0092))]+[('crow_'+side+'_'+z,(.018,.0025,.0092)) for z in ('front_toe_1','front_toe_2','front_toe_3','rear_toe')]:
     z=pose.get((l,t,name))
     if z:
      x,y,q,w=z[3:7];axis=(2*(x*q-y*w),2*(y*q+x*w),1-2*(x*x+y*y));gaps.append(z[2]-sum(abs(a)*b for a,b in zip(axis,h)))
    flags[ind].append(len(gaps)==5 and min(gaps)>.001 and imp[2+ind]<=1e-7 and imp[1-ind]>=.5*.45004*9.81*.02 and imp[6]<=1e-7)
  bs=[bouts(x) for x in flags];order=sorted([(v[0],i) for i in range(2) for v in bs[i]]);alternating=all(order[i][1]!=order[i-1][1] for i in range(1,len(order)));distance=f[l,200]['q'][0]-f[l,101]['q'][0] if (l,200) in f and (l,101) in f else None;walkpass=geomvalid and not missing and min(map(len,bs))>=3 and alternating and not loss and not other and distance is not None and distance>=.015 and max(tilts,default=9)<=.35 and max(speeds,default=9)<=.5
  first=firsts[l];rising=first is not None and first<270 and all((l,t) in g and g[l,t]['v'][2]>.05 and ints(gs,l,t) is not None and sum(ints(gs,l,t)[2:4])<=1e-7 for t in (first,first+1));re=next((t for t in range(first+1,271) if ints(gs,l,t) is not None and sum(ints(gs,l,t)[2:4])+ints(gs,l,t)[6]>1e-7),None) if first is not None else None;end=re-1 if re else 270;rise=max((g[l,t]['q'][2]-g[l,247]['q'][2] for t in range(249,end+1) if (l,t) in g and (l,247) in g),default=-1);legother=max((ints(gs,l,t)[6] for t in range(249,271) if ints(gs,l,t) is not None),default=float('inf'));paired=[t for t in range(1,(first+2 if first else 1)) if not eq(f,g,l,t) or any(fs.get((l,t,k))!=gs.get((l,t,k)) for k in range(16))];legmissing=[t for t in range(249,271) if ints(gs,l,t) is None or (l,t) not in g];legpass=rising and rise>.02 and legother<=1e-7 and not paired and not legmissing
  leg_gaps=[]
  if first:
   for t in (first,first+1):
    for side in ('left','right'):
     for name,h in [('crow_'+side+'_foot',(.010,.01725,.0092))]+[('crow_'+side+'_'+z,(.018,.0025,.0092)) for z in ('front_toe_1','front_toe_2','front_toe_3','rear_toe')]:
      z=pose.get((l,t,name))
      if z:
       x,y,q,w=z[3:7];axis=(2*(x*q-y*w),2*(y*q+x*w),1-2*(x*x+y*y));leg_gaps.append(z[2]-sum(abs(a)*b for a,b in zip(axis,h)))
  geomleg=geomvalid and len(leg_gaps)==20 and min(leg_gaps)>.001;ffirst=next((t for t in range(249,501) if ints(fs,l,t) is not None and sum(ints(fs,l,t)[2:4])<=1e-7),None);ground=None
  if ffirst:
   for t in range(ffirst,501):
    z=ints(fs,l,t)
    if z is not None and z[2]+z[3]+z[6]>1e-7:ground={'step':t,'impulse_Ns':z[2]+z[3]+z[6],'root_position_m':f[l,t]['q'][:3] if (l,t) in f else None};break
  miss=[t for t in range(1,501) if ints(fs,l,t) is None];met={}
  for t in (353,500):
   if (l,t) in f and (l,247) in f:
    q=f[l,t]['q'];v=f[l,t]['v'];met[str(t)]={'height_gain_m':q[2]-f[l,247]['q'][2],'forward_speed_m_s':v[0],'tilt_rad':math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2))))}
  m=met.get('353',{});z=met.get('500',{});positions=[(t,f[l,t]['q'][0]-f[l,247]['q'][0]) for t in range(247,501) if (l,t) in f and (l,247) in f];flightpass=ffirst==first and ground is None and not miss and m.get('height_gain_m',-1)>=.15 and m.get('forward_speed_m_s',-1)>.2 and m.get('tilt_rad',9)<=.35 and z.get('forward_speed_m_s',-1)>.2 and z.get('tilt_rad',9)<=.35
  rows.append({'lane':l,'left_gait_bouts':bs[0],'right_gait_bouts':bs[1],'gait_alternating':alternating,'gait_low_support_substeps':loss,'gait_nonfoot_contacts':other,'gait_distance_m':distance,'gait_maximum_tilt_rad':max(tilts,default=None),'gait_maximum_speed_m_s':max(speeds,default=None),'walk_pass':walkpass,'leg_first_unload':first,'leg_rising':rising,'leg_height_gain_m':rise,'leg_first_recontact':re,'leg_missing_intervals':legmissing,'paired_differences':paired,'leg_minimum_toe_sole_gap_m':min(leg_gaps) if leg_gaps else None,'leg_pass':legpass and geomleg,'first_flight_ground_recontact':ground,'flight_missing_intervals':miss,'flight_metrics':met,'first_relative_x_min':min(positions,key=lambda x:x[1]) if positions else None,'flight_pass':flightpass,'pass':not diff and walkpass and legpass and geomleg and flightpass})
 d=json.loads((A/(seed+'-flight.stdout')).read_text());e=json.loads((A/(seed+'-leg.stdout')).read_text());out[seed]={'before_gait_differences':diff[:20],'before_gait_difference_count':len(diff),'max_replay_position_error_m':maxpos,'max_replay_q_error':maxq,'flight_failed_environment_steps':d['failed_environment_steps'],'flight_termination_count':d['termination_count'],'leg_failed_environment_steps':e['failed_environment_steps'],'leg_termination_count':e['termination_count'],'lanes':rows,'pass':all(x['pass'] for x in rows) and not d['failed_environment_steps'] and not d['termination_count'] and not e['failed_environment_steps'] and not e['termination_count']}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for seed,o in out.items():
 r=o['lanes'];print(seed,'pass',sum(x['pass'] for x in r),'/8','walking',sum(x['walk_pass'] for x in r),'leg',sum(x['leg_pass'] for x in r),'flight',sum(x['flight_pass'] for x in r),'pre-gait differences',o['before_gait_difference_count'])
 for x in r:print(' ',x['lane'],'bouts',len(x['left_gait_bouts']),len(x['right_gait_bouts']),'support gaps',len(x['gait_low_support_substeps']),'leg',x['leg_first_unload'],'paired',x['paired_differences'][:1],'flight',x['flight_pass'],'ground',x['first_flight_ground_recontact'])
