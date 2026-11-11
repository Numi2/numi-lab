from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-bounded-launch-commands-20261111')
def read(path,maxstep):
 sup={};sub={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(line.split(' ',1)[1]);t=d['step']
    if t<=maxstep:sup[d['lane'],t]=d
   elif line.startswith('CROW_WALK_SUBSTEP '):
    v=line.split();lane,t,k=int(v[3]),int(v[1]),int(v[2])
    if t<=maxstep:sub[lane,t,k]=list(map(float,v[6:]))
 return sup,sub
base,_=read(Path('artifacts/crow-v15-contact-flight-handoff-20261111/handoff.stderr.gz'),252)
allout={}
for speed in ('040','070'):
 for height in ('065','085','105'):
  name='v'+speed+'-h'+height;sup,sub=read(A/(name+'.stderr.gz'),500);rows=[]
  for lane in range(8):
   def ints(t):
    values=[sub.get((lane,t,k)) for k in range(16)]
    if any(x is None for x in values):return None
    return [sum(x[i] for x in values) for i in range(7)]
   missing=[t for t in range(1,501) if ints(t) is None]
   diffs=[t for t in range(1,253) if (lane,t) not in base or (lane,t) not in sup or any(base[lane,t][f]!=sup[lane,t][f] for f in ('q','v'))]
   first=next((t for t in range(251,501) if ints(t) is not None and sum(ints(t)[2:4])<=1e-7),None)
   rises=first is not None and all((lane,t) in sup and sup[lane,t]['v'][2]>.05 and ints(t) is not None and sum(ints(t)[2:4])<=1e-7 for t in (first,first+1))
   ground=None
   if first is not None:
    for t in range(first,501):
     z=ints(t)
     if z is not None and (z[2]+z[3]>1e-7 or z[6]>1e-7):
      s=sup.get((lane,t));ground={'step':t,'foot_normal_impulse_Ns':z[2]+z[3],'nonfoot_impulse_Ns':z[6],'root_position_m':s['q'][:3] if s else None,'root_velocity_m_s':s['v'][:3] if s else None};break
   metrics={}
   for t in (353,500):
    s=sup.get((lane,t));baseline=sup.get((lane,247))
    if s and baseline:
     q=s['q'];metrics[str(t)]={'height_gain_m':q[2]-baseline['q'][2],'forward_speed_m_s':s['v'][0],'tilt_rad':math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2)))),'root_position_m':q[:3]}
   seen=[(t,sup[lane,t]['q'][0]) for t in range(247,501) if (lane,t) in sup]
   m353=metrics.get('353',{});m500=metrics.get('500',{})
   ok=(not missing and not diffs and first is not None and first>=253 and rises and ground is None and m353.get('height_gain_m',-1)>=.15 and m353.get('forward_speed_m_s',-1)>.2 and m353.get('tilt_rad',9)<=.35 and m500.get('forward_speed_m_s',-1)>.2 and m500.get('tilt_rad',9)<=.35)
   rows.append({'lane':lane,'missing_substep_intervals':missing,'first_qv_divergence_before_release':min(diffs) if diffs else None,'first_unload_step':first,'first_two_unloaded_intervals_rising':rises,'first_ground_recontact':ground,'minimum_x_m':min(seen,key=lambda x:x[1]) if seen else None,'metrics':metrics,'pass':ok})
  data=json.loads((A/(name+'.stdout')).read_text());out={'failed_environment_steps':data['failed_environment_steps'],'termination_count':data['termination_count'],'lanes':rows,'pass':all(r['pass'] for r in rows) and data['failed_environment_steps']==0 and data['termination_count']==0}
  allout[name]=out
(A/'analysis.json').write_text(json.dumps(allout,indent=2)+'\n')
for name,d in allout.items():
 print(name,'pass',sum(r['pass'] for r in d['lanes']),'/8','failures',d['failed_environment_steps'],'terminations',d['termination_count'])
 for r in d['lanes']:
  print(' lane',r['lane'],'unload',r['first_unload_step'],'ground',r['first_ground_recontact'],'353',r['metrics'].get('353'),'500',r['metrics'].get('500'))
