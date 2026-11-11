from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-propulsive-leg-push-20261111');P=Path('artifacts/crow-v15-bounded-launch-commands-20261111')
def read(path,maxstep=500):
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
base,bs=read(P/'v040-h065.stderr.gz');outputs={}
for hip in ('n50','p10','p70'):
 for ankle in ('n30','p18','p70'):
  name='h'+hip+'-a'+ankle;sup,sub=read(A/(name+'.stderr.gz'));rows=[]
  if name=='hp10-ap18':
   assert sup.keys()==base.keys() and sub==bs and all(all(sup[k][f]==base[k][f] for f in ('q','v')) for k in sup),'baseline mismatch'
  for lane in range(8):
   def ints(t):
    v=[sub.get((lane,t,k)) for k in range(16)]
    return None if any(x is None for x in v) else [sum(x[i] for x in v) for i in range(7)]
   missing=[t for t in range(1,501) if ints(t) is None]
   diffs=[t for t in range(1,249) if (lane,t) not in base or (lane,t) not in sup or any(base[lane,t][f]!=sup[lane,t][f] for f in ('q','v'))]
   first=next((t for t in range(249,501) if ints(t) is not None and sum(ints(t)[2:4])<=1e-7),None)
   rises=first is not None and all((lane,t) in sup and sup[lane,t]['v'][2]>.05 and ints(t) is not None and sum(ints(t)[2:4])<=1e-7 for t in (first,first+1))
   ground=None
   if first is not None:
    for t in range(first,501):
     z=ints(t)
     if z is not None and (z[2]+z[3]>1e-7 or z[6]>1e-7):
      s=sup.get((lane,t));ground={'step':t,'foot_normal_impulse_Ns':z[2]+z[3],'nonfoot_impulse_Ns':z[6],'root_position_m':s['q'][:3] if s else None};break
   metrics={}
   for t in (252,253,275,353,500):
    s=sup.get((lane,t));b=sup.get((lane,247))
    if s and b:
     q=s['q'];metrics[str(t)]={'height_gain_m':q[2]-b['q'][2],'forward_speed_m_s':s['v'][0],'vertical_speed_m_s':s['v'][2],'tilt_rad':math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2)))),'root_position_m':q[:3]}
   seen=[(t,sup[lane,t]['q'][0]) for t in range(247,501) if (lane,t) in sup];m353=metrics.get('353',{});m500=metrics.get('500',{})
   ok=(not missing and not diffs and first is not None and first>=253 and rises and ground is None and m353.get('height_gain_m',-1)>=.15 and m353.get('forward_speed_m_s',-1)>.2 and m353.get('tilt_rad',9)<=.35 and m500.get('forward_speed_m_s',-1)>.2 and m500.get('tilt_rad',9)<=.35)
   rows.append({'lane':lane,'missing_substep_intervals':missing,'first_qv_divergence_before_push':min(diffs) if diffs else None,'first_unload_step':first,'first_two_unloaded_intervals_rising':rises,'first_ground_recontact':ground,'minimum_x_m':min(seen,key=lambda x:x[1]) if seen else None,'metrics':metrics,'pass':ok})
  data=json.loads((A/(name+'.stdout')).read_text());out={'failed_environment_steps':data['failed_environment_steps'],'termination_count':data['termination_count'],'lanes':rows,'pass':all(r['pass'] for r in rows) and data['failed_environment_steps']==0 and data['termination_count']==0};outputs[name]=out
(A/'analysis.json').write_text(json.dumps(outputs,indent=2)+'\n')
for n,o in outputs.items():
 r=o['lanes'];speeds=[x['metrics'].get('353',{}).get('forward_speed_m_s') for x in r];print(n,'pass',sum(x['pass'] for x in r),'ground',sum(x['first_ground_recontact'] is not None for x in r),'first-unload',sorted(set(x['first_unload_step'] for x in r),key=lambda x:-1 if x is None else x),'step353-vx',speeds,'failures',o['failed_environment_steps'],'terminations',o['termination_count'])
