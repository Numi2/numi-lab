from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-delayed-flight-handoff-20261111');P=Path('artifacts/crow-v15-forward-leg-jump-20261111')
def read(path,maxstep):
 sup={};sub={}
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);
    if d['step']<=maxstep:sup[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=maxstep:sub[lane,t,k]=list(map(float,v[6:]))
 return sup,sub
prior=json.loads((P/'analysis.json').read_text());out={}
for name in ('hn50-ap18','hn50-ap30','hn50-ap45','hn70-ap45','hn85-ap45','hn100-ap45'):
 base,bs=read(P/(name+'.stderr.gz'),270);sup,sub=read(A/(name+'.stderr.gz'),500);rows=[]
 for lane in range(8):
  def ints(data,t):
   if not all((lane,t,k) in data for k in range(16)):return None
   return [sum(data[lane,t,k][i] for k in range(16)) for i in range(7)]
  orig=prior[name]['lanes'][lane];first=orig['first_unload_step'];qdiff=[];cdiff=[]
  for t in range(1,271):
   if (lane,t) not in base or (lane,t) not in sup or any(base[lane,t][f]!=sup[lane,t][f] for f in ('q','v')):qdiff.append(t)
   if any(bs.get((lane,t,k))!=sub.get((lane,t,k)) for k in range(16)):cdiff.append(t)
  leg_matches=first is not None and not any(t<=first+1 for t in qdiff+cdiff)
  missing=[t for t in range(1,501) if ints(sub,t) is None]
  initial=next((t for t in range(249,501) if ints(sub,t) is not None and sum(ints(sub,t)[2:4])<=1e-7),None);ground=None
  if initial is not None:
   for t in range(initial,501):
    z=ints(sub,t)
    if z is not None and (z[2]+z[3]>1e-7 or z[6]>1e-7):
     s=sup.get((lane,t));ground={'step':t,'foot_normal_impulse_Ns':z[2]+z[3],'nonfoot_impulse_Ns':z[6],'root_position_m':s['q'][:3] if s else None};break
  metrics={}
  for t in (353,500):
   s=sup.get((lane,t));b=sup.get((lane,247))
   if s and b:
    q=s['q'];metrics[str(t)]={'height_gain_m':q[2]-b['q'][2],'forward_speed_m_s':s['v'][0],'tilt_rad':math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2)))),'root_position_m':q[:3]}
  m353=metrics.get('353',{});m500=metrics.get('500',{});ok=(orig['pass'] and leg_matches and not missing and initial==first and ground is None and m353.get('height_gain_m',-1)>=.15 and m353.get('forward_speed_m_s',-1)>.2 and m353.get('tilt_rad',9)<=.35 and m500.get('forward_speed_m_s',-1)>.2 and m500.get('tilt_rad',9)<=.35)
  rows.append({'lane':lane,'leg_only_pass':orig['pass'],'leg_only_first_unload_step':first,'actual_first_unload_step':initial,'first_qv_divergence':min(qdiff) if qdiff else None,'first_contact_divergence':min(cdiff) if cdiff else None,'exact_through_two_unloaded_rising_intervals':leg_matches,'missing_substeps':missing,'first_ground_recontact':ground,'metrics':metrics,'pass':ok})
 data=json.loads((A/(name+'.stdout')).read_text());out[name]={'lanes':rows,'failed_environment_steps':data['failed_environment_steps'],'termination_count':data['termination_count'],'pass':all(r['pass'] for r in rows) and not data['failed_environment_steps'] and not data['termination_count']}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for n,o in out.items():
 r=o['lanes'];print(n,'pass',sum(x['pass'] for x in r),'parity',sum(x['exact_through_two_unloaded_rising_intervals'] for x in r),'first-qdiff',sorted(set(x['first_qv_divergence'] for x in r),key=lambda x:-1 if x is None else x),'first-contactdiff',sorted(set(x['first_contact_divergence'] for x in r),key=lambda x:-1 if x is None else x),'recontact',sum(x['first_ground_recontact'] is not None for x in r),'step353-vx',[round(x['metrics'].get('353',{}).get('forward_speed_m_s',0),3) for x in r])
