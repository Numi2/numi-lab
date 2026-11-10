from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-gentle-wing-transfer-20261110');out={}
for course in ['gentle15','gentle30']:
 name='development-reserved-final-j-'+course;d=json.loads((A/(name+'.stdout')).read_text());e=d['first_episode_navigation'];lanes={i:{'loaded_streak':0,'longest_loaded_streak':0,'loaded_samples':0,'phase2_samples':0,'max_preperch_angle_rad':0.,'max_after_25_no_contact_rad':0.,'no_contact_samples_after_25':0,'last':None} for i in range(8)};last={};closed=set();nocontact={i:[0,0] for i in range(8)};nav={i:0 for i in range(8)}
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_TASK_STATE '):
    v=l.split();i=int(v[1]);t=int(v[2]);tag='state';key=(tag,i)
    if key in last and t<last[key]:closed.add(key)
    last[key]=t
    if key in closed or t>e['steps_by_environment'][i]:continue
    nav[i]=float(v[4]);
    if nav[i]<5:continue
    a,b=float(v[5]),float(v[6]);ok=a>.1 and b>.1 and a+b>2.2;z=lanes[i];z['loaded_streak']=z['loaded_streak']+1 if ok else 0;z['longest_loaded_streak']=max(z['longest_loaded_streak'],z['loaded_streak']);z['loaded_samples']+=ok;z['phase2_samples']+=float(v[7])>=1.5;z['last']={'step':t,'phase':float(v[7]),'loads_n':[a,b]}
   elif l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);i=x['lane'];t=x['step'];key=('support',i)
    if key in last and t<last[key]:closed.add(key)
    last[key]=t
    if key in closed or t>e['steps_by_environment'][i] or t<800:continue
    for side in range(2):
     label='crow_left_' if side==0 else 'crow_right_';active=False
     for c in x.get('contacts',[]):
      names=[c['a_name'],c['b_name']]
      if any(label in n and ('toe' in n or 'foot' in n) for n in names) and c['impulses'][0]>1e-8:active=True;break
     nocontact[i][side]=0 if active else nocontact[i][side]+1
     q=x['q'][23+4*side:27+4*side]
     if nocontact[i][side]>=25:
      z=lanes[i];z['max_after_25_no_contact_rad']=max(z['max_after_25_no_contact_rad'],*map(abs,q));z['no_contact_samples_after_25']+=1
   elif l.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);i=x['lane'];t=x['step'];key=('flight',i)
    if key in last and t<last[key]:closed.add(key)
    last[key]=t
    if key in closed or t>e['steps_by_environment'][i] or t<800 or nav[i]>=5:continue
    z=lanes[i];z['max_preperch_angle_rad']=max(z['max_preperch_angle_rad'],*map(abs,x['q'][23:31]))
 out[course]={'first_episode':e,'failed_environment_steps':d['failed_environment_steps'],'lanes':lanes}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for c,x in out.items():
 print(c,'survived',sum(not y for y in x['first_episode']['physical_failure_by_environment']),'longest loaded',max(z['longest_loaded_streak'] for z in x['lanes'].values()),'phase2',sum(z['phase2_samples']>0 for z in x['lanes'].values()),'preperch toe max',max(z['max_preperch_angle_rad'] for z in x['lanes'].values()),'no-contact toe max',max(z['max_after_25_no_contact_rad'] for z in x['lanes'].values()))
