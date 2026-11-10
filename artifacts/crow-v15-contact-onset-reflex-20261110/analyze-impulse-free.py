from pathlib import Path
import gzip,json,struct
A=Path('artifacts/crow-v15-contact-onset-reflex-20261110');out={}
for course in ('i','j'):
 name='development-reserved-final-'+course;summary=json.loads((A/(name+'.stdout')).read_text())['first_episode_navigation'];states={};supports={};last={};closed=set()
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for line in f:
   if line.startswith('CROW_TASK_STATE '):
    v=line.split();i=int(v[1]);t=int(v[2]);key=('s',i)
    if key in last and t<last[key]:closed.add(key)
    last[key]=t
    if key in closed or t+1>summary['steps_by_environment'][i] or float(v[4])<5:continue
    bits=list(map(int,v[10:12]));states[(i,t+1)]=[struct.unpack('f',struct.pack('I',u))[0] for u in bits]
   elif line.startswith('CROW_SUPPORT_SAMPLE '):
    v=json.loads(line.split(' ',1)[1]);i=v['lane'];t=v['step'];key=('c',i)
    if key in last and t<last[key]:closed.add(key)
    last[key]=t
    if key in closed or t>summary['steps_by_environment'][i]:continue
    supports[(i,t)]=v['q'][23:31]
 rows=[]
 for i in range(8):
  for side in range(2):
   streak=0;count=0;maximum=0.;at=None
   for t in sorted(t for j,t in states if j==i):
    impulse=states[(i,t)][side];streak=streak+1 if abs(impulse)<=1e-8 else 0
    if streak>=25 and (i,t) in supports:
     count+=1;v=max(map(abs,supports[(i,t)][4*side:4*side+4]))
     if v>maximum:maximum=v;at=t
   rows.append({'lane':i,'side':side,'impulse_free_all_substeps_count':count,'maximum_abs_toe_angle_rad':maximum,'step':at})
 out[course]=rows
(A/'analysis-impulse-free.json').write_text(json.dumps(out,indent=2)+'\n')
for c,rows in out.items():print(c,'side runs',len(rows),'samples',sum(x['impulse_free_all_substeps_count'] for x in rows),'max toe rad',max(x['maximum_abs_toe_angle_rad'] for x in rows),'worst',max(rows,key=lambda x:x['maximum_abs_toe_angle_rad']))
