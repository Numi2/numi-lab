from pathlib import Path
from collections import defaultdict
import gzip,json,math
A=Path('artifacts/crow-v15-grip-search-20261109');B=Path('build/crow-v15-grip-search-20261109');name='symmetric-launch-wings-off-push'
p=json.loads((B/(name+'.json')).read_text())['payload'];names=p['body_names'];st=p['body_state_stride'];fr=p['frames'];contact=defaultdict(lambda:defaultdict(float))
for line in gzip.open(A/(name+'.stderr.gz'),'rt'):
 if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
 s=json.loads(line.split(' ',1)[1]);step=s['step']
 if s['lane']!=1 or not 100<=step<=200:continue
 for c in s['contacts']:
  a,b=c['a_name'],c['b_name']
  for foot in ('crow_left_foot','crow_right_foot'):
   if foot not in (a,b) or 'locomotion_ground' not in (a,b) or c['impulses'][0]<=0:continue
   normal=(-1 if a==foot else 1)*c['normal'][2]*c['impulses'][0]
   if normal>0:contact[step][foot]+=normal
out={};both=0
for foot in ('crow_left_foot','crow_right_foot'):
 idx=names.index(foot);dist=[];loaded=[]
 for step in range(100,201):
  if contact[step][foot]>0.00005:loaded.append(step)
  if step<200 and contact[step][foot]>.00005 and contact[step+1][foot]>.00005:
   x=fr[step-1]['body_states'][idx*st:idx*st+3];y=fr[step]['body_states'][idx*st:idx*st+3];dist.append(math.dist(x[:2],y[:2]))
 out[foot]={'loaded_endpoint_count':len(loaded),'unloaded_endpoint_count':101-len(loaded),'loaded_adjacent_pairs':len(dist),'loaded_planar_distance_sum_m':sum(dist),'loaded_planar_distance_max_per_20ms_m':max(dist,default=0),'loaded_steps':loaded}
for step in range(100,201):both+=all(contact[step][f]>.00005 for f in ('crow_left_foot','crow_right_foot'))
root=lambda step:fr[step-1]['body_states'][:3]
x=root(100);y=root(200);report={'environment':1,'interval_steps':[100,200],'root_start_m':x,'root_end_m':y,'root_forward_displacement_m':y[0]-x[0],'double_support_samples':both,'feet':out,'scope':'sampled endpoint contacts and body-COM displacements; not ground contact-point slip or validated natural gait'}
(A/'walk-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='feet'},indent=2));print({k:{a:b for a,b in v.items() if a!='loaded_steps'} for k,v in out.items()})
