from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-deep-coordinated-perch-20261110');out={'samples':0,'geometric_opposed_bilateral_samples':0,'impulse_opposed_bilateral_samples':0,'per_group_contact_samples':{},'per_group_load_samples':{}};labels=('left_front','left_rear','right_front','right_rear');tc={x:0 for x in labels};ti=dict(tc)
with gzip.open(A/'development-j-deep-coordinated-perch.stderr.gz','rt') as f:
 for l in f:
  if not l.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(l.split(' ',1)[1]);t=x['step']
  if not 4751<=t<=5001:continue
  geom=dict.fromkeys(labels,False);imp=dict.fromkeys(labels,0.)
  for c in x['contacts']:
   a,b=c['a_name'],c['b_name'];n=a if b=='crow_course_perch' else b if a=='crow_course_perch' else None
   if not n:continue
   side=next((s for s in ('left','right') if '_'+s+'_' in n),None);front='_front_toe' in n;rear='_rear_toe' in n
   if side is None or not(front or rear):continue
   sign=-1 if n==a else 1;nx=-sign*c['normal'][0];direction=(front and nx<-.5) or (rear and nx>.5)
   if not direction:continue
   key=side+('_front' if front else '_rear')
   if c['separation']<=0.0005:geom[key]=True
   imp[key]+=c['impulses'][0]
  out['samples']+=1
  for k in labels:tc[k]+=geom[k];ti[k]+=imp[k]>.00005
  out['geometric_opposed_bilateral_samples']+=all(geom.values());out['impulse_opposed_bilateral_samples']+=all(v>.00005 for v in imp.values())
out['per_group_contact_samples']=tc;out['per_group_load_samples']=ti
(A/'toe-state.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
