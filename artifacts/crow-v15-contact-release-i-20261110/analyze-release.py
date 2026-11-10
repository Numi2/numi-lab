from pathlib import Path
import gzip,json,statistics
A=Path('artifacts/crow-v15-contact-release-i-20261110');windows={'before':(5950,5999),'early_after':(6050,6099),'late_after':(6900,6999)};samples={k:[] for k in windows}
with gzip.open(A/'development-i-contact-release.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(line.split(' ',1)[1]);t=x['step']
  for key,(lo,hi) in windows.items():
   if lo<=t<=hi:samples[key].append(x)
out={}
for name,arr in samples.items():
 a=[];rear=[];bilat=opposed=nonfoot=0;pos=[]
 for x in arr:
  a.extend(map(abs,x['q'][23:31]));pos.append(x['q'][:3]);groups={z:{'up':0.,'front':0.,'rear':0.} for z in ('left','right')}
  for c in x['contacts']:
   an,bn=c['a_name'],c['b_name'];body=an if bn=='crow_course_perch' else bn if an=='crow_course_perch' else None
   if body is None:continue
   side=next((z for z in groups if '_'+z+'_' in body),None);imp=c['impulses'][0];sgn=-1 if body==an else 1;up=sgn*c['normal'][2]*imp;nx=-sgn*c['normal'][0]
   if side and ('_foot' in body or '_toe' in body):
    groups[side]['up']+=max(up,0.)
    if 'front_toe' in body and nx<-.5:groups[side]['front']+=imp
    if 'rear_toe' in body and nx>.5:groups[side]['rear']+=imp
   elif imp>1e-8:nonfoot+=1
  bilat+=all(groups[s]['up']>.001 for s in groups);opposed+=all(groups[s][g]>.00005 for s in groups for g in ('front','rear'));rear.append(sum(groups[s]['rear'] for s in groups))
 out[name]={'sample_count':len(arr),'mean_absolute_toe_angle_rad':statistics.mean(a) if a else None,'max_absolute_toe_angle_rad':max(a) if a else None,'bilateral_upward_impulse_samples':bilat,'opposed_positive_impulse_samples':opposed,'mean_rear_opposing_impulse_Ns':statistics.mean(rear) if rear else None,'nonfoot_positive_contacts':nonfoot,'root_position_ranges_m':[[min(v[i] for v in pos),max(v[i] for v in pos)] for i in range(3)] if pos else []}
(A/'release-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
