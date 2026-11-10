from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-touchdown-hold-20261110'); rows=[]
with gzip.open(p/'development-j-touchdown-hold.stderr.gz','rt') as f:
 for l in f:
  if not l.startswith('CROW_SUPPORT_SAMPLE '): continue
  x=json.loads(l.split(' ',1)[1]); t=x['step']
  if t<2788: continue
  cs=[c for c in x['contacts'] if 'crow_course_perch' in (c['a_name'],c['b_name']) and c['impulses'][0]>1e-8]
  if not cs:continue
  rows.append((t,x['q'][0],x['q'][2],x['v'][0],x['v'][2],max(abs(z) for z in x['q'][23:31]),','.join(c['a_name'] if c['b_name']=='crow_course_perch' else c['b_name'] for c in cs)))
print('contacts',len(rows),'first 25:')
for r in rows[:25]:print(*r)
print('last 10:')
for r in rows[-10:]:print(*r)
