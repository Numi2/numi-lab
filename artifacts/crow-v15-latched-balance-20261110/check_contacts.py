from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-latched-balance-20261110');out=[]
for arm in ('positive','negative'):
 p=A/('layout-j-'+arm+'.stderr.gz');first=None;last=None;loaded=0
 for line in gzip.open(p,'rt'):
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  d=json.loads(line.split(' ',1)[1]);t=d['step']
  if d['lane']!=0 or t<2050:continue
  relevant=[]
  for c in d['contacts']:
   a,b=c['a_name'],c['b_name'];imp=c['impulses'][0]
   if imp<=1e-5:continue
   if 'crow_course_perch' in (a,b) and ('crow_left_foot' in (a,b) or 'crow_right_foot' in (a,b) or '_toe' in a or '_toe' in b):loaded+=1
   if 'locomotion_ground' in (a,b):
    bird=b if a=='locomotion_ground' else a
    if bird not in ('crow_left_foot','crow_right_foot') and '_toe' not in bird:relevant.append({'body':bird,'collider':'locomotion_ground','normal_impulse_Ns':imp})
  if relevant and first is None:first={'step':t,'root_position_m':d['q'][:3],'contacts':relevant}
  if t==3000:last={'step':t,'root_position_m':d['q'][:3],'body_ground':relevant}
 out.append({'arm':arm,'first_nonfoot_ground':first,'last':last,'foot_or_toe_perch_contact_records':loaded})
(A/'independent-contact-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
