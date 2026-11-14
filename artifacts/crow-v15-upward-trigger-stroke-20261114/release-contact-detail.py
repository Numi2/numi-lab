from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-upward-trigger-stroke-20261114');rows=[]
for arm in ('push','hold'):
 with gzip.open(p/(arm+'-window.json.gz'),'rt') as f:d=json.load(f)
 for e in (6,7):
  c={(x[1],x[2]):x[3:] for x in d['contact'] if x[0]==e};tar={x[1]:x[2:] for x in d['target'] if x[0]==e};trim=[0.,0.,.35,.35,0.,0.]
  start=next((t for t in range(335,501) if any(abs(tar[t][j]-trim[j-6])>1.e-5 for j in range(6,12))),None)
  detail=[]
  if start:
   for t in range(start-2,start+5):
    loads=[sum(c[t,k][2:4]) for k in range(16)];detail.append({'interval':t,'foot_normal_impulse_sum_Ns':sum(loads),'last_substep_normal_impulse_Ns':loads[-1],'substeps_with_impulse_gt_1e-7':sum(x>1.e-7 for x in loads)})
  rows.append({'arm':arm,'lane':e,'first_wing_target_departure':start,'foot_contacts_around_departure':detail})
out={'rows':rows,'scope':'Substep contact records; target/state epoch alignment is inferred; a last-substep zero is not proof of a whole contact-free interval.'};(p/'release-contact-detail.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
