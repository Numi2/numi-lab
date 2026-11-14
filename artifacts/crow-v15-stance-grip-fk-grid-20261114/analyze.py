from pathlib import Path
import json,math
p=Path('artifacts/crow-v15-stance-grip-fk-grid-20261114');ref=json.loads(Path('artifacts/crow-v15-stance-transfer-diagnostic-20261114/pose-crosscheck.json').read_text())['candidate']['samples'];bytick={x['zero_based_tick']:x for x in ref};out={}
with (p/'output.tsv').open() as f:
 for line in f:
  lane,t,dh,dk,da,foot,front,rear=map(float,line.split());t=int(t);o=out.setdefault(t,{'count':0,'zero':None,'smallest_contact_pose':None});o['count']+=1
  d={'hip_rad':dh,'knee_rad':dk,'ankle_rad':da,'foot_m':foot,'front_m':front,'rear_m':rear,'norm_rad':math.sqrt(dh*dh+dk*dk+da*da)}
  if dh==dk==da==0:o['zero']=d
  if -.0006<=front<=0 and -.0006<=rear<=0 and foot>=-.0002 and (o['smallest_contact_pose'] is None or d['norm_rad']<o['smallest_contact_pose']['norm_rad']):o['smallest_contact_pose']=d
for t,o in out.items():
 assert o['zero'] is not None
 groups=bytick[t]['groups'];o['zero_gap_error_m']={g:abs(o['zero'][g+'_m']-groups[g]['min_box_gap_m']) for g in ('foot','front','rear')};assert max(o['zero_gap_error_m'].values())<1.e-6
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
