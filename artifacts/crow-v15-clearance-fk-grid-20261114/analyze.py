from pathlib import Path
import json,math
p=Path('artifacts/crow-v15-clearance-fk-grid-20261114')
ref=json.loads((p/'pose-crosscheck.json').read_text())['candidate']['samples']; bytick={x['zero_based_tick']:x for x in ref}; out={}
with (p/'output.tsv').open() as f:
 for line in f:
  lane,t,dh,dk,da,foot,front,rear=map(float,line.split());t=int(t); o=out.setdefault(t,{'count':0,'zero':None,'best':None,'smallest_3mm':None});o['count']+=1
  d={'hip_rad':dh,'knee_rad':dk,'ankle_rad':da,'foot_m':foot,'front_m':front,'rear_m':rear,'minimum_m':min(foot,front,rear),'norm_rad':math.sqrt(dh*dh+dk*dk+da*da)}
  if dh==dk==da==0:o['zero']=d
  if o['best'] is None or d['minimum_m']>o['best']['minimum_m']:o['best']=d
  if d['minimum_m']>=.003 and (o['smallest_3mm'] is None or d['norm_rad']<o['smallest_3mm']['norm_rad']):o['smallest_3mm']=d
for t,o in out.items():
 assert o['zero'] is not None
 groups=bytick[t]['groups'];o['zero_gap_error_m']={g:abs(o['zero'][g+'_m']-groups[g]['min_box_gap_m']) for g in ('foot','front','rear')}
 assert max(o['zero_gap_error_m'].values())<1.e-6
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
