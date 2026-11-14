from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-launch-wing-phase-20261114');ds={}
for arm in ('push','hold'):
 x=json.load(gzip.open(p/(arm+'-window.json.gz'),'rt'));ds[arm]={tuple(r[:2]):r[2:] for r in x['target']}
rows=[]
for e in range(8):
 missing=[];diff={}
 for t in range(335,501):
  x=ds['push'].get((e,t));y=ds['hold'].get((e,t))
  if x is None or y is None or len(x)!=40 or len(y)!=40:missing.append(t);continue
  for i,(a,b) in enumerate(zip(x,y)):
   if a!=b:diff.setdefault(i,[]).append(t)
 rows.append({'lane':e,'missing_or_bad_target_records':missing,'first_difference_by_dof':{str(k):v[0] for k,v in diff.items()},'only_leg_dofs_differ':set(diff).issubset(set(range(13,19))),'wing_tail_toe_equal':not any(k in diff for k in list(range(6,13))+list(range(22,38)))})
r={'window':[335,500],'rows':rows,'all_pass':all(not x['missing_or_bad_target_records'] and x['only_leg_dofs_differ'] and x['wing_tail_toe_equal'] for x in rows)};(p/'all-command-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
