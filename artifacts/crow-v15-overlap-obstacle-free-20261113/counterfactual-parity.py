from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-overlap-obstacle-free-20261113');O=Path('artifacts/crow-v15-overlap-long-flight-20261113');ref=json.loads((O/'fatal-contact.json').read_text());bounds={x['lane']:x['first_scene_contact']['tick']-1 for x in ref['rows'] if x['first_scene_contact']};data=[]
for path in (O/'push-wing-20261126-rebound-long/run.stderr.gz',A/'push-wing-20261126-rebound-clear/run.stderr.gz'):
 d={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
    if e in bounds and t<=bounds[e]:d[e,t]=x
 data.append(d)
rows=[]
for e in range(2,8):
 end=bounds.get(e);missing=[] if end is not None else ['reference scene contact missing'];diff={k:[] for k in ('q','v','policy_action','contacts')}
 if end is not None:
  for t in range(1,end+1):
   if any((e,t) not in d for d in data):missing.append(t);continue
   for k in diff:
    if data[0][e,t][k]!=data[1][e,t][k]:diff[k].append(t)
 rows.append({'lane':e,'through_tick':end,'missing_ticks':missing,'mismatches':diff,'matched':not missing and not any(diff.values())})
main=json.loads((A/'analysis.json').read_text());mask=json.loads((A/'mask-audit.json').read_text());p={str(e)+'-'+str(e+1):bool(mask['valid'] and main['paired_long_pass'][str(e)+'-'+str(e+1)] and all(next(x for x in rows if x['lane']==k)['matched'] for k in (e,e+1))) for e in (2,4,6)};out={'rows':rows,'mask_valid':mask['valid'],'paired_matched_obstacle_free_screen_pass':p,'scope':'Exact accepted-state/action/contact equality is checked only before the first scene contact recorded in the obstacle-present first episode. Later trajectories are counterfactual, not the same execution.'};(A/'counterfactual-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
