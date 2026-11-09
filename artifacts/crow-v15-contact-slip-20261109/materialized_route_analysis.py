from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-contact-slip-20261109');reports=[]
for course in ('reserved-final-i','reserved-final-j'):
 a=json.load(gzip.open(A/('materialized-route-'+course+'.replay.json.gz'),'rt'))['payload']['frames'];b=json.load(gzip.open(A/('materialized-late-route-'+course+'.replay.json.gz'),'rt'))['payload']['frames'];same=len(a)==len(b) and all(all(x[k]==y[k] for k in ('q','v','body_states')) for x,y in zip(a,b));samples=[]
 for l in gzip.open(A/('materialized-late-route-'+course+'.stderr.gz'),'rt'):
  if l.startswith('CROW_CLEARANCE_SAMPLE '):samples.append(json.loads(l.split(' ',1)[1]))
 rows=[]
 for lane in (0,1):
  choices=[x for x in samples if x['lane']==lane and x['end_step'] in (3000,3001)];z=next((x for x in choices if x['end_step']==3001),choices[-1] if choices else None);rows.append({'lane':lane,'minimum_sampled_clearance':z})
 report=json.loads((A/('materialized-late-route-'+course+'-summary.json')).read_text());reports.append({'course':course,'selected_lane_observer_state_parity':same,'terminal_support_and_outcomes':report,'clearance':rows})
result={'rows':reports,'clearance_scope':'minimum signed bird-obstacle shape distance over accepted control-step endpoints; excludes perch and is not continuous-time','scope':'two development-exposed lanes per layout; no grip qualification'};(A/'materialized-route-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for r in reports:print(r['course'],'parity',r['selected_lane_observer_state_parity'],'clearance',[(x['lane'],None if not x['minimum_sampled_clearance'] else x['minimum_sampled_clearance']['minimum_m']) for x in r['clearance']])
