from pathlib import Path
import json,statistics
A=Path('artifacts/crow-v15-swing-foot-placement-20261112');P=Path('artifacts/crow-v15-toe-corvid-sequence-audit-20261112')
old=json.loads((P/'analysis.json').read_text());new=json.loads((A/'toe-sequence.json').read_text());physical=json.loads((A/'evaluation.json').read_text());results={}
for seed in (20261126,20261127):
 rows=[]
 for e in range(8):
  measurements={};sidecounts={};all_order=True
  for label,data in (('previous',old),('trial',new)):
   record=data[f'{seed}-candidate'];vs=[];counts={}
   for side in ('left','right'):
    rr=next(r for r in record['rows'] if r['lane']==e and r['side']==side)
    es=[x for x in rr['events'] if x['type']=='touchdown']
    counts[side]=len(es);all_order &= all(x['order']=='hallux_first' for x in es);vs.extend(x['gap_ms'] for x in es)
   measurements[label]=statistics.median(vs) if vs else None;sidecounts[label]=counts
  coverage=all(n>=2 for side in sidecounts.values() for n in side.values());a,b=measurements['previous'],measurements['trial'];ratio=b/a if a is not None and b is not None and a>0 else None
  physical_ok=physical[str(seed)][e]['passes'];toe_ok=coverage and ratio is not None and ratio<=.85
  rows.append({'lane':e,'knee_offset':[0,0,0,0,-.015,-.015,-.03,-.03][e],'previous_and_trial_median_ms':measurements,'event_count_by_foot':sidecounts,'trial_to_previous_ratio':ratio,'physical_gates_pass':physical_ok,'preregistered_toe_gate_pass':toe_ok,'joint_screen_pass':physical_ok and toe_ok,'posthoc_both_arms_hallux_first':all_order})
 results[str(seed)]=rows
(A/'toe-evaluation.json').write_text(json.dumps(results,indent=2)+'\n')
for seed,rows in results.items():
 for r in rows:print(seed,r['lane'],'ratio',None if r['trial_to_previous_ratio'] is None else round(r['trial_to_previous_ratio'],2),'physical',r['physical_gates_pass'],'coverage',r['event_count_by_foot'],'pass',r['joint_screen_pass'])
