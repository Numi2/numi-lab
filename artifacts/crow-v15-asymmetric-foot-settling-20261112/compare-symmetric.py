from pathlib import Path
import json
r=Path('artifacts');A=r/'crow-v15-asymmetric-foot-settling-20261112';P=r/'crow-v15-slewed-foot-settling-20261112';old=json.loads((P/'evaluation.json').read_text());new=json.loads((A/'evaluation.json').read_text());oldtoe=json.loads((P/'toe-evaluation.json').read_text());newtoe=json.loads((A/'toe-evaluation.json').read_text());out={}
for seed in old:
 rows=[]
 for e in range(4,8):
  a,b=old[seed][e],new[seed][e];at,bt=oldtoe[seed][e],newtoe[seed][e]
  rows.append({'lane':e,'unloaded_symmetric':a['unloaded_substeps'],'unloaded_asymmetric':b['unloaded_substeps'],'right_speed_ratio_symmetric':a['speed_p95_ratio']['right'],'right_speed_ratio_asymmetric':b['speed_p95_ratio']['right'],'dx_symmetric_m':a['dx_m'],'dx_asymmetric_m':b['dx_m'],'toe_ratio_symmetric':at['trial_to_previous_ratio'],'toe_ratio_asymmetric':bt['trial_to_previous_ratio'],'slewed_failed_gates':[k for k,v in b['gates'].items() if not v]})
 out[seed]=rows
(A/'symmetric-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for seed,rows in out.items():
 for x in rows:print(seed,x)
