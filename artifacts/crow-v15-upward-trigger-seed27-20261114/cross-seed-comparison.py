from pathlib import Path
import gzip,json
base=Path('artifacts'); old=base/'crow-v15-upward-trigger-stroke-20261114'; new=base/'crow-v15-upward-trigger-seed27-20261114'; out={}
for arm in ('push','hold'):
 with gzip.open(old/(arm+'-window.json.gz'),'rt') as f: x=json.load(f)
 with gzip.open(new/(arm+'-window.json.gz'),'rt') as f: y=json.load(f)
 out[arm]={k:{'old_count':len(x[k]),'new_count':len(y[k]),'exact_equal':x[k]==y[k]} for k in ('state','target','contact','leg','toe','aero','errors')}
out['scope']='Comparison of serialized observer records only; seed difference does not by itself prove an independent starting-state distribution.'
(new/'cross-seed-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
