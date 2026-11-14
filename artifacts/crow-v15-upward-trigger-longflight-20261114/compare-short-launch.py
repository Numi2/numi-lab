from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-upward-trigger-longflight-20261114')
short=json.load(gzip.open('artifacts/crow-v15-upward-trigger-stroke-20261114/push-window.json.gz','rt'))['state']
long=json.load(gzip.open(A/'straight-reference.json.gz','rt'))
a={(x['lane'],x['step']):x for x in short}; b={(x['lane'],x['step']):x for x in long}
rows=[]
for lane in range(8):
 keys=sorted(k for k in a if k[0]==lane and k[1]<=450)
 missing=[k[1] for k in keys if k not in b]
 differences={key:[k[1] for k in keys if k in b and a[k].get(key)!=b[k].get(key)] for key in ('q','v','policy_action','contacts')}
 rows.append({'lane':lane,'compared':len(keys),'missing_steps':missing[:10],'mismatch_count':{k:len(v) for k,v in differences.items()},'first_mismatch':{k:v[0] if v else None for k,v in differences.items()}})
out={'scope':'Exact logged state parity to short launch through available ticks 1-450, used seed 20261126','rows':rows}
(A/'straight-short-parity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
