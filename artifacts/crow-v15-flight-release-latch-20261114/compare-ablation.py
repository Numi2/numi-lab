from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-flight-release-latch-20261114');B=Path('artifacts/crow-v15-upward-trigger-longflight-20261114')
a={(x['lane'],x['step']):x for x in json.load(gzip.open(A/'turn-reference.json.gz','rt'))}; b={(x['lane'],x['step']):x for x in json.load(gzip.open(B/'turn-reference.json.gz','rt'))}; rows=[]
for e in range(8):
 end=897 if e in (6,7) else (760 if e==0 else 764 if e==1 else 1800)
 bad=[t for t in range(1,end+1) if (e,t) not in a or (e,t) not in b or any(a[e,t][k]!=b[e,t][k] for k in ('q','v','policy_action','contacts'))]
 rows.append({'lane':e,'through':end,'mismatch_count':len(bad),'first_mismatches':bad[:10]})
o={'scope':'exact intervention versus failed-turn logged state and contact parity, not independent replay timing','rows':rows,'all_exact':all(not x['mismatch_count'] for x in rows)};(A/'ablation-parity.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
