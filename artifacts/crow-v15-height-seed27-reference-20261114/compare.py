from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-height-seed27-reference-20261114');P=Path('artifacts/crow-v15-height-feedback-strength-20261114/altitude-20261127-rebound-switch/run.stderr.gz')
with gzip.open(A/'reference.json.gz','rt') as f:r=json.load(f)
ref={(x['lane'],x['step']):x for x in r['states']};cur={}
with gzip.open(P,'rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);k=x['lane'],x['step']
   if x['step']<=500:
    assert k not in cur;cur[k]=x
rows=[]
for e in range(8):
 bad=[t for t in range(1,501) if (e,t) not in cur or (e,t) not in ref or any(cur[e,t][k]!=ref[e,t][k] for k in ('q','v','policy_action','contacts'))]
 rows.append({'lane':e,'mismatch_count':len(bad),'first':bad[:10]})
o={'reference_commit':r['source_commit'],'reference_sha256':r['whole_compressed_sha256'],'rows':rows,'pass':all(not x['mismatch_count'] for x in rows)};(A/'comparison.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
