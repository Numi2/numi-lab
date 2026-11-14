from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-corrected-reset-flight-20261114');P=Path('artifacts/crow-v15-matched-zero-twist-straight-20261114/turn-reference.json.gz')
def read(p):return {(x['lane'],x['step']):x for x in json.load(gzip.open(p,'rt'))}
a=read(A/'straight-reference.json.gz');b=read(P);rows=[]
for e in range(8):
 x,y=a[e,1],b[e,1];rows.append({'lane':e,'first_observation_q_max_abs_difference':max(abs(u-v) for u,v in zip(x['q'],y['q'])),'first_observation_v_max_abs_difference':max(abs(u-v) for u,v in zip(x['v'],y['v']))})
out={'rows':rows,'scope':'first logged post-step observations, not independently captured reset states'};(A/'first-observation-difference.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
