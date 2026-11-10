from pathlib import Path
import gzip,json
R=Path.cwd();A=R/'artifacts/crow-v15-toe-free-flight-20261110';rows=[]
for seed in ('20261120','20261121'):
 maps=[]
 for a in ('crow-v15-toe-flight-screen-20261110','crow-v15-toe-free-flight-20261110'):
  data={}
  with gzip.open(R/'artifacts'/a/(seed+'.stderr.gz'),'rt') as f:
   for line in f:
    if line.startswith('CROW_FLIGHT_SAMPLE '):
     x=json.loads(line.split(' ',1)[1]);
     if x['step']<=600:data[x['lane'],x['step']]=(x['q'],x['v'])
  maps.append(data)
 common=set(maps[0])&set(maps[1]);bad=[k for k in common if maps[0][k]!=maps[1][k]];rows.append({'seed':seed,'samples':len(common),'matching_qv':len(common)-len(bad),'first_differences':sorted(bad)[:10]});assert len(common)==480 and not bad
(A/'precontact-comparison.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
