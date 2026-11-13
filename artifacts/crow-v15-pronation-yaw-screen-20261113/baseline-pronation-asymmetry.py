import gzip,json,collections,statistics
from pathlib import Path
p='artifacts/crow-v15-position-flight-screen-20261113/position-flight-20261126-rebound-position/run.stderr.gz';d=collections.defaultdict(lambda:[[],[]])
with gzip.open(p,'rt') as f:
 for line in f:
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(line[20:]);e=x['lane'];t=x['step']
  if e in (4,5,6,7) and 650<=t<=1150:
   for j in range(2):d[e][j].append(x['q'][11+j])
out={str(e):[dict(min=min(v),max=max(v),mean=statistics.mean(v),samples=len(v)) for v in parts] for e,parts in d.items()}
Path('artifacts/crow-v15-pronation-yaw-screen-20261113/baseline-pronation-asymmetry.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
