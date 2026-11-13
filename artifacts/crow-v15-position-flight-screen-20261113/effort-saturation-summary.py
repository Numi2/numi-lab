from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-position-flight-screen-20261113');out={}
for seed in ('20261126','20261127'):
 p=A/f'position-flight-{seed}-rebound-position'/'run.stderr.gz';data=collections.defaultdict(lambda:dict(substeps=0,left_at_bound=0,right_at_bound=0,either_at_bound=0))
 with gzip.open(p,'rt') as f:
  for line in f:
   if not line.startswith('CROW_WING_EFFORT '):continue
   z=line.split();t=int(z[1]);e=int(z[3])
   if e not in (4,5,6,7) or not 500<=t<=1800:continue
   l=abs(float(z[4]))>=2.99999;r=abs(float(z[11]))>=2.99999;d=data[e];d['substeps']+=1;d['left_at_bound']+=l;d['right_at_bound']+=r;d['either_at_bound']+=l or r
 for e,d in data.items():
  assert d['substeps']==1301*16,(seed,e,d)
  d['either_fraction']=d['either_at_bound']/d['substeps']
 out[seed]=data
(A/'effort-saturation-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for seed,data in out.items():
 for lane,d in data.items():print(seed,lane,'either fraction',round(d['either_fraction'],4),'substeps',d['substeps'])
