from pathlib import Path
import gzip,json
R=Path('artifacts');A=R/'crow-v15-synchronized-gait-frequency-20261112';B=R/'crow-v15-toe-preload-matched-trial-20261112';paths={'baseline':B/'candidate-20261127-rebound-low','candidate':A/'candidate-20261127-rebound-phase'}
def read(path):
 d={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);
    if x['step'] in (200,300):d[x['lane'],x['step']]=x
 return d
sets={k:read(p) for k,p in paths.items()};rows=[]
for e in range(8):
 r={'lane':e,'hip_stroke':[.30,.30,.30,.30,.30,.30,.30,.30][e]}
 for k,s in sets.items():
  a,b=s[e,200],s[e,300];r[k]={'dx_m':b['q'][0]-a['q'][0],'dy_m':b['q'][1]-a['q'][1],'height_change_m':b['q'][2]-a['q'][2]}
 rows.append(r)
(A/'displacement.json').write_text(json.dumps({'window_seconds':[4,6],'rows':rows},indent=2)+'\n')
for r in rows:print(r['lane'],r['hip_stroke'],'dx base/new',round(r['baseline']['dx_m'],4),round(r['candidate']['dx_m'],4),'dy',round(r['candidate']['dy_m'],4))
