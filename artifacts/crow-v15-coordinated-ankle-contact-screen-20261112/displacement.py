from pathlib import Path
import gzip,json,sys
seed=int(sys.argv[1]);assert seed in (20261126,20261127)
R=Path('artifacts');A=R/'crow-v15-coordinated-ankle-contact-screen-20261112';paths={'baseline':R/'crow-v15-knee-phase-advance-screen-20261112'/f'candidate-{seed}-rebound-baseline','candidate':A/f'candidate-{seed}-rebound-phase'}
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
 r={'lane':e,'feedback_gain':0}
 for k,s in sets.items():
  a,b=s[e,200],s[e,300];r[k]={'dx_m':b['q'][0]-a['q'][0],'dy_m':b['q'][1]-a['q'][1],'height_change_m':b['q'][2]-a['q'][2]}
 rows.append(r)
(A/f'displacement-{seed}.json').write_text(json.dumps({'window_seconds':[4,6],'rows':rows},indent=2)+'\n')
for r in rows:print(r['lane'],r['feedback_gain'],'dx base/new',round(r['baseline']['dx_m'],4),round(r['candidate']['dx_m'],4),'dy',round(r['candidate']['dy_m'],4))
