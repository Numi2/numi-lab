from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-perch-hip-offset-screen-20261111');p=A/'trial';hits={e:[] for e in range(8)};counts={e:0 for e in range(8)};missing=[];seen=set()
with __import__('contextlib').ExitStack() as stack:
 f=__import__('itertools').chain.from_iterable(stack.enter_context(gzip.open(z,'rt')) for z in sorted(p.glob('stderr-*.gz')))
 for l in f:
  if l.startswith('CROW_TOE_V_SUBSTEP '):
   a=l.split();t,k,e=map(int,a[1:4]);v=list(map(float,a[4:]));
   if t>5000:continue
   counts[e]+=1;seen.add((e,t,k))
   if len(v)!=16:missing.append([e,t,k,'wrong_width'])
   if any(abs(z)>=11.999 for z in v):hits[e].append([t,k,[i for i,z in enumerate(v) if abs(z)>=11.999]])
for e in range(8):
 for t in range(1,5001):
  for k in range(16):
   if (e,t,k) not in seen:missing.append([e,t,k,'absent'])
rows=[{'lane':e,'all_hits':hits[e],'hits_after_ground_handoff_step_355':[v for v in hits[e] if v[0]>=356],'substeps':counts[e]} for e in range(8)]
ends=json.loads((A/'analysis.json').read_text())['first_episode_navigation']['steps_by_environment']
for row in rows:row['first_episode_hits_after_ground_handoff']=[h for h in row['all_hits'] if 356<=h[0]<=ends[row['lane']]+1]
(A/'toe-velocity-analysis.json').write_text(json.dumps({'lanes':rows,'missing':missing,'scope':'All observed development teacher steps, including attempted landing and later auto-reset episodes; first-episode post-handoff hits are reported separately. Absence of saturation is not biological validation.'},indent=2)+'\n')
