from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-toe-early-support-feedback-20261111');p=A/'height/stderr.gz';hits={e:[] for e in range(8)};counts={e:0 for e in range(8)};missing=[];seen=set()
with gzip.open(p,'rt') as f:
 for l in f:
  if l.startswith('CROW_TOE_V_SUBSTEP '):
   a=l.split();t,k,e=map(int,a[1:4]);v=list(map(float,a[4:]));
   if t>600:continue
   counts[e]+=1;seen.add((e,t,k))
   if len(v)!=16:missing.append([e,t,k,'wrong_width'])
   if any(abs(z)>=11.999 for z in v):hits[e].append([t,k,[i for i,z in enumerate(v) if abs(z)>=11.999]])
for e in range(8):
 for t in range(1,601):
  for k in range(16):
   if (e,t,k) not in seen:missing.append([e,t,k,'absent'])
rows=[{'lane':e,'all_hits':hits[e],'hits_after_ground_handoff_step_355':[v for v in hits[e] if v[0]>=356],'substeps':counts[e]} for e in range(8)]
(A/'toe-velocity-analysis.json').write_text(json.dumps({'lanes':rows,'missing':missing,'scope':'Early support feedback screen, development starts only; speed limits and control stability must both pass.'},indent=2)+'\n')
