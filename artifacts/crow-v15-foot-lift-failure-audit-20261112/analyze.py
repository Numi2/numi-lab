from pathlib import Path
import collections,hashlib,importlib.util,json,math
A=Path('artifacts/crow-v15-foot-lift-failure-audit-20261112');P=Path('artifacts/crow-v15-quarter-stance-screen-20261112')
sp=importlib.util.spec_from_file_location('gait','artifacts/crow-v15-whole-gait-audit-20261111/audit.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
out={'arms':{},'input_sha256':{},'scope':'Endpoint geometry and integrated normal impulses in the two-second window on one used development seed. Thresholds reproduce the existing engineering clearance-bout gate, not crow measurements. These are component tests, not continuous foot trajectory or biological validation.'}
for arm,tag in [('sham','sham-20261127-dev2'),('candidate','candidate-20261127-rebound')]:
 path=P/tag
 for filename in ('run.stderr.gz','replay.json.gz'):out['input_sha256'][arm+'/'+filename]=hashlib.sha256((path/filename).read_bytes()).hexdigest()
 sup,sub,body,com=m.load(path);rows=[]
 for e in range(8):
  for si,side in enumerate(('left','right')):
   counts=collections.Counter();maxgap=-math.inf;block=collections.Counter();allgaps=[]
   for t in range(201,301):
    names=m.names(side);clear=[]
    for name in names:
     z=body[e,t,name];x,y,zq,w=z[3:7];rotation=(2*(x*zq-y*w),2*(y*zq+x*w),1-2*(x*x+y*y));h=(.010,.01725,.0092) if name.endswith('_foot') else (.009,.0025,.0092)
     clear.append(z[2]-sum(abs(a)*b for a,b in zip(rotation,h)))
    gap=min(clear);maxgap=max(maxgap,gap);allgaps.append(gap);block[names[clear.index(gap)]]+=1
    normals=[sum(sub[e,t,k][2+i] for k in range(16)) for i in range(2)]
    other=sum(sub[e,t,k][4] for k in range(16))
    tests={'clear':gap>.001,'own_unloaded':normals[si]<=m.EPS,'other_support':normals[1-si]>=m.SUPPORT,'no_nonfoot':other<=m.EPS}
    for n,v in tests.items():counts[n]+=int(v)
    counts['all_conditions']+=int(all(tests.values()))
    counts['clear_with_other_support']+=int(tests['clear'] and tests['other_support'])
   rows.append({'lane':e,'side':side,'max_endpoint_clearance_m':maxgap,'condition_counts':dict(counts),'most_limiting_shapes':block.most_common(3)})
 out['arms'][arm]=rows
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for arm in out['arms']:
 print(arm)
 for r in out['arms'][arm]:
  if r['side']=='right':print(r['lane'],round(r['max_endpoint_clearance_m']*1000,2),r['condition_counts'],r['most_limiting_shapes'][0])
