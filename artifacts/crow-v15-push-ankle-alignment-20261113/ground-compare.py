from pathlib import Path
import collections,gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-push-ankle-alignment-20261113';rows=[]
for arm in ('push','hold'):
 p=(A/'push-wing-20261126-rebound-ankle/run.stderr.gz' if arm=='push' else R/'crow-v15-supported-slow-push-screen-20261113/hold-wing-20261126-rebound-slowpush/run.stderr.gz');D=collections.defaultdict(lambda:collections.defaultdict(float))
 with gzip.open(p,'rt') as f:
  for line in f:
   z=line.split()
   if not z or z[0]!='CROW_GROUND_CONTACT':continue
   t,k,e,a,b=map(int,z[1:6])
   if e<2 or t>365:continue
   body=a if b==35 else b; vals=list(map(float,z[8:])); assert len(vals)==16;Jn=vals[12];assert Jn>0
   D[e,'348-355' if t<=355 else '356-365'][body]+=Jn
 for (e,w),vals in sorted(D.items()):
  total=sum(vals.values());prox=vals[25]+vals[33];rear=prox+vals[26]+vals[34]
  rows.append({'arm':arm,'lane':e,'ticks':w,'total_normal_ns':total,'rear_proximal_normal_ns':prox,'rear_all_normal_ns':rear,'rear_all_fraction':rear/total,'rear_proximal_fraction':prox/total,'body_normal_ns':dict(vals)})
(A/'push-hold-comparison.json').write_text(json.dumps({'scope':'Posthoc summed modeled positive normal impulses by body; contact impulses may include warm-start/solver effects. Does not establish slipping, pathology, or cause of takeoff failure.','rows':rows},indent=2)+'\n')
for r in rows:print(r['arm'],r['lane'],r['ticks'],'rear fraction',round(r['rear_all_fraction'],3),'prox fraction',round(r['rear_proximal_fraction'],3))
