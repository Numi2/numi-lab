from pathlib import Path
import gzip,json,math,sys
A=Path(sys.argv[1]);states={(e,side):[] for e in (0,1) for side in ('left','right')};falls={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x or x[0]!='step':continue
  e,t=int(x[1]),int(x[2]);q=x.index('q');v=x.index('v');pos=list(map(float,x[q+1:v]));vel=list(map(float,x[v+1:]));
  if pos[2]<.1 and e not in falls:falls[e]=t
  for side,start in [('left',23),('right',31)]:states[e,side].append((t,pos[start:start+8],vel[start-1:start+7]))
out=[]
for (e,side),rows in states.items():
 before=[(t,q,v) for t,q,v in rows if t<falls.get(e,450)];hits=[t for t,q,v in rows if max(map(abs,v))>=11.99];pre=[t for t,q,v in before if max(map(abs,v))>=11.99];out.append({'lane':e,'foot':side,'samples':len(rows),'joint_position_range_rad':[min(min(q) for _,q,_ in rows),max(max(q) for _,q,_ in rows)],'configured_joint_position_limit_rad':[-.3,1.2],'first_accepted_speed_threshold_tick':min(hits) if hits else None,'first_threshold_before_root_fall':min(pre) if pre else None,'peak_accepted_speed_rad_s':max(max(map(abs,v)) for _,_,v in rows),'peak_accepted_speed_before_root_fall_rad_s':max(max(map(abs,v)) for _,_,v in before)})
o={'rows':out,'falls_zero_based':falls,'scope':'accepted control states only, not substep extrema or actuator effort'};(A/'bilateral-toe-audit.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
