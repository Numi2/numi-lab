from pathlib import Path
import collections,gzip,json,math,sys
p=Path('artifacts/crow-v15-actor-pronation-slew-20261114')/sys.argv[1];targets={};states={}
with gzip.open(p/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_DRIVE_TARGET '):
   z=l.split();targets[int(z[2]),int(z[1])]=list(map(float,z[3:]))
  elif l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);states[x['lane'],x['step']]=x['q']
def yaw(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
rows=[]
for e in range(8):
 changes=[abs(targets[e,t][j]-targets[e,t-1][j]) for t in range(527,1801) for j in (10,11) if (e,t) in targets and (e,t-1) in targets and len(targets[e,t])>11 and len(targets[e,t-1])>11]
 h0=yaw(states[e,525]) if (e,525) in states else None
 heading=[abs(wrap(yaw(states[e,t])-h0)) for t in range(525,1801) if (e,t) in states and h0 is not None]
 rows.append({'lane':e,'target_change_samples':len(changes),'max_absolute_target_change_rad':max(changes,default=None),'slew_pass':len(changes)==2*(1800-526) and max(changes,default=1.)<=.04001,'heading_samples':len(heading),'max_abs_posthandoff_heading_change_rad':max(heading,default=None),'minimum_heading_movement_pass':len(heading)==1276 and max(heading,default=0.)>=.05})
out={'rows':rows,'all_pass':all(x['slew_pass'] and x['minimum_heading_movement_pass'] for x in rows),'scope':'measured target changes and heading movement only; not requested-heading accuracy or navigation qualification'};(p/'slew-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
