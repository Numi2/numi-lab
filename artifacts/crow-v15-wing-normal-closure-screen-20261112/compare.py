from pathlib import Path
import collections,gzip,json,math,statistics
R=Path('artifacts');A=R/'crow-v15-wing-normal-closure-screen-20261112';B=R/'crow-v15-common-command-pronation-screen-20261112';p='candidate-20261126-rebound-flexed/run.stderr.gz'
def read(path):
 st={};cs=collections.defaultdict(list)
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);st[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();ti,sub,e=map(int,z[1:4]);
    if e<2:cs[e].append(tuple(z[1:]))
 return st,cs
s,c=read(B/p);t,d=read(A/p);diff=[]
for e in (0,1):
 for i in range(1,1201):
  for k in ('q','v','policy_action','contacts'):
   if s[e,i][k]!=t[e,i][k]:diff.append([e,i,k])
 if c[e]!=d[e]:diff.append([e,'contact'])
out={'state_samples':2400,'contact_substeps':sum(len(c[e]) for e in (0,1)),'differences':diff};(A/'control-parity.json').write_text(json.dumps(out,indent=2)+'\n');print('control parity',out);assert not diff,diff[:10]
meta=json.loads((A/'candidate-20261126-rebound-flexed/run.stdout').read_text())['first_episode_navigation'];rows=[]
def angles(q):
 x,y,z,w=q[3:7];return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
for e in range(8):
 end=min(1200,meta['steps_by_environment'][e]);start=min(601,end);seq=[t[e,i] for i in range(start,end+1) if (e,i) in t];assert seq;(a,b)=(seq[0],seq[-1]);ang=[angles(v['q']) for v in seq];yaw=[x[2] for x in ang];unwrapped=[yaw[0]]
 for v in yaw[1:]:unwrapped.append(unwrapped[-1]+math.atan2(math.sin(v-unwrapped[-1]),math.cos(v-unwrapped[-1])))
 dx=b['q'][0]-a['q'][0];dy=b['q'][1]-a['q'][1]
 rows.append({'lane':e,'normal_blend':[0,0,.15,.15,.35,.35,.6,.6][e],'pronation_offset':[0,0,.2,-.2,.2,-.2,.2,-.2][e],'end_step':end,'termination':meta['termination_reason_by_environment'][e],'yaw_change_rad':unwrapped[-1]-unwrapped[0],'trajectory_angle_rad':math.atan2(dy,dx),'dx_m':dx,'dy_m':dy,'max_abs_roll_rad':max(abs(v[0]) for v in ang),'height_range_m':[min(v['q'][2] for v in seq),max(v['q'][2] for v in seq)]})
(A/'flight-summary.json').write_text(json.dumps({'rows':rows,'scope':'Used seed, model sensitivity test with closure changed throughout candidate trajectories, 50 Hz endpoint sampling, no qualified flight or biological force law.'},indent=2)+'\n')
for r in rows:print(r)
