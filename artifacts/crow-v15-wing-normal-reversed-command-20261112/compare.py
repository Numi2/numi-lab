from pathlib import Path
import collections,gzip,json,math
R=Path('artifacts');A=R/'crow-v15-wing-normal-reversed-command-20261112';B=R/'crow-v15-wing-normal-closure-screen-20261112';p='candidate-20261126-rebound-flexed/run.stderr.gz'
def read(path):
 st={};cs=collections.defaultdict(list)
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);st[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();ti,sub,e=map(int,z[1:4]);
    if ti<=600 or e<2:cs[e].append(tuple(z[1:]))
 return st,cs
s,c=read(B/p);t,d=read(A/p);diff=[];states=contacts=0
for e in range(8):
 end=1200 if e<2 else 600
 for i in range(1,end+1):
  for k in ('q','v','policy_action','contacts'):
   if s[e,i][k]!=t[e,i][k]:diff.append([e,i,k])
  states+=1
 aa=[z for z in c[e] if int(z[0])<=end];bb=[z for z in d[e] if int(z[0])<=end];contacts+=len(aa)
 if aa!=bb:diff.append([e,'contact'])
out={'state_samples':states,'contact_substeps':contacts,'differences':diff};(A/'parity.json').write_text(json.dumps(out,indent=2)+'\n');print('pre-command parity',out);assert not diff,diff[:10]
def yaw(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def measure(z,e):
 seq=[z[e,i] for i in range(601,1201)];y=[yaw(v['q']) for v in seq];u=[y[0]]
 for v in y[1:]:u.append(u[-1]+math.atan2(math.sin(v-u[-1]),math.cos(v-u[-1])))
 dx=seq[-1]['q'][0]-seq[0]['q'][0];dy=seq[-1]['q'][1]-seq[0]['q'][1]
 return {'yaw_change_rad':u[-1]-u[0],'trajectory_angle_rad':math.atan2(dy,dx),'dx_m':dx,'dy_m':dy,'end_height_m':seq[-1]['q'][2]}
rows=[]
for e in range(2,8):
 x=measure(s,e);y=measure(t,e);rows.append({'lane':e,'normal_blend':[0,0,.15,.15,.35,.35,.6,.6][e],'original_offset':.2 if e%2==0 else -.2,'original':x,'reversed':y,'yaw_difference_rad':x['yaw_change_rad']-y['yaw_change_rad']})
(A/'comparison.json').write_text(json.dumps({'rows':rows,'limits':'one used seed, obstacle-free model, changed uncalibrated wing closure throughout, 50 Hz body snapshots; no biological or closed-loop flight qualification'},indent=2)+'\n')
for r in rows:print(r)
