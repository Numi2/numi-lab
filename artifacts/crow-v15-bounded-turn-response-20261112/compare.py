from pathlib import Path
import collections,gzip,json,math,statistics
R=Path('artifacts');A=R/'crow-v15-bounded-turn-response-20261112';B=R/'crow-v15-straight-command-second-seed-20261112';p='candidate-20261126-rebound-flexed/run.stderr.gz'
def read(path):
 st={};cs=collections.defaultdict(list)
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);st[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();ti,sub,e=map(int,z[1:4]);
    if ti<=450 or e in (0,1):cs[e].append(tuple(z[1:]))
 return st,cs
s,c=read(B/p);t,d=read(A/p);diff=[];ns=nc=0
for e in range(8):
 end=1200 if e in (0,1) else 450
 for i in range(1,end+1):
  for k in ('q','v','policy_action','contacts'):
   if s[e,i][k]!=t[e,i][k]:diff.append([e,i,k])
  ns+=1
 aa=[z for z in c[e] if int(z[0])<=end];bb=[z for z in d[e] if int(z[0])<=end];nc+=len(aa)
 if aa!=bb:diff.append([e,'contact'])
(A/'parity.json').write_text(json.dumps({'state_samples':ns,'contact_substeps':nc,'differences':diff},indent=2)+'\n');print('matched',ns,nc,'differences',len(diff));assert not diff,diff[:10]
meta=json.loads((A/'candidate-20261126-rebound-flexed/run.stdout').read_text())['first_episode_navigation'];rows=[]
def angles(q):
 x,y,z,w=q[3:7];return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
for e in range(8):
 end=min(1200,meta['steps_by_environment'][e]);seq=[t[e,i] for i in range(601,end+1) if (e,i) in t];a=[angles(v['q']) for v in seq];yaw=[x[2] for x in a];unwrapped=[yaw[0]] if yaw else []
 for v in yaw[1:]:unwrapped.append(unwrapped[-1]+math.atan2(math.sin(v-unwrapped[-1]),math.cos(v-unwrapped[-1])))
 first,last=seq[0],seq[-1];dx=last['q'][0]-first['q'][0];dy=last['q'][1]-first['q'][1];heading=math.atan2(dy,dx)
 rows.append({'lane':e,'turn_command':[0,0,.15,.15,-.15,-.15,.15,-.15][e],'end_step':end,'termination':meta['termination_reason_by_environment'][e],'yaw_change_rad':unwrapped[-1]-unwrapped[0],'trajectory_angle_rad':heading,'dx_m':dx,'dy_m':dy,'max_abs_roll_rad':max(abs(v[0]) for v in a),'height_range_m':[min(v['q'][2] for v in seq),max(v['q'][2] for v in seq)],'mean_abs_world_vy_m_s':statistics.mean(abs(v['v'][1]) for v in seq)})
(A/'flight-summary.json').write_text(json.dumps({'rows':rows,'scope':'Turn demand is a bounded constant relative-heading input, not a target-angle closed-loop maneuver. Yaw and path heading are 50 Hz samples; no biological validation.'},indent=2)+'\n')
for x in rows:print(x['lane'],'turn',x['turn_command'],'end',x['end_step'],'yaw change',round(x['yaw_change_rad'],3),'path angle',round(x['trajectory_angle_rad'],3),'dx dy',round(x['dx_m'],2),round(x['dy_m'],2),'roll',round(x['max_abs_roll_rad'],3),'z',[round(z,2) for z in x['height_range_m']],'reason',x['termination'])
