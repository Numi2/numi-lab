from pathlib import Path
import collections,gzip,json,math,statistics
R=Path('artifacts');A=R/'crow-v15-wing-normal-heading-gain-screen-20261112';B=R/'crow-v15-wing-normal-closure-screen-20261112';p='candidate-20261126-rebound-flexed/run.stderr.gz'
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
meta=json.loads((A/'candidate-20261126-rebound-flexed/run.stdout').read_text())['first_episode_navigation'];rows=[]
def yaw(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
for e in range(2,8):
 end=min(meta['steps_by_environment'][e],1200); target=.35 if e%2==0 else -.35
 seq=[t[e,i] for i in range(1001,end+1) if (e,i) in t];
 if not seq:
  rows.append({'lane':e,'target_rad':target,'end_step':end,'no_final_window':True});continue
 errors=[abs(math.atan2(math.sin(target-yaw(x['q'])),math.cos(target-yaw(x['q'])))) for x in seq];vx=statistics.mean(x['v'][0] for x in seq);vy=statistics.mean(x['v'][1] for x in seq)
 rows.append({'lane':e,'normal_blend':[0,0,.15,.15,.35,.35,.6,.6][e],'target_rad':target,'end_step':end,'termination':meta['termination_reason_by_environment'][e],'mean_abs_body_heading_error_rad':statistics.mean(errors),'max_abs_body_heading_error_rad':max(errors),'final_body_heading_rad':yaw(seq[-1]['q']),'mean_path_heading_rad':math.atan2(vy,vx),'height_range_m':[min(x['q'][2] for x in seq),max(x['q'][2] for x in seq)]})
(A/'heading-summary.json').write_text(json.dumps({'rows':rows,'limits':'one used seed, fixed target, obstacle-free model, uncalibrated wing closure; 50 Hz samples, not biological or route qualification'},indent=2)+'\n')
for r in rows:print(r)
