from pathlib import Path
import collections,gzip,json,math,statistics
R=Path('artifacts');A=R/'crow-v15-flight-height-feedback-screen-20261112';B=R/'crow-v15-straight-command-second-seed-20261112';p='candidate-20261126-rebound-flexed/run.stderr.gz'
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
for e in range(8):
 end=min(1200,meta['steps_by_environment'][e]);seq=[t[e,i] for i in range(601,end+1) if (e,i) in t];z=[v['q'][2] for v in seq];vx=[v['v'][0] for v in seq];vy=[v['v'][1] for v in seq];fw=[]
 for v in seq:
  x,y,k,w=v['q'][3:7];yaw=math.atan2(2*(w*k+x*y),1-2*(y*y+k*k));fw.append(math.cos(yaw)*v['v'][0]+math.sin(yaw)*v['v'][1])
 last=t[e,end];target=.55
 rows.append({'lane':e,'gain':[0,0,.4,.4,.8,.8,1.2,1.2][e],'end_step':end,'termination':meta['termination_reason_by_environment'][e],'samples':len(seq),'height_range_m':[min(z),max(z)] if z else None,'mean_height_error_m':statistics.mean(z)-.95 if z else None,'mean_body_forward_m_s':statistics.mean(fw) if fw else None,'mean_abs_forward_error_m_s':statistics.mean(abs(v-target) for v in fw) if fw else None,'mean_abs_world_lateral_speed_m_s':statistics.mean(map(abs,vy)) if vy else None,'final_position_m':last['q'][:3],'final_velocity_m_s':last['v'][:3]})
(A/'flight-summary.json').write_text(json.dumps({'rows':rows,'scope':'Used seed. 50 Hz endpoint samples; body-forward velocity projected into yaw frame, not a continuous error bound.'},indent=2)+'\n')
for x in rows:print(x['lane'],'gain',x['gain'],'end',x['end_step'],'mean body forward',round(x['mean_body_forward_m_s'],3) if x['samples'] else None,'MAE',round(x['mean_abs_forward_error_m_s'],3) if x['samples'] else None,'height err',round(x['mean_height_error_m'],3) if x['samples'] else None,'mean |vy|',round(x['mean_abs_world_lateral_speed_m_s'],3) if x['samples'] else None,'reason',x['termination'])
