from pathlib import Path
import collections,gzip,json,math,statistics
R=Path('artifacts');A=R/'crow-v15-commanded-straight-flight-20261112';B=R/'crow-v15-speed-feedback-long-flight-20261112';p='candidate-20261127-rebound-flexed/run.stderr.gz'
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
s,c=read(B/p);t,d=read(A/p);diff=[];ns=0;nc=0
for e in range(8):
 end=1200 if e in (0,1) else 450
 for i in range(1,end+1):
  for k in ('q','v','policy_action','contacts'):
   if s[e,i][k]!=t[e,i][k]:diff.append([e,i,k])
  ns+=1
 aa=[z for z in c[e] if int(z[0])<=end];bb=[z for z in d[e] if int(z[0])<=end];nc+=len(aa)
 if aa!=bb:diff.append([e,'contact'])
print('matched samples',ns,'contact substeps',nc,'differences',len(diff));(A/'parity.json').write_text(json.dumps({'state_samples':ns,'contact_substeps':nc,'differences':diff},indent=2)+'\n');assert not diff,diff[:10]
meta=json.loads((A/'candidate-20261127-rebound-flexed/run.stdout').read_text())['first_episode_navigation'];rows=[]
for e in range(8):
 end=min(1200,meta['steps_by_environment'][e]);seq=[t[e,i] for i in range(601,end+1) if (e,i) in t];z=[v['q'][2] for v in seq];vx=[v['v'][0] for v in seq];vy=[v['v'][1] for v in seq];last=t[e,end]
 rows.append({'lane':e,'command':'baseline' if e<2 else {'speed_m_s': .35 if e<4 else .55 if e<6 else .75,'height_m': .95 if e%2==0 else 1.1},'end_step':end,'termination':meta['termination_reason_by_environment'][e],'height_range_m':[min(z),max(z)] if z else None,'mean_world_vx_m_s':statistics.mean(vx) if vx else None,'mean_abs_world_vy_m_s':statistics.mean(map(abs,vy)) if vy else None,'final_position_m':last['q'][:3],'final_velocity_m_s':last['v'][:3]})
(A/'flight-summary.json').write_text(json.dumps({'rows':rows,'scope':'50 Hz endpoint samples; world velocity differs from body-relative requested speed.'},indent=2)+'\n')
for x in rows:print(x['lane'],x['command'],'end',x['end_step'],'reason',x['termination'],'zrange',[round(z,2) for z in x['height_range_m']] if x['height_range_m'] else None,'meanvx',round(x['mean_world_vx_m_s'],3) if x['mean_world_vx_m_s'] is not None else None,'meanabsvy',round(x['mean_abs_world_vy_m_s'],3) if x['mean_abs_world_vy_m_s'] is not None else None,'vfinal',[round(v,3) for v in x['final_velocity_m_s']])
