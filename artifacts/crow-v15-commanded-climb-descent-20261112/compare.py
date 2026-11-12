from pathlib import Path
import collections,gzip,json,math,statistics
R=Path('artifacts');A=R/'crow-v15-commanded-climb-descent-20261112';B=R/'crow-v15-straight-command-second-seed-20261112';p='candidate-20261126-rebound-flexed/run.stderr.gz'
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
 end=min(1200,meta['steps_by_environment'][e]);segments={}
 for start,stop,label in [(551,600,'before'),(751,800,'climb'),(951,1000,'descent'),(1151,1200,'recovery')]:
  seq=[t[e,i] for i in range(start,stop+1) if (e,i) in t and i<=end]
  if seq:segments[label]={'samples':len(seq),'height_mean_m':statistics.mean(x['q'][2] for x in seq),'vz_mean_m_s':statistics.mean(x['v'][2] for x in seq),'world_vx_mean_m_s':statistics.mean(x['v'][0] for x in seq),'world_abs_vy_mean_m_s':statistics.mean(abs(x['v'][1]) for x in seq)}
 rows.append({'lane':e,'end_step':end,'termination':meta['termination_reason_by_environment'][e],'segments':segments})
(A/'flight-summary.json').write_text(json.dumps({'rows':rows,'scope':'50 Hz endpoint samples, four one-second windows; no continuous extrema or biological validation.'},indent=2)+'\n')
for x in rows:print(x['lane'],'end',x['end_step'],'height windows',[(k,round(v['height_mean_m'],3)) for k,v in x['segments'].items()],'reason',x['termination'])
