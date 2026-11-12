from pathlib import Path
import collections,gzip,json
R=Path('artifacts');A=R/'crow-v15-reset-height-sensitivity-20261112';B=R/'crow-v15-single-reset-perturbation-20261112';p='candidate-20261127-rebound-flexed/run.stderr.gz'
def read(path):
 st={};contact=collections.defaultdict(list)
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);st[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();t,k,e=map(int,z[1:4]);contact[e].append((t,k,float(z[6])+float(z[7]),float(z[8])+float(z[9])))
 return st,contact
s,c=read(B/p);t,d=read(A/p);diff=[]
for e in (0,1):
 for i in range(1,501):
  for k in ('q','v','policy_action','contacts'):
   if s[e,i][k]!=t[e,i][k]:diff.append([e,i,k])
 if c[e]!=d[e]:diff.append([e,'contact'])
assert not diff,diff[:10]
rows=[];offset=[0,0,2,2,4,4,6,6]
for e in range(8):
 def f(st,cs):
  first=st[e,1];return {'first_step_z':first['q'][2],'first_step_vz':first['v'][2],'first_interval_up_impulse':sum(v[2] for v in cs[e] if v[0]==1),'early_unloaded_substeps':sum(v[3]<=1e-7 for v in cs[e] if v[0]<=200),'walking_unloaded_substeps':sum(v[3]<=1e-7 for v in cs[e] if 201<=v[0]<=300),'at10_position':st[e,500]['q'][:3]}
 rows.append({'lane':e,'offset_mm':offset[e],'original':f(s,c),'candidate':f(t,d)})
(A/'comparison.json').write_text(json.dumps({'zero_offset_parity':{'lanes':[0,1],'state_samples':1000,'contact_substeps':16000,'differences':diff},'rows':rows},indent=2)+'\n')
for r in rows:
 print(r['lane'],r['offset_mm'],'unloaded original/new',r['original']['early_unloaded_substeps'],r['candidate']['early_unloaded_substeps'],'first impulse',round(r['candidate']['first_interval_up_impulse'],3),'vz',round(r['candidate']['first_step_vz'],3))
