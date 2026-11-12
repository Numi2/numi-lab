from pathlib import Path
import gzip,json,math,collections
R=Path('artifacts');A=R/'crow-v15-slow-push-crouch-control-20261112';B=R/'crow-v15-slow-push-uniform-dev-20261112';p='candidate-20261127-rebound-flexed/run.stderr.gz'
def read(path):
 states={};contacts=collections.defaultdict(list)
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);t=x['step'];e=x['lane']
    if t<=500:states[e,t]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if t<=500:contacts[e].append((t,k,float(z[8])+float(z[9]),float(z[10])))
 return states,contacts
s,c=read(B/p);t,d=read(A/p)
assert all((e,i) in s and (e,i) in t for e in range(8) for i in range(1,501))
pre=collections.Counter();diff=[]
for e in range(8):
 for i in range(1,348):
  for k in ('q','v','policy_action','contacts'):
   if s[e,i][k]!=t[e,i][k]:diff.append([e,i,k])
  pre['state_samples']+=1
 ac=[x for x in c[e] if x[0]<=347];bc=[x for x in d[e] if x[0]<=347];pre['contact_substeps']+=len(ac)
 if ac!=bc:diff.append([e,'pre-contact-mismatch'])
print('pre',dict(pre),'differences',len(diff));assert not diff,diff[:10]
def summary(states,cs,e):
 a=states[e,500]; q=a['q'];v=a['v'];la=[z for z in cs[e] if 348<=z[0]<=500];first=next(((z[0],z[1]) for z in la if z[2]<=1e-7),None)
 return {'position':q[:3],'velocity':v[:3],'first_unloaded_substep':first,'unloaded_8_10_seconds':sum(z[2]<=1e-7 for z in la if 401<=z[0]<=500),'nonfoot_ground_substeps':sum(z[3]>1e-7 for z in la)}
rows=[]
for e in range(8):
 x=summary(s,c,e);y=summary(t,d,e);rows.append({'lane':e,'extension':x,'crouch_hold':y,'extension_minus_crouch_x_m':x['position'][0]-y['position'][0]})
out={'pre_intervention_parity':dict(pre),'differences':diff,'rows':rows,'scope':'Active crouch hold is not passive leg removal; other dynamics including the wings and later leg actions remain active. All starts and layout are used for development.'}
(A/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['lane'],'x ext/control',round(r['extension']['position'][0],3),round(r['crouch_hold']['position'][0],3),'delta',round(r['extension_minus_crouch_x_m'],3),'z',round(r['extension']['position'][2],3),round(r['crouch_hold']['position'][2],3),'unloaded 8-10',r['extension']['unloaded_8_10_seconds'],r['crouch_hold']['unloaded_8_10_seconds'])
