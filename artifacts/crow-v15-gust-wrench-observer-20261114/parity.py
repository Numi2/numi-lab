from pathlib import Path
import gzip, json, math
A=Path('artifacts/crow-v15-gust-wrench-observer-20261114'); D=A/'push-20261126-rebound-gust'
B=A/'push-20261126-rebound-control'
def read(path):
 d={k:{} for k in ('state','target','foot','toe','force')};d['errors']=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);table='state';key=(x['lane'],x['step']);value=x
   else:
    z=line.split()
    if not z: continue
    kind=z[0]
    if 'OBSERVER_ERROR' in kind or kind=='CROW_SUBSTEP_ERROR': d['errors'].append(line.strip());continue
    if kind=='CROW_DRIVE_TARGET':
     t,e=map(int,z[1:3]);table='target';key=e,t;value=list(map(float,z[3:]))
    elif kind in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_PRE_AERO_WRENCH'):
     t,sub,e=map(int,z[1:4]);key=e,t,sub
     if not 0<=sub<16:d['errors'].append('invalid substep '+str(key))
     if kind=='CROW_TWO_TOE_SUBSTEP':
      if int(z[4])>int(z[5]):d['errors'].append('contact overflow '+str(key))
      table='foot';value=list(map(float,z[6:]))
     elif kind=='CROW_TOE_V_SUBSTEP':table='toe';value=list(map(float,z[4:]))
     else:table='force';value=list(map(float,z[4:]))
    else:continue
   if key in d[table]:d['errors'].append('duplicate '+table+' '+str(key))
   d[table][key]=value
 return d
pairs=[('control',B,Path('artifacts/crow-v15-descent-observer-extension-20261114/push-20261126-rebound-observed')),('gust',D,Path('artifacts/crow-v15-launch-gust-20261114/push-20261126-rebound-gust'))]
result=[]
for label,new,old in pairs:
 n=read(new/'run.stderr.gz');b=read(old/'run.stderr.gz');row={'case':label,'errors':n['errors']+b['errors'],'mismatch':{}}
 for kind in ('state','target','foot','toe'):
  nk=n[kind];bk=b[kind];keys=set(nk)|set(bk);bad=[]
  for key in keys:
   if key not in nk or key not in bk:bad.append(key);continue
   if kind=='state':
    if any(nk[key].get(k)!=bk[key].get(k) for k in ('q','v','policy_action','contacts')):bad.append(key)
   elif nk[key]!=bk[key]:bad.append(key)
  row['mismatch'][kind]={'count':len(bad),'first':sorted(bad)[:5],'new_count':len(nk),'old_count':len(bk)}
 row['pass']=not row['errors'] and all(x['count']==0 for x in row['mismatch'].values());result.append(row)
out={'cases':result,'pass':all(x['pass'] for x in result),'scope':'matched accepted state and observed target/contact/toe records; not independent observer epoch synchronization'};(A/'parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
