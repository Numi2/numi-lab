from pathlib import Path
import gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts')
A=R/'crow-v15-neutral-wing-handoff-20261113'
paths={'reference':R/'crow-v15-aero-wrench-observer-20261113/push-wing-20261126-rebound-aero','neutral':A/'push-wing-20261126-rebound-neutral'}
result={}
for arm,p in paths.items():
 kin={};aero={};errors=[]
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=='CROW_KIN_STATE':
    t,k,e,nq,nv=map(int,z[1:6]);v=list(map(float,z[6:]));assert len(v)==nq+nv and (e,t,k) not in kin;kin[e,t,k]=v[nq:]
   elif z[0]=='CROW_AERO_FORCE':
    t,k,e,stride=map(int,z[1:5]);v=list(map(float,z[5:]));assert stride>=35 and len(v)==12 and v[-1]==4 and all(map(math.isfinite,v)) and (e,t,k) not in aero;aero[e,t,k]=v
   elif z[0]=='CROW_AERO_OBSERVER_ERROR':errors.append(line.strip())
 assert len(aero)==len(kin)==8*28*16 and not errors,(arm,len(aero),len(kin),errors)
 rows=[]
 for e in range(2,8):
  keys=[(e,t,k) for t in range(348,376) for k in range(16)]
  m=max(keys,key=lambda x:max(abs(kin[x][8]),abs(kin[x][9])))
  f=max(keys,key=lambda x:aero[x][2])
  rows.append({'lane':e,'max_substep_flap_speed_rad_s':max(abs(kin[m][8]),abs(kin[m][9])),'speed_step':m[1],'speed_substep':m[2],'force_at_speed_peak_n':aero[m][2],'max_vertical_aero_force_n':aero[f][2],'force_step':f[1],'force_substep':f[2]})
 result[arm]={'rows':rows,'records':len(kin)}
(A/'substep-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for x,y in zip(result['reference']['rows'],result['neutral']['rows']):
 print('lane',x['lane'],'speed',round(x['max_substep_flap_speed_rad_s'],2),'->',round(y['max_substep_flap_speed_rad_s'],2),'peak model force',round(x['max_vertical_aero_force_n'],2),'->',round(y['max_vertical_aero_force_n'],2))
