from pathlib import Path
import gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-supported-slow-push-screen-20261113'
P={'push':A/'push-wing-20261126-rebound-slowpush','hold':A/'hold-wing-20261126-rebound-slowpush'}
out={}
for name,p in P.items():
 wing={};aero={};errors=[]
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   z=l.split()
   if not z:continue
   if z[0]=='CROW_WING_EFFORT':
    t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==14 and all(map(math.isfinite,v)) and (e,t,k) not in wing;wing[e,t,k]=[v[:7],v[7:]]
   elif z[0]=='CROW_AERO_FORCE':
    t,k,e=map(int,z[1:4]);v=list(map(float,z[5:]));assert len(v)==12 and all(map(math.isfinite,v)) and v[-1]==4 and (e,t,k) not in aero;aero[e,t,k]=v
   elif z[0] in ('CROW_AERO_OBSERVER_ERROR','CROW_SUBSTEP_ERROR'):errors.append(l.strip())
 assert len(wing)==len(aero)==8*28*16 and not errors,(name,len(wing),len(aero),errors)
 rows=[]
 for e in range(2,8):
  keys=[(e,t,k) for t in range(348,376) for k in range(16)];m=max(keys,key=lambda x:max(abs(wing[x][0][4]),abs(wing[x][1][4])));f=max(keys,key=lambda x:aero[x][2]);maxeff=max(abs(wing[x][j][0]) for x in keys for j in (0,1))
  rows.append({'lane':e,'max_abs_effort':maxeff,'max_substep_flap_speed_rad_s':max(abs(wing[m][0][4]),abs(wing[m][1][4])),'speed_step':m[1],'speed_substep':m[2],'peak_vertical_aero_force_n':aero[f][2],'force_step':f[1],'force_substep':f[2]})
 out[name]=rows
(A/'substep-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for name in ('push','hold'):
 for r in out[name]:print(name,r['lane'],'max effort',round(r['max_abs_effort'],5),'max speed',round(r['max_substep_flap_speed_rad_s'],3),'peak model aero',round(r['peak_vertical_aero_force_n'],3))
 if any(r['max_abs_effort']>(3.0+0.00001) for r in out[name]):raise SystemExit('Bound violated in '+name)
