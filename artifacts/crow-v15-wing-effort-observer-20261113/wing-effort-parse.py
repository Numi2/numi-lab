from pathlib import Path
import gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-wing-effort-observer-20261113');p=R/'push-wing-20261126-rebound-effort/run.stderr.gz';eff={};aero={}
with gzip.open(p,'rt') as f:
 for l in f:
  z=l.split()
  if not z:continue
  if z[0]=='CROW_WING_EFFORT':
   t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==14 and all(map(math.isfinite,v)) and (e,t,k) not in eff;eff[e,t,k]=[v[:7],v[7:]]
  if z[0]=='CROW_AERO_FORCE':
   t,k,e=map(int,z[1:4]);v=list(map(float,z[5:]));assert len(v)==12 and (e,t,k) not in aero;aero[e,t,k]=v
assert len(eff)==len(aero)==8*28*16
rows=[]
for e in range(2,8):
 keys=[(e,t,k) for t in range(348,376) for k in range(16)];key=max(keys,key=lambda x:max(abs(eff[x][0][4]),abs(eff[x][1][4])))
 rows.append({'lane':e,'step':key[1],'substep':key[2],'wings':[dict(zip(('working_effort_buffer','target_rad','start_q_rad','end_q_rad','start_v_rad_s','presolve_v_rad_s','end_v_rad_s'),v)) for v in eff[key]],'aero_vertical_force_n':aero[key][2]})
(R/'wing-effort-peaks.json').write_text(json.dumps({'records':len(eff),'scope':'working effort buffer is not established as total wing actuator torque','rows':rows},indent=2)+'\n')
for r in rows:print(r['lane'],r['step'],r['substep'],'efforts',*[round(v['working_effort_buffer'],4) for v in r['wings']],'targets',*[round(v['target_rad'],4) for v in r['wings']])
