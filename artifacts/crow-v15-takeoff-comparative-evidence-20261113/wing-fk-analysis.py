from pathlib import Path
import gzip,json,os,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-takeoff-comparative-evidence-20261113';p=R/'artifacts/crow-v15-continuous-push-transition-20261113/push-wing-20261126-rebound-continuous/run.stderr.gz';states={}
with gzip.open(p,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
   if e>=2 and 348<=t<=370:states[e,t]=x
assert len(states)==6*23
keys={};inputs=[]
for i,(key,x) in enumerate(sorted(states.items())):
 keys[i]=key;inputs.append(str(i)+' '+' '.join(map(str,x['q']+x['v'])))
i=len(inputs);x=states[2,350];q=x['q'];v=x['v'];qplus=q.copy();qplus[9]+=.1;qplus[10]+=.1
for ix,qq in ((i,q),(i+1,qplus)):inputs.append(str(ix)+' '+' '.join(map(str,qq+v)))
exe=R/'build/crow-v15-leg-wing-transition-20261113/wing-fk';env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/crow-v15-presolve-contact-audit-20261113')};r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env=env);out={}
for l in r.stdout.splitlines():
 z=l.split();out[int(z[0])]=list(map(float,z[1:]));assert len(out[int(z[0])])==6
assert len(out)==len(inputs)
cal=[out[i+1][j]-out[i][j] for j in (2,5)]
rows=[]
for ix,(e,t) in keys.items():
 x=states[e,t];rows.append({'lane':e,'step':t,'left_wing_com_relative_z_m':out[ix][2],'right_wing_com_relative_z_m':out[ix][5],'flap_joint_q_rad':x['q'][9:11],'flap_joint_v_rad_s':x['v'][8:10]})
(A/'wing-fk-analysis.json').write_text(json.dumps({'reference':'continuous push development seed 20261126','positive_flap_delta_point1_relative_z_changes_m':cal,'rows':rows,'scope':'Wing body center of mass, not wingtip; relative to torso COM in world axes. Endpoint sampling can miss substep reversals. Numerical perturbation changes q indices 9/10; inspect model DOF mapping before identifying stroke direction.'},indent=2)+'\n');print('positive q perturbation relative wing COM z change',cal)
for e in (2,7):
 for x in rows:
  if x['lane']==e and x['step'] in (350,354,355,356,357,358,359,360,361,362,363):print(e,x['step'],round(x['left_wing_com_relative_z_m'],4),round(x['right_wing_com_relative_z_m'],4))
