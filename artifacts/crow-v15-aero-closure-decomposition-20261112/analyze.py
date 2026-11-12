from pathlib import Path
import gzip,json,math,os,statistics,subprocess,collections
A=Path('artifacts/crow-v15-aero-closure-decomposition-20261112');D=A/'candidate-20261126-rebound-aero2';states={};allloads={};closure={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_AERO_STATE':
   t,k,e=map(int,z[1:4]);states[e,t,k]=list(map(float,z[4:]))
  elif z[0] in ('CROW_AERO_WRENCH','CROW_CLOSURE_WRENCH'):
   t,k,e,b=map(int,z[1:5]);d=allloads if z[0]=='CROW_AERO_WRENCH' else closure;key=e,t,k,b;assert key not in d;d[key]=list(map(float,z[5:]))
assert len(states)==8*50*16 and len(allloads)==8*50*16*4 and len(closure)==8*50*16*2,(len(states),len(allloads),len(closure))
payload=''.join(f'{e} {(t-651)*16+k} '+' '.join(map(str,s))+'\n' for (e,t,k),s in sorted(states.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=payload,text=True,capture_output=True,check=True,env=env)
bodies={}
for line in p.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body' and z[3] in ('crow_body','crow_left_wing','crow_right_wing'):bodies[int(z[1]),int(z[2]),z[3]]=list(map(float,z[4:]))
assert len(bodies)==len(states)*3
cross=lambda a,b:[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
values=collections.defaultdict(list);transverse=[]
for (e,t,k),qv in sorted(states.items()):
 i=(t-651)*16+k;root=bodies[e,i,'crow_body'];x,y,z,ww=qv[3:7];up=[2*(x*z+ww*y),2*(y*z-ww*x),1-2*(x*x+y*y)]
 for body,name in ((5,'left'),(6,'right')):
  pos=bodies[e,i,'crow_'+name+'_wing'][:3];arm=[pos[j]-root[j] for j in range(3)]
  forward=[1-2*(y*y+z*z),2*(x*y+ww*z),2*(x*z-ww*y)]
  direction=[up[j]+.11875*forward[j] for j in range(3)];norm=math.sqrt(sum(v*v for v in direction));direction=[v/norm for v in direction]
  cf=closure[e,t,k,body][:3];along=sum(cf[j]*direction[j] for j in range(3));transverse.append(math.sqrt(sum((cf[j]-along*direction[j])**2 for j in range(3))))
  for label,loads in (('total',allloads),('closure',closure)):
   a=loads[e,t,k,body];assert all(math.isfinite(v) for v in a);m=cross(arm,a[:3]);torque=[a[j+3]+m[j] for j in range(3)]
   values[e,name,label,'yaw'].append(sum(torque[j]*up[j] for j in range(3)))
   values[e,name,label,'vertical'].append(a[2])
   values[e,name,label,'forward_world'].append(a[0])
rows=[]
for e in range(8):
 row={'lane':e}
 for side in ('left','right'):
  for label in ('total','closure'):
   for axis in ('yaw','vertical','forward_world'):
    row[f'{side}_{label}_mean_{axis}']=statistics.mean(values[e,side,label,axis])
  row[f'{side}_closure_fraction_mean_vertical']=row[f'{side}_closure_mean_vertical']/row[f'{side}_total_mean_vertical']
 rows.append(row)
out={'sampling':{'control_steps':[651,700],'physics_substeps':16,'lanes':8,'states':len(states)},'method':'counterfactual closure-only copy of identical wing kernel at each source q/v, in zeroed scratch buffer, never fed to solver; force in world coordinates, yaw moment about torso COM projected onto torso up','max_transverse_closure_force_n':max(transverse),'rows':rows,'limits':'Original wing wrench may include previously accumulated physical actuator contributions; total-minus-closure is not guaranteed to be aerodynamic blade load alone. One one-second window, one used seed, no real crow force measurements, uncalibrated model. State/contact parity is required separately.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');[print(r) for r in rows]
