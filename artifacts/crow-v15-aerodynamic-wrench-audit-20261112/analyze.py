from pathlib import Path
import gzip,json,os,statistics,subprocess,math,collections
A=Path('artifacts/crow-v15-aerodynamic-wrench-audit-20261112');D=A/'candidate-20261126-rebound-aero';states={};w={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_AERO_STATE':
   t,k,e=map(int,z[1:4]);key=e,t,k;assert key not in states;states[key]=list(map(float,z[4:]))
  elif z[0]=='CROW_AERO_WRENCH':
   t,k,e,b=map(int,z[1:5]);key=e,t,k,b;assert key not in w;w[key]=list(map(float,z[5:]));assert len(w[key])==6
assert len(states)==8*50*16 and len(w)==8*50*16*4,(len(states),len(w))
assert all(len(v)==81 and all(math.isfinite(a) for a in v) for v in states.values())
# The input index is arbitrary and provides a reversible mapping to (t,sub).
payload=''.join(f'{e} {(t-651)*16+k} '+' '.join(map(str,s))+'\n' for (e,t,k),s in sorted(states.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=payload,text=True,capture_output=True,check=True,env=env)
bodies={}
for line in p.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body' and z[3] in ('crow_body','crow_left_wing','crow_right_wing','crow_tail'):bodies[int(z[1]),int(z[2]),z[3]]=list(map(float,z[4:]))
assert len(bodies)==len(states)*4
names={0:'crow_body',5:'crow_left_wing',6:'crow_right_wing',7:'crow_tail'}
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
values=collections.defaultdict(list)
for (e,t,k),qv in sorted(states.items()):
 i=(t-651)*16+k;root=bodies[e,i,'crow_body'];q=qv[3:7];x,y,z,ww=q;up=[2*(x*z+ww*y),2*(y*z-ww*x),1-2*(x*x+y*y)]
 total=[0.,0.,0.]
 for body,name in names.items():
  a=w[e,t,k,body];pos=bodies[e,i,name][:3];arm=[pos[j]-root[j] for j in range(3)];moment=cross(arm,a[:3]);torque=[a[j+3]+moment[j] for j in range(3)];yaw=sum(torque[j]*up[j] for j in range(3));values[e,name+'_yaw'].append(yaw);values[e,name+'_fx'].append(a[0]);values[e,name+'_fz'].append(a[2]);total=[total[j]+torque[j] for j in range(3)]
 values[e,'total_yaw'].append(sum(total[j]*up[j] for j in range(3)))
rows=[]
for e in range(8):
 row={'lane':e}
 for name in ('crow_body','crow_left_wing','crow_right_wing','crow_tail','total'):
  a=values[e,name+'_yaw'];row[name+'_mean_yaw_nm']=statistics.mean(a);row[name+'_yaw_range_nm']=[min(a),max(a)]
 for name in ('crow_left_wing','crow_right_wing'):
  row[name+'_mean_fx_n']=statistics.mean(values[e,name+'_fx']);row[name+'_mean_fz_n']=statistics.mean(values[e,name+'_fz'])
 rows.append(row)
out={'sampling':{'control_steps':[651,700],'physics_substeps':16,'lanes':8,'states':len(states)},'method':'sum world-frame body wrenches and moment arms about torso COM, project onto torso up axis; kinematics reconstructed from source q/v at wing-kernel time','rows':rows,'limitations':'Wrench buffer is captured immediately after wing-load kernel, but may contain previously accumulated physical actuator contributions; torso includes fuselage and any separately accumulated root wrenches. This is an uncalibrated model diagnostic, not measured crow aerodynamics.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');[print(r) for r in rows]
