from pathlib import Path
import gzip,json,math,os,statistics,subprocess
p=Path('artifacts/crow-v15-wing-bilateral-symmetry-audit-20261112');d=Path('artifacts/crow-v15-differential-pronation-screen-20261112/candidate-20261126-rebound-flexed');samples={}
with gzip.open(d/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);e,t=x['lane'],x['step']
   if 651<=t<=750 and t%2:samples[e,t]=x['q']+x['v']
assert len(samples)==400
payload=''.join(f'{e} {t} '+' '.join(map(str,v))+'\n' for (e,t),v in sorted(samples.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
r=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=payload,text=True,capture_output=True,check=True,env=env)
bodies={}
for line in r.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body' and z[3] in ('crow_body','crow_left_wing','crow_right_wing'):bodies[int(z[1]),int(z[2]),z[3]]=list(map(float,z[4:]))
with gzip.open(d/'replay.json.gz','rt') as f: replay=json.load(f)['payload']
assert replay['environment']==5 and replay['body_state_stride']==13
replay_error=0.; comparisons=0
for frame in replay['frames']:
 t=frame['step']+1
 if (5,t) not in samples:continue
 for name in ('crow_body','crow_left_wing','crow_right_wing'):
  index=replay['body_names'].index(name);actual=frame['body_states'][index*13:index*13+13];expected=bodies[5,t,name]
  replay_error=max(replay_error,max(abs(a-b) for a,b in zip(actual,expected)));comparisons+=1
assert comparisons==150 and replay_error<1e-4,(comparisons,replay_error)
def mul(a,b):
 x,y,z,w=a;X,Y,Z,W=b;return [w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z]
def inv(a):return [-a[0],-a[1],-a[2],a[3]]
def rel(root,wing):return mul(inv(root),wing)
def err(a,b):
 d=abs(sum(x*y for x,y in zip(a,b)))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b)));return 2*math.acos(min(1.,max(0.,d)))
rows=[]
for e in range(8):
 errors=[]
 for _,t in sorted(k for k in samples if k[0]==e):
  root=bodies[e,t,'crow_body'][3:7];l=rel(root,bodies[e,t,'crow_left_wing'][3:7]);r=rel(root,bodies[e,t,'crow_right_wing'][3:7]);mr=[-r[0],r[1],-r[2],r[3]];errors.append(err(l,mr))
 rows.append({'lane':e,'samples':len(errors),'mean_reflection_error_rad':statistics.mean(errors),'p95_reflection_error_rad':sorted(errors)[int(.95*(len(errors)-1))],'max_reflection_error_rad':max(errors)})
out={'interval_s':[13.02,15.0],'source':'accepted first-episode q/v, independent articulated forward kinematics on v15 body','method':'compare torso-relative left wing orientation with reflection of right wing orientation across sagittal plane, quaternion convention xyzw','validation':{'replay_component_max_abs':replay_error,'compared_bodies':comparisons},'rows':rows,'limitations':'Orientation symmetry only; does not validate wing surfaces, aerodynamic forces, or biological motion. Oscillatory flight sampled at 25 Hz; a high-frequency envelope may differ.'};(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');[print(x) for x in rows]
