from pathlib import Path
import gzip,json,math,os,subprocess
A=Path('artifacts/crow-v15-gait-slip-observer-20261112')
p=Path('artifacts/crow-v15-substep-target-interpolation-screen-20261112/sham-20261126');s={}
with gzip.open(p/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1])
   if x['lane']==5 and 201<=x['step']<=300:s[x['step']]=x
assert len(s)==100
inp=''.join('5 '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for t,x in sorted(s.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
out=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,env=env,check=True).stdout
b={}
for l in out.splitlines():
 z=l.split()
 if z[0]=='body':b[int(z[2]),z[3]]=list(map(float,z[4:]))
r=json.load(gzip.open(p/'replay.json.gz','rt'))['payload'];names=[n for n in r['body_names'] if n.startswith(('crow_left_','crow_right_')) and ('foot' in n or 'toe' in n)]
pe=ve=ae=0.
for x in r['frames']:
 if not 200<=x['step']<=299:continue
 t=x['step']+1
 for n in names:
  j=r['body_names'].index(n)*13;v=x['body_states'][j:j+13];z=b[t,n]
  pe=max(pe,math.dist(v[:3],z[:3]));ve=max(ve,math.dist(v[7:10],z[7:10]));ae=max(ae,math.dist(v[10:13],z[10:13]))
res={'states':len(s),'foot_and_toe_bodies':len(names),'max_position_error_m':pe,'max_linear_velocity_error_m_s':ve,'max_angular_velocity_error_rad_s':ae,'pass':pe<1e-5 and ve<1e-4 and ae<1e-4}
(A/'reference-kinematics-validation.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
assert res['pass']
