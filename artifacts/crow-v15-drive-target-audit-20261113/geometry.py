from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-drive-target-audit-20261113';src=R/'artifacts/crow-v15-presolve-contact-audit-20261113/push-nowings-20261127-rebound-presolve/run.stderr.gz';exe=R/'build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics';lib=R/'build/crow-v15-hip-roll-journey-integration-20261111/libmetalrobo.dylib'
state=None
with gzip.open(src,'rt') as f:
 for line in f:
  if line.startswith('CROW_KIN_STATE 348 0 5 '):
   z=line.split();assert tuple(map(int,z[1:6]))==(348,0,5,41,40);state=list(map(float,z[6:]));break
assert state and len(state)==81
variants=[('baseline',None,0.)]+[(name,inds,sign*.001) for name,inds in (('hip',(14,17)),('knee',(15,18)),('ankle',(16,19))) for sign in (-1,1)]
lines=[]
for i,(name,inds,delta) in enumerate(variants):
 v=state.copy()
 if inds:
  for k in inds:v[k]+=delta
 lines.append('5 '+str(i)+' '+' '.join(map(str,v)))
r=subprocess.run([str(exe)],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});samples={i:{} for i in range(len(variants))}
for line in r.stdout.splitlines():
 z=line.split();e,i=map(int,z[1:3]);assert e==5
 if z[0]=='com':samples[i]['com']=list(map(float,z[4:7]))
 elif z[0]=='body' and z[3] in ('crow_left_foot','crow_right_foot','crow_body'):samples[i][z[3]]=list(map(float,z[4:7]))
assert all(set(x)=={'com','crow_left_foot','crow_right_foot','crow_body'} for x in samples.values())
jac={}
for j,name in enumerate(('hip','knee','ankle')):
 minus=samples[1+j*2];plus=samples[2+j*2];jac[name]={}
 for side in ('crow_left_foot','crow_right_foot'):
  jac[name][side+'_minus_com_m_per_rad']=[((plus[side][k]-plus['com'][k])-(minus[side][k]-minus['com'][k]))/.002 for k in range(3)]
out={'lane':5,'step':348,'q':state[:41],'finite_difference_rad':.001,'relative_foot_com_jacobian':jac,'kinematics_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'scope':'Posthoc instantaneous fixed-root symmetric-joint finite differences about one used-seed development posture. Foot body position relative to whole-model COM, not contact point, does not include forces, contact constraints, dynamics, or demonstrate a viable push.'};(A/'geometry.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(jac,indent=2))
