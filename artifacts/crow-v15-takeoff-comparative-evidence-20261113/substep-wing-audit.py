from pathlib import Path
import collections,gzip,json,os,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-takeoff-comparative-evidence-20261113';p=R/'artifacts/crow-v15-continuous-push-transition-20261113/push-wing-20261126-rebound-continuous/run.stderr.gz';q={};contact={}
with gzip.open(p,'rt') as f:
 for line in f:
  if line.startswith('CROW_Q_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if e>=2 and 350<=t<=365:q[e,t,k]=list(map(float,z[4:]))
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if e>=2 and 350<=t<=365:contact[e,t,k]=sum(map(float,z[8:10]))
assert len(q)==len(contact)==6*16*16
assert all(len(v)==41 for v in q.values());keys={};inputs=[]
for i,(key,v) in enumerate(sorted(q.items())):keys[i]=key;inputs.append(str(i)+' '+' '.join(map(str,v+[0.]*40)))
exe=R/'build/crow-v15-leg-wing-transition-20261113/wing-fk';env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/crow-v15-presolve-contact-audit-20261113')};r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env=env);values={}
for line in r.stdout.splitlines():
 z=line.split();i=int(z[0]);v=list(map(float,z[1:]));assert len(v)==6;values[keys[i]]=(v[2],v[5])
assert len(values)==len(q)
rows=[]
for e in range(2,8):
 samples=[(t,k) for t in range(355,361) for k in range(16)];peak=max(samples,key=lambda z:sum(values[(e,*z)]))
 touches=[(t,k) for t in range(350,366) for k in range(16) if contact[e,t,k]>1e-7];last=max(touches) if touches else None
 def idx(x):return 16*x[0]+x[1]
 rows.append({'lane':e,'peak_relative_wing_com_step_substep':peak,'last_foot_contact_step_substep_350_365':last,'peak_minus_last_contact_ms':(idx(peak)-idx(last))*1.25 if last else None,'peak_relative_wing_com_z_m':list(values[(e,*peak)])})
out={'scope':'Posthoc 1.25 ms sampled substep forward kinematics. Wing-body COM relative to torso in world axes is a proxy for stroke phase, not a wingtip. Contact means summed measured foot normal impulse >1e-7 per substep; last contact is only in recorded 350–365 and not necessarily physically exact toe-off. Does not measure aerodynamic force.','rows':rows};(A/'substep-wing-audit.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x)
