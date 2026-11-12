from pathlib import Path
import collections,gzip,hashlib,json
A=Path('artifacts/crow-v15-toe-cap-timing-audit-20261112');P=Path('artifacts/crow-v15-release-perturbation-screen-20261112/release');window={};caps=[]
with gzip.open(P/'trace.log.gz','rt') as f:
 for l in f:
  z=l.split()
  if not z or z[0] not in ('step','control','event','contact') or int(z[1])!=3:continue
  t=int(z[2])+1
  if not 259<=t<=264:continue
  w=window.setdefault(t,{'contacts':[]})
  if z[0]=='step':
   i=z.index('q');j=z.index('v');q=list(map(float,z[i+1:j]));v=list(map(float,z[j+1:]));assert len(q)==41 and len(v)==40
   w['right_rear_proximal_q_rad']=q[37];w['right_rear_proximal_v_rad_s']=v[36];w['root_height_m']=q[2]
  elif z[0]=='control':
   a=list(map(float,z[3:]));w['right_rear_target_rad']=a[3];w['right_front_target_rad']=a[4]
  elif z[0]=='event':
   w['release_active']=bool(int(z[3]));w['release_finished']=bool(int(z[11]));w['left_up_impulse']=float(z[5]);w['right_up_impulse']=float(z[6])
  elif z[0]=='contact' and ('rear_toe' in z[3] or 'foot' in z[3]):w['contacts'].append({'a':z[3],'b':z[4],'normal_impulse':float(z[5]),'separation':float(z[6])})
with gzip.open(P/'substeps.log.gz','rt') as f:
 for l in f:
  z=l.split()
  if not z or z[0]!='CROW_TOE_VELOCITY':continue
  t,k,e=map(int,z[1:4])
  if e!=3 or not 259<=t<=264:continue
  v=list(map(float,z[-18:-2]));assert len(v)==16
  if max(map(abs,v))>=11.99:caps.append({'interval':t,'substep':k,'slot':max(range(16),key=lambda i:abs(v[i])),'velocity':max(v,key=abs)})
assert [(x['interval'],x['substep'],x['slot']) for x in caps]==[(262,5,14),(262,6,14)]
out={'input_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (P/'trace.log.gz',P/'substeps.log.gz')},'caps':caps,'window':window,'index_mapping':'Bench source maps toe velocity slots 0-7 to left, 8-15 to right, digit 3 to rear and segment 0 to proximal; nv=40 and nq=41. Trace step is zero-based; substep interval is one-based.','scope':'One constructed perturbed perch bench, sampled endpoint contacts and velocities, not direct drive torques or continuous flight.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'caps':caps,'interval261':window[261],'interval262':window[262]},indent=2))
