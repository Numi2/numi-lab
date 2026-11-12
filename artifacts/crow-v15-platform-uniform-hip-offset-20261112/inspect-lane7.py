from pathlib import Path
import subprocess,gzip,json,collections,math
A=Path('artifacts/crow-v15-platform-uniform-hip-offset-20261112');D=A/'autonomous-20261126-posture';meta=json.loads((D/'run.stdout').read_text())['first_episode_navigation'];end=meta['steps_by_environment'][7];assert end==4800 and not meta['physical_failure_by_environment'][7]
p=subprocess.Popen(['cat',*[str(x) for x in sorted(D.glob('run.stderr.gz.part[0-9][0-9][0-9]'))]],stdout=subprocess.PIPE);counts=collections.Counter();caps=collections.Counter();scene=collections.Counter();overflow=[];errors=[]
with gzip.open(p.stdout,'rt') as f:
 for l in f:
  if l.startswith('CROW_TOE_V_SUBSTEP '):
   x=l.split();t,k,e=map(int,x[1:4]);
   if e==7 and 3800<=t<=end:counts['toe']+=1;caps['toe']+=int(max(abs(float(v)) for v in x[4:])>=11.99)
  elif l.startswith('CROW_BODY_JOINT_V_SUBSTEP '):
   x=l.split();t,k,e=map(int,x[1:4]);
   if e==7 and 3800<=t<=end:counts['body']+=1
  elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
   x=l.split();t,k,e=map(int,x[1:4]);
   if e==7 and 3800<=t<=end:
    counts['contact']+=1
    if int(x[4])>int(x[5]):overflow.append([t,k])
  elif l.startswith('CROW_SCENE_CONTACT '):
   x=l.split();t,k,e,a,b=map(int,x[1:6]);
   if e==7 and 3800<=t<=end and float(x[6])>1e-7 and 40 in (a,b):
    body=b if a==40 else a;scene['foot' if body in (11,15) or 19<=body<=34 else 'nonfoot']+=1
  elif l.startswith('CROW_SUBSTEP_ERROR'):errors.append(l.strip())
assert p.wait()==0
print('lane 7 end',end,'expected substeps',(end-3800+1)*16,'coverage',dict(counts),'toe caps',dict(caps),'scene platform contacts',dict(scene),'overflow',len(overflow),'errors',len(errors))
r={'lane':7,'first_episode_end':end,'expected_substeps':(end-3800+1)*16,'coverage':dict(counts),'toe_caps':dict(caps),'platform_contact_counts':dict(scene),'overflow':overflow,'errors':errors,'scope':'Velocity streams are only qualified where coverage is complete. Nonfoot counts cover platform contacts with positive normal impulse, not every possible collider. Used development start and old flight law.'};(A/'lane7-diagnostic.json').write_text(json.dumps(r,indent=2)+'\n')
