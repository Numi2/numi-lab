from pathlib import Path
import gzip,json,os,subprocess
p=Path('artifacts/crow-v15-perch-contact-alignment-screen-20261112');src=Path('artifacts/crow-v15-perch-regrip-start-envelope-20261112/high/trace.log.gz')
with gzip.open(src,'rt') as f:
 for line in f:
  z=line.split()
  if z[:3]==['step','3','0']:
   a=z.index('q');b=z.index('v');q=list(map(float,z[a+1:b]));v=list(map(float,z[b+1:]));break
 else:raise RuntimeError('missing recorded state')
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};rows=[]
for index,body in ((14,'crow_left_front_toe_2_proximal'),(17,'crow_right_front_toe_2_proximal')):
 for delta in (-.08,0,.08):
  a=q.copy();a[index]+=delta;msg='0 1 '+' '.join(map(str,a+v))+'\n';res=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=msg,text=True,capture_output=True,check=True,env=env)
  for line in res.stdout.splitlines():
   z=line.split()
   if z and z[0]=='body' and z[3]==body:rows.append({'hip_q_index':index,'hip_offset_rad':delta,'toe_body':body,'position_m':list(map(float,z[4:7]))});break
  else:raise RuntimeError('missing toe')
out={'source':'lane 3 endpoint of first control interval from retained +6 mm constructed start','method':'independent articulated forward kinematics, all other coordinates and velocities held fixed','rows':rows,'limitations':'local geometry only; does not predict resulting contact, force or controller stability'};(p/'initial-foot-kinematics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
