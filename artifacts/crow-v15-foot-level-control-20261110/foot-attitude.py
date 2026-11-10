from pathlib import Path
import gzip,json,subprocess,math
R=Path.cwd();A=R/'artifacts/crow-v15-foot-level-control-20261110';out={}
for p in sorted(A.glob('diagnostic-*.stderr.gz')):
 name=p.name[:-len('.stderr.gz')];stdout=A/(name+'.stdout')
 if not stdout.exists() or not stdout.stat().st_size:continue
 end=json.loads(stdout.read_text())['first_episode_navigation']['steps_by_environment'][0];states={}
 for l in gzip.open(p,'rt'):
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   z=json.loads(l.split(' ',1)[1]);t=z['step']
   if t<=end and t in (2500,2550,2600,2650,2700,2750,2800,2850,2900,3000,3500,4000,4500):states[t]=z['q']
 inp=''.join(str(t)+' '+' '.join(map(str,q))+'\n' for t,q in sorted(states.items()))
 result=subprocess.run([str(R/'build/crow-v15-foot-attitude-kinematics-20261110/fk_probe')],input=inp,text=True,capture_output=True,check=True);rows=[]
 for l in result.stdout.splitlines():
  a=list(map(float,l.split()));t=int(a[0]);x,y,z,w=a[4:8];q=states[t];rx,ry,rz,rw=q[3:7];rows.append({'step':t,'foot_pitch':math.asin(max(-1,min(1,2*(w*y-z*x)))),'root_pitch':math.asin(max(-1,min(1,2*(rw*ry-rz*rx)))),'q14_16':q[14:17]})
 out[p.name]=rows
(A/'foot-attitude.json').write_text(json.dumps(out,indent=2)+'\n')
