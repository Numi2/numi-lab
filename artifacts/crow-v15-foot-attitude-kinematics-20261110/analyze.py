from pathlib import Path
import gzip,json,subprocess,math
R=Path.cwd();A=R/'artifacts/crow-v15-foot-attitude-kinematics-20261110';p=R/'artifacts/crow-v15-landing-support-alignment-20261110/diagnostic-reserved-final-j-ahead-original.stderr.gz';d={}
for l in gzip.open(p,'rt'):
 if l.startswith('CROW_SUPPORT_SAMPLE '):
  z=json.loads(l.split(' ',1)[1]);t=z['step']
  if t in (2550,2600,2650,2700,2750,2800,2850):d[t]=z['q']
q=d[2600];cases=[('reference',q)]
for i in (14,15,16):
 x=q.copy();x[i]+=.1;cases.append(('joint'+str(i)+'+0.1',x))
a=q.copy();x,y,z,w=q[3:7];s=math.sin(.05);c=math.cos(.05);a[3:7]=[c*x+s*z,c*y+s*w,c*z-s*x,c*w-s*y];cases.append(('root_pitch+0.1',a))
for t,v in d.items():cases.append(('trace'+str(t),v))
inp=''.join(str(i)+' '+' '.join(map(str,q))+'\n' for i,(_,q) in enumerate(cases));p=subprocess.run([str(R/'build/crow-v15-foot-attitude-kinematics-20261110/fk_probe')],input=inp,text=True,capture_output=True,check=True)
out=[]
for l in p.stdout.splitlines():
 a=list(map(float,l.split()));idx=int(a[0]);x,y,z,w=a[4:8];pitch=math.asin(max(-1,min(1,2*(w*y-z*x))));q=cases[idx][1];rx,ry,rz,rw=q[3:7];root=math.asin(max(-1,min(1,2*(rw*ry-rz*rx))));out.append({'case':cases[idx][0],'foot_position':a[1:4],'foot_pitch':pitch,'root_pitch':root,'q14_16':q[14:17]})
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
