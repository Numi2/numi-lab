from pathlib import Path
import gzip,subprocess,json
A=Path('artifacts/crow-v15-foot-placement-sensitivity-20261112');p=Path('artifacts/crow-v15-pitch-response-separated-20261111/zero/trace.log.gz');q=None
with gzip.open(p,'rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='step' and int(z[1])==1 and int(z[2])==9:
   a=z.index('q');b=z.index('v');q=list(map(float,z[a+1:b]));break
assert q
cases=[]
for joint in (14,15,16):
 for delta in (-.2,-.1,0.,.1,.2):
  x=q.copy();x[joint]+=delta;cases.append((joint,delta,x))
inp=''.join(str(i)+' '+' '.join(map(str,x))+'\n' for i,(_,_,x) in enumerate(cases))
p=subprocess.run(['build/crow-v15-foot-placement-sensitivity-20261112/fk'],input=inp,text=True,capture_output=True,check=True)
out=[]
for i,(joint,delta,x) in enumerate(cases):
 bodies={}
 for l in p.stdout.splitlines():
  z=l.split()
  if int(z[0])==i:bodies[z[1]]=list(map(float,z[2:]))
 out.append({'joint_q_index':joint,'delta_rad':delta,'bodies':bodies})
(A/'analysis.json').write_text(json.dumps({'case':'negative-pitch lane at step 10; isolated joint perturbation in a static kinematic copy','rows':out,'limits':'Geometric sensitivity only: no contact, actuator dynamics or realized motion. Joint signs follow v15 model coordinates.'},indent=2)+'\n')
for x in out:print(x['joint_q_index'],x['delta_rad'],x['bodies'].get('crow_left_foot'))
