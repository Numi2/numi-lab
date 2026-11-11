from pathlib import Path
import gzip,hashlib,json,math,subprocess
D=Path('artifacts/crow-v15-walk-all-lanes-geometry-20261111');S=Path('artifacts/crow-v15-walk-16substep-screen-20261111');support={};sub={}
with gzip.open(S/'zero16.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   d=json.loads(l.split(' ',1)[1]);
   if 101<=d['step']<=201:support[d['lane'],d['step']]=d
  elif l.startswith('CROW_WALK_SUBSTEP '):
   v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
   if 101<=t<=201:sub[lane,t,k]=list(map(float,v[6:]))
assert len(support)==8*101 and len(sub)==8*101*16
text=''.join(str(lane)+' '+str(t)+' '+' '.join(map(str,d['q']+d['v']))+'\n' for (lane,t),d in sorted(support.items()))
p=subprocess.run(['build/crow-v15-walk-all-lanes-geometry-20261111/kinematics'],input=text,text=True,capture_output=True,check=True);pose={}
for l in p.stdout.splitlines():
 v=l.split();pose[int(v[0]),int(v[1]),v[2]]=list(map(float,v[3:]))
assert len(pose)==8*101*10
with gzip.open(S/'zero16.replay.json.gz','rt') as f:replay=json.load(f)['payload']
frames={f['step']:f for f in replay['frames'] if 100<=f['step']<=200};assert replay['environment']==1 and len(frames)==101
maxpos=maxrot=maxq=0.
for t,f in frames.items():
 d=support[1,t+1];maxq=max(maxq,max(abs(a-b) for a,b in zip(f['q'],d['q'])))
 for name in replay['body_names']:
  if (1,t+1,name) not in pose:continue
  z=pose[1,t+1,name];i=replay['body_names'].index(name)*13;v=f['body_states'][i:i+13];maxpos=max(maxpos,math.sqrt(sum((a-b)**2 for a,b in zip(z[:3],v[:3]))));maxrot=max(maxrot,min(math.sqrt(sum((a-b)**2 for a,b in zip(z[3:7],v[3:7]))),math.sqrt(sum((a+b)**2 for a,b in zip(z[3:7],v[3:7])))))
assert maxq<1e-6 and maxpos<1e-5 and maxrot<1e-5,(maxq,maxpos,maxrot)
def longest(x):
 best=cur=0
 for y in x:cur=cur+1 if y else 0;best=max(best,cur)
 return best
def bouts(x):
 z=[];a=None
 for i,on in enumerate(x+[False]):
  if on and a is None:a=i
  if not on and a is not None:
   if i-a>=3:z.append([100+a,99+i,i-a])
   a=None
 return z
rows=[]
for lane in range(8):
 row={'lane':lane,'forward_displacement_m':support[lane,201]['q'][0]-support[lane,101]['q'][0]}
 for side,ind in (('left',0),('right',1)):
  flags=[]
  for t in range(101,202):
   x=[sub[lane,t,k] for k in range(16)];imp=[sum(z[i] for z in x) for i in range(7)];gaps=[]
   for name,h in [('crow_'+side+'_foot',(.010,.01725,.0092))]+[('crow_'+side+'_'+n,(.018,.0025,.0092)) for n in ('front_toe_1','front_toe_2','front_toe_3','rear_toe')]:
    z=pose[lane,t,name];x,y,q,w=z[3:7];rr=(2*(x*q-y*w),2*(y*q+x*w),1-2*(x*x+y*y));gaps.append(z[2]-sum(abs(a)*b for a,b in zip(rr,h)))
   flags.append(min(gaps)>.001 and imp[2+ind]<=1e-7 and imp[1-ind]>=.5*.45004*9.81*.02 and imp[6]<=1e-7)
  row[side]={'qualified_samples':sum(flags),'longest_consecutive':longest(flags),'bouts_3_or_more':bouts(flags)}
 rows.append(row)
out={'source_stderr_sha256':hashlib.sha256((S/'zero16.stderr.gz').read_bytes()).hexdigest(),'source_replay_sha256':hashlib.sha256((S/'zero16.replay.json.gz').read_bytes()).hexdigest(),'max_q_mismatch':maxq,'max_position_mismatch_m':maxpos,'max_quaternion_mismatch':maxrot,'lanes':rows,'scope':'accepted-pose geometry plus all-substep contact, two seconds, development starts; no within-step clearance, slip, landing or full-route qualification'};(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
