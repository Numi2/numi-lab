from pathlib import Path
import gzip,json,math,statistics,subprocess,os
A=Path('artifacts/crow-v15-strong-pronation-turn-screen-20261112');B=Path('artifacts/crow-v15-wing-root-moment-observer-20261112')
def load(path):
 f={};w={};err=[]
 with gzip.open(path,'rt') as src:
  for l in src:
   if l.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);f[x['lane'],x['step']]=[x['q'],x['v'],x['policy_action']]
   if l.startswith('CROW_AERO_ERROR'):err.append(l.strip())
   if l.startswith('CROW_WING_WRENCH '):
    z=l.split();t,k,e=map(int,z[1:4]);values=list(map(float,z[4:]));assert (e,t,k) not in w and all(math.isfinite(x) for x in values);w[e,t,k]=values
 return f,w,err
f,w,err=load(A/'bank.stderr.gz');rf,rw,rerr=load(B/'bank.stderr.gz');assert not err and not rerr; pre=[k for k in f if k[1]<450];assert len(pre)==48 and all(f[k]==rf[k] for k in pre);assert len(w)==9600 and all(len(w[k])==103 for k in w)
inp=''.join(str(e)+' '+str(t*16+k)+' '+' '.join(map(str,x[22:]))+'\n' for (e,t,k),x in sorted(w.items()));env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};proc=subprocess.run(['build/crow-v15-wing-root-moment-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env);pos={}
for l in proc.stdout.splitlines():
 z=l.split();pos[int(z[0]),int(z[1])]=list(map(float,z[2:]))
assert len(pos)==9600
with gzip.open(A/'bank-replay.json.gz','rt') as inpfile:replay=json.load(inpfile)['payload']
e=replay['environment'];x=w[e,701,0];matches=[]
for frame in replay['frames']:
 if 695<=frame['step']<=705:
  delta=max(abs(a-b) for a,b in zip(x[22:63],frame['q']));matches.append((delta,frame))
qerr,fr=min(matches,key=lambda v:v[0]);assert qerr<1e-6,qerr
names=replay['body_names'];p=pos[e,701*16];poseerr=0
for i,name in enumerate(('crow_body','crow_left_wing','crow_right_wing')):
 base=names.index(name)*replay['body_state_stride'];poseerr=max(poseerr,math.dist(p[i*3:i*3+3],fr['body_states'][base:base+3]))
assert poseerr<1e-5,poseerr
def cross(r,f):return [r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]]
def moments(e,t,k):
 x=w[e,t,k];p=pos[e,t*16+k];root=p[:3];out=[]
 for i,off in enumerate((4,10)):
  r=[p[3+i*3+j]-root[j] for j in range(3)];lever=cross(r,x[off:off+3]);out.append([lever[j]+x[off+3+j] for j in range(3)])
 out.append(x[19:22]);out.append([sum(z[j] for z in out) for j in range(3)]);return out
def summarize(seq):return {'mean':statistics.mean(seq),'min':min(seq),'max':max(seq),'signed_impulse_1s':sum(seq)*.00125}
rows=[]
for lane in range(6):
 for lo,hi in [(451,500),(701,750)]:
  vals=[moments(lane,t,k) for t in range(lo,hi+1) for k in range(16)]
  rows.append({'lane':lane,'steps':[lo,hi],**{name:{axis:summarize([v[i][j] for v in vals]) for j,axis in enumerate(('x','y','z'))} for i,name in enumerate(('left_about_root','right_about_root','root_kernel','combined_kernel_about_root'))}})
out={'pre_intervention_sample_count':len(pre),'pre_intervention_mismatches':0,'substep_wrench_records':len(w),'replay_validation':{'lane':e,'source_step':701,'source_substep':0,'replay_step':fr['step'],'max_q_error':qerr,'max_body_com_error_m':poseerr},'rows':rows,'scope':'World-axis moment about root body COM of wing and root wrenches added by the flapping kernel. Includes root kernel wrench; omits the separate tail wrench and all other applied wrenches, joint constraint impulses, angular momentum derivative or biological validation. COM pose reconstructed by CPU forward kinematics from source q/v at each substep. One replay state validates frame alignment; further states and model assumptions remain limitations.'};(Path('artifacts/crow-v15-pronation-wrench-analysis-20261112')/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
