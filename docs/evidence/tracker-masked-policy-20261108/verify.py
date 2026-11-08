from array import array
from pathlib import Path
import hashlib, math, re
p=Path(__file__).resolve().parent
def read(prefix,s,t):
 a=array(t);a.frombytes((p/(prefix+'.'+s)).read_bytes()); return a
for suffix in ('q','v','scene','status','reset','history','actor','inputs','policy','actions'):
 a=(p/('run3.'+suffix)).read_bytes(); b=(p/('run4.'+suffix)).read_bytes()
 assert a==b, suffix+' nondeterministic'
 print('repeat',suffix,'identical sha256',hashlib.sha256(a).hexdigest())
 if suffix!='actions': assert a==(p/('run1.'+suffix)).read_bytes(),suffix+' changed with action capture'
reset=read('run3','reset','I'); x=read('run3','inputs','f'); policy=read('run3','policy','f'); actions=read('run3','actions','f')
pix=19200; stride=38526; envs=2; nact=9
assert len(x)==30*envs*stride and len(actions)==30*envs*nact
assert (p/'run3.actor').read_bytes()==(p/'run3.inputs').read_bytes()
log=(p/'run3.log').read_text()
assert 'published=1 success_steps=55 failure_steps=5' in log
trans={int(i):int(e) for i,e in re.findall(r'transition\[(\d+)\] done=\d+ timeout=\d+ physics_error=(\d+)',log)}
assert len(trans)==60
maxerr=0.0; positives=0; resets=[]
for step in range(30):
 for env in range(envs):
  index=step*envs+env; base=index*stride
  depth=x[base+102:base+102+pix]
  assert all(math.isfinite(v) and 0<=v<=1 for v in depth)
  ref=sum(1-v for v in depth)/pix
  val=policy[index*2]; trace_action=policy[index*2+1]; action=actions[index*nact]
  maxerr=max(maxerr,abs(ref-val)); positives += val>0
  assert abs(val-ref)<1e-6
  assert abs(trace_action-0.01*val)<1e-8 and action==trace_action
  assert all(a==0 for a in actions[index*nact+1:(index+1)*nact])
  if reset[index]:
   tracks=[v for j in range(3) for v in x[base+j*34+27:base+j*34+34]]
   depths=x[base+102:base+102+2*pix]; features=x[base+102+2*pix:base+stride]
   print('reset',step,env,'sample',step%5==0,'signal',val,'action',action,'depth_nonfar',sum(v!=1 for v in depths),'feature_nonzero',sum(v!=0 for v in features),'track_nonzero',sum(v!=0 for v in tracks),'other_signal',policy[(step*2+1-env)*2],'physics_error',trans[index])
   if step in (14,27,28):
    assert all(v==1 for v in depths) and all(v==0 for v in features+array('f',tracks))
    assert val==0 and action==0
   resets.append((step,env))
assert resets==[(0,0),(0,1),(14,0),(15,0),(27,0),(28,0),(28,1)]
assert policy[(13*2)*2]>0 and policy[(26*2)*2]>0
assert policy[(14*2+1)*2]>0 and policy[(27*2+1)*2]>0
assert policy[(15*2)*2]>0
assert trans[14*2]==1 and trans[27*2]==1 and trans[28*2]==0 and trans[28*2+1]==0
print('cpu_gpu_max_abs',maxerr,'nonzero_policy_inputs',positives,'PASS')
