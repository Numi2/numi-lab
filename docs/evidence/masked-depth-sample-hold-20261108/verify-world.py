from array import array
from pathlib import Path
import hashlib,json,math,re,sys
root=Path(__file__).resolve().parent
p=Path(sys.argv[1]) if len(sys.argv)>1 else root/"captures"
suffixes=("q","v","scene","status","reset","history","actor","inputs","policy","actions","observation")
hashes={}
for suffix in suffixes:
 a=(p/("run1."+suffix)).read_bytes();b=(p/("run2."+suffix)).read_bytes()
 assert a==b,(suffix,"repeat mismatch")
 hashes[suffix]=hashlib.sha256(a).hexdigest()
def read(s,t):
 a=array(t);a.frombytes((p/("run1."+s)).read_bytes());return a
reset=read("reset","I");inputs=read("inputs","f");policy=read("policy","f");actions=read("actions","f");obs=read("observation","f")
actor=read("actor","f")
pix=19200;stride=38526;envs=2;nact=9
assert len(inputs)==len(actor)==30*envs*stride
assert len(obs)==30*3*envs*stride and len(actions)==30*envs*nact
log=(root/"run1.log").read_text()
trans={int(i):int(e) for i,e in re.findall(r"transition\[(\d+)\] done=\d+ timeout=\d+ physics_error=(\d+)",log)}
assert len(trans)==60
counts={"accepted":0,"rejected_reset":0,"rejected_no_reset":0,"accepted_changed":0,"rejected_changed":0,"mixed_steps":0,"positive_feedback":0,"nonsampled_reset_with_positive_neighbor":0}
maxerr=0.; details=[]
def row(step,phase,env):
 i=((step*3+phase)*envs+env)*stride
 return obs[i:i+stride]
track=set(j*34+k for j in range(3) for k in range(27,34))
for step in range(30):
 counts["mixed_steps"]+=int(trans[step*2]!=trans[step*2+1])
 for env in range(envs):
  i=step*2+env
  a,b,c=(row(step,phase,env) for phase in range(3))
  inp=inputs[i*stride:(i+1)*stride]
  assert b==inp,(step,env,"input differs from tracker")
  assert c==actor[i*stride:(i+1)*stride],(step,env,"output differs from actor")
  depth=inp[102:102+pix]
  assert all(math.isfinite(v) and 0<=v<=1 for v in depth)
  ref=sum(1-v for v in depth)/pix
  signal=policy[2*i]; action=actions[i*nact]
  maxerr=max(maxerr,abs(ref-signal))
  assert abs(ref-signal)<1e-6 and abs(policy[2*i+1]-.01*signal)<1e-8 and action==policy[2*i+1]
  assert all(v==0 for v in actions[i*nact+1:(i+1)*nact])
  counts["positive_feedback"]+=int(signal>0)
  if not trans[i]:
   assert c==b,(step,env,"accepted output");counts["accepted"]+=1
   counts["accepted_changed"]+=int(a!=b)
  elif not reset[i]:
   assert c==a,(step,env,"rejected output");counts["rejected_no_reset"]+=1
   counts["rejected_changed"]+=int(a!=b)
  else:
   assert all(c[k]==(0. if k in track else a[k]) for k in range(102)),(step,env,"reset tracks")
   assert all(v==1. for v in c[102:102+2*pix]),(step,env,"reset depth")
   assert all(v==0. for v in c[102+2*pix:]),(step,env,"reset features")
   counts["rejected_reset"]+=1
  if reset[i] and step%5!=0:
   assert all(v==0. for k,v in enumerate(inp[:102]) if k in track)
   assert all(v==1. for v in inp[102:102+2*pix])
   assert all(v==0. for v in inp[102+2*pix:])
   assert signal==0 and action==0
   other=policy[2*(step*2+1-env)]
   if other>0:counts["nonsampled_reset_with_positive_neighbor"]+=1
   details.append({"step":step,"env":env,"failed":bool(trans[i]),"neighbor_signal":other})
assert all(counts[k]>0 for k in counts),counts
result={"counts":counts,"nonsampled_resets":details,"max_cpu_gpu_depth_signal_difference":maxerr,"repeat_sha256":hashes}
(root/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
