from pathlib import Path
import gzip,hashlib,json,os,re,subprocess,threading
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-toe-damping-narrow-screen-20261111';B=R/'build/crow-v15-two-segment-toe-damping-narrow-screen-20261111';P=R/'artifacts/crow-v15-two-segment-gait-uniform-adaptive-20261111';L=R/'build/crow-v15-two-segment-walk-frozen-preflight-20261111';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('d006','d008'):
 D=B/name;O=A/name;lib=D/'libmetalrobo.dylib';cmd=json.loads((P/'command.json').read_text());cmd[cmd.index('--metallib')+1]=str(R/'build/crow-v15-two-segment-gait-uniform-adaptive-20261111/tracking.metallib');cmd[cmd.index('--state-trace')+1]=str(D/'trace.tsv');cmd[cmd.index('--crow-replay-pack')+1]=str(D/'replay.json');env={**os.environ,'DYLD_LIBRARY_PATH':str(D),'CROW_SUBSTEP_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 probe=cmd.copy();probe[probe.index('--steps')+1]='1';p=subprocess.run(probe,cwd=R,env=env,capture_output=True,text=True);(O/'fingerprint-preflight.stderr').write_text(p.stderr);(O/'fingerprint-preflight.stdout').write_text(p.stdout);m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
 if not m:raise RuntimeError(name+' expected new task fingerprints: '+p.stderr[-1500:])
 world,task,obs,act=map(int,m.groups());reports=[]
 for option,label in [('--policy-pack','landing'),('--base-policy-pack','route')]:
  source=Path(cmd[cmd.index(option)+1]);src=read_policy_pack(source,library_path=str(L/'libmetalrobo.dylib'));assert not src.critic_layers and len(src.action_log_standard_deviation)==0;layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];target=D/(label+'.policypack');write_policy_pack(target,policy_id='crow_toe_damping_'+name+'_'+label,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib));dst=read_policy_pack(target,library_path=str(lib));assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers));assert all(np.array_equal(getattr(src,k),getattr(dst,k)) for k in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'));assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip;cmd[cmd.index(option)+1]=str(target);reports.append({'actor':label,'source_sha256':h(source),'target_sha256':h(target),'identical_layers_and_transforms':True})
 (O/'policy-binding.json').write_text(json.dumps(reports,indent=2)+'\n');probe=cmd.copy();probe[probe.index('--steps')+1]='2';p=subprocess.run(probe,cwd=R,env=env,capture_output=True,text=True,timeout=180);(O/'rebound-preflight.stdout').write_text(p.stdout);(O/'rebound-preflight.stderr').write_text(p.stderr)
 if p.returncode:raise RuntimeError(name+' rebound preflight failed: '+p.stderr[-1500:])
 (O/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 def pump(src,path,zipped):
  with path.open('wb') as f:
   if zipped:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
     for b in iter(lambda:src.read(1<<16),b''):z.write(b)
   else:
    for b in iter(lambda:src.read(1<<16),b''):f.write(b)
 ts=[threading.Thread(target=pump,args=(p.stdout,O/'stdout',False)),threading.Thread(target=pump,args=(p.stderr,O/'stderr.gz',True))]
 for t in ts:t.start()
 rc=p.wait()
 for t in ts:t.join()
 for src,dst in [(D/'trace.tsv',O/'trace.tsv.gz'),(D/'replay.json',O/'replay.json.gz')]:
  if src.exists():
   with src.open('rb') as f,dst.open('wb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
     for b in iter(lambda:f.read(1<<16),b''):z.write(b)
 (O/'run-result.json').write_text(json.dumps({'exit':rc},indent=2)+'\n');print(name,'exit',rc,flush=True)
