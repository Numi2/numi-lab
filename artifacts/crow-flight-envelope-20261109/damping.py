from pathlib import Path
import gzip,hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-flight-envelope-20261109';B=R/'build/crow-flight-envelope-20261109'
lib=B/'damping-lib/libmetalrobo.dylib'
old=R/'build/crow-envelope-restore-policy/build/crow-flight-source-diagnostics-01/versioned-flight-bench.policypack'
src=read_policy_pack(old,library_path=str(lib))
def layers(xs):return [PolicyDenseLayerArtifact(weights=x.weights,bias=x.bias,activation=x.activation) for x in xs]
for value in ['0.15','0.05']:
 env={**os.environ,'CROW_ROOT_DAMPING_DIAGNOSTIC':value,'DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1','CROW_CONTACT_TRACE':'1'}
 probe=json.loads((A/'yaw-negative-analysis.json').read_text())['argv'].copy()
 probe[probe.index('--steps')+1]='1'
 p=subprocess.run(probe,cwd=R,env=env,text=True,capture_output=True,timeout=120)
 m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
 if not p.returncode or not m:raise RuntimeError('original pack was not rejected: '+p.stderr[-4000:])
 world,task,obs,act=map(int,m.groups())
 new=B/('damping-'+value+'.policypack')
 write_policy_pack(new,policy_id='crow_v15_diagnostic_damping_ablation',revision=1,contract_version=src.contract_version,
 world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,
 layers=layers(src.layers),critic_layers=layers(src.critic_layers),observation_mean=src.observation_mean,
 observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,critic_observation_mean=src.critic_observation_mean,
 critic_observation_inverse_standard_deviation=src.critic_observation_inverse_standard_deviation,
 action_log_standard_deviation=src.action_log_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,
 observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
 dest=read_policy_pack(new,library_path=str(lib))
 same=all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) for x,y in zip(src.layers,dest.layers))
 if not same:raise RuntimeError('actor tensors changed')
 rec={'diagnostic_damping':value,'original_policy_rejected':True,'actor_weights_identical':same,'world':world,'task':task,'observation':obs,'action':act,'old_policy_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'rebound_policy_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'rejection':p.stderr[-2000:]}
 (A/('damping-'+value+'-contract.json')).write_text(json.dumps(rec,indent=2)+'\n')
 for action in ['yaw-negative','yaw-zero']:
  name='damping-'+value+'-'+action
  cmd=json.loads((A/(action+'-analysis.json')).read_text())['argv'].copy()
  cmd[cmd.index('--policy-pack')+1]=str(new)
  cmd[cmd.index('--state-trace')+1]=str(B/(name+'-lane0.tsv'))
  run=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180)
  (A/(name+'.stdout')).write_text(run.stdout)
  with (A/(name+'.stderr.gz')).open('wb') as f:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(run.stderr.encode())
  if run.returncode:raise RuntimeError(run.stderr[-4000:])
  data=json.loads(run.stdout)
  rec={'name':name,'argv':cmd,'diagnostic_damping':value,'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'first_episode':data['first_episode_navigation'],'failed_environment_steps':data['failed_environment_steps']}
  (A/(name+'-results.json')).write_text(json.dumps(rec,indent=2)+'\n')
  print(name,'physical_failures',data['first_episode_navigation']['physical_failure_count'],'failed_steps',data['failed_environment_steps'],flush=True)
