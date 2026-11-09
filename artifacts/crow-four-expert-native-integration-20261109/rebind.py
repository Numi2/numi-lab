from pathlib import Path
import gzip,hashlib,json,os,re,shutil,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-four-expert-native-integration-20261109';B=R/'build/crow-four-expert-native-integration-20261109';lib=R/'build/lib/libmetalrobo.dylib'
oldroute=R/'build/crow-four-expert-experiment-20261109/stage4-independent-40.policypack';oldlanding=R/'build/crow-dual-expert-native-integration-20261109/landing.policypack'
cmd=json.loads((R/'artifacts/crow-expanded-candidate-eval-20261109/baseline/reserved-final-c/results.json').read_text())['argv'].copy()
for k,v in [('--birdflow-journey-variant','v15-triple-waypoint-expert-navigation'),('--steps','1'),('--metallib',str(R/'build/shaders/MetalRobo.metallib')),('--state-trace',str(B/'contract.tsv'))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120)
(A/'old-contract.stderr').write_text(p.stderr)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
if not p.returncode or not m:raise RuntimeError('Expected new contract rejection: '+p.stderr[-4000:])
world,task,obs,act=map(int,m.groups());reports=[]
for label,source in [('route',oldroute),('landing',oldlanding)]:
 src=read_policy_pack(source,library_path=str(lib))
 assert src.world_fingerprint==world and src.observation_fingerprint==obs and src.action_fingerprint==act,(label,src.world_fingerprint,world,src.observation_fingerprint,obs,src.action_fingerprint,act)
 assert src.task_fingerprint!=task and not src.critic_layers and len(src.action_log_standard_deviation)==0
 layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers]
 out=B/(label+'.policypack')
 write_policy_pack(out,policy_id='crow_triple_waypoint_expert_'+label,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
 dst=read_policy_pack(out,library_path=str(lib))
 assert all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) for x,y in zip(src.layers,dst.layers))
 for name in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:assert np.array_equal(getattr(src,name),getattr(dst,name))
 with out.open('rb') as f,(A/(label+'.policypack.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
 assert hashlib.sha256(gzip.decompress((A/(label+'.policypack.gz')).read_bytes())).hexdigest()==hashlib.sha256(out.read_bytes()).hexdigest()
 reports.append({'policy':label,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rebound_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'old_task_fingerprint':src.task_fingerprint,'actor_weights_and_transforms_unchanged':True})
report={'old_policy_rejected':True,'new_fingerprints':{'world':world,'task':task,'observation':obs,'action':act},'policies':reports,'scope':'explicit contract migration for unchanged deterministic evaluated actors; physical equivalence requires separate test'}
(A/'contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
