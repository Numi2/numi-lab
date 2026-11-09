from pathlib import Path
import gzip,hashlib,json,os,shutil,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-late-landing-training-20261109';B=R/'build/crow-late-landing-training-20261109';lib=R/'build/lib/libmetalrobo.dylib';epoch=os.environ['CROW_EPOCH']
srcpath=R/f'build/crow-late-landing-training-20261109/checkpoint-{epoch}.policypack';refpath=R/'build/crow-four-expert-native-integration-20261109/landing.policypack'
src=read_policy_pack(srcpath,library_path=str(lib));ref=read_policy_pack(refpath,library_path=str(lib))
for k in ['contract_version','world_fingerprint','observation_fingerprint','action_fingerprint','observation_clip','action_clip']:assert getattr(src,k)==getattr(ref,k),k
for k in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:assert np.array_equal(getattr(src,k),getattr(ref,k)),k
layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers]
path=B/f'landing-{epoch}.policypack'
write_policy_pack(path,policy_id=f'crow_third_landing_{epoch}',revision=src.revision+1,contract_version=ref.contract_version,world_fingerprint=ref.world_fingerprint,task_fingerprint=ref.task_fingerprint,observation_fingerprint=ref.observation_fingerprint,action_fingerprint=ref.action_fingerprint,layers=layers,observation_mean=ref.observation_mean,observation_inverse_standard_deviation=ref.observation_inverse_standard_deviation,action_bias=ref.action_bias,action_scale=ref.action_scale,observation_clip=ref.observation_clip,action_clip=ref.action_clip,library_path=str(lib))
p=read_policy_pack(path,library_path=str(lib));assert not p.critic_layers and not len(p.action_log_standard_deviation)
x=np.random.default_rng(72).normal(size=(64,816)).astype(np.float32);error=float(np.max(np.abs(src.actor_mean(x)-p.actor_mean(x))));assert error<2e-5,error
with path.open('rb') as f,(A/f'landing-{epoch}.policypack.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert hashlib.sha256(gzip.decompress((A/f'landing-{epoch}.policypack.gz').read_bytes())).hexdigest()==sha(path)
(A/f'landing-{epoch}.json').write_text(json.dumps({'source':str(srcpath),'source_sha256':sha(srcpath),'candidate_sha256':sha(path),'max_actor_error':error,'task_fingerprint':ref.task_fingerprint},indent=2)+'\n')
print('landing',epoch,'error',error,flush=True)
