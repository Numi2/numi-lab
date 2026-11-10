from pathlib import Path
import gzip,hashlib,json,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-stage-aware-learning-20261110';B=R/'build/crow-v15-stage-aware-learning-20261110';lib=R/'build/crow-v15-long-perch-host-20261110/libmetalrobo.dylib';srcpath=B/'corrective.policypack';refpath=R/'build/crow-v15-long-perch-host-20261110/route.policypack';src=read_policy_pack(srcpath,library_path=str(lib));ref=read_policy_pack(refpath,library_path=str(lib))
for k in ['contract_version','world_fingerprint','task_fingerprint','observation_fingerprint','action_fingerprint','observation_clip','action_clip']:assert getattr(src,k)==getattr(ref,k),k
for k in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:assert np.array_equal(getattr(src,k),getattr(ref,k)),k
layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];out=B/'corrective-actor.policypack'
write_policy_pack(out,policy_id='crow_v15_corrective_deterministic_development',revision=src.revision+1,contract_version=ref.contract_version,world_fingerprint=ref.world_fingerprint,task_fingerprint=ref.task_fingerprint,observation_fingerprint=ref.observation_fingerprint,action_fingerprint=ref.action_fingerprint,layers=layers,observation_mean=ref.observation_mean,observation_inverse_standard_deviation=ref.observation_inverse_standard_deviation,action_bias=ref.action_bias,action_scale=ref.action_scale,observation_clip=ref.observation_clip,action_clip=ref.action_clip,library_path=str(lib))
p=read_policy_pack(out,library_path=str(lib));assert not p.critic_layers and not len(p.action_log_standard_deviation);x=np.random.default_rng(72).normal(size=(16,816)).astype(np.float32);error=float(np.max(np.abs(src.actor_mean(x)-p.actor_mean(x))));assert error<2e-5,error
with out.open('rb') as inp,(A/'corrective-actor.policypack.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:gz.write(inp.read())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'export.json').write_text(json.dumps({'trained_pack_sha256':sha(srcpath),'actor_pack_sha256':sha(out),'max_actor_error':error,'critic_removed':True,'diagnostic':'Training pack rejected by native residual interface; deterministic actor export retains identical actor weights and outputs.'},indent=2)+'\n');print('actor export max error',error)
