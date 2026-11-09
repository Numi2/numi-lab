from pathlib import Path
import json,os,subprocess
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-dual-expert-native-integration-20261109';B=R/'build/crow-dual-expert-native-integration-20261109';lib=R/'build/lib/libmetalrobo.dylib'
f=json.loads((A/'contract.json').read_text())['new_fingerprints']
src=read_policy_pack(R/'build/crow-stage2-adapter-training-20261109/recovery-imitation.policypack',library_path=str(lib))
def ls(xs):return [PolicyDenseLayerArtifact(weights=x.weights,bias=x.bias,activation=x.activation) for x in xs]
pack=B/'incompatible-stochastic.policypack'
write_policy_pack(pack,policy_id='crow_expert_guard_test',revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=f['world'],task_fingerprint=f['task'],observation_fingerprint=f['observation'],action_fingerprint=f['action'],layers=ls(src.layers),critic_layers=ls(src.critic_layers),observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,critic_observation_mean=src.critic_observation_mean,critic_observation_inverse_standard_deviation=src.critic_observation_inverse_standard_deviation,action_log_standard_deviation=src.action_log_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
cmd=json.loads((R/'artifacts/crow-dual-native-parity-20261109/integrated/reserved-final-c/results.json').read_text())['argv'].copy()
i=cmd.index('--base-policy-pack');del cmd[i:i+2]
for k,v in [('--steps','1'),('--policy-pack',str(pack))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120)
expected='crow waypoint expert execution requires a deterministic actor-only policy'
r={'returncode':p.returncode,'expected':expected,'rejected_with_expected_message':p.returncode!=0 and expected in p.stderr,'stderr':p.stderr,'argv':cmd}
(A/'guard-results.json').write_text(json.dumps(r,indent=2)+'\n')
print('stochastic single policy rejected',r['rejected_with_expected_message'],flush=True)
assert r['rejected_with_expected_message']
