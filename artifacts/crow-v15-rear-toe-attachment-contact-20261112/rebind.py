from pathlib import Path
import gzip,hashlib,json,re,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/"artifacts/crow-v15-rear-toe-attachment-contact-20261112";B=R/"build/crow-v15-rear-toe-attachment-contact-20261112";C=A/"candidate-preflight-20261127";cmd=json.loads((C/"command.json").read_text())
with gzip.open(C/"run.stderr.gz","rt") as f:err=f.read()
pairs=re.search(r"policy=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\) task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)",err);assert pairs,err
p=list(map(int,pairs.groups()));assert p[0]!=p[4] and json.loads((A/"layout-audit.json").read_text())["same_other_observation_action_hashes"];world,task,obs,act=p[4:];reports=[];h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
for option,label in [("--policy-pack","landing"),("--base-policy-pack","route")]:
 source=Path(cmd[cmd.index(option)+1]);src=read_policy_pack(source,library_path=str(B/"libmetalrobo.dylib"));assert not src.critic_layers and len(src.action_log_standard_deviation)==0
 layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];target=B/(label+"-attachment.policypack")
 write_policy_pack(target,policy_id="crow_toe_attachment_diagnostic_"+label,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(B/"libmetalrobo.dylib"))
 dst=read_policy_pack(target,library_path=str(B/"libmetalrobo.dylib"));assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers));assert all(np.array_equal(getattr(src,k),getattr(dst,k)) for k in ("observation_mean","observation_inverse_standard_deviation","action_bias","action_scale"));assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip
 reports.append({"actor":label,"source_sha256":h(source),"target_sha256":h(target),"identical_layers_and_transforms":True})
(A/"policy-binding.json").write_text(json.dumps({"old_fingerprints":p[:4],"new_fingerprints":p[4:],"reports":reports,"scope":"Diagnostic rebinding to changed world/task and observation signatures. World differs; geometry-dependent radii are reported separately; all other diagnostic hash inputs match. Exact weights and transforms are checked. Sensor values, physical behavior and policy validity on the changed body are not assumed."},indent=2)+"\n")
