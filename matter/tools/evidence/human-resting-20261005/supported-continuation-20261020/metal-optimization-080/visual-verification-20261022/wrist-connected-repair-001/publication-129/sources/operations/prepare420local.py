from pathlib import Path
import json,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/postfit-backtrack-420/run-001';P.mkdir(parents=True,exist_ok=False)
cfg=json.loads((D/'D/postfit-backtrack-417/run-001/backtrack-config.json').read_text())
helper=D/'D/local-witness-backtracking-183/final-a644/local_witness_backtracking.py';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(helper)=='a644de24e2f23e90807eb75578309618d89c1a3f9d95d4f6a6f045f4d935c31a'
print('old helper',[(k,v)for k,v in cfg.items()if 'helper' in k or 'uniform' in k])
cfg.update(helper=str(helper),helper_entry='reduce_witness_node_scales',helper_sha256=sha(helper),uniform_failure_backtracking=False,out=str(P/'backtrack-373'))
(P/'backtrack-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(P/'preflight.json').write_text(json.dumps(dict(status='EXPLICIT_LOCAL_WITNESS_HELPER_SELECTED',helper_sha256=sha(helper),helper_entry=cfg['helper_entry'],uniform_failure_backtracking=False,configuration_sha256=sha(P/'backtrack-config.json'),native_admission=False,accepted_physical_steps=0),indent=2)+'\n')

