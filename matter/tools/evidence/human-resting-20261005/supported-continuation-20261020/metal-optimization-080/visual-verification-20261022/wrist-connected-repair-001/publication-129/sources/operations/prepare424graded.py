from pathlib import Path
import json,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/postfit-backtrack-424/run-001';P.mkdir(parents=True,exist_ok=False)
cfg=json.loads((D/'D/postfit-backtrack-417/run-001/backtrack-config.json').read_text())
helper=D/'D/graded-local-step-423/graded_local_step_423.py';ctx=D/'D/graded-local-step-context-423/run-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(helper)=='dbe0516dff8cc88593f904fef04dd824606ad77f3a822590f91204af932d89d6'
cfg.update(helper=str(helper),helper_entry='reduce_witness_node_scales',helper_sha256=sha(helper),uniform_failure_backtracking=False,out=str(P/'backtrack-373'),local_step_context_npz=str(ctx/'context.npz'),local_step_context_sha256=sha(ctx/'context.npz'),local_step_context_report=str(ctx/'report.json'),local_step_context_report_sha256=sha(ctx/'report.json'),local_step_context_kappa=.1)
(P/'backtrack-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps(dict(config=str(P/'backtrack-config.json'),sha256=sha(P/'backtrack-config.json'))))

