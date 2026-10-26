from pathlib import Path
import json,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/postfit-backtrack-427/run-001';P.mkdir(parents=True,exist_ok=False)
j=json.loads((D/'D/postfit-backtrack-424/run-001/backtrack-config.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
h=D/'D/graded-cut-step-426/graded_cut_step_426.py';j.update(helper=str(h),helper_sha256=sha(h),out=str(P/'backtrack-373'),initial_scales=str(D/'D/postfit-backtrack-424/run-001/backtrack-373/pass-013/node-step-scales.npy'))
(P/'backtrack-config.json').write_text(json.dumps(j,indent=2)+'\n');print(sha(P/'backtrack-config.json'))

