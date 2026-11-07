from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

root = Path('/Users/n/Developer/numi-crow-rl-20261107')
run = root/'.numi/runs/american-crow/20261107-feeding-learning-campaign-01'
processes = subprocess.check_output(['ps','-axo','pid,ppid,args'], text=True)
active = [line for line in processes.splitlines() if '20261107-feeding-learning-campaign-01' in line]
assert not active, active
analyses = [json.loads(path.read_text()) for path in sorted(run.glob('lifetime-*/analysis.json'))]
value = {'schema':'numi.crow-interruption.v1', 'created_utc':datetime.now(timezone.utc).isoformat(),
    'status':'interrupted_invalid_control', 'completed_lifetime_analyses':analyses,
    'reason':'The initial.policypack baseline contains unit stochastic exploration. The registered control required a deterministic zero-output actor. This was detected after eight completed lifetimes; SIGINT stopped the owning Python process during lifetime 09. No native workers remained at audit.',
    'original_plan_sha256':hashlib.sha256((run/'plan.json').read_bytes()).hexdigest(),
    'baseline_invalid_for_registered_control':True,
    'deterministic_screen_after_eight_lifetimes':'zero intake on all three development seeds',
    'prediction_status':'inconclusive: original 32-lifetime budget not completed; invalid baseline',
    'retention':'All artifacts, including incomplete lifetime-09 outputs, remain untouched.',
    'next_test':'Frozen initial and lifetime-08 policies, explicit stochastic/deterministic modes, direct per-lane native resource accounting, prospectively selected unused development seeds; no extra learning.'}
with (run/'interruption.json').open('x') as handle:
    json.dump(value,handle,indent=2);handle.write('\n')
print(json.dumps({'completed_lifetimes':len(analyses),'status':value['status']}))
