"""Exercise the new host for one step or until policy-contract rejection."""
import json, os, subprocess
from pathlib import Path
r=Path('/Users/n/Developer/numi-crow-rl-20261107')
a=r/'artifacts/crow-v15-course-collision-restored-20261114'
b=r/'build/crow-v15-course-collision-restored-20261114'
source=r/'artifacts/crow-v15-uniform-handoff-second-seed-20261114/push-20261127-rebound-uniform/command.json'
cmd=json.loads(source.read_text())
for key,value in [('--steps','1'),('--envs','1'),('--state-trace-environment','0'),('--state-trace',str(b/'smoke.trace.tsv')),('--crow-replay-pack',str(b/'smoke.replay.json'))]:
    cmd[cmd.index(key)+1]=value
env=dict(os.environ, DYLD_LIBRARY_PATH=str(b), CROW_AERO_WRENCH_TRACE='1', CROW_SUBSTEP_TRACE='1', CROW_CONTACT_TRACE='1', CROW_SUPPORT_TRACE='1', CROW_EARLY_FLIGHT_TRACE='1', CROW_EXPERIMENT_GAIT_SMOOTH='1')
(a/'smoke-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
p=subprocess.run(cmd,cwd=r,env=env,capture_output=True,text=True)
(a/'smoke.stdout').write_text(p.stdout)
(a/'smoke.stderr').write_text(p.stderr)
lines=[line for line in p.stderr.splitlines() if line.startswith('CROW_SCENE_MASK_RESTORED ')]
(a/'smoke-result.json').write_text(json.dumps({'exit_code':p.returncode,'mask_lines':lines,'scope':'Manifest and policy-contract smoke only; no physical contact test.'},indent=2)+'\n')
print('exit',p.returncode,'restored mask lines',len(lines))
print('\n'.join(lines))
print('\n'.join(p.stderr.splitlines()[-8:]))
assert len(lines)==5
