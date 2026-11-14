import json,os,subprocess,hashlib
from pathlib import Path
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114';run=a/'smoke-reset-audit';run.mkdir(exist_ok=False)
src=r/'artifacts/crow-v15-uniform-handoff-second-seed-20261114/push-20261127-rebound-uniform/command.json'
cmd=json.loads(src.read_text())
for k,v in [('--steps','1'),('--envs','1'),('--state-trace-environment','0'),('--state-trace',str(run/'trace.tsv')),('--crow-replay-pack',str(run/'replay.json')),('--policy-pack',str(b/'landing-rebound.policypack')),('--base-policy-pack',str(b/'route-rebound.policypack')),('--metallib',str(b/'restart.metallib'))]:cmd[cmd.index(k)+1]=v
inp=Path('/tmp/crow-48s-multiview-20261114/restart-input.txt');env=dict(os.environ,DYLD_LIBRARY_PATH=str(b/'reset-observer'),CROW_RECORDED_STATE_RESTART=str(inp),CROW_RESTART_RESET_AUDIT='1',CROW_AERO_WRENCH_TRACE='1',CROW_SUBSTEP_TRACE='1',CROW_CONTACT_TRACE='1',CROW_SUPPORT_TRACE='1',CROW_EARLY_FLIGHT_TRACE='1',CROW_EXPERIMENT_GAIT_SMOOTH='1')
(run/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
(run/'inputs.json').write_text(json.dumps({'restart_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'host_sha256':hashlib.sha256((b/'reset-observer/libmetalrobo.dylib').read_bytes()).hexdigest(),'scope':'one-step reconstructed-state diagnostic; not a full native checkpoint'},indent=2)+'\n')
p=subprocess.run(cmd,cwd=r,env=env,capture_output=True,text=True)
(run/'stdout').write_text(p.stdout);(run/'stderr').write_text(p.stderr)
(run/'result.json').write_text(json.dumps({'exit_code':p.returncode},indent=2)+'\n')
print('exit',p.returncode);print('\n'.join(line for line in p.stderr.splitlines() if line.startswith(('CROW_RECORDED','CROW_SCENE_MASK_RESTORED','CROW_RESET_AUDIT'))));print('\n'.join(p.stderr.splitlines()[-4:]))
