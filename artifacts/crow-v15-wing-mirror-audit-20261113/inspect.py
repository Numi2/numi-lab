from pathlib import Path
import json,hashlib,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-wing-mirror-audit-20261113';B=R/'build/crow-v15-wing-mirror-audit-20261113';P=R/'artifacts/crow-v15-position-flight-screen-20261113/position-flight-20261126-rebound-position';D=A/'one-step-20261126';D.mkdir(exist_ok=False);cmd=json.loads((P/'command.json').read_text())
cmd[cmd.index('--steps')+1]='1'
for option in ('--state-trace','--state-trace-environment','--crow-replay-pack'):
 i=cmd.index(option);del cmd[i:i+2]
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_EXPERIMENT_GAIT_SMOOTH':'1'}
for k in ('CROW_AERO_WRENCH_TRACE','CROW_SUBSTEP_TRACE','CROW_CONTACT_TRACE','CROW_SUPPORT_TRACE','CROW_EARLY_FLIGHT_TRACE'):env.pop(k,None)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();inputs={'host_sha256':h(B/'libmetalrobo.dylib'),'metallib_sha256':h(cmd[cmd.index('--metallib')+1]),'landing_policy_sha256':h(cmd[cmd.index('--policy-pack')+1]),'route_policy_sha256':h(cmd[cmd.index('--base-policy-pack')+1])};(D/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');(D/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
r=subprocess.run(cmd,cwd=R,env=env,capture_output=True);(D/'run.stdout').write_bytes(r.stdout);(D/'run.stderr').write_bytes(r.stderr);(D/'run-result.json').write_text(json.dumps({'exit_code':r.returncode},indent=2)+'\n');print('one-step exit',r.returncode);raise SystemExit(r.returncode)
