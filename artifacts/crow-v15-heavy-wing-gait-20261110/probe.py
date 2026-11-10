from pathlib import Path
import os,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-heavy-wing-gait-20261110';B=R/'build/crow-v15-heavy-wing-gait-20261110';other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air'))
for name in ('push-trigger','control-trigger'):
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(R/'build/crow-v15-gait-coordinated-confirmation-20261110'/(name+'.air')),'-o',str(B/(name+'.metallib'))],check=True)
cmd=json.loads((R/'artifacts/crow-v15-gait-coordinated-confirmation-20261110/38942573-push-trigger-bindings.json').read_text())['argv']
for k,v in [('--metallib',str(B/'push-trigger.metallib')),('--steps','1'),('--envs','1'),('--state-trace',str(B/'probe.tsv')),('--state-trace-environment','0')]:cmd[cmd.index(k)+1]=v
p=subprocess.run(cmd,cwd=R,env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=300);(A/'probe.stdout').write_text(p.stdout);(A/'probe.stderr').write_text(p.stderr);print('exit',p.returncode,'error',p.stderr[-2000:])
