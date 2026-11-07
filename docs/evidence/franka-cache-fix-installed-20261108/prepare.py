import hashlib,json,pathlib,subprocess
base=pathlib.Path('/Users/numi/numi-franka-cache-fix-smoke-20261108')
pkg=pathlib.Path('/Users/numi/numi-lab-cache-fix-installed-20261108')
src=pathlib.Path('/Users/numi/numi-lab-autoload-package-source-20261108')
run=base/'workspace/.numi/runs/window-train-smoke'
python=pathlib.Path('/Users/numi/numi-window-integration-venv-20261107/bin/python')
fixture=base/'fixture'
inputs=[pkg/'bin/numi',pkg/'libexec/numi/train',pkg/'bin/metalrobo_task_rollout',pkg/'bin/metalrobo_task_train',pkg/'lib/libmetalrobo.dylib',pkg/'lib/metalrobo/MetalRobo.metallib',pkg/'share/numi/python/metalrobo/policy_selection.py',pkg/'share/numi/python/metalrobo/mlx_policy_worker.py',pkg/'share/numi/REVISION',python.resolve(),src/'apps/NumiPolicyStatus.swift',src/'apps/task_rollout.swift',base/'decision.swift',base/'decision-check',base/'generate.cpp',base/'generate',*sorted(fixture.iterdir())]
def hashes(): return {str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs}
revision=(pkg/'share/numi/REVISION').read_text().strip()
assert revision=='4aee60dc1b607b179dd2bb93278ee11f31a15bcc'
assert not subprocess.check_output(['git','status','--short'],cwd=src,text=True).strip()
assert not subprocess.check_output(['git','diff',revision,'HEAD','--','apps/NumiPolicyStatus.swift','apps/task_rollout.swift'],cwd=src,text=True).strip()
assert not run.exists()
plan={'revision':revision,'sha256':hashes(),'workspace':str(base/'workspace'),'timeout':120,
 'scope':'synthetic non-G1 Franka WorldPack with one short training/selection; shared decision on real output plus copied identity mutations; no UI or physical qualification',
 'argv':[str(pkg/'bin/numi'),'train','--world-pack',str(fixture/'world.mrworld'),'--task-pack',str(fixture/'task.taskpack'),'--robot-actuator-pack',str(fixture/'actuators.actuatorpack'),'--sensor-program-pack',str(fixture/'sensors.sensorpack'),'--reality-program-pack',str(fixture/'reality.realitypack'),'--envs','1','--steps','8','--updates','1','--chunk','1','--update-epochs','1','--minibatch-size','8','--seed','20261107'],
 'environment':{'NUMI_LAB_ROOT':str(pkg),'NUMI_WORKSPACE_ROOT':str(base/'workspace'),'NUMI_RUN_DIR':str(run),'NUMI_MLX_PYTHON':str(python),'NUMI_SELECTION_ENVS':'1','NUMI_SELECTION_SEED':'4081019'}}
(base/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
print(json.dumps({'revision':revision,'inputs':len(inputs),'world':'authored Franka WorldPack','timeout':120},indent=2))
