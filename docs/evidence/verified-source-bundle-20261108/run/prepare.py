import hashlib,json,pathlib,subprocess
base=pathlib.Path('/Users/numi/numi-verified-bundle-smoke-20261108')
pkg=pathlib.Path('/Users/numi/numi-verified-final-offline-20261108/installed')
bundle=pathlib.Path('/Users/numi/numi-verified-final-offline-20261108/extracted')
run=base/'workspace/.numi/runs/window-train-smoke'
python=pathlib.Path('/Users/numi/numi-window-integration-venv-20261107/bin/python')
fixture=base/'fixture'
inputs=[pkg/'bin/numi',pkg/'libexec/numi/train',pkg/'bin/metalrobo_task_rollout',pkg/'bin/metalrobo_task_train',pkg/'lib/libmetalrobo.dylib',pkg/'lib/metalrobo/MetalRobo.metallib',pkg/'share/numi/python/metalrobo/policy_selection.py',pkg/'share/numi/python/metalrobo/mlx_policy_worker.py',pkg/'share/numi/REVISION',python.resolve(),bundle/'BUNDLE-MANIFEST.json',bundle/'GIT-PROOF.json',pathlib.Path('/Users/numi/numi-verified-final-bundle-20261108.sha256'),pkg/'share/numi/SOURCE-PROVENANCE.json',pkg/'share/numi/SOURCE-BUNDLE.tar.gz',base/'decision.swift',base/'decision-check',base/'generate.cpp',*sorted(fixture.iterdir())]
hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs}
assert (pkg/'share/numi/REVISION').read_text().strip()=='verified-source:a182448ec9d987ef9446fea384560d61cb5ed4ee'
assert not (bundle/'source/.git').exists() and not run.exists()
expected_sha=pathlib.Path('/Users/numi/numi-verified-final-bundle-20261108.sha256').read_text().split()[0]
subprocess.run([str(pkg/'libexec/numi/numi_source_provenance.py'),'audit','--prefix',str(pkg),'--expected-revision','a182448ec9d987ef9446fea384560d61cb5ed4ee','--expected-archive-sha256',expected_sha],check=True)
plan={'revision':'verified-source:a182448ec9d987ef9446fea384560d61cb5ed4ee','sha256':hashes,'workspace':str(base/'workspace'),'timeout':120,'scope':'network-denied synthetic non-G1 Franka one-update training/selection; external host MLX environment; no UI or physics qualification',
 'argv':[str(pkg/'bin/numi'),'train','--world-pack',str(fixture/'world.mrworld'),'--task-pack',str(fixture/'task.taskpack'),'--robot-actuator-pack',str(fixture/'actuators.actuatorpack'),'--sensor-program-pack',str(fixture/'sensors.sensorpack'),'--reality-program-pack',str(fixture/'reality.realitypack'),'--envs','1','--steps','8','--updates','1','--chunk','1','--update-epochs','1','--minibatch-size','8','--seed','20261107'],
 'environment':{'NUMI_LAB_ROOT':str(pkg),'NUMI_WORKSPACE_ROOT':str(base/'workspace'),'NUMI_RUN_DIR':str(run),'NUMI_MLX_PYTHON':str(python),'NUMI_SELECTION_ENVS':'1','NUMI_SELECTION_SEED':'4081019'}}
(base/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
print('offline installed plan prepared:',len(inputs),'pinned files')
