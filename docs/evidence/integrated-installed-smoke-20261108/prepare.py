import hashlib,json,pathlib,subprocess
base=pathlib.Path('/Users/numi/numi-integrated-installed-smoke-20261108')
pkg=pathlib.Path('/Users/numi/numi-lab-integrated-install-relocated-20261108')
src=pathlib.Path('/Users/numi/numi-lab-integrated-install-20261108')
scene=base/'workspace/.numi/runs/synthetic/diagnostic.json'
run=base/'workspace/.numi/runs/window-train-smoke'
python=pathlib.Path('/Users/numi/numi-window-integration-venv-20261107/bin/python')
inputs=[pkg/'bin/numi',pkg/'libexec/numi/window',pkg/'libexec/numi/train',pkg/'bin/metalrobo_task_rollout',pkg/'bin/metalrobo_task_train',pkg/'lib/libmetalrobo.dylib',pkg/'lib/metalrobo/MetalRobo.metallib',pkg/'share/numi/python/metalrobo/policy_selection.py',pkg/'share/numi/python/metalrobo/mlx_policy_worker.py',pkg/'share/numi/REVISION',python.resolve(),base/'catalog-check',base/'catalog.swift',scene,scene.with_name('diagnostic.mrvpack'),scene.with_name('synthetic.numi-window.json')]
def hashes(): return {str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs}
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()
assert revision=='be52afa5659a0331d6f5d6c26f7ea10f7366cb65'
assert not subprocess.check_output(['git','status','--short'],cwd=src,text=True).strip()
assert (pkg/'share/numi/REVISION').read_text().strip()==revision
assert not run.exists()
plan={'revision':revision,'sha256':hashes(),'workspace':str(base/'workspace'), 'scope':'synthetic authored scene, one short train update and held-out screen; no live UI or learned-control claim',
'window':{'timeout':60,'argv':[str(pkg/'bin/numi'),'window','--check','--window-scene',str(scene),'--envs','1','--steps','16','--chunk','1','--robot-source','unitree-g1','--scene','ground','--task','velocity','--zero-actions']},
 'train':{'timeout':120,'argv':[str(pkg/'bin/numi'),'train','--envs','1','--steps','8','--updates','1','--chunk','1','--update-epochs','1','--minibatch-size','8','--scene','ground','--task','velocity','--seed','20261107']},
 'environment':{'NUMI_LAB_ROOT':str(pkg),'NUMI_WORKSPACE_ROOT':str(base/'workspace'),'NUMI_RUN_DIR':str(run),'NUMI_MLX_PYTHON':str(python),'NUMI_SELECTION_ENVS':'1','NUMI_SELECTION_SEED':'4081019','NUMI_WINDOW_SCENE_CATALOG':str(base/'workspace/.numi/runs'),'NUMI_WINDOW_POLICY_CATALOG':str(base/'workspace/.numi/policies')}}
(base/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
print(json.dumps({'revision':revision,'inputs':len(inputs),'window_timeout':60,'training_timeout':120},indent=2))
