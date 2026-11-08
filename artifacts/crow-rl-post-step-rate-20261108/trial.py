"""Execute the sealed fixed-budget matched-rate trial, with no tuning/retries."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time

root=Path.cwd(); source=root/'artifacts/crow-rl-post-step-rate-20261108'
plan=json.loads((source/'plan.json').read_text())
out=root/'.numi/runs/american-crow/post-step-rate-01'
def write(p,x):
    with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['sealed_revision'] if 'sealed_revision'in plan else True
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
assert sha(out/'calibration/before.policypack')==plan['initial_policy_sha256']
assert sha(root/plan['protected_deployment'])==plan['protected_deployment_sha256']
processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True)
assert not any(x.rsplit('/',1)[-1]in('metalrobo_task_train','metalrobo_task_rollout','metalrobo_crow_memory_causal_check')for x in processes.splitlines())
write(out/'worker-admission.json',{'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'free_bytes':shutil.disk_usage(root).free,'processes':processes,'plan_sha256':sha(source/'plan.json')})
assert shutil.disk_usage(root).free>=140*1024**3
env={k:v for k,v in os.environ.items()if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
original=json.loads((out/'calibration/command.json').read_text())
remove={'--resume-training-checkpoint','--updated-policy-pack','--deployment-policy-pack','--incumbent-policy-pack',
        '--rollout-pack','--output-learner-state','--checkpoint-directory','--checkpoint-interval'}
base=[];i=0
while i<len(original):
    if original[i]in remove:i+=2;continue
    if original[i]=='--fixed-learning-rate':i+=1;continue
    base.append(original[i]);i+=1
def replace(args,key,val):args[args.index(key)+1]=str(val)
def execute(directory,command,timeout):
    write(directory/'command.json',command);started=time.monotonic()
    with (directory/'stdout').open('x')as stdout,(directory/'stderr').open('x')as stderr:
        run=subprocess.run(command,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=timeout)
    write(directory/'receipt.json',{'returncode':run.returncode,'wall_seconds':time.monotonic()-started})
    if run.returncode:raise RuntimeError('Trial execution failed: '+str(directory))
    return json.loads((directory/'stdout').read_text())
for arm in plan['training_order']:
    directory=out/arm;directory.mkdir()
    command=list(base)
    for k,v in {'--updates':plan['updates_per_arm'],'--seed':plan['training_seed'],
                '--learner-seed':plan['learner_seed']}.items():replace(command,k,v)
    command+=['--policy-pack',str(out/'calibration/before.policypack'),
              '--adaptive-kl-mode','post-step-decision-groups',
              '--target-kl',str(plan['target_kl']),'--minimum-learning-rate',str(plan['minimum_learning_rate']),
              '--maximum-learning-rate',str(plan['maximum_learning_rate']),
              '--updated-policy-pack',str(directory/'updated.policypack'),
              '--deployment-policy-pack',str(directory/'deployment.policypack'),
              '--incumbent-policy-pack',str(directory/'incumbent.policypack'),
              '--rollout-pack',str(directory/'training.rolloutpack'),
              '--output-learner-state',str(directory/'learner.safetensors'),
              '--checkpoint-directory',str(directory/'checkpoints'),'--checkpoint-interval','31']
    if arm=='fixed':command+=['--fixed-learning-rate']
    result=execute(directory,command,plan['training_timeout_seconds'])
    expected={'initial_policy_revision':65,'final_policy_revision':97,'training_samples':655360,
              'failed_environment_steps':0,'learner_state_restored':False,
              'birdflow_journey_variant':'v13-retinal-feeding','birdflow_journey_teacher':False}
    for key,value in expected.items():assert result.get(key)==value,(arm,key,result.get(key),value)
    assert len(result['learning_update_history'])==32
    write(directory/'summary.json',result)
    print(json.dumps({'stage':'training_complete','arm':arm,'samples':655360,
                      'final_revision':97,'tps':result.get('training_transitions_per_second')}),flush=True)
held=out/'heldout';held.mkdir()
policies={'initial':out/'calibration/before.policypack',
          'fixed':out/'fixed/updated.policypack','adaptive':out/'adaptive/updated.policypack'}
for block,order in enumerate(plan['evaluation_order']):
    for arm in order:
        directory=held/(arm+'-'+str(block));directory.mkdir()
        seed=plan['heldout_seed_bases'][block]
        command=['tools/numi','crow-env','memory-causal-admission','--policy',str(policies[arm]),
                 '--seed-base',str(seed),'--output',str(directory/'evaluation')]
        # Numi wrapper stdout is JSON; its native diagnostic is stored inside evaluation.
        execute(directory,command,plan['evaluation_timeout_seconds'])
        summary=json.loads((directory/'evaluation/native-evidence/summary.json').read_text())
        assert summary['status']=='passed_engineering_controls'and summary['stochastic']is True
        assert summary['policy_sha256']==sha(policies[arm])
        assert summary['accepted_transitions']==245760
        print(json.dumps({'stage':'heldout_complete','arm':arm,'block':block,
                          'choice':summary['correct_head_side'],'intake':summary['intake_lifetimes']}),flush=True)
assert sha(root/plan['protected_deployment'])==plan['protected_deployment_sha256']
write(out/'trial-complete.json',{'status':'executed_fixed_budget','training_transitions':1310720,
      'evaluation_transitions':2211840,'new_policy_promotions':0})
