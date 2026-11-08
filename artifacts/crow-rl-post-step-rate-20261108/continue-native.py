"""Two engineering continuation controls; primary trial endpoints stay frozen."""
from pathlib import Path
import hashlib,json,os,subprocess,time

r=Path.cwd();out=r/'.numi/runs/american-crow/post-step-rate-01';dest=out/'native-continuation'
dest.mkdir(exist_ok=False)
def write(p,v):
    with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items()if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
results={}
for arm in ('fixed','adaptive'):
    d=dest/arm;d.mkdir();command=json.loads((out/arm/'command.json').read_text())
    i=command.index('--policy-pack');command[i:i+2]=['--resume-training-checkpoint',str(out/arm/'checkpoints/revision-00000096.trainingcheckpoint')]
    for key,value in {'--updates':'1','--checkpoint-interval':'1','--checkpoint-directory':str(d/'checkpoints'),
        '--updated-policy-pack':str(d/'updated.policypack'),'--deployment-policy-pack':str(d/'deployment.policypack'),
        '--incumbent-policy-pack':str(d/'incumbent.policypack'),'--rollout-pack':str(d/'training.rolloutpack'),
        '--output-learner-state':str(d/'learner.safetensors')}.items():command[command.index(key)+1]=value
    write(d/'command.json',command);start=time.monotonic()
    with(d/'stdout').open('x')as stdout,(d/'stderr').open('x')as stderr:
        run=subprocess.run(command,cwd=r,env=env,stdout=stdout,stderr=stderr,timeout=180)
    write(d/'receipt.json',{'returncode':run.returncode,'wall_seconds':time.monotonic()-start})
    assert run.returncode==0,arm
    summary=json.loads((d/'stdout').read_text());assert summary['training_samples']==20480 and summary['failed_environment_steps']==0
    assert summary['initial_policy_revision']==96 and summary['final_policy_revision']==97 and summary['learner_state_restored']is True
    compared={}
    for name in ('updated.policypack','deployment.policypack','training.rolloutpack','learner.safetensors'):
        a,b=sha(out/arm/name),sha(d/name);assert a==b,(arm,name,a,b);compared[name]=a
    final=d/'checkpoints/revision-00000097.trainingcheckpoint'
    assert(final/'COMMITTED').is_file()and(final/'life.native').is_file()
    results[arm]={'status':'four_canonical_files_byte_exact','compared':compared,'physics_failures':0,
        'engineering_replayed_transitions':20480,'final_native_checkpoint':str(final.relative_to(r)),
        'process_local_incumbent_compared':False}
write(dest/'summary.json',{'status':'passed','results':results,
    'scope':'Exact next-update continuation of both original native+learner states on the original runtime. These40960 engineering transitions reproduce trial outcomes; they are not extra independent learning data or new heldout evaluations.'})
print(json.dumps({'status':'passed','canonical_files_exact_each':4,'engineering_replayed_transitions':40960}))
