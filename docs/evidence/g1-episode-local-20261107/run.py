from pathlib import Path
import subprocess

root = Path('/Users/numi/numi-heldout-g1-20261107/episode-local')
build = Path('/Users/numi/numi-g1-policy-origin-build-20261107')
policy = Path('/Users/numi/numi-g1-policy-origin-20261107/artifacts/g1-legs-locomotion-60min-20260805/retained-policy/candidate.policypack')
common = [str(build/'bin/metalrobo_task_rollout'), '--envs','8','--steps','1000',
          '--repeats','1','--chunk','1','--physics-substeps','4',
          '--velocity-iterations','4','--final-velocity-iterations','2',
          '--scene','ground','--task','g1-legs-locomotion','--metallib',
          str(build/'shaders/MetalRobo.metallib'), '--no-scheduled-resets']
arms = [('policy-4081007',4081007,2,['--policy-pack',str(policy)]),
        ('policy-4081008',4081008,0,['--policy-pack',str(policy)]),
        ('zero-4081007',4081007,2,['--zero-actions'])]
for name, seed, env, actor in arms:
    args = common + ['--seed',str(seed),'--state-trace',str(root/(name+'.trace.tsv')),
                     '--state-trace-environment',str(env)] + actor
    (root/(name+'.command.txt')).write_text('\n'.join(args)+'\n')
    with (root/(name+'.stdout')).open('wb') as out, (root/(name+'.stderr')).open('wb') as err:
        try:
            result = subprocess.run(args, stdout=out, stderr=err, timeout=120, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    (root/(name+'.exit')).write_text(str(code)+'\n')
    print(f'{name} exit={code}', flush=True)
