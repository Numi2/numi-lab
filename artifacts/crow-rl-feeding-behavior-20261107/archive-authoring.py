from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from metalrobo.crow_environment import validate_feeding_accounting, validate_native

root=Path('/Users/n/Developer/numi-crow-rl-20261107')
runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-feeding-behavior-20261107'
archive.mkdir(exist_ok=False)

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

comparison=runs/'20261107-feeding-frozen-policy-comparison-01'
summary=json.loads((comparison/'summary.json').read_text())
assert summary['prediction_supported'] is True and len(summary['records'])==12
for cell in comparison.glob('seed-*/*'):
    seed=int(cell.parent.name[5:])
    data=json.loads((cell/f'seed-{seed}.json').read_text())
    receipt=json.loads((cell/f'seed-{seed}.receipt.json').read_text())
    assert receipt['exit_code']==0 and not receipt['timed_out'] and not receipt['interrupted']
    assert digest(cell/f'seed-{seed}.json')==receipt['stdout_sha256']
    assert digest(cell/f'seed-{seed}.stderr')==receipt['stderr_sha256']
    validate_native(data,envs=512,steps=1600,seed=seed,policy=True,variant='v12-feeding-station')
    validate_feeding_accounting(data,512,1600)
    assert data['host_requested_resets']==0 and data['termination_count']==0

inventory=[]
names=['feeding-update-history-control-01','feeding-learning-campaign-01',
       'feeding-inventory-control-01','feeding-inventory-control-02',
       'feeding-frozen-policy-comparison-01','feeding-accounting-api-admission-01',
       'feeding-policy-mode-rejection-01','feeding-learned-partition-01']
for name in names:
    for path in sorted((runs/('20261107-'+name)).rglob('*')):
        if not path.is_file():continue
        copied=path.stat().st_size<1_000_000 and path.suffix not in ('.policypack','.safetensors','.rolloutpack')
        entry={'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':digest(path),
               'copied_to_source_archive':copied}
        inventory.append(entry)
        if copied:
            target=archive/'trials'/path.relative_to(runs)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            assert digest(target)==entry['sha256']

campaign=runs/'20261107-feeding-learning-campaign-01'
selected={'initial-stochastic.policypack':campaign/'lifetime-01/initial.policypack',
          'initial-deterministic.policypack':campaign/'lifetime-01/incumbent.policypack',
          'trained-stochastic.policypack':campaign/'lifetime-08/updated.policypack',
          'trained-deterministic.policypack':campaign/'lifetime-08/deployment.policypack',
          'trained-learner.safetensors':campaign/'lifetime-08/learner.safetensors'}
(archive/'policies').mkdir()
selected_receipt=[]
for name,source in selected.items():
    target=archive/'policies'/name
    shutil.copyfile(source,target)
    assert digest(source)==digest(target)
    selected_receipt.append({'source':str(source),'archive':str(target.relative_to(root)),
                             'bytes':target.stat().st_size,'sha256':digest(target)})
(archive/'selected-policy-receipt.json').write_text(json.dumps(selected_receipt,indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
shutil.copyfile(root/'build/Testing/Temporary/LastTest.log',archive/'native-accounting-check.log')
shutil.copyfile(Path(__file__),archive/'archive-authoring.py')
shutil.copyfile(root.parent/'numi-crow-rl-20261107-close-campaign.py',archive/'interruption-audit.py')
(archive/'.gitattributes').write_text('*.log whitespace=-blank-at-eof\n')
for name,command in [('preflight',['tools/numi','crow-env','doctor']),
                     ('python-checks',['python/.venv/bin/python','-m','unittest','discover','-s','tests','-p','crow_*test.py'])]:
    result=subprocess.run(command,cwd=root,text=True,capture_output=True)
    (archive/(name+'.stdout')).write_text(result.stdout)
    (archive/(name+'.stderr')).write_text(result.stderr)
    (archive/(name+'.receipt.json')).write_text(json.dumps({'command':command,'exit_code':result.returncode})+'\n')
    assert result.returncode==0

training=[json.loads(path.read_text()) for path in sorted(campaign.glob('lifetime-*/analysis.json'))]
report={'schema':'numi.crow-feeding-behavior-evidence.v1','status':'development-mechanism-supported',
        'comparison':summary,'completed_training_lifetimes':len(training),
        'accepted_training_samples_in_complete_lifetimes':sum(row['samples'] for row in training),
        'whole_process_training_steps_per_second_range':[min(row['whole_process_steps_per_second'] for row in training),
                                                       max(row['whole_process_steps_per_second'] for row in training)],
        'maximum_reported_mlx_peak_bytes':max(row['mlx_peak_memory_bytes'] for row in training),
        'incomplete_lifetime09_excluded_from_complete_training_totals':True,
        'original_32_lifetime_prediction':'inconclusive; interrupted invalid deterministic baseline',
        'additional_learning_for_comparison':0,'full_life_checkpoint_tested':False,
        'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
(archive/'README.md').write_text('''# American crow: frozen learned feeding comparison

macmini M4 Pro, 7 November 2026. This is a native v12 fixed-station feeding
primitive for the lifelong-intelligence project, not a completed ecology.

The learned stochastic policy fully consumed 45 mg in 512/512 lanes on each
of three unused development seeds: 1,536/1,536 total. The initial stochastic
control completed in 7, 7 and 6 lanes: 20/1,536 total. Both deterministic modes
consumed zero food. The four-mode comparison has 6,144 native trajectories and
9,830,400 accepted test transitions, zero physical terminations, resets and
device failures. The preregistered per-seed prediction (trained >=95% complete,
initial stochastic <=30%) was supported. This is one trained learner, not
thousands of independently trained animals.

Every lane retains integer remaining/consumed micrograms, crop payload, bite
counts and accepted clock. Inventory, reward and typed intake outcome agree.
Receipt hashes and native validity were rechecked before this archive copy.
Actor-mode pairs have identical coefficients, normalization and transforms.
Learned variance remains part of the stochastic policy. Different policy
revisions use different RNG keys; equal seeds do not create identical noise.

The parent campaign completed eight finite-world lifetimes, four PPO updates
each: 2,097,152 accepted training transitions and revisions 1 through 33. Each
update uses 512 environments x128 steps, four PPO epochs, minibatch 8,192,
fixed learning rate 0.0001 and initial unit exploration. Resource depletion
persists across all four updates in a lifetime. A new process deliberately
creates an independent new finite world while restoring learner/Adam state.
Whole-process throughput was about 18,700-18,900 transitions/s; MLX peak was
about 458 MB. This excludes no completed lifetime and is not visual ecological
throughput or a full-process memory qualification.

The original 32-lifetime campaign was interrupted after detecting that its
registered deterministic baseline used a stochastic pack. Lifetime 09 is
incomplete and retained, excluded from complete training totals. The original
prediction is inconclusive. The corrected frozen-policy comparison added no
learning. The first inventory export also failed because Swift inferred Void
arrays inside an erased JSON dictionary; typed arrays and a JSON validity
guard corrected it. Both negative attempts remain in trials/.

Native C API controls reject null/wrong-sized outputs, absent resource programs
and uninitialized resident state without changing caller memory. Repeated
ledger publication is exact. Positive, yaw-miss and held-closed controls agree
across chunks 8/64. The frozen learned actor replays exactly across chunks 1/8
for ordinary and residual sampling. The native finite-resource CTest and all
19 Python crow checks passed; synthetic qualification fixtures are not animal
qualification. An actual stochastic pack presented as deterministic is now
rejected before native execution, with a retained negative trial.

policies/ contains hash-verified initial/trained stochastic and deterministic
artifacts and the trained learner checkpoint. payload-inventory.json hashes
all retained larger macmini rollouts and intermediate policy/learner artifacts.
All per-update metrics are retained; native bounded storage retains the final
rollout pack in each lifetime rather than every intermediate rollout.

Scope: estimated supported crow with independent physical head/jaw controls,
one fixed nutritional intake-volume tracer, capacity-limited crop and local
nutrition inputs. Other controls are held by the standing curriculum. No
qualified eye images, search, solid-food pickup/carrying, swallowing, water or
energy physiology, learned memory, full-life restore, or social learning.
The complete ecological environment is unqualified. Original qualification
seeds and layouts remain unused. The working feeding mode requires stochastic
actions; the deterministic actor must not be promoted as a feeding skill.
''')
print(json.dumps({'archive':str(archive),'inventoried_files':len(inventory),
                  'training_samples':report['accepted_training_samples_in_complete_lifetimes'],
                  'policy_bytes':sum(row['bytes'] for row in selected_receipt)}))
