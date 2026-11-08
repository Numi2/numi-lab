"""A bounded native object-learning study; no Python transitions or tuning loop."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .crow_environment import execute, preflight, runtime, sha, validate_native, write
from .mlx_policy_learning import read_policy_pack, read_policy_rollout_pack
from .native import PolicyDenseLayerArtifact, write_policy_pack

GOAL = np.asarray([.19, .25, .257], dtype=np.float32).astype(np.float64)
ARCHIVE = 'artifacts/crow-rl-object-curriculum-20261108'


def frozen_mean(root: Path, source: Path, target: Path, *, head_bias: float | None = None) -> None:
    p = read_policy_pack(source, library_path=runtime(root)['library'])
    layers = [PolicyDenseLayerArtifact(x.weights.copy(), x.bias.copy(), x.activation) for x in p.layers]
    if head_bias is not None:
        layers[-1].bias[15] = head_bias
    write_policy_pack(target, policy_id=p.id+'-frozen-mean', revision=p.revision,
        **{k: getattr(p,k) for k in ('contract_version','world_fingerprint','task_fingerprint',
            'observation_fingerprint','action_fingerprint','observation_mean',
            'observation_inverse_standard_deviation','action_bias','action_scale',
            'observation_clip','action_clip','actor_memory')}, layers=layers,
        library_path=runtime(root)['library'])


def endpoint(output: Path, name: str, n: int, *, trace: bool = False) -> dict:
    """Validate endpoint using independent FP64 distance and typed reward identity."""
    root = Path(__file__).resolve().parents[2]
    v = json.loads((output/(name+'.state.json')).read_text())
    if v['schema'] != 'numi.task-final-state.v1' or v['environment_count'] != n or v['body_state_width'] != 13:
        raise RuntimeError('Invalid final native state schema')
    names = v['body_names']
    if len(names) != v['body_count'] or len(set(names)) != len(names):
        raise RuntimeError('Invalid native body identities')
    bodies = np.asarray(v['body_states'], dtype=np.float64).reshape(n,len(names),13)
    q = np.asarray(v['q'], dtype=np.float64).reshape(n,v['configuration_count'])
    velocity = np.asarray(v['v'], dtype=np.float64).reshape(n,v['velocity_count'])
    if not all(np.isfinite(x).all() for x in (bodies,q,velocity)):
        raise RuntimeError('Invalid final accepted state')
    position = bodies[:,names.index('crow_manipulandum'),:3]
    distance = np.linalg.norm(position-GOAL,axis=1)
    pack = read_policy_rollout_pack(output/(name+'.rolloutpack'), library_path=runtime(root)['library'])
    if pack.environment_count != n or pack.control_step_count != 160:
        raise RuntimeError('Wrong endpoint generating rollout')
    recorded = pack.outcomes['object_goal_score'].reshape(160,n)[-1]/np.float64(np.float32(.02))
    expected = np.exp(-distance**2/np.float64(np.float32(.01)))
    error = float(np.max(np.abs(recorded-expected)))
    if error > 2e-6:
        raise RuntimeError('Final world coordinates disagree with FP64 reward identity')
    if (np.any(np.abs(position[:,2]-.257) > .0005) or np.any(position[:,0] < .16) or
        np.any(position[:,0] > .80) or np.any(np.abs(position[:,1]) > .50)):
        raise RuntimeError('Object left the admitted support envelope')
    if trace:
        t = np.loadtxt(output/(name+'.trace.tsv'),comments='#')
        scene = bodies[0,-3:,:].reshape(-1)
        if t.shape != (160,1+q.shape[1]+scene.size) or not np.allclose(t[-1,1:],np.r_[q[0],scene],rtol=0,atol=1e-7):
            raise RuntimeError('Endpoint disagrees with accepted lane0 q/scene trace')
    return {'position_m':position.tolist(),'distance_m':distance.tolist(),
        'success_within_20mm':(distance<=.02).tolist(),'mean_distance_m':float(distance.mean()),
        'successes':int(np.count_nonzero(distance<=.02)),'reward_FP64_max_error':error}


def evaluation(root: Path, output: Path, name: str, policy: Path, seed: int, n: int,
               chunk: int = 8, *, trace: bool = False) -> dict:
    paths=runtime(root)
    c=['/usr/bin/env','DYLD_PRINT_LIBRARIES=1',str(paths['rollout']),
       '--birdflow-american-crow-journey','--birdflow-journey-variant','v14-object-push',
       '--visual-observation-config',str(root/'environments/american-crow/object-push-v1/retinal-observation.json'),
       '--object-push-curriculum',str(root/'environments/american-crow/object-push-v1/curriculum.json'),
       '--metallib',str(paths['metallib']),'--policy-pack',str(policy),'--envs',str(n),'--steps','160',
       '--chunk',str(chunk),'--seed',str(seed),'--repeats','1','--no-scheduled-resets',
       '--minimum-difficulty-band','0','--maximum-difficulty-band','0',
       '--save-life-checkpoint',str(output/(name+'.life')),
       '--rollout-pack',str(output/(name+'.rolloutpack')),
       '--final-state-json',str(output/(name+'.state.json'))]
    if trace:c += ['--state-trace',str(output/(name+'.trace.tsv'))]
    v=execute(output,name,c,root,180)
    validate_native(v,envs=n,steps=160,seed=seed,policy=True,variant='v14-object-push')
    if v['accepted_lifetime_control_ticks_by_environment'] != [160]*n:
        raise RuntimeError('Evaluation has warmup or incorrect clocks')
    loaded=[s for s in (output/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
    if not loaded or any(not s.endswith(str(paths['library'])) for s in loaded):
        raise RuntimeError('Wrong native evaluation library')
    result=endpoint(output,name,n,trace=trace)
    write(output/(name+'.endpoint.json'),result)
    return result


def calibration(root: Path, output: Path) -> None:
    output.mkdir(parents=True,exist_ok=False)
    write(output/'calibration-plan.json',{
        'question':'Does optional final-state publication preserve native execution and report every accepted endpoint correctly?',
        'preflight':preflight(root,json.loads((root/'profiles/american-crow-rl-v1.json').read_text())),
        'prediction':'33-lane neutral chunks1/8 yield identical final state/rollout bytes; lane0 agrees with trace; all endpoint FP64 goal score errors<=2e-6. Constant head0.6 moves at least one object>1cm compared with neutral.',
        'calibration':'Existing admitted contact fixture and independently checked FP64 accepted-state reward; this tests exporter semantics, not physics calibration or learned behavior.',
        'budget':'Three160-tick33-lane native executions plus one rejected overwrite request;180s each;128MiB raw; no video or retries',
        'seed':2651317001,'learner_initialization_seed':2651317002,
        'initial_policy':'Fresh GRU32/encoder32, zero output, std0.2; no scripted motor bias in learning baseline',
        'source_template':'Read exact task fingerprints from an archived policy; no template weights imported',
    })
    spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    entries=a.tree()
    with a.Objects() as objects:
        metadata=json.loads(objects.read(entries[ARCHIVE+'/runtime-recovery.json'][1]))
        manifest=json.loads(objects.read(entries[ARCHIVE+'/manifest.json'][1]))
        source=metadata['checkpoint']['input/push.policypack']
        record=next(x for x in manifest['files'] if x['path']==source)
        template=output/'contract-template.policypack'
        template.write_bytes(b''.join(a.content(record,entries,objects)))
        if sha(template)!=record['sha256']:raise RuntimeError('Git task template recovery failed')
    p=read_policy_pack(template,library_path=runtime(root)['library'])
    c=[sys.executable,'-m','metalrobo.mlx_policy_worker','initialize',
       '--actor-observations','1247','--critic-observations','100','--actions','18',
       '--policy-id','crow-object-learning-exploration-v1','--output',str(output/'initial.policypack'),
       '--native-library',str(runtime(root)['library']),'--seed','2651317002',
       '--recurrent-hidden-size','32','--hidden-sizes','32','--recurrent-sequence-length','160',
       '--recurrent-memory-timescale-ticks','64','--minibatch-size','20480',
       '--initial-log-standard-deviation',str(math.log(.2)),'--zero-actor-output']
    for name in ('world','task','observation','action'):
        c += ['--'+name+'-fingerprint',str(getattr(p,name+'_fingerprint'))]
    execute(output,'initialize',c,root,60)
    frozen_mean(root,output/'initial.policypack',output/'initial-mean.policypack')
    frozen_mean(root,output/'initial.policypack',output/'scripted-positive.policypack',head_bias=.6)
    neutral1=evaluation(root,output,'neutral1',output/'initial-mean.policypack',2651317001,33,1,trace=True)
    evaluation(root,output,'neutral8',output/'initial-mean.policypack',2651317001,33)
    positive=evaluation(root,output,'positive',output/'scripted-positive.policypack',2651317001,33)
    for suffix in ('state.json','rolloutpack'):
        if sha(output/('neutral1.'+suffix))!=sha(output/('neutral8.'+suffix)):
            raise RuntimeError('Final publication changes native chunk equivalence')
    motion=np.linalg.norm(np.asarray(positive['position_m'])-np.asarray(neutral1['position_m']),axis=1)
    if motion.max()<=.01:raise RuntimeError('Positive control lacked physical movement')
    c=json.loads((output/'neutral1.receipt.json').read_text())['command']
    before=sha(output/'neutral1.state.json')
    p=subprocess.run(c,cwd=root,capture_output=True,text=True,timeout=30)
    (output/'overwrite.stdout').write_text(p.stdout);(output/'overwrite.stderr').write_text(p.stderr)
    if p.returncode==0 or '--final-state-json requires a new file' not in p.stderr or sha(output/'neutral1.state.json')!=before:
        raise RuntimeError('Final-state overwrite guard failed')
    raw=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if raw>128*1024**2:raise RuntimeError('Calibration storage bound exceeded')
    write(output/'calibration-summary.json',{'status':'passed','accepted_transitions':15840,
        'neutral_partition_state_and_rollout_exact':True,'trace_lane0_exact_within_1e-7':True,
        'all_endpoint_FP64_reward_identities_checked':True,'positive_motion_m':motion.tolist(),
        'overwrite_rejected_before_execution':True,'raw_bytes':raw})


def training(root: Path, output: Path) -> None:
    if json.loads((output/'calibration-summary.json').read_text())['status']!='passed':
        raise RuntimeError('Exporter calibration required')
    d=output/'training';d.mkdir(exist_ok=False);paths=runtime(root)
    c=['/usr/bin/env','DYLD_PRINT_LIBRARIES=1',str(paths['train']),
       '--birdflow-american-crow-journey','--birdflow-journey-variant','v14-object-push',
       '--visual-observation-config',str(root/'environments/american-crow/object-push-v1/retinal-observation.json'),
       '--object-push-curriculum',str(root/'environments/american-crow/object-push-v1/curriculum.json'),
       '--envs','128','--steps','160','--updates','16','--chunk','8','--seed','2651317101',
       '--learner-seed','2651317102','--minimum-difficulty-band','0','--maximum-difficulty-band','0',
       '--metallib',str(paths['metallib']),'--native-library',str(paths['library']),
       '--mlx-python',sys.executable,'--python-root',str(root/'python'),
       '--policy-pack',str(output/'initial.policypack'),'--fixed-learning-rate','--learning-rate','0.0003',
       '--discount','0.999','--gae-lambda','1','--update-epochs','2','--minibatch-size','20480',
       '--recurrent-sequence-length','160','--checkpoint-life','--checkpoint-interval','8',
       '--checkpoint-directory',str(d/'checkpoints'),'--updated-policy-pack',str(d/'updated.policypack'),
       '--deployment-policy-pack',str(d/'candidate.policypack'),'--incumbent-policy-pack',str(d/'incumbent.policypack'),
       '--rollout-pack',str(d/'rollout.rolloutpack'),'--output-learner-state',str(d/'learner.safetensors')]
    write(output/'training-plan.json',{
        'question':'Can ordinary PPO learn useful physical pushing from an unbiased fresh policy in the native episode curriculum?',
        'purpose':'Exploratory first learning trial; one learner seed, same development apparatus, unused placement streams',
        'preflight':preflight(root,json.loads((root/'profiles/american-crow-rl-v1.json').read_text())),
        'calibration_summary_sha256':sha(output/'calibration-summary.json'),'initial_policy_sha256':sha(output/'initial.policypack'),
        'model':'PPO can discover forward physical contact within16 updates; alternative is inadequate exploration/credit or undersized/destabilizing updates',
        'prediction':'Frozen final mean improves paired mean endpoint goal distance by>=.02m AND success within.02m by>=.20 over frozen initial mean',
        'refutation_rule':'Both quantitative thresholds required. Otherwise prediction contradicted; invalid physics/missing evidence makes it inconclusive. No extension, checkpoint selection or adaptive LR after viewing results.',
        'primary_metrics':['all-lane accepted final world-coordinate object goal distance m','endpoint success distance<=.02m'],
        'unit':'64 independent counter-RNG placement draws conditional on one frozen learned policy in one repeated apparatus; not independent learners or geometries',
        'evaluation':{'seeds':[2651317901,2651317902],'lanes_per_seed':32,'ticks':160,'mode':'frozen mean','order':['initial','final','final','initial']},
        'analysis':'Report every paired endpoint, mean distance reduction and success delta; exact paired sign test with1um tie tolerance. No biological/general-geometry confidence claim.',
        'held_fixed':'32 physics substeps,50Hz control,15Hz retina+2tick delay,18 actions, GRU32/encoder32, fixed LR.0003,gamma.999,lambda1,two full-batch epochs; default joint gradient clipping unchanged',
        'mechanism_limit':'Placement generalization in a narrow fixed apparatus only. No visual-dependence claim without separate causal ablation; no learned solid-food handling claim.',
        'budget':'One16-update128-lane training process timeout900s; four160-tick32-lane evaluations timeout180s each; stop on invalid physics. Raw study including calibration<=1GiB, no video, no retry. Retain native/model/Adam checkpoints8/16 and final generating rollout; intermediate generating rollouts are intentionally overwritten by the ordinary trainer and reconstructible from exact seeded inputs.',
        'command':c})
    v=execute(output,'training-run',c,root,900)
    if (v['training_samples']!=327680 or v['failed_environment_steps']!=0 or
        v['accepted_lifetime_control_ticks_by_environment']!=[2560]*128 or len(v['learning_update_history'])!=16):
        raise RuntimeError('Learning trial violates native accounting')
    frozen_mean(root,d/'updated.policypack',output/'final-mean.policypack')
    records={}
    for name,policy,seed in [('initial-a','initial-mean.policypack',2651317901),('final-a','final-mean.policypack',2651317901),
                             ('final-b','final-mean.policypack',2651317902),('initial-b','initial-mean.policypack',2651317902)]:
        records[name]=evaluation(root,output,name,output/policy,seed,32)
    initial=np.concatenate([records['initial-'+s]['distance_m'] for s in ('a','b')])
    final=np.concatenate([records['final-'+s]['distance_m'] for s in ('a','b')])
    delta=initial-final
    improvements=int(np.count_nonzero(delta>1e-6));regressions=int(np.count_nonzero(delta < -1e-6))
    n=improvements+regressions
    sign_p=min(1.,2*sum(math.comb(n,k) for k in range(min(improvements,regressions)+1))/2**n) if n else 1.
    success_delta=float(np.mean(final<=.02)-np.mean(initial<=.02))
    prediction=bool(delta.mean()>=.02 and success_delta>=.20)
    raw=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if raw>1024**3:raise RuntimeError('Study raw storage bound exceeded')
    write(output/'training-summary.json',{'status':'valid','prediction':'supported' if prediction else 'contradicted',
        'mean_initial_distance_m':float(initial.mean()),'mean_final_distance_m':float(final.mean()),
        'mean_paired_distance_reduction_m':float(delta.mean()),'successes_initial':int(np.count_nonzero(initial<=.02)),
        'successes_final':int(np.count_nonzero(final<=.02)),'cases':64,'success_fraction_delta':success_delta,
        'paired_distance_sign_test_two_sided_p':sign_p,'improvements':improvements,'regressions':regressions,
        'paired_distance_reductions_m':delta.tolist(),'raw_bytes_before_summary':raw,
        'training_samples':v['training_samples'],'failed_environment_steps':0,
        'end_to_end_training_transitions_per_second':v['end_to_end_environment_steps_per_second'],
        'learned_visual_manipulation_qualified':False,'ecological_environment_qualified':False})


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('phase',choices=['calibration','training'])
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    try:
        (calibration if args.phase=='calibration' else training)(root,output)
    except Exception as exc:
        if output.is_dir():write(output/(args.phase+'-failure.json'),{'type':type(exc).__name__,'error':str(exc)})
        raise
    print((output/(args.phase+'-summary.json')).read_text())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
