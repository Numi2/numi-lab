"""Sealed behavioral endpoints, seed-cluster inference and retained-update diagnosis."""
from pathlib import Path
import csv,itertools,json,math
import mlx.core as mx
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_rollout_pack
from metalrobo.mlx_recurrent_policy_learning import MLXRecurrentPolicyBatch

root=Path.cwd();out=root/'.numi/runs/american-crow/post-step-rate-01'
plan=json.loads((root/'artifacts/crow-rl-post-step-rate-20261108/plan.json').read_text())
assert (out/'trial-complete.json').exists()
data={};cases={};diagnosis={}
def paired(a,b,key):
    assert [(x['seed'],x['lane'],x['mask'])for x in a]==[(x['seed'],x['lane'],x['mask'])for x in b]
    wins=sum(y[key]and not x[key]for x,y in zip(a,b));loss=sum(x[key]and not y[key]for x,y in zip(a,b))
    n=wins+loss
    discordant=min(1.,2*sum(math.comb(n,k)for k in range(min(wins,loss)+1))/2**n)if n else 1.
    differences=[sum(y[key]-x[key]for x,y in zip(a,b)if x['seed']==seed)
                 for seed in sorted({x['seed']for x in a})]
    observed=abs(sum(differences));permuted=[sum(s*d for s,d in zip(signs,differences))
        for signs in itertools.product((-1,1),repeat=len(differences))]
    return {'gain_count':wins-loss,'gain_fraction':(wins-loss)/len(a),
        'candidate_only':wins,'reference_only':loss,'exact_discordant_pair_p':discordant,
        'per_seed_count_differences':differences,'positive_seeds':sum(d>0 for d in differences),
        'exact_seed_sign_flip_p':sum(abs(x)>=observed for x in permuted)/len(permuted)}
for name in ('initial','fixed','adaptive'):
    arms=[[]for _ in range(4)];summaries=[]
    for block in range(3):
        native=out/f'heldout/{name}-{block}/evaluation/native-evidence'
        summary=json.loads((native/'summary.json').read_text());summaries.append(summary)
        for seed in range(plan['heldout_seed_bases'][block],plan['heldout_seed_bases'][block]+3):
            for arm in range(4):
                with (native/f'{seed}-arm{arm}/lifetimes.tsv').open()as f:raw=list(csv.DictReader(f,delimiter='\t'))
                assert len(raw)==64
                for x in raw:
                    mask=int(x['allocation_mask']);head=float(x['head_action_tick64']);food=int(x['consumed_micrograms'])
                    assert 0<=food<=45000 and mask in (1,2)
                    arms[arm].append({'seed':seed,'lane':int(x['lane']),'mask':mask,'head':head,
                        'food':food,'correct':int(head>0 if mask==1 else head<0),'intake':int(food>0)})
    data[name]=arms
    cases[name]={'policy_revision':summaries[0]['policy_revision'],'policy_sha256':summaries[0]['policy_sha256'],
        'choice':[sum(x['correct']for x in arm)for arm in arms],
        'intake':[sum(x['intake']for x in arm)for arm in arms],
        'consumed_micrograms':[sum(x['food']for x in arm)for arm in arms],
        'by_cue':{str(mask):{'lifetimes':sum(x['mask']==mask for x in arms[0]),
            'choice':sum(x['correct']for x in arms[0]if x['mask']==mask),
            'intake':sum(x['intake']for x in arms[0]if x['mask']==mask)}for mask in (1,2)},
        'causal_controls':{str(arm):{k:paired(arms[arm],arms[0],k)for k in ('correct','intake')}
                           for arm in (1,2,3)}}
    assert all(v['policy_sha256']==cases[name]['policy_sha256']for v in summaries)
for name in ('fixed','adaptive'):
    r=read_policy_rollout_pack(out/name/'training.rolloutpack')
    _,meta=mx.load(out/name/'learner.safetensors',return_metadata=True);c=json.loads(meta['configuration'])
    batch=MLXRecurrentPolicyBatch.from_rollout(r,discount=c['discount'],gae_lambda=c['gae_lambda'])
    T,N,A=r.control_step_count,r.environment_count,r.action_count
    sel=r.action_selections().reshape(T,N,A).astype(bool);head=sel[:,:,15]
    rewards=r.transitions['reward'].reshape(T,N);values=r.old_values.reshape(T,N)
    assert np.array_equal(np.argwhere(head)[:,0],np.full(N,64))
    returns=np.zeros((T,N),'f8');running=np.zeros(N,'f8')
    for tick in range(T-1,-1,-1):running=rewards[tick]+c['discount']*running;returns[tick]=running
    advantage=batch.advantages[head].astype('f8');mc=returns[head]-values[head];intake=rewards.sum(axis=0)>0
    history=json.loads((out/name/'summary.json').read_text())['learning_update_history']
    summary=json.loads((out/name/'summary.json').read_text())
    identity=returns[head]+c['discount']**(T-64)*r.bootstrap_values-values[head]
    diagnosis[name]={'actual_configuration':c,'head_tick':64,'head_count':int(head.sum()),
        'head_actual_advantage_mean_intake':float(advantage[intake].mean())if intake.any()else None,
        'head_actual_advantage_mean_no_intake':float(advantage[~intake].mean())if (~intake).any()else None,
        'head_reward_only_MC_correlation':float(np.corrcoef(advantage,mc)[0,1])if np.std(mc)>0 else None,
        'lambda1_FP64_telescoping_max_abs':float(np.max(np.abs(identity-advantage))),
        'actual_128_tick_TD_coefficient':float((c['discount']*c['gae_lambda'])**128),
        'default_128_tick_TD_coefficient':float((.99*.95)**128),
        'final_batch_intake_lifetimes':int(intake.sum()),'updates':history,
        'training_wall_seconds':summary['elapsed_seconds'],
        'training_transitions_per_second':summary['end_to_end_environment_steps_per_second'],
        'maximum_mlx_peak_memory_bytes':max(x['mlx_peak_memory_bytes']for x in history),
        'minimum_joint_gradient_clip_scale':min(x['minimum_gradient_clip_scale']for x in history)}
comparisons={}
for before,after in (('fixed','adaptive'),('initial','fixed'),('initial','adaptive')):
    comparisons[before+'->'+after]={k:paired(data[before][0],data[after][0],k)for k in ('correct','intake')}
primary=comparisons['fixed->adaptive']
endpoint_pass={k:v['gain_fraction']>=.05 and v['exact_seed_sign_flip_p']<=.05 and v['positive_seeds']>=7
               for k,v in primary.items()}
control_pass=all(cases['adaptive']['causal_controls'][str(arm)][k]['gain_count']>0
                 for arm in (1,2,3)for k in ('correct','intake'))
success=all(endpoint_pass.values())and control_pass and comparisons['initial->adaptive']['intake']['gain_count']>0
result={'status':'measured','prediction':'supported'if success else 'contradicted',
    'lifetimes_per_policy_control':576,'accepted_training_transitions':1310720,
    'accepted_evaluation_transitions':2211840,'failed_physics_steps':0,'new_policy_promotions':0,
    'cases':cases,'comparisons':comparisons,'primary_endpoint_pass':endpoint_pass,
    'causal_control_direction_pass':control_pass,'retained_update_diagnosis':diagnosis,
    'scope':plan['limits'],'inference_unit':'Nine evaluation seeds conditional on one matched training pair; not independent training replicates or algorithm-wide generalization.'}
with (out/'analysis.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'prediction':result['prediction'],'cases':{k:{x:v[x]for x in ('choice','intake')}for k,v in cases.items()},
                  'primary':primary,'primary_endpoint_pass':endpoint_pass,'causal_controls':control_pass}))
