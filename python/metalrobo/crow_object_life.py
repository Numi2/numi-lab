"""Bounded ordinary native/Swift/MLX life continuation for object manipulation.

This orchestrates whole native runs and reads their publications. It does not
simulate or modify physical state. The scripted policy is an engineering input.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import numpy as np


def _envelope(data: bytes, kind: bytes) -> tuple[bytes, int]:
    tag, abi, reserved, identity, size, digest = struct.unpack_from('<8sIIQQ32s', data)
    if tag != kind or abi != 1 or reserved or size != len(data)-64 or hashlib.sha256(data[64:]).digest() != digest:
        raise RuntimeError('Invalid checkpoint envelope in independent readback')
    return data[64:], identity


def _parts(data: bytes) -> tuple[bytes, list[bytes], int]:
    payload, identity = _envelope(data, b'NUMILF01')
    offset = 40
    parts = []
    for _ in range(3):
        count, = struct.unpack_from('<Q', payload, offset); offset += 8
        parts.append(payload[offset:offset+count]); offset += count
    if offset != len(payload) or parts[2]:
        raise RuntimeError('Object life must have exactly world/retina and an empty resource component')
    return payload[:40], parts, identity


def _rewrite(data: bytes, index: int, replacement: bytes) -> bytes:
    counters, parts, identity = _parts(data); parts[index] = replacement
    payload = counters + b''.join(struct.pack('<Q', len(p))+p for p in parts)
    return struct.pack('<8sIIQQ32s', b'NUMILF01', 1, 0, identity, len(payload), hashlib.sha256(payload).digest())+payload


def run(root: Path, output: Path, paths: dict, *, reuse_completed: bool = False) -> dict:
    from .crow_environment import execute, sha, write
    from .mlx_policy_learning import read_policy_pack, read_policy_rollout_pack
    from .native import PolicyDenseLayerArtifact, write_policy_pack
    import mlx.core as mx

    config = root/'environments/american-crow/object-push-v1/retinal-observation.json'
    plan = {
        'question': 'Does ordinary v14 preserve physical object/contact, RNG, pending/visible retina, GRU, model/Adam and accepted lifetime clocks across a fresh process?',
        'prediction': 'All causal native components, future rollout records and checkpoint model/Adam tensors/metadata exact across chunks1/8; no warmup ticks. Six malformed/mismatched restores reject before a new physical transition.',
        'units': 'Two native lanes with counter-based action randomness in one fixed engineering apparatus; not independent geometries or learned behavior.',
        'budget': 'One1-tick initialization update;3-update reference,1-update split and two2-update resumes at47ticks x2lanes;160-tick fixed-policy reference,47-tick prefix and two113-tick resumes; six negative requests. Each process180s, stop first unexpected result, no retries. Bounded state trace160rows, no video; expected evidence under100MiB excluding exact runtime.',
        'policy': 'Fresh GRU32/encoder32; native writer changes only final head bias to0.6 and sets latent std0.01. All18 actions retain authority. Input controller, not a learned manipulation claim.',
        'learning': 'fixed LR0.0003, gamma0.999, lambda1, two full-batch epochs; no adaptive-rate selection',
        'seeds': [2651315001, 2651315002],
        'scope': 'Exact-runtime same-device engineering continuation, not ecological, material, biological or sustained performance qualification',
    }
    if reuse_completed:
        if json.loads((output/'object-life-plan.json').read_text()) != plan:
            raise RuntimeError('Reanalysis must retain the original prospective plan')
    else:
        write(output/'object-life-plan.json', plan)
    shared = ['--birdflow-american-crow-journey','--birdflow-journey-variant','v14-object-push',
              '--visual-observation-config',str(config),'--envs','2','--seed','2651315001',
              '--minimum-difficulty-band','0','--maximum-difficulty-band','0','--metallib',str(paths['metallib'])]
    train = ['/usr/bin/env','DYLD_PRINT_LIBRARIES=1',str(paths['train'])]+shared+[
        '--native-library',str(paths['library']),'--mlx-python',sys.executable,'--python-root',str(root/'python'),
        '--learner-seed','2651315002','--fixed-learning-rate','--learning-rate','0.0003','--discount','0.999','--gae-lambda','1',
        '--update-epochs','2','--minibatch-size','94','--recurrent-sequence-length','47',
        '--checkpoint-life','--checkpoint-interval','1']

    def training(name, updates, chunk, resume=None, bootstrap=False):
        d=output/name
        if not (reuse_completed and d.is_dir()):
            d.mkdir()
        c=train+['--steps','1' if bootstrap else '47','--updates',str(updates),'--chunk',str(chunk),
            '--updated-policy-pack',str(d/'updated.policypack'),'--deployment-policy-pack',str(d/'candidate.policypack'),
            '--incumbent-policy-pack',str(d/'incumbent.policypack'),'--rollout-pack',str(d/'rollout.rolloutpack'),
            '--output-learner-state',str(d/'learner.safetensors'),'--checkpoint-directory',str(d/'checkpoints')]
        if resume:c+=['--resume-training-checkpoint',str(resume)]
        elif bootstrap:c+=['--policy-pack',str(d/'initial.policypack'),'--initialize-policy','crow-object-life-input',
            '--zero-actor-output','--recurrent-hidden-size','32','--recurrent-encoder-sizes','32',
            '--recurrent-memory-timescale-ticks','64','--initial-log-standard-deviation',str(math.log(.01))]
        else:c+=['--policy-pack',str(output/'push.policypack')]
        return c

    def execute_native(name, c):
        if reuse_completed:
            receipt=json.loads((output/(name+'.receipt.json')).read_text())
            if (receipt['command'] != c or receipt['exit_code'] != 0 or
                receipt['timed_out'] or receipt['interrupted'] or
                receipt['stdout_sha256'] != sha(output/(name+'.json')) or
                receipt['stderr_sha256'] != sha(output/(name+'.stderr'))):
                raise RuntimeError('Retained execution receipt differs: '+name)
            v=json.loads((output/(name+'.json')).read_text())
        else:
            v=execute(output,name,c,root,180)
        loaded=[s for s in (output/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
        if not loaded or any(not s.endswith(str(paths['library'])) for s in loaded):raise RuntimeError('Unexpected native library: '+name)
        if v['failed_environment_steps']!=0:raise RuntimeError('Rejected native transition: '+name)
        return v

    execute_native('initialize',training('initialize',1,1,bootstrap=True))
    p=read_policy_pack(output/'initialize/initial.policypack',library_path=paths['library'])
    layers=[PolicyDenseLayerArtifact(x.weights.copy(),x.bias.copy(),x.activation)for x in p.layers]
    layers[-1].bias[15]=.6
    authored_policy=output/('reanalysis-scripted.policypack' if reuse_completed else 'push.policypack')
    write_policy_pack(authored_policy,policy_id='crow-object-life-scripted-push',revision=1,
        **{k:getattr(p,k)for k in ('contract_version','world_fingerprint','task_fingerprint','observation_fingerprint','action_fingerprint',
            'observation_mean','observation_inverse_standard_deviation','critic_observation_mean','critic_observation_inverse_standard_deviation',
            'action_log_standard_deviation','action_bias','action_scale','observation_clip','action_clip','actor_memory')},
        layers=layers,critic_layers=[PolicyDenseLayerArtifact(x.weights,x.bias,x.activation)for x in p.critic_layers],library_path=paths['library'])
    if reuse_completed and sha(authored_policy) != sha(output/'push.policypack'):
        raise RuntimeError('Retained scripted input differs from the original policy writer')
    def snapshot(name, revision):return output/name/'checkpoints'/f'revision-{revision:08}.trainingcheckpoint'
    training_records=[]
    source=snapshot('split',2)
    for name,updates,chunk,resume in [('reference',3,8,None),('split',1,8,None),('resume1',2,1,source),('resume8',2,8,source)]:
        v=execute_native(name,training(name,updates,chunk,resume))
        ticks=141 if resume or name=='reference' else 47
        if v['training_samples']!=updates*94 or v['accepted_lifetime_control_ticks_by_environment']!=[ticks]*2 or v['life_state_restored']!=bool(resume):
            raise RuntimeError('Accepted lifetime/sample/resume accounting differs: '+name)
        training_records.append({k:v[k]for k in ('training_samples','accepted_lifetime_control_ticks_by_environment','end_to_end_environment_steps_per_second','life_checkpoint_seconds','retained_buffer_bytes','transient_private_bytes','life_state_restored')})
    comparisons=[]
    for name,revisions in [('split',(2,)),('resume1',(3,4)),('resume8',(3,4))]:
        for revision in revisions:
            a,b=snapshot('reference',revision),snapshot(name,revision)
            am,bm=[json.loads((d/'manifest.json').read_text())for d in (a,b)]
            for m in (am,bm):
                if m['acceptedControlTicks']!=[47*(revision-1)]*2 or m['completedUpdates']!=revision-1:raise RuntimeError('Checkpoint tick/update count differs')
                if (a if m is am else b).joinpath('COMMITTED').read_text()!=sha((a if m is am else b)/'manifest.json'):raise RuntimeError('Bad checkpoint publication marker')
            for f in ('policy.policypack','candidate.policypack','rollout.rolloutpack'):
                if sha(a/f)!=sha(b/f):raise RuntimeError('Future artifact differs: '+name+'/'+f)
            x,xm=mx.load(a/'learner.safetensors',return_metadata=True);y,ym=mx.load(b/'learner.safetensors',return_metadata=True)
            if xm!=ym or set(x)!=set(y):raise RuntimeError('Model/Adam topology or metadata differs')
            for key in x:
                if np.asarray(x[key]).tobytes()!=np.asarray(y[key]).tobytes():raise RuntimeError('Model/Adam tensor differs: '+key)
            if _parts((a/'life.native').read_bytes())[1]!=_parts((b/'life.native').read_bytes())[1]:raise RuntimeError('Native physical/GRU/retinal continuation differs')
            comparisons.append({'stage':name,'revision':revision,'model_Adam_tensors_exact':len(x),'all_causal_components_exact':True})

    rollout=['/usr/bin/env','DYLD_PRINT_LIBRARIES=1',str(paths['rollout'])]+shared+[
        '--policy-pack',str(output/'push.policypack'),'--repeats','1','--no-scheduled-resets']
    for name,steps,chunk,resume in [('full',160,1,None),('prefix',47,1,None),('future1',113,1,output/'prefix.life'),('future8',113,8,output/'prefix.life')]:
        c=rollout+['--steps',str(steps),'--chunk',str(chunk),'--rollout-pack',str(output/(name+'.rolloutpack')),'--save-life-checkpoint',str(output/(name+'.life'))]
        if resume:c+=['--resume-life-checkpoint',str(resume)]
        if name=='full':c+=['--state-trace',str(output/'accepted-state.tsv')]
        v=execute_native(name,c)
        if v['accepted_lifetime_control_ticks_by_environment']!=[47 if name=='prefix' else 160]*2:raise RuntimeError('Rollout restored an incorrect lifetime clock')
    full=read_policy_rollout_pack(output/'full.rolloutpack',library_path=paths['library'])
    for name in ('future1','future8'):
        future=read_policy_rollout_pack(output/(name+'.rolloutpack'),library_path=paths['library'])
        for field in ('actor_observations','critic_observations','latents','old_log_probabilities','old_values','transitions','memory_before','memory_resets'):
            x=np.asarray(getattr(full,field));y=np.asarray(getattr(future,field))
            # The native reader returns flat storage, including multi-feature fields.
            # Preserve environment-interleaved sample order before selecting the suffix.
            if (x.size % 320 or y.size % 226 or x.size//320 != y.size//226 or
                x.reshape(320,-1)[94:].tobytes()!=y.reshape(226,-1).tobytes()):
                raise RuntimeError('Fixed-policy future stream differs: '+name+'/'+field)
        if full.bootstrap_values.tobytes()!=future.bootstrap_values.tobytes() or full.outcomes.keys()!=future.outcomes.keys():raise RuntimeError('Future terminal value/outcome schema differs')
        for key in full.outcomes:
            if full.outcomes[key][94:].tobytes()!=future.outcomes[key].tobytes():raise RuntimeError('Future outcome differs: '+key)
        if _parts((output/'full.life').read_bytes())[1]!=_parts((output/(name+'.life')).read_bytes())[1]:raise RuntimeError('Final native hidden components differ')
    trace=np.loadtxt(output/'accepted-state.tsv',comments='#');xyz=trace[:,1+23+2*13:1+23+2*13+3]
    if trace.shape!=(160,1+23+3*13) or np.linalg.norm(xyz[0]-[.225,.03,.257])>.001:raise RuntimeError('Bound native object-state trace schema/initial position differs')
    motion=float(np.linalg.norm(xyz[-1]-xyz[0]))
    if motion<=.01 or not np.any(np.asarray(full.memory_before)!=0):raise RuntimeError('Continuation lacked object movement or live GRU memory')

    negatives=[]
    for name,reason in [('missing-marker','COMMITTED'),('wrong-clock','accepted lifetime clock disagrees'),('unexpected-resource','resource component contract mismatch'),('bad-retina','checksum mismatch'),('truncated-world','schema or length mismatch'),('changed-learner','scheduler/learner configuration mismatch')]:
        bad=output/('invalid-'+name);shutil.copytree(source,bad)
        if name=='missing-marker':(bad/'COMMITTED').unlink()
        else:
            manifest=json.loads((bad/'manifest.json').read_text())
            if name=='wrong-clock':manifest['acceptedControlTicks']=[48,48]
            elif name in ('unexpected-resource','bad-retina','truncated-world'):
                p=bad/'life.native';data=p.read_bytes();parts=_parts(data)[1]
                if name=='unexpected-resource':data=_rewrite(data,2,b'X')
                elif name=='bad-retina':
                    corrupt=bytearray(parts[1]);corrupt[-1]^=1;data=_rewrite(data,1,bytes(corrupt))
                else:data=_rewrite(data,0,parts[0][:-1])
                p.write_bytes(data);manifest['files']['life.native']={'sha256':sha(p),'bytes':p.stat().st_size}
            (bad/'manifest.json').write_text(json.dumps(manifest,sort_keys=True));(bad/'COMMITTED').write_text(sha(bad/'manifest.json'))
        c=training('reject-'+name,1,8,bad)
        if name=='changed-learner':c[c.index('--discount')+1]='0.998'
        try:execute(output,'reject-'+name,c,root,60)
        except RuntimeError:
            receipt=json.loads((output/('reject-'+name+'.receipt.json')).read_text())
            if receipt['timed_out'] or receipt['exit_code'] in (None,0) or reason.lower() not in (output/('reject-'+name+'.stderr')).read_text().lower():raise RuntimeError('Unexpected rejection cause: '+name)
            if list((output/('reject-'+name)).rglob('life.native')):raise RuntimeError('Invalid restore published a new life')
            negatives.append(name)
        else:raise RuntimeError('Invalid checkpoint accepted: '+name)
    return {'status':'passed','verification_reuses_completed_positive_runs':reuse_completed,'training_records':training_records,'coupled_boundary_comparisons':comparisons,
        'fixed_policy_future_fields_exact':True,'native_world_retina_GRU_future_exact':True,'empty_resource_component_required':True,
        'accepted_lifetime_ticks_exact':True,'rejected_controls':negatives,'object_motion_m':motion,
        'object_final_position_m':xyz[-1].tolist(),'pending_retinal_save_tick':47,'scripted_controller':True,
        'learned_manipulation_qualified':False,'ecological_environment_qualified':False}
