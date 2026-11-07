"""Native/Swift recurrent rollout handoff admission; no simulation or PPO."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import mlx.core as mx
import numpy as np
from .gru_memory import gru_sequence
from .native import learning_pack_content_hash
from .mlx_policy_learning import read_policy_pack, read_policy_rollout_pack
from .recurrent_policy_admission import dense, write


def sequence_policy(p,observations: np.ndarray,initial: np.ndarray,resets: np.ndarray):
    m=p.actor_memory
    if m is None:raise ValueError("Sequence requires recurrent PolicyPack")
    x=mx.array(observations)
    x=mx.clip((x-mx.array(p.effective_observation_mean))*mx.array(p.effective_observation_inverse_standard_deviation),-p.observation_clip,p.observation_clip)
    parameters={k:mx.array(getattr(m,k)) for k in ('input_weights','recurrent_weights','input_bias','recurrent_bias')}
    states,before=gru_sequence(dense(x,p.layers[:m.after_layer]),mx.array(initial),mx.array(resets.astype(bool)),parameters)
    latents=dense(states,p.layers[m.after_layer:]);mx.eval(before,latents)
    return np.array(before),np.array(latents)


def run_cli(source: Path,output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    write(output/'plan.json',{'schema':'numi.crow-recurrent-rollout-cli-sequence-plan.v1',
        'prediction':'regular CLI37+13 native-life continuation at chunks1/8 and retained runtime has exact canonical bytes; MLX evaluation of the complete50-tick sequence matches all saved pre-cell/latent values<=5e-6; continuation initial hidden nonzero and first reset masks0; modeled zero-initial-hidden continuation changes action mean>=1e-4',
        'unit':'two supported native v13 lanes on unused development seed2651109501; exact replay engineering instrument',
        'stopping':'one fixed batch instrument; preserve failures; no retries/tolerance changes',
        'limitations':'modeled hidden-state counterfactual and execution continuity, not native memory intervention, PPO or ecological memory behavior'})
    summary=json.loads((source/'cli-life-summary.json').read_text())
    if summary.get('status')!='passed' or summary.get('accepted_ticks')!=[37,50] or summary.get('unreported_warmup_steps')!=0:
        raise ValueError('Requires passed ordinary recurrent CLI lifetime/partition admission')
    inputs={}
    for label in ('cli-start','cli-resume'):
        a=(source/(label+'-chunk1.rolloutpack')).read_bytes();b=(source/(label+'-chunk8.rolloutpack')).read_bytes()
        if a!=b:raise RuntimeError('Regular CLI chunk-dependent recurrent artifact')
    if (source/'cli-retained-resume.rolloutpack').read_bytes()!=(source/'cli-resume-chunk8.rolloutpack').read_bytes():
        raise RuntimeError('Retained CLI recurrent artifact differs')
    first=read_policy_rollout_pack(source/'cli-start-chunk1.rolloutpack')
    tail=read_policy_rollout_pack(source/'cli-resume-chunk1.rolloutpack')
    p=read_policy_pack(source/'native-evidence/first.policypack')
    if (first.environment_count,first.control_step_count,tail.control_step_count,first.memory_hidden_count,tail.memory_hidden_count)!=(2,37,13,7,7):
        raise ValueError('Regular CLI recurrent sequence dimensions invalid')
    if first.policy_revision!=p.revision or first.policy_fingerprint!=tail.policy_fingerprint or tail.policy_revision!=p.revision:
        raise ValueError('Regular CLI sequence changed generating policy')
    obs=np.concatenate((first.actor_observations,tail.actor_observations)).reshape(50,2,p.actor_observation_count)
    before=np.concatenate((first.memory_before,tail.memory_before)).reshape(50,2,7)
    latents=np.concatenate((first.latents,tail.latents)).reshape(50,2,p.action_count)
    resets=np.concatenate((first.memory_resets,tail.memory_resets)).reshape(50,2)
    if not np.all(resets[0]==1) or np.any(tail.memory_resets[:2]!=0) or float(np.linalg.norm(before[37]))<1e-3:
        raise RuntimeError('Regular CLI continuation reset or discarded recurrent hidden history')
    b,a=sequence_policy(p,obs,before[0],resets)
    maximum=max(float(np.max(np.abs(b-before))),float(np.max(np.abs(a-latents))))
    zero_b,zero_a=sequence_policy(p,obs[37:],np.zeros((2,7),np.float32),resets[37:])
    difference=float(np.max(np.abs(zero_a-latents[37:])))
    np.savez(output/'cli-sequence-parity.npz',mlx_before=b,mlx_latents=a,native_before=before,native_latents=latents,
        resets=resets,zero_initial_hidden_before=zero_b,zero_initial_hidden_latents=zero_a)
    if maximum>5e-6 or difference<1e-4:raise RuntimeError('Regular CLI full-sequence parity or modeled counterfactual failed')
    for name in ('cli-start-chunk1.rolloutpack','cli-start-chunk8.rolloutpack','cli-resume-chunk1.rolloutpack','cli-resume-chunk8.rolloutpack','cli-retained-resume.rolloutpack','cli-life-summary.json'):
        inputs[name]=hashlib.sha256((source/name).read_bytes()).hexdigest()
    write(output/'input-receipt.json',inputs)
    result={'schema':'numi.crow-recurrent-rollout-cli-sequence-evidence.v1','status':'passed','real_native_slots':100,
        'MLX_full50tick_native_sequence_max_abs':maximum,'continuation_zero_hidden_action_difference':difference,
        'continuation_initial_hidden_norm':float(np.linalg.norm(before[37])),
        'modeled_hidden_counterfactual_only':True,'recurrent_PPO_integrated':False,'embodied_memory_qualified':False}
    write(output/'summary.json',result);return result


def run(native: Path, swift: Path, output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    write(output/'plan.json',{'schema':'numi.crow-recurrent-rollout-learner-plan.v1',
        'prediction':'native format8 and two Swift collection paths expose every real learner/hidden/reset value exactly; MLX evaluation of both artifact sequences matches latents/pre-cell state<=5e-6;11 malformed/flat-consumption controls reject',
        'unit':'same two native v13 development lanes replayed through C++ and Swift;80 real slots, not independent animals',
        'conditions':'40ticks, native lane reset9, weight publication18; two constant-policy segments18/22; extra native intermediate previews, aligned endpoints',
        'stopping':'one fixed instrument, no retries/tolerance changes; preserve all failures',
        'limitations':'artifact/forward-sequence engineering admission; no PPO, behavioral learning or production throughput claim'})
    n=json.loads((native/'summary.json').read_text());s=json.loads((swift/'summary.json').read_text())
    if n.get('status')!='passed' or n.get('recurrent_rollout_format')!=8 or n.get('recurrent_rollout_rejection_controls')!=10:
        raise ValueError('Requires passed native recurrent rollout8 admission')
    if s.get('status')!='passed' or s.get('rejection_controls')!=10 or not s.get('native_swift_all_segment_bytes_exact'):
        raise ValueError('Requires passed Swift recurrent collection admission')
    evidence={};maximum=0.;inputs={};rejections=[];slots=0
    def rejects(name,function,phrase):
        try:function()
        except ValueError as error:
            if phrase not in str(error):raise RuntimeError('Wrong rejection cause for '+name+': '+str(error)) from error
            rejections.append({'name':name,'error':str(error)});return
        raise RuntimeError('Invalid recurrent rollout accepted: '+name)
    for name,raw,policy,count in [('segment-first','first-segment','first.policypack',18),('segment-next','tail','next.policypack',22)]:
        file=native/(name+'.rolloutpack');data=file.read_bytes();inputs[str(file)]=hashlib.sha256(data).hexdigest()
        if struct.unpack_from('<I',data,8)[0]!=8:raise ValueError('Native recurrent record is not format8')
        r=read_policy_rollout_pack(file);p=read_policy_pack(native/policy);H=r.memory_hidden_count
        if (r.environment_count,r.control_step_count,H,r.policy_revision)!=(2,count,7,p.revision):
            raise ValueError('Recurrent rollout lost its generating dimensions/revision')
        for attribute,filename,dtype in [('actor_observations','actor.f32',np.float32),('critic_observations','critic.f32',np.float32),
                ('motion_features','motion.f32',np.float32),('latents','latents.f32',np.float32),('old_log_probabilities','logp.f32',np.float32),
                ('old_values','values.f32',np.float32),('bootstrap_values','bootstrap.f32',np.float32),
                ('memory_before','before.f32',np.float32),('memory_resets','reset.u32',np.uint32)]:
            if not np.array_equal(getattr(r,attribute),np.fromfile(native/raw/filename,dtype)):
                raise RuntimeError('Python reader differs from native '+name+': '+attribute)
        for mode in ('concat-chunk1','append-chunk8'):
            path=swift/mode/(name+'.rolloutpack')
            if path.read_bytes()!=data:raise RuntimeError('Swift canonical sequence bytes differ: '+mode+'/'+name)
            inputs[str(path)]=hashlib.sha256(data).hexdigest()
        m=p.actor_memory
        if m is None or np.asarray(m.recurrent_weights).shape[1]!=H:raise ValueError('Policy/rollout recurrent shape mismatch')
        b,a=sequence_policy(p,r.actor_observations.reshape(count,2,r.actor_observation_count),
            r.memory_before.reshape(count,2,H)[0],r.memory_resets.reshape(count,2))
        maximum=max(maximum,float(np.max(np.abs(b-r.memory_before.reshape(count,2,H)))),float(np.max(np.abs(a-r.latents.reshape(count,2,r.action_count)))))
        evidence[name+'-mlx-before']=b;evidence[name+'-mlx-latents']=a;evidence[name+'-native-before']=r.memory_before;evidence[name+'-native-latents']=r.latents;evidence[name+'-resets']=r.memory_resets
        slots+=r.sample_count
        rejects(name+'-flat-consumption',lambda r=r:r.policy_batch(),'contiguous sequence PPO')
        if name=='segment-first':
            memory_offset=len(data)-(24+4*r.memory_before.size+4*r.memory_resets.size)
            reset_count_offset=memory_offset+16+4*r.memory_before.size
            reset_data_offset=reset_count_offset+8
            cases=[('ABI',memory_offset,'<I',2,'recurrent ABI'),('hidden-width',memory_offset+4,'<I',0,'recurrent ABI'),
                ('before-count',memory_offset+8,'<Q',r.memory_before.size-1,'vector count'),
                ('reset-count',reset_count_offset,'<Q',r.memory_resets.size-1,'vector count'),
                ('nonfinite-state',memory_offset+16+4*2*H,'<f',float('nan'),'pre-cell/reset'),
                ('state-bound',memory_offset+16+4*2*H,'<f',1.01,'pre-cell/reset'),
                ('reset-mask',reset_data_offset,'<I',2,'pre-cell/reset'),
                ('reset-nonzero-state',memory_offset+16,'<f',.1,'pre-cell/reset'),
                ('v7-disguise',8,'<I',7,'trailing payload')]
            for label,offset,fmt,value,phrase in cases:
                corrupt=bytearray(data);struct.pack_into(fmt,corrupt,offset,value)
                struct.pack_into('<Q',corrupt,24,learning_pack_content_hash(np.frombuffer(corrupt,dtype=np.uint8,offset=32)))
                target=output/(label+'.corrupt.rolloutpack');target.write_bytes(corrupt)
                rejects(label,lambda target=target:read_policy_rollout_pack(target),phrase)
    np.savez(output/'artifact-sequence-parity.npz',**evidence)
    write(output/'rejections.json',rejections);write(output/'input-receipt.json',inputs)
    if maximum>5e-6:raise RuntimeError('Artifact MLX/native sequence error exceeds prospective tolerance')
    result={'schema':'numi.crow-recurrent-rollout-learner-evidence.v1','status':'passed','real_native_slots':slots,
        'all_native_learner_hidden_reset_arrays_exact':True,'both_swift_paths_canonical_bytes_exact':True,
        'MLX_native_artifact_sequence_max_abs':maximum,'parser_and_flat_consumption_rejections':len(rejections),
        'recurrent_rollout_format_integrated':True,'recurrent_PPO_integrated':False,'embodied_memory_qualified':False}
    write(output/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native',type=Path);parser.add_argument('--swift',type=Path);parser.add_argument('--cli',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.cli:result=run_cli(args.cli.resolve(strict=True),args.output.resolve())
        else:
            if not args.native or not args.swift:parser.error("--native and --swift are required unless --cli is supplied")
            result=run(args.native.resolve(strict=True),args.swift.resolve(strict=True),args.output.resolve())
        print(json.dumps(result,indent=2))
    except Exception as error:
        if args.output.is_dir() and not (args.output/'failure.json').exists():write(args.output/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise
if __name__=='__main__':main()
