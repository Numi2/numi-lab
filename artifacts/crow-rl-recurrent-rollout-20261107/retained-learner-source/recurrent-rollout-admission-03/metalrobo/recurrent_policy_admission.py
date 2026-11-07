"""Native PolicyPack5 to MLX sequence algebra admission, never a simulator."""
from __future__ import annotations
import argparse
import ctypes as ct
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import mlx.core as mx
import numpy as np
from . import native
from .gru_memory import gru_sequence
from .mlx_policy_learning import MLXPolicyLearner, read_policy_pack


def write(path: Path, value: object) -> None:
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False);f.write('\n')


def publish(p, target: Path) -> None:
    native.write_policy_pack(target, policy_id=p.id, revision=p.revision,
        contract_version=p.contract_version, world_fingerprint=p.world_fingerprint,
        task_fingerprint=p.task_fingerprint, observation_fingerprint=p.observation_fingerprint,
        action_fingerprint=p.action_fingerprint,
        layers=[native.PolicyDenseLayerArtifact(x.weights,x.bias,x.activation) for x in p.layers],
        critic_layers=[native.PolicyDenseLayerArtifact(x.weights,x.bias,x.activation) for x in p.critic_layers],
        observation_mean=p.observation_mean, observation_inverse_standard_deviation=p.observation_inverse_standard_deviation,
        critic_observation_mean=p.critic_observation_mean,
        critic_observation_inverse_standard_deviation=p.critic_observation_inverse_standard_deviation,
        action_log_standard_deviation=p.action_log_standard_deviation, action_bias=p.action_bias, action_scale=p.action_scale,
        observation_clip=p.observation_clip, action_clip=p.action_clip, actor_memory=p.actor_memory)


def dense(x, layers):
    for layer in layers:
        x=x@mx.array(layer.weights).T+mx.array(layer.bias)
        if layer.activation==1:x=mx.maximum(x,0)
        elif layer.activation==2:x=mx.tanh(x)
        elif layer.activation==3:x=mx.where(x>=0,x,mx.expm1(mx.minimum(x,0)))
        elif layer.activation==4:x=x*mx.sigmoid(x)
    return x


def run(source: Path, output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    write(output/'plan.json', {'schema':'numi.crow-recurrent-policy-plan.v1',
        'purpose':'canonical native/Python PolicyPack5 interoperability and real native observation sequence algebra',
        'prediction':'first/next recurrent and dense-control policies roundtrip byte exactly; MLX actor/pre-cell max error<=5e-6 on all native observations; malformed re-signed policies and invalid API/learner uses reject',
        'conditions':'two native v13 lanes,40ticks,reset9,weights18 from passed recurrent actor admission; no simulated transitions here',
        'stopping':'one fixed bounded instrument; preserve every failure; no retries or tolerance change',
        'limitations':'no PPO, behavioral learning, sequence rollout format, recurrent optimizer or throughput qualification'})
    summary=json.loads((source/'summary.json').read_text())
    if summary.get('status')!='passed' or not summary.get('fresh_process_full_life_exact'):
        raise ValueError('Requires passed native recurrent actor evidence')
    names=('first.policypack','next.policypack','dense-drop-control.policypack')
    packs=[read_policy_pack(source/name) for name in names]
    for p,name in zip(packs,names):
        publish(p,output/name)
        if (source/name).read_bytes()!=(output/name).read_bytes():
            raise RuntimeError('Native/Python canonical policy bytes differ: '+name)
    first,next_policy,_=packs
    if first.actor_memory is None or next_policy.actor_memory is None:
        raise ValueError('Native artifact has no recurrent memory')
    N,T,H,F=2,40,7,first.actor_observation_count
    observations=np.fromfile(source/'actor.f32',np.float32).reshape(T,N,F)
    expected_before=np.fromfile(source/'before.f32',np.float32).reshape(T,N,H)
    resets=np.fromfile(source/'reset.u32',np.uint32).reshape(T,N)
    expected_actions=np.fromfile(source/'actions.f32',np.float32).reshape(T,N,18)
    if np.any(resets>1):raise ValueError('Native reset traces invalid')
    all_actions=[];all_before=[]
    for begin,end,p in ((0,18,first),(18,40,next_policy)):
        m=p.actor_memory
        x=mx.array(observations[begin:end])
        x=mx.clip((x-mx.array(p.effective_observation_mean))*mx.array(p.effective_observation_inverse_standard_deviation),-p.observation_clip,p.observation_clip)
        x=dense(x,p.layers[:m.after_layer])
        parameters={key:mx.array(getattr(m,key)) for key in ('input_weights','recurrent_weights','input_bias','recurrent_bias')}
        states,before=gru_sequence(x,mx.array(expected_before[begin]),mx.array(resets[begin:end].astype(bool)),parameters)
        actions=dense(states,p.layers[m.after_layer:])
        actions=mx.clip(actions*mx.array(p.effective_action_scale)+mx.array(p.effective_action_bias),-p.action_clip,p.action_clip)
        mx.eval(actions,before);all_actions.append(np.array(actions));all_before.append(np.array(before))
    actions=np.concatenate(all_actions);before=np.concatenate(all_before)
    maximum=max(float(np.max(np.abs(actions-expected_actions))),float(np.max(np.abs(before-expected_before))))
    np.savez(output/'sequence-parity.npz',actions=actions,before=before,expected_actions=expected_actions,expected_before=expected_before,resets=resets)
    if maximum>5e-6:raise RuntimeError('MLX/native recurrent policy sequence error exceeds prospective tolerance')
    rejections=[]
    def rejects(name,function,phrase):
        try:function()
        except (ValueError,native.MetalRoboError) as error:
            if phrase not in str(error):raise RuntimeError('Wrong rejection cause for '+name+': '+str(error)) from error
            rejections.append({'name':name,'error':str(error)});return
        raise RuntimeError('Invalid contract accepted: '+name)
    for name,function in (
        ('flat-actor',lambda:first.actor_mean(observations[0])),
        ('flat-learner-restore',lambda:MLXPolicyLearner.from_policy_pack(source/'first.policypack')),
        ('flat-actor-initialization',lambda:MLXPolicyLearner.from_actor_policy_pack(source/'first.policypack',97)),
        ('flat-actor-projection',lambda:MLXPolicyLearner.from_projected_actor_policy_pack(source/'first.policypack',97,tuple(range(F))))):
        rejects(name,function,'sequence')
    m=first.actor_memory
    rejects('writer-invalid-boundary',lambda:publish(replace(first,actor_memory=replace(m,after_layer=0)),output/'forbidden-boundary.policypack'),'GRU')
    rejects('writer-dense-boundary-mismatch',lambda:publish(replace(first,actor_memory=replace(m,after_layer=1,recurrent_weights=np.zeros((18,6),np.float32),input_weights=np.zeros((18,5),np.float32),input_bias=np.zeros(18,np.float32),recurrent_bias=np.zeros(18,np.float32))),output/'forbidden-shape.policypack'),'boundary')
    rejects('writer-nonfinite',lambda:publish(replace(first,actor_memory=replace(m,input_bias=np.full(21,np.nan,np.float32))),output/'forbidden-nan.policypack'),'finite')
    # Test the additive descriptor guard directly before policy conversion.
    bindings=native._load_bindings();function=bindings.lib.mr_write_recurrent_policy_pack
    function.argtypes=[ct.POINTER(native._PolicyPackC),ct.POINTER(native._PolicyGRUMemoryC),ct.c_char_p];function.restype=ct.c_int
    dummy=native._PolicyPackC()
    for name,version,size in (('C-version',2,ct.sizeof(native._PolicyGRUMemoryC)),('C-size',1,0),('C-dimensions',1,ct.sizeof(native._PolicyGRUMemoryC))):
        descriptor=native._PolicyGRUMemoryC(abi_version=version,struct_size=size)
        target=output/(name+'.forbidden')
        status=function(ct.byref(dummy),ct.byref(descriptor),str(target).encode())
        phrase='ABI/size' if name!='C-dimensions' else 'dimensions/tensor counts'
        if status==0 or phrase not in bindings.last_error() or target.exists():raise RuntimeError('C descriptor guard failed: '+name)
        rejections.append({'name':name,'error':bindings.last_error()})
    # Correctly re-sign malformed PolicyPack5 payloads so parsing tests the
    # semantic descriptor rather than stopping at a checksum mismatch.
    original=(source/'first.policypack').read_bytes()
    memory_bytes=16+4*8+4*sum(np.asarray(getattr(m,k)).size for k in ('input_weights','recurrent_weights','input_bias','recurrent_bias'))
    offset=len(original)-memory_bytes
    corruptions=[('after-zero',0,'<I',0),('after-end',0,'<I',len(first.layers)),('input-width',4,'<I',6),('hidden-zero',8,'<I',0),('nonfinite-bound',12,'<f',float('nan')),('wrong-weight-count',16,'<Q',1),('nonfinite-weight',24,'<f',float('inf'))]
    for name,relative,fmt,value in corruptions:
        data=bytearray(original);struct.pack_into(fmt,data,offset+relative,value)
        struct.pack_into('<Q',data,24,native.learning_pack_content_hash(np.frombuffer(data,dtype=np.uint8,offset=32)))
        target=output/(name+'.corrupt.policypack');target.write_bytes(data)
        rejects('reader-'+name,lambda target=target:read_policy_pack(target),'PolicyPack' if name not in ('wrong-weight-count',) else 'vector count')
    if any(output.glob('forbidden-*.policypack')):raise RuntimeError('Rejected writer published output')
    write(output/'rejections.json',rejections)
    write(output/'input-receipt.json',{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in (*names,'actor.f32','before.f32','reset.u32','actions.f32','summary.json')})
    result={'schema':'numi.crow-recurrent-policy-evidence.v1','status':'passed','canonical_recurrent_policies_byte_exact':2,
        'canonical_dense_control_byte_exact':True,'MLX_native_sequence_max_abs':maximum,'contract_rejections':len(rejections),
        'actual_native_observation_slots':80,'recurrent_PPO_integrated':False,'embodied_memory_qualified':False}
    write(output/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:print(json.dumps(run(args.native.resolve(strict=True),args.output.resolve()),indent=2))
    except Exception as error:
        if args.output.is_dir() and not (args.output/'failure.json').exists():write(args.output/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise
if __name__=='__main__':main()
