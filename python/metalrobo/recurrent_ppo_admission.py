"""Bounded native stochastic recurrent PPO and optimizer recovery instrument."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from .crow_environment import execute
from .mlx_policy_learning import read_policy_pack, read_policy_rollout_pack
from .mlx_policy_worker import _write_learner_state, _restore_learner_state
from .mlx_recurrent_policy_learning import (MLXRecurrentPPOConfiguration,
    MLXRecurrentPolicyBatch, MLXRecurrentPolicyLearner)
from .recurrent_policy_admission import publish


def write(path, value):
    with path.open("x") as f: json.dump(value,f,indent=2,allow_nan=False);f.write("\n")


def state(learner):
    return {prefix+name:np.array(value) for prefix,tree in
        (("model.",learner.model.state),("optimizer.",learner.optimizer.state)) for name,value in tree_flatten(tree)}


def same(a,b):
    return set(a)==set(b) and all(a[k].dtype==b[k].dtype and np.array_equal(a[k],b[k]) for k in a)


def fp64_loss(parameters, p, args, config):
    actor,critic,latents,old_logp,old_means,old_logstd,old_values,adv,returns,initial,resets,valid=args
    def dense(x,prefix,layers):
        for i,layer in enumerate(layers):
            x=x@parameters[f"{prefix}.{i}.weight"].T+parameters[f"{prefix}.{i}.bias"]
            if layer.activation==1:x=np.maximum(x,0)
            elif layer.activation==2:x=np.tanh(x)
            elif layer.activation==3:x=np.where(x>=0,x,np.expm1(np.minimum(x,0)))
            elif layer.activation==4:x=x/(1+np.exp(-x))
        return x
    m=p.actor_memory;H=m.recurrent_weights.shape[1]
    x=np.clip((actor-p.effective_observation_mean)*p.effective_observation_inverse_standard_deviation,-p.observation_clip,p.observation_clip)
    x=dense(x,"actor.encoder",p.layers[:m.after_layer]);hidden=initial.copy();states=[]
    for t in range(len(x)):
        hidden=np.where(resets[t,:,None],0,hidden)
        a=x[t]@parameters["actor.memory.input_weights"].T+parameters["actor.memory.input_bias"]
        b=hidden@parameters["actor.memory.recurrent_weights"].T+parameters["actor.memory.recurrent_bias"]
        r=1/(1+np.exp(-(a[:,:H]+b[:,:H])));z=1/(1+np.exp(-(a[:,H:2*H]+b[:,H:2*H])))
        hidden=(1-z)*np.tanh(a[:,2*H:]+r*b[:,2*H:])+z*hidden;states.append(hidden)
    means=dense(np.stack(states),"actor.decoder",p.layers[m.after_layer:])
    v=np.clip((critic-p.effective_critic_observation_mean)*p.effective_critic_observation_inverse_standard_deviation,-p.observation_clip,p.observation_clip)
    values=dense(v,"critic",p.critic_layers)[...,0]
    logstd=np.clip(parameters["log_standard_deviation"],-5,2)
    logp=np.sum(-.5*((latents-means)*np.exp(-logstd))**2-logstd-.5*math.log(2*math.pi),axis=-1)
    ratio=np.exp(logp-old_logp);average=lambda a:np.sum(a*valid)/np.sum(valid)
    policy=-average(np.minimum(ratio*adv,np.clip(ratio,1-config.clip_ratio,1+config.clip_ratio)*adv))
    clipped=old_values+np.clip(values-old_values,-config.clip_ratio,config.clip_ratio)
    value=average(np.maximum((values-returns)**2,(clipped-returns)**2))
    entropy=np.sum(logstd+.5*(1+math.log(2*math.pi)))
    return float(policy+config.value_coefficient*value-config.entropy_coefficient*entropy)


def run(root: Path, out: Path):
    write(out/"plan-stochastic-v2.json",{"schema":"numi.crow-sequence-ppo-plan.v2",
        "source_inspection":"Earlier fixture policies are deterministic; they cannot generate on-policy stochastic PPO records. Preserve them as controls; add logstd=-2.5 with a distinct policy identity/revision31 before collecting new native data.",
        "unit":"Two native v13 lanes, new development seed2651109503;37+13tick checkpoint continuation; one13tick post-update rollout. Paired ordinary training3 vs2+1 updates of37ticks may follow as a separately bounded integration check.",
        "prediction":"Native generating state/value <=5e-5, Gaussian logp <=5e-4; FP64 selected PPO gradients <=3e-5; reset and initial-state gradient cuts exact zero; padding gradient difference <=3e-6; all Adam/model arrays and next update exact after fresh learner restore.",
        "stopping":"One fixed instrument for this source revision; preserve failure; no threshold relaxation; no qualification seeds or ecological claims."})
    snapshots=out/"sources-stochastic";snapshots.mkdir()
    sources={}
    for name in ("mlx_policy_learning.py","mlx_policy_worker.py","mlx_recurrent_policy_learning.py","gru_memory.py","recurrent_ppo_admission.py"):
        p=root/"python/metalrobo"/name;data=p.read_bytes();(snapshots/name).write_bytes(data);sources[name]=hashlib.sha256(data).hexdigest()
    write(out/"sources-stochastic.json",sources)
    if not (out/"inputs").exists():
        value=execute(out,"native-seed",[str(root/"build/bin/metalrobo_crow_recurrent_actor_check"),
            str(out/"inputs"),str(root/"artifacts/crow-rl-recurrent-actor-20261107/inputs/PolicyABI3.metallib")],root,180)
        if value.get("status")!="passed" or value.get("recurrent_rollout_format")!=8:
            raise RuntimeError("Native recurrent seed fixture admission failed")
    visual=json.loads((out/"inputs/retinal-observation.json").read_text())
    for asset in visual["packs"]:
        if not (out/"inputs"/asset["path"]).is_file():
            raise ValueError("Retinal SensorPack dependency is absent: "+asset["path"])
    p=read_policy_pack(out/"inputs/first.policypack")
    stochastic=replace(p,id="crow-recurrent-ppo-development",revision=31,action_log_standard_deviation=np.full(p.action_count,-2.5,np.float32))
    publish(stochastic,out/"stochastic-seed.policypack")
    base=[str(root/"build/bin/metalrobo_task_rollout"),"--metallib",str(root/"build/shaders/MetalRobo.metallib"),
        "--birdflow-american-crow-journey","--birdflow-journey-variant","v13-retinal-feeding","--envs","2",
        "--repeats","1","--no-scheduled-resets","--minimum-difficulty-band","0","--maximum-difficulty-band","0",
        "--seed","2651109503","--visual-observation-config",str(out/"inputs/retinal-observation.json")]
    for name,ticks in (("stochastic-start",37),("stochastic-resume",13)):
        command=base+["--policy-pack",str(out/"stochastic-seed.policypack"),"--steps",str(ticks),"--chunk","8",
            "--rollout-pack",str(out/(name+".rolloutpack")),"--save-life-checkpoint",str(out/(name+".lifecheckpoint"))]
        if "resume" in name:command += ["--resume-life-checkpoint",str(out/"stochastic-start.lifecheckpoint")]
        native=execute(out,name,command,root,120)
        if native["failed_environment_steps"]:raise RuntimeError("Native stochastic recurrent rollout failed")
    r=read_policy_rollout_pack(out/"stochastic-start.rolloutpack")
    tail=read_policy_rollout_pack(out/"stochastic-resume.rolloutpack")
    config=MLXRecurrentPPOConfiguration(sequence_length=8,minibatch_size=16,update_epochs=2,
        learning_rate=.0003,adaptive_learning_rate=False,target_kl=None,seed=2651109503)
    learner=MLXRecurrentPolicyLearner.from_policy_pack(out/"stochastic-seed.policypack",config)
    batch=MLXRecurrentPolicyBatch.from_rollout(r)
    windows=batch.windows(8);counts=np.zeros(r.sample_count,np.uint32)
    for window in windows:
        tables,indices,valid=batch.gather(window[None],8);counts[indices[valid]]+=1
        if not np.all(np.diff(indices[valid])==r.environment_count):raise RuntimeError("Sequence lost lane/time continuity")
    if not np.all(counts==1):raise RuntimeError("Sequence coverage duplicated or omitted a real transition")
    # A single padded tail exercises gradients independently from full windows.
    tables,indices,valid=batch.gather(windows[-2:],8)
    means,_=learner.model.actor_mean(mx.array(tables["actor_observations"]),mx.array(tables["initial"]),mx.array(tables["resets"]))
    mx.eval(means)
    adv=(batch.advantages-batch.advantages.mean())/(batch.advantages.std()+1e-8)
    args=[tables["actor_observations"],tables["critic_observations"],tables["latents"],tables["old_log_probabilities"],
        np.array(means),np.array(learner.model.log_standard_deviation),tables["old_values"],
        np.where(valid,adv.reshape(-1)[indices],0),tables["returns"],tables["initial"],tables["resets"],tables["valid"]]
    (loss,metrics),grads=learner._loss_and_gradient(*[mx.array(x) for x in args]);mx.eval(loss,metrics,grads)
    parameters={k:np.array(v,dtype=np.float64) for k,v in tree_flatten(learner.model.trainable_parameters())}
    references=[];refargs=[np.asarray(x,dtype=np.float64) if np.asarray(x).dtype!=bool else x for x in args]
    for name,gradient in tree_flatten(grads):
        g=np.array(gradient);coordinate=np.unravel_index(np.argmax(np.abs(g)),g.shape);original=parameters[name][coordinate];eps=1e-5
        parameters[name][coordinate]=original+eps;plus=fp64_loss(parameters,stochastic,refargs,config)
        parameters[name][coordinate]=original-eps;minus=fp64_loss(parameters,stochastic,refargs,config)
        parameters[name][coordinate]=original;reference=(plus-minus)/(2*eps);error=abs(reference-float(g[coordinate]))
        references.append({"parameter":name,"coordinate":[int(x) for x in coordinate],"MLX":float(g[coordinate]),"FP64":reference,"abs_error":error})
    write(out/"gradient-reference.json",references)
    if max(x["abs_error"] for x in references)>3e-5:raise RuntimeError("FP64 PPO gradient error exceeded preregistered bound")
    real=int(windows[-1,2]);short=[mx.array(x[:real] if np.asarray(x).ndim>=2 and i not in (5,9) else x) for i,x in enumerate(args)]
    (_, _),shortgrads=learner._loss_and_gradient(*short);mx.eval(shortgrads)
    padding_error=max(float(np.max(np.abs(np.array(a)-np.array(b)))) for (_,a),(_,b) in zip(tree_flatten(grads),tree_flatten(shortgrads),strict=True))
    if padding_error>3e-6:raise RuntimeError("Padded sequence changed real-transition gradients")
    x=mx.array(r.actor_observations.reshape(37,2,p.actor_observation_count)[:8,:1]);h=mx.zeros((1,7));resets=mx.array(np.array([[t==4] for t in range(8)]))
    observation_grad=mx.grad(lambda observations:mx.sum(learner.model.actor_mean(observations,h,resets)[0][-1]))(x)
    hidden_grad=mx.grad(lambda initial:mx.sum(learner.model.actor_mean(x,initial,resets)[0][-1]))(h)
    mx.eval(observation_grad,hidden_grad)
    if np.any(np.array(observation_grad)[:4]!=0) or np.any(np.array(hidden_grad)!=0) or not np.any(np.array(observation_grad)[4:]!=0):
        raise RuntimeError("Recurrent temporal/reset/initial-state gradient cut failed")
    rejections=[]
    def rejects(name,call):
        before=state(learner)
        try:call()
        except ValueError as error:rejections.append({"name":name,"error":str(error)})
        else:raise RuntimeError("Invalid recurrent contract accepted: "+name)
        if not same(before,state(learner)):raise RuntimeError("Rejected update changed model/Adam state")
    rejects("deterministic-seed",lambda:MLXRecurrentPolicyLearner.from_policy_pack(out/"inputs/first.policypack",config))
    manual=out/"inputs/manual-reset.rolloutpack"
    if not manual.exists():manual=out/"inputs/segment-first.rolloutpack"
    rejects("manual-reset-without-terminal",lambda:MLXRecurrentPolicyBatch.from_rollout(read_policy_rollout_pack(manual)))
    rejects("flat-batch",lambda:learner.update(object()))
    rejects("stale-revision",lambda:learner.update(replace(batch,rollout=replace(r,policy_revision=1))))
    rejects("changed-GAE",lambda:learner.update(replace(batch,returns=batch.returns+1)))
    bad=r.transitions.copy();bad["physics_error"][0]=1
    rejects("physics-error",lambda:MLXRecurrentPolicyBatch.from_rollout(replace(r,transitions=bad)))
    invalid=r.memory_before.copy();invalid[0]=np.nan
    rejects("nonfinite-hidden",lambda:MLXRecurrentPolicyBatch.from_rollout(replace(r,memory_before=invalid)))
    rejects("wrong-hidden-width",lambda:learner.update(replace(batch,rollout=replace(r,memory_hidden_count=6))))
    initial=state(learner);result=learner.update(batch);after=state(learner)
    changed=[k for k in initial if k.startswith("model.actor.memory.") and not np.array_equal(initial[k],after[k])]
    if len(changed)!=4:raise RuntimeError("Adam did not train all four GRU parameter families")
    learner.write_policy_pack(out/"trained.policypack");_write_learner_state(learner,out/"trained.safetensors")
    recovered=MLXRecurrentPolicyLearner.from_policy_pack(out/"trained.policypack",config)
    if not _restore_learner_state(recovered,out/"trained.safetensors") or not same(state(learner),state(recovered)):
        raise RuntimeError("Recurrent model/Adam restoration is not exact")
    native=execute(out,"post-update",base+["--policy-pack",str(out/"trained.policypack"),"--steps","13","--chunk","8",
        "--rollout-pack",str(out/"post-update.rolloutpack")],root,120)
    if native["failed_environment_steps"]:raise RuntimeError("Native updated recurrent policy failed")
    next_batch=MLXRecurrentPolicyBatch.from_rollout(read_policy_rollout_pack(out/"post-update.rolloutpack"))
    a=learner.update(next_batch);b=recovered.update(next_batch)
    if not same(state(learner),state(recovered)):raise RuntimeError("Resumed recurrent next update differs")
    learner.write_policy_pack(out/"next.policypack");recovered.write_policy_pack(out/"next-recovered.policypack")
    if (out/"next.policypack").read_bytes()!=(out/"next-recovered.policypack").read_bytes():raise RuntimeError("Resumed recurrent policy bytes differ")
    write(out/"rejections.json",rejections)
    summary={"status":"passed","native_stochastic_samples":r.sample_count+tail.sample_count,
        "sequence_windows":len(windows),"sequence_coverage_exact":True,"padding_gradient_max_abs":padding_error,
        "FP64_PPO_gradient_max_abs":max(x["abs_error"] for x in references),"gradient_parameter_families":len(references),
        "reset_and_initial_gradient_cuts_exact":True,"GRU_parameter_families_trained":changed,
        "optimizer_restore_and_next_update_exact":True,"policy_recovery_bytes_exact":True,
        "nonzero_life_continuation_hidden_norm":float(np.linalg.norm(tail.memory_before[:14])),
        "rejection_controls":len(rejections),"first_update":result,"next_update":a,
        "ecological_or_learned_memory_qualified":False}
    write(out/"summary.json",summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    try:print(json.dumps(run(root,out),indent=2))
    except Exception as error:
        if not (out/"failure.json").exists():write(out/"failure.json",{"type":type(error).__name__,"error":str(error)})
        raise
if __name__=="__main__":main()
