"""Contiguous truncated-BPTT PPO over native recurrent rollout artifacts.

Only learner algebra lives here. Swift/Metal owns the world, episode identity,
collection and recurrent state. Captured initial states are detached; sequence
windows are shuffled as units, and padding has no loss or optimizer authority.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from .gru_memory import PARAMETER_NAMES, gru_sequence
from .mlx_policy_learning import (MLXActorCritic, MLXPPOConfiguration,
    MLXPolicyLearner, NativePolicyDenseLayer, NativePolicyPack, NativePolicyRollout, read_policy_pack)
from .native import PolicyDenseLayerArtifact, PolicyGRUMemoryArtifact, write_policy_pack


@dataclass(frozen=True, slots=True)
class MLXRecurrentPPOConfiguration(MLXPPOConfiguration):
    sequence_length: int = 32
    adaptive_kl_mode: str = "legacy-pre-step"

    def validate(self) -> None:
        super(MLXRecurrentPPOConfiguration, self).validate()
        if self.sequence_length <= 0 or self.minibatch_size < self.sequence_length:
            raise ValueError("Recurrent PPO requires positive sequence_length <= minibatch_size")
        if self.adaptive_kl_mode not in ("legacy-pre-step", "post-step-decision-groups"):
            raise ValueError("Unknown recurrent adaptive KL mode")


@dataclass(frozen=True, slots=True)
class MLXRecurrentPolicyBatch:
    rollout: NativePolicyRollout
    advantages: np.ndarray
    returns: np.ndarray

    @classmethod
    def from_rollout(cls, rollout: NativePolicyRollout, *, discount: float = .99,
                     gae_lambda: float = .95) -> "MLXRecurrentPolicyBatch":
        if not 0 <= discount <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("Recurrent GAE discount/lambda must be in [0, 1]")
        T, N, H = rollout.control_step_count, rollout.environment_count, rollout.memory_hidden_count
        if min(T, N, H) <= 0:
            raise ValueError("Contiguous sequence PPO requires a recurrent rollout")
        rollout.action_selections()  # Validate even directly constructed batches.
        hidden = rollout.memory_before.reshape(T, N, H)
        resets = rollout.memory_resets.reshape(T, N)
        transitions = rollout.transitions.reshape(T, N)
        done = transitions["done"].astype(bool)
        timeout = transitions["timeout"].astype(bool)
        if (np.any(resets > 1) or not np.isfinite(hidden).all()
                or np.any(np.abs(hidden) > np.float32(1.000001))
                or np.any(hidden[resets.astype(bool)] != 0)):
            raise ValueError("Recurrent captured hidden/reset state is invalid")
        if (np.any(transitions["done"] > 1) or np.any(transitions["timeout"] > 1)
                or np.any(transitions["policy_revision"] != rollout.policy_revision)
                or np.any(transitions["physics_error"]) or np.any(timeout & ~done)):
            raise ValueError("Recurrent rollout contains a physics failure or nonterminal timeout")
        # A manual state reset without its terminal transition has no recorded
        # pre-reset bootstrap value. Refuse to invent a return for that edge.
        if np.any(resets[1:].astype(bool) != done[:-1]):
            raise ValueError("Recurrent reset/terminal boundary mismatch; pre-reset bootstrap is unavailable")
        if rollout.teacher_actions.size:
            raise ValueError("Recurrent PPO does not accept flattened teacher/retention targets")
        values = rollout.old_values.reshape(T, N)
        rewards = transitions["reward"]
        advantages = np.empty((T, N), np.float32)
        running = np.zeros(N, np.float32)
        for t in range(T - 1, -1, -1):
            next_value = rollout.bootstrap_values if t == T - 1 else values[t + 1]
            continuing = (~done[t]).astype(np.float32)
            bootstrap = timeout[t] * transitions["timeout_bootstrap_value"][t]
            delta = rewards[t] + discount * (bootstrap + continuing * next_value) - values[t]
            running = delta + discount * gae_lambda * continuing * running
            advantages[t] = running
        result = cls(rollout, advantages, advantages + values)
        if not np.isfinite(result.advantages).all() or not np.isfinite(result.returns).all():
            raise ValueError("Recurrent GAE produced nonfinite values")
        return result

    def windows(self, length: int) -> np.ndarray:
        """Rows are (first tick, lane, real length), each transition once."""
        T, N = self.rollout.control_step_count, self.rollout.environment_count
        if length <= 0:
            raise ValueError("Sequence length must be positive")
        return np.array([(start, lane, min(length, T-start))
                         for start in range(0, T, length) for lane in range(N)], dtype=np.int64)

    def gather(self, windows: np.ndarray, length: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        r = self.rollout
        T, N, H = r.control_step_count, r.environment_count, r.memory_hidden_count
        starts, lanes, sizes = windows.T
        valid = np.arange(length)[:, None] < sizes[None, :]
        ticks = np.minimum(starts[None, :] + np.arange(length)[:, None], T-1)
        indices = ticks * N + lanes[None, :]
        def take(array):
            value = np.asarray(array)[indices]
            mask = valid.reshape(valid.shape + (1,) * (value.ndim-2))
            return np.where(mask, value, np.zeros((), value.dtype))
        tables = {"actor_observations": take(r.actor_observations.reshape(r.sample_count,r.actor_observation_count)),
                  "critic_observations": take(r.critic_observations.reshape(r.sample_count,r.critic_observation_count)),
                  "latents": take(r.latents.reshape(r.sample_count,r.action_count)),
                  "old_log_probabilities": take(r.old_log_probabilities), "old_values": take(r.old_values)}
        tables["advantages"] = take(self.advantages.reshape(-1))
        tables["returns"] = take(self.returns.reshape(-1))
        tables["valid"] = valid.astype(np.float32)
        tables["initial"] = r.memory_before.reshape(T, N, H)[starts, lanes]
        tables["resets"] = np.where(valid, r.memory_resets.reshape(T, N)[ticks, lanes], 1).astype(bool)
        return tables, indices, valid


class _Dense(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.weight = mx.array(layer.weights)
        self.bias = mx.array(layer.bias)
        self.activation = layer.activation

    def __call__(self, x):
        x = x @ self.weight.T + self.bias
        if self.activation == 1: return mx.maximum(x, 0)
        if self.activation == 2: return mx.tanh(x)
        if self.activation == 3: return mx.where(x >= 0, x, mx.expm1(mx.minimum(x, 0)))
        if self.activation == 4: return x * mx.sigmoid(x)
        return x


class _RecurrentActor(nn.Module):
    def __init__(self, pack: NativePolicyPack):
        super().__init__()
        memory = pack.actor_memory
        self.encoder = [_Dense(x) for x in pack.layers[:memory.after_layer]]
        self.decoder = [_Dense(x) for x in pack.layers[memory.after_layer:]]
        self.memory = {key: mx.array(getattr(memory, key)) for key in PARAMETER_NAMES}

    def __call__(self, observations, initial, resets):
        x = observations
        for layer in self.encoder: x = layer(x)
        x, before = gru_sequence(x, initial, resets, self.memory)
        for layer in self.decoder: x = layer(x)
        return x, before


class MLXRecurrentActorCritic(MLXActorCritic):
    def __init__(self, pack: NativePolicyPack):
        nn.Module.__init__(self)
        self.actor = _RecurrentActor(pack)
        self.critic = [_Dense(x) for x in pack.critic_layers]
        self.actor_observation_mean = mx.array(pack.effective_observation_mean)
        self.actor_observation_inverse_standard_deviation = mx.array(pack.effective_observation_inverse_standard_deviation)
        self.critic_observation_mean = mx.array(pack.effective_critic_observation_mean)
        self.critic_observation_inverse_standard_deviation = mx.array(pack.effective_critic_observation_inverse_standard_deviation)
        self.log_standard_deviation = mx.array(pack.action_log_standard_deviation)
        self.observation_clip = pack.observation_clip
        self.freeze(recurse=False, keys=["actor_observation_mean", "actor_observation_inverse_standard_deviation",
            "critic_observation_mean", "critic_observation_inverse_standard_deviation"], strict=True)

    def actor_mean(self, observations, initial=None, resets=None):
        if initial is None or resets is None:
            raise ValueError("Recurrent actor requires contiguous sequence initial/reset state")
        x = mx.clip((observations-self.actor_observation_mean)*self.actor_observation_inverse_standard_deviation,
                    -self.observation_clip, self.observation_clip)
        return self.actor(x, initial, resets)

    def value(self, observations):
        x = mx.clip((observations-self.critic_observation_mean)*self.critic_observation_inverse_standard_deviation,
                    -self.observation_clip, self.observation_clip)
        for layer in self.critic: x = layer(x)
        return x.squeeze(-1)


class MLXRecurrentPolicyLearner(MLXPolicyLearner):
    recurrent = True

    def _create_model(self, *args):
        return MLXRecurrentActorCritic(self._source_pack)

    @classmethod
    def from_policy_pack(cls, path: str | Path,
                         configuration: MLXRecurrentPPOConfiguration = MLXRecurrentPPOConfiguration(),
                         *, library_path=None):
        pack = read_policy_pack(path, library_path=library_path)
        return cls._from_pack(pack, configuration)

    @classmethod
    def _from_pack(cls, pack, configuration):
        memory = pack.actor_memory
        if memory is None or not pack.critic_layers or pack.action_log_standard_deviation.size != pack.action_count:
            raise ValueError("Recurrent PPO requires a stochastic recurrent PolicyPack with critic")
        if pack.layers[memory.after_layer-1].activation != 2 or memory.maximum_absolute_input < 1:
            raise ValueError("Trainable recurrent encoder must end in tanh with a bound >=1")
        if pack.layers[-1].activation != 0:
            raise ValueError("Recurrent Gaussian actor requires an identity output")
        if not isinstance(configuration, MLXRecurrentPPOConfiguration):
            raise ValueError("Recurrent PPO requires sequence configuration")
        configuration = replace(configuration, observation_clip=pack.observation_clip,
            hidden_sizes=tuple(x.output_count for x in pack.layers[:-1]),
            critic_hidden_sizes=tuple(x.output_count for x in pack.critic_layers[:-1]))
        learner = cls.__new__(cls)
        learner._source_pack = pack
        MLXPolicyLearner.__init__(learner, pack.actor_observation_count,
            pack.critic_observation_count, pack.action_count, configuration)
        learner.policy_id, learner.revision = pack.id, pack.revision
        for name in ("contract_version", "world_fingerprint", "task_fingerprint", "observation_fingerprint", "action_fingerprint"):
            setattr(learner, name, getattr(pack, name))
        learner.set_action_contract(action_bias=pack.effective_action_bias,
            action_scale=pack.effective_action_scale, action_clip=pack.action_clip)
        return learner

    @classmethod
    def initialize(cls, actor_observation_count: int, critic_observation_count: int,
                   action_count: int, hidden_count: int,
                   configuration: MLXRecurrentPPOConfiguration = MLXRecurrentPPOConfiguration(),
                   *, maximum_memory_timescale_ticks: int = 0):
        """Seed a bounded tanh encoder, native GRU and Gaussian actor/critic.

        Dense and GRU rows use a seeded fan-in uniform initializer; biases are
        zero. The last encoder tanh guarantees the native cell's input bound.
        This authors learner parameters, never simulator or recurrent life state.
        """
        if not isinstance(configuration, MLXRecurrentPPOConfiguration):
            raise ValueError("Recurrent initialization requires sequence configuration")
        configuration.validate()
        if min(actor_observation_count, critic_observation_count, action_count, hidden_count) <= 0:
            raise ValueError("Recurrent policy dimensions must be positive")
        if not configuration.hidden_sizes:
            raise ValueError("Recurrent initialization requires a nonempty tanh encoder")
        if (not isinstance(maximum_memory_timescale_ticks, int) or isinstance(maximum_memory_timescale_ticks, bool)
                or maximum_memory_timescale_ticks not in (0,) and not 2 <= maximum_memory_timescale_ticks <= 4096):
            raise ValueError("Maximum initial memory timescale must be 0 (legacy) or an integer in 2..4096 accepted ticks")
        generator = np.random.default_rng(configuration.seed)
        def weights(rows, columns):
            bound = 1 / math.sqrt(columns)
            return generator.uniform(-bound, bound, (rows, columns)).astype(np.float32)
        def dense(inputs, outputs, activation):
            return NativePolicyDenseLayer(inputs, outputs, activation,
                weights(outputs, inputs), np.zeros(outputs, np.float32))
        encoder = []
        width = actor_observation_count
        for output in configuration.hidden_sizes:
            encoder.append(dense(width, output, 2))
            width = output
        memory = PolicyGRUMemoryArtifact(len(encoder), weights(3*hidden_count, width),
            weights(3*hidden_count, hidden_count), np.zeros(3*hidden_count, np.float32),
            np.zeros(3*hidden_count, np.float32), 1.)
        if maximum_memory_timescale_ticks:
            # For an isolated unforced cell, z=exp(-1/tau) preserves h with
            # exact nominal exponential timescale tau. A log-spaced population
            # retains short and long responses. Ordinary learned gates remain
            # fully trainable; this changes initialization only, not execution.
            tau = np.geomspace(2., float(maximum_memory_timescale_ticks), hidden_count)
            memory.input_bias[hidden_count:2*hidden_count] = -np.log(np.expm1(1./tau)).astype(np.float32)
        layers = tuple(encoder + [dense(hidden_count, action_count, 0)])
        critic = []
        width = critic_observation_count
        critic_sizes = (configuration.critic_hidden_sizes if configuration.critic_hidden_sizes is not None
                        else configuration.hidden_sizes)
        for output in critic_sizes:
            critic.append(dense(width, output, 3))
            width = output
        critic.append(dense(width, 1, 0))
        empty = np.empty(0, np.float32)
        pack = NativePolicyPack(id="", revision=1, contract_version=0,
            world_fingerprint=0, task_fingerprint=0, observation_fingerprint=0, action_fingerprint=0,
            observation_mean=empty, observation_inverse_standard_deviation=empty, layers=layers,
            critic_observation_mean=empty, critic_observation_inverse_standard_deviation=empty,
            critic_layers=tuple(critic), action_log_standard_deviation=np.full(action_count,
                configuration.initial_log_standard_deviation, np.float32), action_bias=empty,
            action_scale=empty, observation_clip=configuration.observation_clip,
            action_clip=float(np.finfo(np.float32).max), content_hash=0, format_version=5, actor_memory=memory)
        return cls._from_pack(pack, configuration)

    def zero_actor_output(self):
        output = self.model.actor.decoder[-1]
        output.weight, output.bias = mx.zeros_like(output.weight), mx.zeros_like(output.bias)
        self.optimizer = optim.Adam(learning_rate=self.configuration.learning_rate, bias_correction=True)
        self.optimizer.init(self.model.trainable_parameters())
        self.refresh_compiled_training_state()
        mx.eval(self.model.parameters(), self.optimizer.state)

    def recurrent_contract(self):
        p = self._source_pack
        return json.dumps({"schema": 1, "after_layer": p.actor_memory.after_layer,
            "maximum_absolute_input": p.actor_memory.maximum_absolute_input,
            "actor_layers": [(x.input_count, x.output_count, x.activation) for x in p.layers],
            "critic_layers": [(x.input_count, x.output_count, x.activation) for x in p.critic_layers],
            "actor_normalization": [np.array(self.model.actor_observation_mean).tolist(),
                np.array(self.model.actor_observation_inverse_standard_deviation).tolist()],
            "critic_normalization": [np.array(self.model.critic_observation_mean).tolist(),
                np.array(self.model.critic_observation_inverse_standard_deviation).tolist()],
            "action_bias": self.action_bias.tolist(), "action_scale": self.action_scale.tolist(),
            "action_clip": self.action_clip, "semantic_contract": [self.contract_version,
                self.world_fingerprint, self.task_fingerprint, self.observation_fingerprint, self.action_fingerprint]},
            sort_keys=True, separators=(",", ":"), allow_nan=False)

    def _loss(self, actor, critic, latents, old_logp, old_means, old_logstd,
              old_values, advantages, returns, initial, resets, valid, selections=None):
        means, _ = self.model.actor_mean(actor, initial, resets)
        values = self.model.value(critic)
        logstd = mx.clip(self.model.log_standard_deviation, -5, 2)
        count = mx.maximum(mx.sum(valid), 1)
        average = lambda x: mx.sum(x * valid) / count
        if selections is None:
            logp = self.model.log_probability(means, logstd, latents)
            entropy = self.model.entropy(means, logstd)
            actor_average = average
        else:
            components = lambda m, s: -.5*(mx.square((latents-m)*mx.exp(-s))+2*s+math.log(2*math.pi))
            logp = mx.sum(components(means, logstd)*selections, axis=-1)
            old_logp = mx.sum(components(old_means, old_logstd)*selections, axis=-1)
            entropy = mx.sum((.5*(1+math.log(2*math.pi))+logstd)*selections, axis=-1)
            actor_valid = valid*(mx.sum(selections, axis=-1)>0)
            actor_count = mx.maximum(mx.sum(actor_valid), 1)
            actor_average = lambda x: mx.sum(x*actor_valid)/actor_count
        log_ratio = logp-old_logp
        ratio = mx.exp(log_ratio)
        policy = -actor_average(mx.minimum(ratio*advantages,
            mx.clip(ratio, 1-self.configuration.clip_ratio, 1+self.configuration.clip_ratio)*advantages))
        clipped_values = old_values + mx.clip(values-old_values,
            -self.configuration.clip_ratio, self.configuration.clip_ratio)
        value = average(mx.maximum(mx.square(values-returns), mx.square(clipped_values-returns)))
        kl_components = logstd-old_logstd + .5*(mx.exp(2*(old_logstd-logstd))
            + mx.square(old_means-means)*mx.exp(-2*logstd)-1)
        kl = actor_average(mx.sum(kl_components if selections is None else kl_components*selections, axis=-1))
        entropy_mean = actor_average(entropy)
        loss = policy+self.configuration.value_coefficient*value-self.configuration.entropy_coefficient*entropy_mean
        return loss, {"loss": loss, "policy_loss": policy, "value_loss": value, "entropy": entropy_mean,
            "kl_divergence": kl, "approximate_kl": actor_average(ratio-1-log_ratio),
            "clip_fraction": actor_average((mx.abs(ratio-1)>self.configuration.clip_ratio).astype(mx.float32))}

    def _training_step(self, *args):
        (loss, metrics), gradients = self._loss_and_gradient(*args)
        _, metrics["actor_gradient_norm"] = optim.clip_grad_norm(
            {"actor": gradients["actor"], "log_standard_deviation": gradients["log_standard_deviation"]},
            self.configuration.maximum_gradient_norm)
        _, metrics["critic_gradient_norm"] = optim.clip_grad_norm(
            {"critic": gradients["critic"]}, self.configuration.maximum_gradient_norm)
        gradients, metrics["gradient_norm"] = optim.clip_grad_norm(gradients, self.configuration.maximum_gradient_norm)
        # Preserve the optimizer algebra. Report the joint clipping actually
        # applied, then measure Adam's parameter change (moments make it
        # different from learning_rate times the clipped gradient).
        metrics["gradient_clip_scale"] = mx.minimum(1.0,
            self.configuration.maximum_gradient_norm / (metrics["gradient_norm"] + 1e-6))
        _, metrics["clipped_actor_gradient_norm"] = optim.clip_grad_norm(
            {"actor": gradients["actor"], "log_standard_deviation": gradients["log_standard_deviation"]},
            self.configuration.maximum_gradient_norm)
        before = dict(tree_flatten({"actor": self.model.actor.trainable_parameters(),
            "log_standard_deviation": self.model.log_standard_deviation}))
        self.optimizer.update(self.model, gradients)
        after = dict(tree_flatten({"actor": self.model.actor.trainable_parameters(),
            "log_standard_deviation": self.model.log_standard_deviation}))
        metrics["actor_parameter_delta_norm"] = mx.sqrt(sum(mx.sum(mx.square(after[k]-v)) for k,v in before.items()))
        return loss, metrics

    def _update_impl(self, batch):
        if not isinstance(batch, MLXRecurrentPolicyBatch):
            raise ValueError("Recurrent PPO requires a contiguous recurrent batch, never flat samples")
        r = batch.rollout
        if (r.actor_observation_count, r.critic_observation_count, r.action_count,
                r.memory_hidden_count, r.policy_revision) != (self.actor_observation_count,
                self.critic_observation_count, self.action_count,
                self._source_pack.actor_memory.recurrent_weights.shape[1], self.revision):
            raise ValueError("Recurrent rollout dimensions/behavior revision disagree with learner")
        # Rebuild derived targets so direct dataclass construction cannot bypass
        # the reset, terminal, physics and return contract.
        checked = MLXRecurrentPolicyBatch.from_rollout(r, discount=self.configuration.discount,
                                                       gae_lambda=self.configuration.gae_lambda)
        if not np.array_equal(checked.advantages, batch.advantages) or not np.array_equal(checked.returns, batch.returns):
            raise ValueError("Recurrent batch targets disagree with terminal-safe GAE")
        length = min(self.configuration.sequence_length, r.control_step_count)
        windows = batch.windows(length)
        per_batch = max(1, self.configuration.minibatch_size // length)
        old_means = np.empty((r.sample_count, self.action_count), np.float32)
        old_logstd = mx.clip(self.model.log_standard_deviation, -5, 2)
        max_state_error = max_logp_error = max_value_error = 0.
        # Freeze the generating distribution once, before any Adam update.
        for offset in range(0, len(windows), per_batch):
            tables, indices, valid = batch.gather(windows[offset:offset+per_batch], length)
            means, before = self.model.actor_mean(mx.array(tables["actor_observations"]),
                mx.array(tables["initial"]), mx.array(tables["resets"]))
            logp = self.model.log_probability(means, old_logstd, mx.array(tables["latents"]))
            values = self.model.value(mx.array(tables["critic_observations"]))
            mx.eval(means, before, logp, values)
            max_state_error = max(max_state_error, float(np.max(np.abs(np.array(before)[valid]-r.memory_before.reshape(r.sample_count,r.memory_hidden_count)[indices[valid]]))))
            max_logp_error = max(max_logp_error, float(np.max(np.abs(np.array(logp)[valid]-r.old_log_probabilities[indices[valid]]))))
            max_value_error = max(max_value_error, float(np.max(np.abs(np.array(values)[valid]-r.old_values[indices[valid]]))))
            old_means[indices[valid]] = np.array(means)[valid]
        if max_state_error > 5e-5 or max_value_error > 5e-5 or max_logp_error > 5e-4:
            raise ValueError("Recurrent generating policy/hidden/value/log-probability parity failed")
        selections = r.action_selections()
        if selections is None:
            normalized = (batch.advantages-np.mean(batch.advantages))/(np.std(batch.advantages)+1e-8)
        else:
            decision = np.any(selections != 0, axis=-1).reshape(batch.advantages.shape)
            chosen = batch.advantages[decision]
            normalized = np.zeros_like(batch.advantages)
            if chosen.size:
                normalized[decision] = (chosen-np.mean(chosen))/(np.std(chosen)+1e-8)
        actor_before = {k: np.array(v) for k,v in tree_flatten({
            "actor": self.model.actor.trainable_parameters(),
            "log_standard_deviation": self.model.log_standard_deviation})}
        totals = {}; updates = 0; weight = 0; minimum_clip_scale = 1.0
        start_time = time.perf_counter()
        for epoch in range(self.configuration.update_epochs):
            order = np.random.default_rng(self._permutation_seed(self.configuration.seed, self.revision, epoch)).permutation(len(windows))
            for offset in range(0, len(windows), per_batch):
                tables, indices, valid = batch.gather(windows[order[offset:offset+per_batch]], length)
                means = np.where(valid[...,None], old_means[indices], 0)
                adv = np.where(valid, normalized.reshape(-1)[indices], 0)
                args = [tables["actor_observations"], tables["critic_observations"], tables["latents"],
                    tables["old_log_probabilities"], means, old_logstd, tables["old_values"], adv,
                    tables["returns"], tables["initial"], tables["resets"], tables["valid"]]
                if selections is not None:
                    args.append(np.where(valid[...,None], selections[indices], 0))
                loss, metrics = self._compiled_training_step(*[mx.array(x) for x in args])
                mx.eval(loss, metrics, self.model.state, self.optimizer.state)
                numeric = {key: float(value.item()) for key, value in metrics.items()}
                if not all(math.isfinite(x) for x in numeric.values()):
                    raise FloatingPointError("Recurrent PPO produced nonfinite minibatch metrics")
                count = int(valid.sum()); weight += count; updates += 1
                minimum_clip_scale = min(minimum_clip_scale, numeric["gradient_clip_scale"])
                if (self.configuration.adaptive_kl_mode == "legacy-pre-step" and
                        self.configuration.adaptive_learning_rate and self.configuration.target_kl is not None):
                    lr = float(self.optimizer.learning_rate.item()); kl = numeric["kl_divergence"]
                    if kl > 2*self.configuration.target_kl: lr = max(self.configuration.minimum_learning_rate, lr/1.5)
                    elif 0 < kl < .5*self.configuration.target_kl: lr = min(self.configuration.maximum_learning_rate, lr*1.5)
                    self.optimizer.learning_rate = lr
                numeric["learning_rate"] = float(self.optimizer.learning_rate.item())
                for key, value in numeric.items(): totals[key] = totals.get(key, 0.)+value*count
        # Audit the final published distribution, after *every* optimizer
        # step. Factor metrics average only its selected decisions; the rare
        # head factor cannot be diluted by frequent jaw decisions.
        post_kl = np.zeros(r.action_count, np.float64)
        post_approx = np.zeros(r.action_count, np.float64)
        post_clip = np.zeros(r.action_count, np.float64)
        factor_counts = np.zeros(r.action_count, np.int64)
        pooled_kl = 0.0; pooled_count = 0
        group_kl = {}; group_counts = {}
        learning_rate_used = float(self.optimizer.learning_rate.item())
        final_logstd = mx.clip(self.model.log_standard_deviation, -5, 2)
        for offset in range(0, len(windows), per_batch):
            tables, indices, valid = batch.gather(windows[offset:offset+per_batch], length)
            means, _ = self.model.actor_mean(mx.array(tables["actor_observations"]),
                mx.array(tables["initial"]), mx.array(tables["resets"]))
            old = mx.array(np.where(valid[...,None], old_means[indices], 0))
            latent = mx.array(tables["latents"])
            kl = final_logstd-old_logstd + .5*(mx.exp(2*(old_logstd-final_logstd))
                + mx.square(old-means)*mx.exp(-2*final_logstd)-1)
            log_ratio = -.5*(mx.square((latent-means)*mx.exp(-final_logstd))
                - mx.square((latent-old)*mx.exp(-old_logstd)) + 2*(final_logstd-old_logstd))
            approx = mx.expm1(log_ratio)-log_ratio
            clip = (mx.abs(mx.expm1(log_ratio))>self.configuration.clip_ratio).astype(mx.float32)
            mx.eval(kl, approx, clip)
            kl_values = np.array(kl)
            selected = np.broadcast_to(valid[...,None], valid.shape+(r.action_count,)).copy()
            if selections is not None: selected &= selections[indices].astype(bool)
            selected_kl = np.where(selected,kl_values,0)
            post_kl += np.sum(selected_kl,axis=(0,1),dtype=np.float64)
            post_approx += np.sum(np.where(selected,np.array(approx),0),axis=(0,1),dtype=np.float64)
            post_clip += np.sum(np.where(selected,np.array(clip),0),axis=(0,1),dtype=np.float64)
            factor_counts += np.sum(selected,axis=(0,1))
            decisions = np.any(selected,axis=-1)
            pooled_kl += float(np.sum(selected_kl,dtype=np.float64))
            pooled_count += int(np.count_nonzero(decisions))
            # Condition on the complete native selection mask, rather than
            # averaging a rare head choice together with frequent jaw choices.
            # Ordinary recurrent packs select every action on every real row.
            if selections is None:
                masks = None
                groups = ((1 << r.action_count)-1,)
            else:
                masks = np.sum(selected.astype(np.uint32) <<
                    np.arange(r.action_count,dtype=np.uint32),axis=-1,dtype=np.uint32)
                groups = np.unique(masks)
            joint_kl = np.sum(selected_kl,axis=-1,dtype=np.float64)
            for mask in groups:
                mask = int(mask)
                if not mask: continue
                rows = valid if masks is None else valid & (masks == mask)
                group_kl[mask] = group_kl.get(mask,0.) + float(np.sum(joint_kl[rows],dtype=np.float64))
                group_counts[mask] = group_counts.get(mask,0) + int(np.count_nonzero(rows))
        post = {"post_update_kl_divergence": pooled_kl/max(pooled_count,1),
            "minimum_gradient_clip_scale": minimum_clip_scale}
        for action in range(r.action_count):
            post[f"post_update_action_{action}_decision_count"] = float(factor_counts[action])
            if factor_counts[action]:
                post[f"post_update_action_{action}_kl"] = float(post_kl[action]/factor_counts[action])
                post[f"post_update_action_{action}_approximate_kl"] = float(post_approx[action]/factor_counts[action])
                post[f"post_update_action_{action}_clip_fraction"] = float(post_clip[action]/factor_counts[action])
        for mask,count in group_counts.items():
            post[f"post_update_decision_group_{mask}_count"] = float(count)
            post[f"post_update_decision_group_{mask}_kl"] = group_kl[mask]/count
        post["post_update_max_decision_group_kl"] = max(
            (group_kl[mask]/count for mask,count in group_counts.items()), default=0.)
        actor_after = dict(tree_flatten({"actor": self.model.actor.trainable_parameters(),
            "log_standard_deviation": self.model.log_standard_deviation}))
        post["actor_total_parameter_delta_norm"] = math.sqrt(sum(float(np.sum(
            np.square(np.array(actor_after[k],dtype=np.float64)-v))) for k,v in actor_before.items()))
        if not all(math.isfinite(value) for value in post.values()):
            raise FloatingPointError("Recurrent PPO final-policy metrics are nonfinite")
        if self.configuration.adaptive_kl_mode == "post-step-decision-groups":
            # Every optimizer step above uses one fixed rate. Adapt once for
            # the *next* update, after measuring the final published policy.
            rate = learning_rate_used
            if self.configuration.adaptive_learning_rate and self.configuration.target_kl is not None:
                kl = post["post_update_max_decision_group_kl"]
                if kl > 2*self.configuration.target_kl:
                    rate = max(self.configuration.minimum_learning_rate,rate/1.5)
                elif 0 < kl < .5*self.configuration.target_kl:
                    rate = min(self.configuration.maximum_learning_rate,rate*1.5)
                self.optimizer.learning_rate = rate
                post["learning_rate_adaptation_kl"] = kl
            post["learning_rate_used"] = learning_rate_used
            post["learning_rate_next_update"] = rate
        self.revision += 1
        return {**{key: value/weight for key,value in totals.items()}, **post,
            "pre_step_kl_divergence": totals["kl_divergence"]/weight, "update_seconds": time.perf_counter()-start_time,
            "minibatch_updates": float(updates), "policy_revision": float(self.revision),
            "sequence_windows": float(len(windows)), "sequence_length": float(length),
            "real_samples_per_epoch": float(r.sample_count),
            "selected_decision_samples": float(r.sample_count if selections is None else np.count_nonzero(np.any(selections,axis=-1))),
            "selected_action_factors": float(r.sample_count*r.action_count if selections is None else selections.sum()), "generating_state_max_abs": max_state_error,
            "generating_logp_max_abs": max_logp_error, "generating_value_max_abs": max_value_error,
            "mean_action_standard_deviation": float(np.mean(np.exp(np.array(mx.clip(self.model.log_standard_deviation,-5,2)))))}

    def write_policy_pack(self, output, *, policy_id=None, stochastic=True, library_path=None, **overrides):
        if overrides:
            raise ValueError("Recurrent policy publication requires its bound normalization/action contract")
        if policy_id is not None: self.policy_id = policy_id
        def layers(source):
            return [PolicyDenseLayerArtifact(np.array(x.weight), np.array(x.bias), x.activation) for x in source]
        mx.eval(self.model.state)
        memory = self._source_pack.actor_memory
        actor_memory = PolicyGRUMemoryArtifact(memory.after_layer,
            *[np.array(self.model.actor.memory[key]) for key in PARAMETER_NAMES], memory.maximum_absolute_input)
        return write_policy_pack(output, policy_id=self.policy_id, revision=self.revision,
            contract_version=self.contract_version, world_fingerprint=self.world_fingerprint,
            task_fingerprint=self.task_fingerprint, observation_fingerprint=self.observation_fingerprint,
            action_fingerprint=self.action_fingerprint,
            layers=layers(self.model.actor.encoder)+layers(self.model.actor.decoder),
            critic_layers=layers(self.model.critic) if stochastic else (), actor_memory=actor_memory,
            observation_mean=np.array(self.model.actor_observation_mean),
            observation_inverse_standard_deviation=np.array(self.model.actor_observation_inverse_standard_deviation),
            critic_observation_mean=np.array(self.model.critic_observation_mean) if stochastic else (),
            critic_observation_inverse_standard_deviation=np.array(self.model.critic_observation_inverse_standard_deviation) if stochastic else (),
            action_log_standard_deviation=np.array(mx.clip(self.model.log_standard_deviation,-5,2)) if stochastic else (),
            action_bias=self.action_bias, action_scale=self.action_scale,
            observation_clip=self.model.observation_clip, action_clip=self.action_clip, library_path=library_path)
