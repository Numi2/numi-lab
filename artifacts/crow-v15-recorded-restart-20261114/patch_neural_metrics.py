from pathlib import Path
import hashlib,json
r=Path('/Users/n/Developer/numi-crow-rl-20261107');b=r/'build/crow-v15-recorded-restart-20261114';a=r/'artifacts/crow-v15-recorded-restart-20261114';p=b/'neural-masked-python/metalrobo/mlx_policy_learning.py';s=p.read_text()
needle='''        metrics["gradient_norm"] = gradient_norm
        return loss, metrics''';assert s.count(needle)==1
s=s.replace(needle,'''        metrics["gradient_norm"] = gradient_norm
        clip_scale = mx.minimum(1.0, self.configuration.maximum_gradient_norm / (gradient_norm + 1.0e-6))
        metrics["actor_clipped_gradient_norm"] = actor_gradient_norm * clip_scale
        metrics["critic_clipped_gradient_norm"] = critic_gradient_norm * clip_scale
        return loss, metrics''')
needle='''        result["update_seconds"] = time.perf_counter() - start''';assert s.count(needle)==1
s=s.replace(needle,'''        result["behavior_logprob_max_abs_error"] = max_parity
        # Re-evaluate on retained behavior observations after the final optimizer step.
        after_chunks = []
        for offset in range(0, sample_count, self.configuration.minibatch_size):
            means_after = self.model.actor_mean(mx.array(batch.actor_observations[offset:offset+self.configuration.minibatch_size], dtype=mx.float32))
            mx.eval(means_after)
            after_chunks.append(np.asarray(means_after, dtype=np.float32))
        after = np.concatenate(after_chunks, axis=0)[..., :4]
        old = old_means[..., :4]
        current_std = mx.clip(self.model.log_standard_deviation, -5.0, 2.0)
        mx.eval(current_std)
        after_logstd = np.asarray(current_std, dtype=np.float32)[:4]
        before_logstd = old_std_np[:4]
        kl_dims = after_logstd - before_logstd + 0.5 * (np.exp(2.0*(before_logstd-after_logstd)) + np.square(old-after)*np.exp(-2.0*after_logstd) - 1.0)
        weights = np.asarray(batch.policy_weights, dtype=np.float32).reshape(-1)
        denom = max(float(np.sum(weights)), 1.0)
        result["final_balance_kl"] = float(np.sum(np.sum(kl_dims, axis=-1)*weights)/denom)
        result["final_balance_actor_mean_shift_rms"] = float(np.sqrt(np.mean(np.square(after-old))))
        for k in range(4): result[f"final_balance_kl_action_{k}"] = float(np.sum(kl_dims[:,k]*weights)/denom)
        result["update_seconds"] = time.perf_counter() - start''')
p.write_text(s);(a/'masked-balance-metrics-patch.json').write_text(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'post-final-step four-action KL, actor mean change and post-clip gradient norms'},indent=2)+'\n');print('patched isolated learner diagnostics')
