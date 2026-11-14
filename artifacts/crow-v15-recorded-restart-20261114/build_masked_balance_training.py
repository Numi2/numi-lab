from pathlib import Path
import json,hashlib,shutil,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114';p=b/'neural-masked-python';p.mkdir(exist_ok=True)
shutil.copytree(r/'python/metalrobo',p/'metalrobo',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
mod=p/'metalrobo/mlx_policy_learning.py';s=mod.read_text()
needle='''        standardized = (
            latent - mean
        ) * mx.exp(-log_standard_deviation)''';assert s.count(needle)==1
s=s.replace(needle,'''        # Dedicated balance task: only actor slots 0..3 are executed.
        mean, latent, log_standard_deviation = mean[..., :4], latent[..., :4], log_standard_deviation[..., :4]
        standardized = (
            latent - mean
        ) * mx.exp(-log_standard_deviation)''')
needle='''        entropy = mx.sum(
            log_standard_deviation
            + 0.5 * (1.0 + math.log(2.0 * math.pi))
        )''';assert s.count(needle)==1
s=s.replace(needle,'''        entropy = mx.sum(
            log_standard_deviation[..., :4]
            + 0.5 * (1.0 + math.log(2.0 * math.pi))
        )''')
needle='''        kl_per_sample = mx.sum(
                current_log_standard_deviation
                - old_log_standard_deviation''';assert s.count(needle)==1
s=s.replace(needle,'''        kl_per_sample = mx.sum(
                current_log_standard_deviation[:4]
                - old_log_standard_deviation[:4]''')
s=s.replace('''                            old_log_standard_deviation
                            - current_log_standard_deviation''','''                            old_log_standard_deviation[:4]
                            - current_log_standard_deviation[:4]''')
s=s.replace('''                    + mx.square(old_means - current_means)
                    * inverse_current_variance''','''                    + mx.square(old_means[..., :4] - current_means[..., :4])
                    * inverse_current_variance[:4]''')
needle='''        mx.eval(old_log_standard_deviation)
        totals: dict[str, float] = {}''';assert s.count(needle)==1
s=s.replace(needle,'''        mx.eval(old_log_standard_deviation)
        # Reconstruct behavior log-probability for the four physically executed latents.
        old_std_np = np.asarray(old_log_standard_deviation, dtype=np.float32)
        standardized = (batch.latents - old_means) * np.exp(-old_std_np)
        per_dim = -0.5 * np.square(standardized) - old_std_np - 0.5 * math.log(2.0 * math.pi)
        old_full = np.sum(per_dim, axis=-1)
        max_parity = float(np.max(np.abs(old_full - batch.old_log_probabilities)))
        if not math.isfinite(max_parity) or max_parity > 0.02:
            raise ValueError(f"native full log probability parity failed: {max_parity}")
        old_masked_log_probabilities = np.sum(per_dim[..., :4], axis=-1)
        totals: dict[str, float] = {"behavior_logprob_max_abs_error": max_parity}''')
needle='batch.old_log_probabilities[indices]';assert s.count(needle)==1;s=s.replace(needle,'old_masked_log_probabilities[indices]')
mod.write_text(s)
s=(a/'task_train-balance-16.swift').read_text()
needle='''       options.birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation ||
       options.birdFlowJourneyVariant == .v15ConditionalLandingNavigation {''';assert s.count(needle)==1
s=s.replace(needle,'''       options.birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation {''')
trainer=a/'task_train-masked-balance-16.swift';trainer.write_text(s)
out=b/'metalrobo_task_train_masked_balance_16';cmd=['xcrun','swiftc','-O','-swift-version','6','-parse-as-library','-I',str(r/'include/metalrobo'),str(trainer),str(r/'apps/TrainingLifeCheckpoint.swift'),str(r/'apps/RunInspector.swift'),str(r/'bindings/swift/MetalRoboTaskRollout.swift'),'-L',str(b/'round-neural-host'),'-lmetalrobo','-framework','AppKit','-o',str(out)]
subprocess.run(cmd,check=True,cwd=r)
shader=a/'restart-round-neural-balance.metal';s=shader.read_text();old='''state.episode.x>=2450u &&
            (state.status.w & 63u)>=12u && ((state.status.w>>6u)&63u)>=12u &&
            (action==7u''';assert s.count(old)==1
s=s.replace(old,'''state.episode.x>=2400u &&
            (action==7u''')
src=a/'restart-round-neural-balance-masked.metal';src.write_text(s)
build=json.loads((a/'shader-build.json').read_text());cc=build['compile'];cc[cc.index('-c')+1]=str(src);air=b/'restart-round-neural-balance-masked.air';lib=b/'restart-round-neural-balance-masked.metallib';cc[cc.index('-o')+1]=str(air);subprocess.run(cc,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(air);link[link.index('-o')+1]=str(lib);subprocess.run(link,check=True)
(a/'masked-balance-build.json').write_text(json.dumps({'trainer_command':cmd,'shader_compile':cc,'shader_link':link,'sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (trainer,out,mod,src,lib)}},indent=2)+'\n');print('built isolated four-action masked balance trainer')
