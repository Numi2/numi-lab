"""Frozen retinal input calibration from accepted native observations only.

This authors initial PolicyPack parameters at a publication boundary. It never
advances physics, reads food labels, changes images, or fits a running policy.
"""
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import numpy as np

RGB = slice(97, 1249)


def fit_rgb_normalization(rgb, valid, minimum_standard_deviation=.05):
    """FP64 population moments; bounded gain on constant/low-variance pixels."""
    if (not isinstance(minimum_standard_deviation, (float, int))
            or isinstance(minimum_standard_deviation, bool)
            or not math.isfinite(minimum_standard_deviation)
            or not .01 <= minimum_standard_deviation <= 1):
        raise ValueError("Minimum RGB standard deviation must be finite in[0.01,1]")
    rgb, valid = np.asarray(rgb), np.asarray(valid)
    if (rgb.ndim != 2 or rgb.shape[1] != 1152 or valid.shape != (len(rgb),)
            or not np.isfinite(rgb).all() or not np.isin(valid, [0, 1]).all()
            or np.any(rgb < 0) or np.any(rgb > 1)):
        raise ValueError("Native RGB/validity calibration shape or values invalid")
    sample = rgb[valid.astype(bool)].astype(np.float64)
    if len(sample) < 2:
        raise ValueError("Calibration requires at least two valid native RGB frames")
    mean = np.mean(sample, axis=0)
    standard_deviation = np.sqrt(np.mean(np.square(sample-mean), axis=0))
    inverse = 1/np.maximum(standard_deviation, minimum_standard_deviation)
    return mean.astype(np.float32), inverse.astype(np.float32)


def calibrate_initial_policy(policy_path: Path, rollout_path: Path, output: Path,
                             *, minimum_standard_deviation=.05):
    from .crow_environment import write
    from .mlx_policy_learning import read_policy_pack, read_policy_rollout_pack
    from .recurrent_policy_admission import publish
    p, r = read_policy_pack(policy_path), read_policy_rollout_pack(rollout_path)
    identities = ('world_fingerprint', 'task_fingerprint', 'observation_fingerprint', 'action_fingerprint')
    if (p.revision != 1 or r.policy_revision != 1 or p.actor_memory is None
            or p.actor_observation_count != 1251 or p.critic_observation_count != 97
            or p.action_count != 18 or r.actor_observation_count != 1251
            or r.critic_observation_count != 97 or r.action_count != 18
            or r.format_version != 9 or r.control_step_count != 64
            or r.environment_count < 8 or r.teacher_actions.size
            or p.task_fingerprint != r.task_fingerprint):
        raise ValueError("Calibration requires matching fresh recurrent crow policy and64-tick native v9 pre-choice rollout")
    selection = r.action_selections()
    if selection is None or np.any(selection):
        raise ValueError("Calibration must end before any native action decision")
    if (np.any(r.transitions['physics_error']) or np.any(r.transitions['done'])
            or np.any(r.transitions['reward'])
            or np.any(r.memory_resets.reshape(64,r.environment_count)[1:])):
        raise ValueError("Calibration cannot contain failures, terminals, reward or episode resets")
    mean, inverse = p.effective_observation_mean.copy(), p.effective_observation_inverse_standard_deviation.copy()
    if np.any(mean[RGB] != 0) or np.any(inverse[RGB] != 1):
        raise ValueError("Calibration requires fresh identity RGB normalization")
    actor = r.actor_observations.reshape(r.sample_count, 1251)
    if np.count_nonzero(actor[:,1249] == 1) < 1024 or np.any(actor[:,84:87]):
        raise ValueError("Calibration requires1024 valid frames and neutral action history")
    mean[RGB], inverse[RGB] = fit_rgb_normalization(actor[:,RGB], actor[:,1249], minimum_standard_deviation)
    target = output/'initial.policypack'
    if target.exists():
        raise ValueError("Calibration refuses to overwrite an initial policy")
    calibrated = replace(p, id=p.id+'-frozen-retinal-normalization', observation_mean=mean,
                         observation_inverse_standard_deviation=inverse)
    publish(calibrated, target)
    restored = read_policy_pack(target)
    if (not np.array_equal(restored.effective_observation_mean, mean)
            or not np.array_equal(restored.effective_observation_inverse_standard_deviation, inverse)):
        raise RuntimeError("Published frozen normalization readback differs")
    # Revert only id and normalization on the newly decoded pack: exact source
    # bytes prove every other parameter and contract survived publication.
    roundtrip = output/'source-roundtrip.policypack'
    reverted = replace(restored, id=p.id, observation_mean=p.observation_mean,
        observation_inverse_standard_deviation=p.observation_inverse_standard_deviation)
    publish(reverted,roundtrip)
    if roundtrip.read_bytes() != policy_path.read_bytes():
        raise RuntimeError("Source PolicyPack canonical roundtrip differs")
    source_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'status':'passed','schema':'numi.crow-frozen-retinal-normalization.v1',
        'source_policy_sha256':source_sha(policy_path),'native_rollout_sha256':source_sha(rollout_path),
        'calibrated_policy_sha256':source_sha(target),'identities':{key:getattr(p,key) for key in identities},
        'rgb_feature_begin':97,'rgb_feature_count':1152,'valid_frames':int(np.count_nonzero(actor[:,1249]==1)),
        'minimum_standard_deviation':minimum_standard_deviation,'maximum_inverse_standard_deviation':float(inverse[RGB].max()),
        'mean_minmax':[float(mean[RGB].min()),float(mean[RGB].max())],
        'inverse_standard_deviation_minmax':[float(inverse[RGB].min()),float(inverse[RGB].max())],
        'policy_revision':1,'normalization_frozen_in_policypack':True,'all_other_parameters_and_contracts_byte_exact':True,'optimizer_updates':0,'simulation_steps':0,
        'calibration_uses_only_native_RGB_and_validity':True,'learned_memory_qualified':False}
    write(output/'calibration.json',result)
    return result
