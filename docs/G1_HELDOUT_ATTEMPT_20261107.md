# G1 fixed-evaluation attempt

The preregistered 7 November evaluation did not produce a learned-policy
rollout. Both seeds were rejected before execution because the historical
executor compiled world, task, observation, and action fingerprints that did
not match the retained PolicyPack. The rigid-projectile variant was also
rejected, with a different scene contract. The errors are preserved in
docs/evidence/g1-fixed-evaluation-20261107; no policy contract was bypassed.

The matched zero-action control completed 8,000 environment steps (8
environments by 1,000 control steps): zero physical-step failures, 93
height/tilt terminations, and zero environments without a termination. Mean root
height was 0.7308 m and maximum tilt was 1.2644 rad. This result characterizes
only the zero-action control, not the retained policy's ability.

The task actor inputs are proprioceptive, command, gait-phase and action
history rather than an instance-segmented image, but the measurements are
idealized simulation values, not calibrated real hardware. The task's
authored push velocity is zero. The seeds differ from the retained selection
seed; complete training-seed history has not been established. The test
cannot establish held-out disturbance recovery, visual transfer or
real-robot behavior.

The run used a historical checkout at
a4334df01211f4c9c2991da98780ece0743ecf59 on Apple M6, not current main. Its
tracked worktree had unrelated Human modifications listed in the evidence.
The full dirty patch is retained locally at
/Users/numi/numi-heldout-g1-20261107/source-dirty.patch with SHA-256
6cdb8affd1a4017bcf092b5cd865fecb1a441eab5fdaea539d1d0552bc4a97d2.
Other identities:

- Retained PolicyPack: 23930cbf1af7f40b3a68a9456e865ef88986426f7847e6a42669e2582ff3b3d3.
- Rollout executable: 94033bba246a1add565dcf531b18a1d14af8c21c15c7fc79358e4f46cbe96ffa.
- Native library: 77117a4e0e6b98d7b3c8a97eb306103f6722a541979a0b124668dbb578c1f934.
- Metal library: c1b7aa5ea0edb823ca47cf55a403d32d0c0ded54bc38fa593b8f2364153998c3.
- Preregistered plan: c866de77107ef012c4e2dd7661d333b96a33dea7286cd86b1582ad4911bc0934.
- Zero-action result: e0f68739f998a8f137060e4581e2fdf04f92d2c43b0992fae69ab6df50dff747.

Repository history places the PolicyPack in a commit dated 6 August 2026.
Recovering its exact originating build and contract is necessary before this
artifact can support a fresh compatible evaluation.

## Fingerprint-matched reconstruction

A later CPU diagnostic compiled the artifact-introduction revision
791fb586934f8222fcf8269c411c17eb2a774087. That revision's authored G1 task
contained a training-only motion stream that differed from the stored pack's
task and observation fingerprints. Removing that stream in an isolated source
checkout restored exact matches for world, task, observation and action.
This is a narrowly reconstructed configuration, not proof of the full
historical producing source or binary. The source patch, host contract probe,
failed diagnostic and successful contract output are retained in the evidence.
The normal policy compiler remained enabled.

The amended plan repeated the two fixed seeds and zero-action control on
Apple M6 with 8 environments and 1,000 control steps each:

| Arm | Physics-failed steps | Height/tilt terminations | Environments with no termination |
| --- | ---: | ---: | ---: |
| Policy, seed 4081007 | 0 | 13 | 1/8 |
| Policy, seed 4081008 | 0 | 11 | 1/8 |
| Zero action, seed 4081007 | 0 | 118 | 0/8 |

The policy's mean tilt was 0.2145 and 0.1844 rad across the two seeds; maximum
tilt was 1.3092 and 1.3441 rad. The zero-action mean and maximum were 0.3097
and 1.2623 rad. The policy had lower mean tilt and fewer terminations in the
matched first seed; neither policy run avoided termination in more
than one of eight environments. Multiple terminations can occur per
environment because the task resets.

Forward-progress reporting was unavailable in this historical executor:
its emitted progress arrays are empty and its availability flag is false.
The accompanying zero placeholders must not be interpreted as measured
displacement. The authored task still has zero push velocity. An attempted
rigid projectile changed the compiled world/observation/action contract and
was rejected before simulation; there is no measured disturbance response.
These small runs do not qualify policy generalization or deployment.

The first matched run spent approximately 80 seconds initializing historical
Metal pipelines; no timeout was preregistered. All requested arms completed
or reported their error without retry. Retained identities:

- Configuration patch: 171c583d9e15cb36bb7a0dd17984c3cb86428cc0bcef9e43a29f52372e042b67.
- Rollout executable: c4e752d80ce8c52113a44e374ce2072a4b6befbb50a67d57e3fe0bf406c24df2.
- Native library: 80663b5c54b80798c7cfcfeb67dfb344de6992e2ecc9a48db531ecdaedd2209a.
- Metal library: 32d9ed14031ab83f87dfcf8281b1d3c65c6315a02e04ce15ab2728772ef35fab.
- Amended plan: 74a8b0be23a00bb177328d8eae84cab82733f5faf0a4976029c38efb9f1d8225.

## Source audit of the outcome metrics

All 24 policy terminations had reason 2: the authored maximum-tilt gate at
1.22173048 rad (70 degrees). Per-environment termination counts were
[1, 0, 3, 3, 1, 1, 3, 1] for seed 4081007 and
[3, 1, 1, 1, 2, 2, 1, 0] for seed 4081008. The available aggregate does not
establish why the tilt threshold was crossed.

The historical evaluator generated 30 scheduled reset requests per arm. Its
reset scheduler forces resets every 257 control steps, staggered across
environments. Its field named clean_horizon_environment_rate counts
environments with no task termination, but those environments did not run an
uninterrupted 1,000-step horizon. This distinction applies to both the
matched runs and the earlier cross-revision zero-action control.

Forward progress was unavailable because the historical evaluator enables
its position readback and progress calculations only when --state-trace is
specified; the fixed plan did not specify it. Its progress calculation is
relative to the first sampled position and does not rebase after scheduled
resets, so it cannot by itself establish continuous-episode displacement
through reset boundaries. A new declared, reset-aware diagnostic is required;
no missing result has been inferred.
