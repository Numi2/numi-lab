# G1 fixed-evaluation attempt

The preregistered 7 November evaluation did not produce a learned-policy
rollout. Both seeds were rejected before execution because the historical
executor compiled world, task, observation, and action fingerprints that did
not match the retained PolicyPack. The rigid-projectile variant was also
rejected, with a different scene contract. The errors are preserved in
docs/evidence/g1-fixed-evaluation-20261107; no policy contract was bypassed.

The matched zero-action control completed 8,000 environment steps (8
environments by 1,000 control steps): zero physical-step failures, 93
height/tilt terminations, and zero clean-horizon environments. Mean root
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
