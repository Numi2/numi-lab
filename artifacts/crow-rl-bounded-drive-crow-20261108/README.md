# Experimental bounded motors remain unqualified for the crow

The native motor equations are now checked independently, but the actual crow
fixture fails physically. With the final twofold motor solve, 368 control
transitions are accepted with zero GPU rejections. At tick 91, lane 3 reaches
1.20000863 rad of body tilt and triggers task reason 2 (tilt). Maximum observed
beak penetration is 5.39133 mm against the unchanged 0.8 mm gate. The terminal
capture reproduces all 364 earlier trace rows byte for byte and retains q/v,
scene bodies, world/contact statuses and all four terminal transitions.

This mode is an explicit C++ opt-in, limited to contact graphs with nv<=40.
Default RunProfile/C API selection and protected deployment97 are unchanged.
Numerical success must not be used to promote this contact model. The current
smooth-force predictor and endpoint contact solve remain split; this trajectory
does not isolate which coupled physical mechanism causes the support/grasp
failure. Qualifying a manufactured motor-plus-contact reaction is the next
useful experiment before controller gain or time-step tuning.

## Numerical fixes and preserved failures

- The original valid motor solve stopped at tick 21 after 256 sweeps. Protected
  first-failure snapshots reproduce its public failure and preserve physical
  Cholesky/inverse, constitutive inputs and failed iterate.
- The first SIMD compensation revision retained an old lane-0-only exit.
  Single-coordinate tests passed, while coupled/capacity tests published wrong
  accelerations. Its three downstream crow runs are invalid for physical
  qualification and are explicitly rejected in `precision/rejection.json`.
- Reference gates now execute inside the native probe after every batch. The
  retained bad shader is rejected after the first bad batch (272 rows / 200
  native environment steps), before later conditions or any crow run.
- Removing the stale exit restored all component gates, but fast-math still
  invalidated compensated arithmetic. The motor kernel now has its own strictly
  compiled translation unit. On the retained failure, its scaled residual
  agrees with FP64 to 7.0e-14; the previous 1.39698e-6 becomes 8.99204e-8.
- The strictly compiled FP32 torque iterate then stopped at tick 53. Rounding
  the FP64 reference torque back to FP32 also exceeded the unchanged 4-epsilon
  certificate. Retaining both high and low torque components removes that
  numerical stop; published acceleration/state and effort remain FP32.
  BDF2 failure snapshots retain both internal components. No physical coefficient,
  torque cap, 256-sweep bound or acceptance threshold was changed.

The final component study passes all 514 native steps / 976 rows: maximum
velocity error 4.08833e-7 rad/s, q integration error 7.11916e-9 rad and torque
error 2.24851e-9 Nm. The 448 legacy-mode rows are unchanged byte for byte.
These are numerical regression conditions, not independent biological trials.
The full motor/contact interaction, mechanical energy integral, reset/checkpoint
contracts and end-to-end training performance remain unqualified.

## Integration and optimizer decision

Full Release succeeds. The default native contact and actual crow object-push
owners pass. A fresh default-mode 640-transition grasp/lift/release replay
matches the previously archived complete trace, q/v, scene trajectory and summary
exactly; see `verification.json`. This preserves the prior default behavior,
including its previously reported cadence-convergence limitation. Protected
policy SHA256 remains
`6a009417eac3e04831c95f46c24e78f943e825892124b972a58b5ee9eeed0389`.

The requested final-step head/beak KL, long-return attribution and applied actor
update checks are completed in
[the optimizer study](../crow-rl-post-step-rate-20261108/README.md).
Actual gamma=.999/lambda=1 retains 0.879797 across 128 ticks. Adaptive rates made
larger actor updates but reduced held-out cue-conditioned choice (466 vs 482
of 576) and intake (98 vs 105); no adaptive policy was promoted. Reliable
stochastic feeding and the wider lifelong embodied goal remain unfinished.

All attempts, including invalid instruments and physical failures, are retained
in the constitutive study's `payload/manifest.json` through actual Git objects.
[Recovery controls](../crow-rl-git-recovery-bounded-drive-20261108/) define the
bounded native replay and exact cleanup. Archival recovery is separate from
scientific or whole-environment qualification.
