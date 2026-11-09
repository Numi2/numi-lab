# Late rejection across the native owner boundary

A native MetalWorld/Matter probe uses a genuine one-degree-of-freedom
articulated body in contact with a deformable pad. The baseline produced a
positive contact impulse and a change in body velocity. This is the existing
coupled contact law, not the experimental midpoint law; the numbers are not
force, torque, interface-work, spatial, or midpoint qualification.

The first useful injected late-failure attempt (`coupled-late-rollback-004.log`)
exposed a real defect: Matter restored its FEM nodes, while MetalWorld
published the provisional rigid position and velocity after the late failure.
The repair reconciles rigid coordinates to the control-step checkpoint after
post-commit device physics, for the articulated-only free-motion graph with
no scene bodies, rods, or native task. Other graphs have additional state
and are not covered by this repair.

A further check (`coupled-late-rollback-007.log`) found that the exported
rigid generalized-candidate buffer still contained the rejected trial. Matter
now checkpoints that buffer and restores the failed environments on both
rollback paths. The prior retry happened to reproduce the physical result,
but that did not justify retaining a rejected candidate in a field documented
as accepted/published state.

The final native probes separately inject a rigid-owner status failure and
an invalid projected body pose, which triggers Matter contact-clearance
rejection after provisional rigid publication. They check failure category,
rigid q/v, FEM nodes and fields, material state, contact history, scheduler,
and generalized candidate. Both test initial and previously loaded,
nonzero checkpoints, and both retries reproduce their accepted reference.
The invalid pose is a targeted failure injection, not a prescribed-motion
physical rollout. Logs and hashes for the final executable, libraries and
sources are retained beside this report.

The selected four-test regression (`coupled-rollback-ctest-004.log`) passes.
The broader attempt (`coupled-rollback-ctest-001.log`) also ran the pre-existing
`matter.integration.articulated_rigid_reaction` fixture; it failed the
existing rigid-world topology admission guard before reaching the solve.
The guard and that fixture were not changed here; the cause has not been
qualified further. Earlier fixture setup failures and the uncorrected and
partially corrected checks remain in the numbered logs.

This does not implement the requested two-way midpoint contact transaction,
prove equal-and-opposite impulse, torque, or interface work, establish spatial
accuracy, or establish an accuracy-adjusted speed benefit. The scoped
midpoint owner guard remains in place, and the two original experimental
contact/CCD options remain disabled.
