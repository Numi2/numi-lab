# Bounded motor equations pass; physical crow qualification remains open

The revised native constitutive trial completed 514 accepted environment
microsteps and 976 component rows, including coupled joints, dry friction,
multiple articulations and all 40 generalized-velocity slots. Independent FP64
mass and active-region references bound velocity error to 4.08833e-7 rad/s,
position integration error to 7.11916e-9 rad and torque error to 2.24851e-9 Nm.
The frozen gates remain 2e-6 rad/s, 2e-7 rad and 1e-7 Nm. Existing implicit
actuation keeps its original operator flags; bounded actuation uses physical mass.

The original trial retained a slightly excessive velocity error (2.05884e-6)
and an invalid 80-body capacity fixture. An intermediate 41-body single-tree
calibration also exceeded the per-articulation body limit. These attempts are
preserved. The final wide fixture has two fixed-root stars with 20 scalar joints
each, within both body limits. No acceptance threshold was relaxed.

These are manufactured numerical conditions without geometric contact. They do
not qualify beak grasping, force/contact coupling, task energy integration,
checkpoint/reset/partition semantics, throughput, or learning. The actual crow
follow-up, numerical instrument repairs and rejected revisions are in
[the crow contact study](../crow-rl-bounded-drive-crow-20261108/README.md).

`payload/manifest.json` archives the complete raw build, calibration, runtime
snapshots, trial evidence and subsequent crow follow-up through deduplicated Git
objects. Git verification, executed restored native checks and exact cleanup
receipts live in
[the recovery controls](../crow-rl-git-recovery-bounded-drive-20261108/).
