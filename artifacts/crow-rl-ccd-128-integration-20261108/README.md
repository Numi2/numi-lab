# Integrated crow contact correctness and open convergence gate

The native translating sphere/box contact-time correction and articulated
hybrid future-FK/motion-margin repairs are integrated on the research branch.
The full Release build passes. All4096 primitive records and all128 projection
conditions match the qualified candidates. The native contact owner rejects
substeps0/129, executes all128, and passes its existing controls. Integrated32
and64 full crow traces, final q/v/scene and scene trajectories are byte exact
to the candidate.

The actual128 run accepts640 transitions with zero rejection. All four poses
pass the sampled grasp/lift/release gates: minimum lift23.85mm, maximum head
relative slip0.160mm, opposed jaw impulses32/32 hold samples, and sampled jaw
penetration below0.040mm. These are prescribed controls on a rigid surrogate,
not ingestion, learned manipulation or biological calibration.

The unchanged full-trajectory numerical gate FAILS:64vs128 differs by62.607mm
(limit5mm);32vs64 previously differed by118.168mm. The largest error is during
release/landing. Sampled jaw separation starts at ticks117,114..115 and113 for
32,64 and128; joint paths differ before release. Some control-boundary height
samples miss an impact and include bounce. See the corrected labels in
release-analysis.json; no certified continuous release/impact time is claimed.

Two output-only diagnostic replays (32/64) reproduce every original physical
trace/final/trajectory byte. Published final-substep residuals and joint
velocities are retained in the payload. They do not establish solver iteration
count or CCD split handling as the cause; aggregate residuals omit earlier
microsteps and do not isolate jaw rows. Source inspection also finds that the
hybrid event rematerializer documents an explicit-drive assumption while this
fixture uses implicit drives. That is an untested owning-source question, not
an explanation proven by these runs.

The first object-push invocation failed before physics because the retained
runtime omitted RetinalStream.metallib. Its failure and the packaging preflight
mistake are preserved. After adding the exact built retinal shader, one
unchanged native owner execution passes7296 transitions, solved contact/miss
controls, RGB visibility, FP64 reward checks, exact partition and world/retina
continuation. No physical failure was silently retried.

No crow profile or protected policy was promoted, no event budget or tolerance
was relaxed, and no further cadence escalation is selected. Full mechanical
trajectory qualification remains open. Git payload recovery and scoped native
reexecution are required before raw evidence is pruned; their receipts live in
artifacts/crow-rl-git-recovery-ccd-sweep-20261108.
