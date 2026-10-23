# Physical cardboard integration: October 23

The adhesive activation blocker has a measured small-fixture repair. The full
box has **not** been physically scored, assembled or crushed. OpenCourant has
not yet passed the corrugated score/fold/release admission fixture, so there is
no qualified capacity curve or justified heavy-object mass.

## Implemented and measured

- The published [adhesive repair](adhesive-repair-20261023.md) fixes the TYPE43
  initial matrix path, undefined LAW117 restart slot, and first-active area/frame.
  A stationary explicit birth window between implicit stages preserves the
  current overlap reference. Monolithic implicit activation remains defective
  and is not used.
- The new nonuniform 2-by-2 cohesive overlay attaches to deformable solid faces
  through native TYPE2 interpolation. Its rotated approach, activation and
  mixed opening/shear sequence completes with direct MUMPS. At first activation,
  maximum traction is about 5.9e-15 MPa and cohesive energy 7.0e-29 N mm.
  Final Gauss-point path work is 50.516957 N mm versus native cohesive energy
  50.516950 N mm. The largest nontrivial sampled relative difference is 1.55e-6.
  These are synthetic elastic interface controls, not calibrated fracture data.
- A declared 50 micrometre hot-melt film with a published nearby product density
  contributes 20 mg over the 20-by-20 mm patch. Native total mass matches the
  substrate plus this film; no added nodal mass appears. Density is an analogue;
  film thickness is an engineering assumption and strength remains unidentified.
- The [LAW112 predictor repair](paper-predictor-20261023/README.md) supplies its
  orthotropic elastic initial iteration matrix, previously missing from MMATS.
  The unchanged diagnostic tensile coupon now completes 2.2478386 s in 121
  cycles, versus the previous stop near 0.067 s. This is not a consistent plastic
  or damage tangent. PCG remains inadmissible; the measured path uses direct MUMPS.
- Published liner/medium tensile bilinear fits supply nonzero MD/CD hardening.
  The [measurement report](paper-predictor-20261023/material-measurements.json)
  separates nominal stress, actual prescribed motion, source readout sensitivity
  and sampled work. The original endpoint command window lost the final MD
  increment; the extended-window control fixes it. All four coupons now meet
  the endpoint-time tolerance (1.1e-6 s), commanded-displacement and native
  energy-balance checks. Two of four still exceed the retained 1% source-line
  diagnostic; all lie within the declared visual-readout sensitivity envelope.
  This envelope is not a confidence interval. Maximum sampled axial work
  discrepancy is 0.01815%. See [the corrected-window report](paper-predictor-20261023/material-measurements-07-endpoint-window.json).
- An all-solid corrugated strip and an explicit-flute shipping blank are exported.
  The [blank preflight](physical-blank-preflight-20261023/preflight.json) contains
  227,480 HEX8 elements and 489,268 nodes. Its positive initial Jacobians and
  Python memory measurement are geometry evidence, not a native resource forecast.

The strip tie implementation initially added 77.512% mass at Engine startup,
although Starter reported zero added mass. TYPE2 Spotflag 27 eliminates that
startup mass. Explicit gripper inertia was also removed: Ispher 2 derives it
from the physical board nodes. Starter's `TOTAL ADDED INERTIA = 0` alone did not
validate the old input because separately specified rigid-body inertia was
excluded from that number. The old mass/inertia attempt is invalid for physics.

A runner timeout could leave a solver child alive after `/usr/bin/time` exited.
Timeout handling now kills the entire owned group even if the wrapper exits
first; a child-heartbeat regression checks this. The affected attempt and
termination audit are retained.

## Gate decisions

| Gate | Decision and reason |
|---|---|
| Entire mesh/time campaign before the first box sequence | Defer until the provisional sequence works, then freeze and qualify it. |
| Exact-recipe fracture/impact calibration as an integration prerequisite | Remove as a prerequisite: public data do not identify it. Preserve separate sensitivity and validation verdicts. |
| Exactly zero contact search gap | Remove. TYPE7 requires a positive numerical gap; declare 0.001 mm, measure penetration, and check half-gap sensitivity before qualification. Never treat it as extra board thickness. |
| Relative increment energy/displacement at a zero-work hold | Do not use as sole pass/fail tests; use absolute force/energy continuity with measured noise floors. |
| Native `C` convergence flag | Insufficient. Inspect the reported force residual: the native displacement fallback can print `C` while the requested force tolerance is not met. |
| 1% agreement with digitized bilinear fit as backend admission | Diagnostic only. Compare achieved nominal stress and source readout sensitivity; do not pretend a fitted line is exact raw measurement. |
| No artificial mass/inertia, valid contact, unchanged accepted history, correct activation reference | Retain. Removing them changes the physics being requested. |
| Fixture release, settled dimensions and shape, energy/momentum, full history coverage | Retain for assembly/crush claims. A parser pass or accepted prefix cannot replace them. |
| Original refinement and impact-peak tolerances | Retain for qualification after a complete sequence. Time refinement and loading-rate sensitivity stay separate. |

The old strip Engine used a timestep growth factor of 1 and cut back after only
15 iterations. Its later 50-iteration control still cut back while residuals
contracted; near the retained stop, the contraction factor was about 0.836 and
would require approximately 71 iterations to reach 1e-6 from the observed state.
The next bounded control permits 100 iterations and growth after easy steps.
This changes the nonlinear iteration budget and timestep controller, not the
material, force tolerance, physical speed or mass.

**Correction to earlier certificate interpretation:** the native implementation
in `nl_solv.F` can accept when the displacement increment is below its fallback
threshold and the relative residual is below 0.1. In the old strip06, all 47
accepted rows exceeded the requested 1e-6 force tolerance (largest 1.311e-5).
Those prefixes must not be described as satisfying that configured force gate.
The small overlay04 and inspected paper controls did satisfy the reported
force threshold. Relative residuals also need a meaningful force scale near
zero; they are not independent physical validation.

## Remaining work and stop rule

The first indispensable outcome is a complete native corrugated strip through
indentation, withdrawal, fold and fixture release, with valid contact and no
mass alteration. Internal starch ties in this fixture are a perfect-bond
control, not a debonding model. LAW117 compression response must not be counted
again by separate compression contact; post-failure contact and irreversible
mixed-mode failure still need their own complete fixture.

Only then can the full blank receive physically generated creases, coupon-based
score allowance, mapped manufacturer/flap closure patches and fixture release.
No assembled checkpoint currently exists. Compression, gravity and impact
therefore remain unrun. The shipping mesh requires a measured native resource
forecast; its export size is not evidence that factorization fits in 16 GiB.

Do not launch a long box campaign or an open-ended solver-repair campaign behind
a failed admission fixture. Keep the exact failure and continuation inputs.
Every retained run receipt binds its source, binary, libraries, input and parent
checkpoint; process completion is explicitly separate from physical qualification.

## Private BLAS control

A private LP64, single-thread OpenBLAS package is staged without changing system
alternatives or the original runtime. The four corrected-window coupon inputs
and Engine binaries match the reference-provider runs. All 85 CSV fields have
identical sample grids; principal endpoint stress, displacement and work match
at output precision. The largest mixed-unit absolute field difference is
1.13e-11, in a reaction-force field. Their median Engine times are 0.27564 s
(reference) and 0.28061 s (OpenBLAS); no speedup is established by these tiny tests.
See the [provider comparison](paper-predictor-20261023/material-history-comparison-07-v-openblas.json).

The nonuniform overlay also agrees between providers on the active-stage
reaction, internal-energy and external-work CSV values. The largest displacement
difference is 1.73e-17 mm. Quiet-stage reaction differences need an absolute
noise scale; do not divide by a near-zero force. Both providers retain the same
0.18–0.39% upper/lower resultant mismatch, so regression equivalence is not an
independent force-balance qualification. See the [overlay provider audit](integration-admission-20261023/candidate05-openblas/comparison.md).

The current strip deck now includes all exposed paper faces in self-contact,
parks the punch above the fold envelope, dwells before release and turns both
gripper rigid bodies off. Its Newmark average-acceleration release input uses
gamma 0.5 and beta 0.25, with physical mass and no added damping. These release
commands are implemented but remain unexecuted until Engine 1 completes. The
old explicit release remains an option: Starter alone suggests at least 97.3
million solid-controlled steps per second, while its much smaller nodal
estimate means the actual active timestep is unknown. No fast explicit-release
claim follows from that estimate.

## Latest bounded outcome

Strip12 Starter completed with zero errors and zero warnings. The 180-second
Engine limit retained only 1.2 s of the requested 26.7454679 s first stage
(4.49%); scoring, folding and release did not complete. There were two iteration
cutbacks. All twelve C-marked rows exceeded the requested force tolerance, with
a largest reported residual of 1.093e-5. At the final interrupted attempt the
residual grew again after its initial decrease. Increasing the iteration budget
therefore did not resolve admission. The [strip12 audit](integration-admission-20261023/strip12-controller-audit.json)
retains the complete partial history and interpretation. Tiny reported added
mass (3.44e-21 Mg versus 3.95e-7 Mg model mass) is numerical roundoff, not
meaningful physical mass scaling.

Review also found that the clamp bands omitted internal starch ties to avoid
secondary-node conflicts. Once RBODY/OFF removed the clamps, those interfaces
would remain untied. Consequently strip12's unexecuted release input is not
physically valid, regardless of its time integrator. The next export must grip
paper nodes while retaining every starch interface tie; a clean Starter pass
will still not prove that release completes.

The current collector now reports force-residual exceedances separately from
the solver's C flag. Failed, partial and invalid runs remain available. See the
[machine-readable verdicts](integration-verdicts-20261023.json) for the distinct
backend, convergence, calibration, validation, assembly and impact status.

Case13 corrects that release topology: all 225 starch secondary nodes per
interface stay tied, while each gripper contains 333 paper nodes and excludes
162 starch nodes. Native Starter completes with zero errors and warnings.
Its Engine stages have **not** been run; this is a preflight correction, not
release qualification or evidence that the scoring convergence blocker is gone.
The final local Python suite passes 64 tests. No solver worker remains running.
