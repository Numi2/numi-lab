# Native cardboard precision, scoring and assembly groundwork

This evidence extends the manufactured liner/flute/glue FEM instrument using
published inputs. It preserves rejected hypotheses and failed trajectories.
It does not qualify a physical scored crease, starch debonding or assembled box.

## Questions and interventions

The previous instrument could not complete its first loaded step at a `1e-6`
residual tolerance, and its 20 µm tool path penetrated the accepted board.
`endpoint-plan.json`, `gradient-plan.json` and `persistent-u-plan.json` record the
successive predictions before their native tests. Merely changing the prescribed
endpoint did not repair contact. Computing displacement by subtracting absolute
FP32 positions removed rest drift but did not meet the tighter loaded gate.

The final precision change stores authoritative displacement independently and
uses `F = I + grad(u)` against an immutable authored reference. Absolute positions
remain `X + u` for contact and output. Matter ABI 40 stores 80 bytes per FEM node;
old packages must be recooked. The native reference check covers translated thin
geometry, force/tangent accuracy, rigid rotation, reset, snapshot identity,
replay, rollback and rejection of old ABI or altered displacement state.

The translating-capsule option supplies a feasible initial Newton candidate,
then requires the ordinary native material, residual, volume and contact gates.
It also checks the swept node/capsule path, motion continuity, realized endpoint,
and rollback. Initial orientation is declared fixed for the first interval;
its start position is inferred as endpoint minus timestep times velocity.
Subsequent starts bind to the previous accepted native endpoint. Coverage is
nodal, not continuous collision detection against all triangle interiors.

## Retained native outcomes

| Trial | Result | Boundary |
|---|---|---|
| `endpoint20-001` | 16 accepted, then rejected | Correct endpoint alone was insufficient |
| `gradient-tight-001` | First loaded step rejected at `1e-6` | Subtracting absolute coordinates is insufficient |
| `gradient-base-001` | 144 accepted at `1e-4` | Earlier displacement-from-position arithmetic; not the final persistent path |
| `persistent-tight-001` | 48/48 loading/hold steps accepted at `1e-6` | Stationary reference exactly at rest; maximum free-force imbalance L2 0.0400 N |
| `persistent-1e8-001` | First loaded step rejected | Residual 2.31384e-7; no tighter accepted endpoint substituted |
| `predictor20-001` | 96/96 load/hold/withdraw/settle steps accepted | Minimum sampled gap 80.03 nm; peak punch reaction 0.7313 N; stationary punch force zero |
| `predictor50-001` | 15 accepted, then rejected | Deeper scoring remains unqualified |
| `box-compile-001` | Compiled complete slotted flat blank | 4,308 nodes; 10,356 tetrahedra |
| `predictor50-trace-001` | Reproduced the same 15 accepted states, then rejected | Native trace identifies the clearance limiter while Newton remains unconverged |
| `regional-1e8-001`, `regional-1e6-001` | Both rejected their first loaded step | Experimental tangent diagonal did not improve board robustness |
| `persistent-halfdt-001` | 96/96 loading/hold steps accepted at `1e-6` | Narrow endpoint moment/plastic-strain screen passes; unequal equivalent force tolerance |
| `persistent-nx16-001` | First loaded step rejected at `1e-6` | Residual `5.46494e-5`; no mesh-convergence endpoint |
| `persistent-thickness2-001` | 2 accepted, then rejected at `1e-6` | Thickness refinement remains unresolved |
| `barrier50-scale1-001` | 15 accepted, then rejected | Observation/tool CSVs reproduce `predictor50-001` byte-for-byte after the code addition |
| `barrier50-scale2-001` | 96/96 accepted | Minimum sampled gap 53.46 nm; peak punch force 2.316 N |
| `barrier50-scale4-001` | 96/96 accepted | Minimum sampled gap 80.16 nm; peak punch force 2.745 N |
| `barrier20-tight-001` | 96/96 accepted at `1e-6` | Minimum sampled gap 219.56 nm; peak punch force 0.3155 N; maximum force imbalance L2 0.03975 N |
| `matched-baseline-001` | 48/48 accepted, `dt=25 µs`, tolerance `1e-6` | Same-build baseline; byte-identical observations/material phase states to prior tight baseline |
| `matched-halfdt-001` | 96/96 accepted, `dt=12.5 µs`, tolerance `5e-7` | Equal tolerance/timestep ratio; moment and plastic RMS endpoint screen passes |
| `barrier50-scale1-tight-001` | 48 accepted, first withdrawal rejected at `1e-6` | Tighter tolerance alone does not recover the default-factor cycle |
| `barrier50-scale2-tight-001` | 96/96 accepted at `1e-6` | Minimum gap 417.15 nm; peak punch force 0.3878 N |
| `barrier50-scale4-tight-001` | 96/96 accepted at `1e-6` | Minimum gap 806.90 nm; peak punch force 0.3907 N |
| `nx16-k256-first-001` | First increment rejected with 256 Krylov columns | More linear work did not recover length refinement |
| `thickness2-k256-first3-001` | 3/3 increments accepted with 256 Krylov columns | Recovers the prior third-increment rejection; short diagnostic only |
| `thickness-k256-baseline-001` | 29 accepted, then status-5 rejection | More Krylov work does not preserve the full coarse trajectory |
| `thickness2-k256-full-001` | 48/48 loading/hold accepted | Recovers the full refined-thickness horizon; no valid paired convergence estimate |
| `nx16-k64-newton-trace-002`, `nx16-k256-newton-trace-002` | Both first increments rejected; complete 28-iteration traces | Byte-identical CSV/mesh/initial geometry to their untraced predecessors |
| `box-feature-rest-003` | 2/2 rest steps; complete standalone layout references | Four panels, joint tab and both ends resolve every action; package/mesh/observations unchanged |
| `box-feature-rest-002` | 2/2 rest steps; same geometry and observations | Typed layout v2 identities; still unexecuted targets |
| `barrier-exact1-compile-001`, `barrier-rounded1-compile-001` | Identical cooked fingerprint | Effective FP32 factor 1; requested near-1 double retained separately |
| `box-rest-001` | 2/2 zero-load steps accepted | Exactly zero motion and force imbalance; no physical assembly |

The tool trials use the exploratory `1e-4` residual tolerance unless marked tight. Their reaction
forces are native instrument outputs, not measured or physically calibrated
forces. A zero-gravity rest check does not establish bending, scoring or folding.

`corrected-path-followup-plan.json` predicts the 20/50 µm tool outcomes and blank
rest check. The 50 µm prediction was contradicted. `line-search-diagnostic-plan.json`
asked whether globalization rejected an otherwise converged trajectory. The
trace contradicts that interpretation: the linear solve converged, but the
nonlinear residual remained `1.23119e-4`, above `1e-4`. A later certificate
value on the rejected state cannot override the failed native status. `tight-resolution-plan.json` defines loading/hold-only time and mesh
comparisons against the accepted tight baseline; it does not replace the earlier
failed full-cycle convergence screen. The half-timestep arm keeps the same
relative impulse threshold, so its equivalent force bound is twice as large.

## Deeper-score diagnosis and time refinement

At the failed 50 µm step, Newton iteration 12 proposes inward motion of node 99
at rigid contact pair 198. Its current gap is 44.8199 nm, below the conservative
45.2042 nm feasible-distance floor (collision floor plus FP32 allowance). The
rigid limiter correctly changes the object step fraction from 1 to 0. The
proposed correction would close another 13.051 nm. The trace changes neither
physics nor acceptance; it reproduces the earlier accepted trajectory exactly.
This is a contact-conditioning failure, not permission to relax the distance
floor or accept an unconverged state.

The preregistered `barrier-conditioning-plan.json` tests fixed multipliers 1,
2 and 4 on the same native build. Scale 1 reproduces the prior trajectory
byte-for-byte in both CSVs. Scales 2 and 4 each complete the 50 µm cycle without
changing distance floors, activation distance or residual tolerance. They
scale barrier impulse, curvature and line-search potential consistently. Peak
punch force changes by 15.64% between the two successful arms. Maximum accepted
free-force imbalance L2 is 3.974 N and 2.416 N respectively at the exploratory
`1e-4` impulse threshold. These are conditioning successes with substantial
remaining force sensitivity, not calibrated deep scoring. Scale 1 remains the
default. The shallow 20 µm arm also completes at `1e-6`, but its peak reaction
changes from 0.7313 to 0.3155 N versus the loose run; this is strong residual-
tolerance sensitivity. `tight-barrier-plan.json` therefore tests both successful
50 µm factors at the tighter gate before attempting a quantitative paired
force estimate. The fixed-factor study is motivated by
[IPC supplement section 6](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf)
but does not implement its adaptive stiffness algorithm.

The persistent-displacement half-timestep comparison passes the declared 10%
loading/hold endpoint screen: moment differences are 3.474% at loading end and
1.268% at hold end; paper plastic-strain RMS differences are 0.887% and 0.066%.
Maximum force imbalance L2 is 0.039999 N for the baseline and 0.079940 N for the
half-timestep arm. Consequently this is not an equal-force-accuracy comparison.
It also does not qualify unloading, release, static equilibrium or physical
crease behavior. Full curve diagnostics remain in `tight-temporal-comparison.json`: loading
reaction-force relative-RMS differences reach 42.0% in X and 13.2% in Z.
The compared runs used related but different source/binary builds; their
matched controls do not make this a same-binary timestep-only experiment.
`force-matched-time-plan.json` defines a stricter same-build follow-up with
equal residual-tolerance/timestep ratios.

The opt-in regional tangent preconditioner passed an independent direct Metal
diagonal/finite-difference oracle, yet both board trials failed at the first
step with native status 5. Its implementation check is not a performance or
nonlinear robustness qualification. The scalar preconditioner remains default.

The tighter factor-2 and factor-4 cycles both complete. Their peak-force
spread is 0.746%, substantially below the loose comparison's 15.64%. Minimum
sampled gaps increase to 417.15 and 806.90 nm. This motivated `default-tight-scoring-plan.json`: test whether tighter
acceptance alone recovers the default-factor cycle. That prediction is
contradicted: factor 1 accepts all 48 loading/hold increments, then rejects the
first withdrawal with residual `2.97255e-5` against `1e-6`. Its accepted minimum
gap is 186.18 nm; the rejected endpoint is not substituted for an accepted one.

The complete factor-2/factor-4 pair uses identical native inputs, with only the
barrier factor and cooked world fingerprint differing. Punch-force relative-RMS
differences are 1.95% in loading, 3.59% in hold, and 22.60% in withdrawal; the
last corresponds to a small absolute RMS difference of 0.00213 N. Both settle
to zero punch force. Thus the close peak comparison does not establish
full-curve convergence. The incomplete factor-1 arm is excluded from this
full-cycle estimate, and the failed three-arm prediction remains explicit in
`completed-tight-barrier-pair-comparison.json`. Numerical agreement is distinct
from damage calibration, mesh convergence and physical scoring.

## Same-build time comparison with matched tolerance/timestep

`force-matched-time-comparison.json` records the completed follow-up. Native
binary, metallib and all selected source/material hashes match exactly across
the two arms and remain unchanged during both runs. The tolerance/timestep
ratio is 0.04 in both; because the residual normalization scale is not exported,
this ratio is reported as the force equivalent at the 1 N s normalization
floor, not a generally proven absolute force gate. Actual maximum free-force
imbalance L2 is 0.0399994 N and 0.0399992 N.

At loading/hold endpoints respectively, moment differences are 0.202%/0.927%
and paper plastic-strain RMS differences are 0.169%/0.225%. Reaction-force
relative-RMS differences over common physical-time samples range from 0.73%
to 6.22% across the three directions and two phases. The preregistered narrow
10% endpoint screen passes. This pair supports improved temporal reliability
for this loading/hold experiment; two resolutions do not establish asymptotic
convergence, mesh convergence, quasi-static behavior, release or a physical
fold.

After this pair, a host allocation-only follow-up limits prescribed-pose
scratch arenas to worlds that enable prescribed motion; other worlds receive
binding sentinels. Later tight-barrier/mesh trials and native regressions use
that rebuilt host library. The timestep pair keeps its exact earlier build
bindings, and is not relabeled as having run a later binary.

## Mesh recovery and nonlinear globalization

`mesh-budget-plan.json` holds the Newton budget and acceptance thresholds fixed
while increasing the Krylov budget from 64 to 256. The finer-length mesh still
rejects its first increment. Its terminal residual is `1.74501e-4`, versus
`5.46494e-5` in the earlier 64-column trial, and neither accepts a microstep.
This contradicts a simple linear-work-budget explanation; the pair does not
establish that additional iterations caused the deterioration. Mesh and initial
geometry hashes and the first grip increment match, but the old and new host
builds are retained separately.

Thickness refinement accepts its first three increments with 256 columns,
including the increment rejected by the 64-column run. This short success is
not a full-horizon convergence result. `thickness-full-horizon-plan.json`
preregisters a complete 48-increment comparison and a matching 256-column
coarse control. The refined-thickness arm completes all 48 increments in
614.3 s, with maximum accepted force imbalance L2 0.0399283 N. The coarse
control rejects increment 29 after 29 accepted increments (761.9 s total),
reporting status 5, tetrahedron 854 and determinant 0.997205. Zero certificate
fields following that failure are not evidence of zero residual or accuracy.
The reported determinant alone does not identify the underlying material-update
failure. Both runs bind identical native inputs unchanged throughout. The
strict spatial analyzer returns `failed_run`; it does not compare a truncated
coarse endpoint or silently substitute the earlier 64-column baseline.

Source audit identifies a specific unresolved solver mechanism: FEM
backtracking checks determinant admissibility and mixed-volume merit, with a
bounded last-trial fallback, before contact limiters choose the common step.
It does not backtrack on the actual nonlinear force residual or constitutive
admissibility. The final reassembled certificate can therefore reject a
trajectory whose linear solve completed. The compact diagnostic in `nonlinear-iterate-diagnostic-plan.json` now
captures the stored Newton-start norm, four alpha boundaries, inner-cycle
status and terminal certificate. Its first live callback emitted empty records
because three values outlived a reference-capturing encoding closure. The
decoder rejected that capture (`nx16-k64-newton-trace-001`); it remains retained.
`newton-capture-recovery-plan.json` fixes only callback ownership, verifies a
native rest capture, and repeats the same pair.

Both corrected traces contain all 28 iterations and reproduce their respective
earlier observation CSVs, mesh and initial geometry byte-for-byte. Every step
has final alpha 1. The 64-column run has eight residual increases exceeding
10%, none paired with an accepted inner cycle. The 256-column run has fifteen
such increases; iteration 4 has an accepted inner cycle and increases the
reassembled norm from 8.01625e-6 to 1.36814e-5 (70.67%). Thus the preregistered
screen is supported in that arm and contradicted in the 64-column arm.
The Arnoldi estimate is not a direct `b-Ax` residual, and the status iteration
count is a microstep maximum rather than per-Newton work. These observations
identify a diagnostic target, not the cause of the instability. The next
prediction is that a direct native linearized-residual check will distinguish
inner-solve error from nonlinear step/globalization error at the first growth
iteration. Failed candidates stay rejected; no residual threshold, collision
floor or globalization rule is changed in this work.

## Separate calibrated material

The Tryding/Biel Mode I coupon uses source parameters for a single-ply bleached,
clay-coated paperboard, distinct from corrugated starch glue. Its native material
evaluator passed 11,894 checks on Apple M4. The 512-step source-work comparison
has 5.55e-6 J/m² error, versus 1.471e-3 J/m² at 32 steps. These cumulative
work values include a shared source-oracle pre-seed work offset; native stress
integration begins at the positive opening seed. One thousand repeated holds
produce no damage creep. The source loading/unloading junction has distinct
one-sided tangents, and the evaluator uses an explicitly reported numerical
selection there. These checks execute zero physical FEM steps. They do not
implement a native cohesive face or make the corrugated interfaces separable.

## Verification and views

Nine final native Metal suites pass after the allocation guard, corrected compact
trace instrument and FP32 barrier identity correction, covering FEM
release/reference state, prescribed contact/rollback, elastic and plastic paper,
regional materials, material frames, delamination evaluation and the experimental
preconditioner oracle. Eleven targeted compiler/geometry suites pass (the
stateful round-trip fixture was updated to reject selector 2 because selector
1 is now defined). The three Python analysis suites pass, with 40 synthetic
regression cases, including fail-closed rejection of incomplete Newton traces
and duplicate terminal records. Retained earlier failed build/test attempts remain in the
instrument logs. The callback-lifetime repair is exercised by the native rest and loaded
trace runs and the repeated final nine-suite regression. Each receipt retains
its actual build hashes. The final box exporter rebuild additionally passes
a native two-step rest check and standalone layout validation, with package,
mesh and observation CSV byte-identical to the preceding rest export.
A passing regression suite is not a physical fold experiment.

Views use actual accepted numerical exports:

- [Every Newton iterate in the failed refined-length runs](instrument-checks/newton-residual-comparison.png)
- [Matched timestep comparison](instrument-checks/force-matched-time.png)
- [Tight 50 µm barrier comparison, including failed withdrawal](instrument-checks/tight-barrier-comparison.png)
- [Earlier loose 50 µm barrier sensitivity](instrument-checks/barrier-sensitivity.png)
- [Connected flat box blank](instrument-checks/box-blank.png)

## Evidence identity and reproduction

`runs/` retains manifests, results, complete observation CSVs, receipts, phase-end
material state and phase-end or failed/rolled-back geometry. Larger geometry,
packages and traces are gzip-compressed. `source-snapshots/` deduplicates exact
source copies by SHA-256. Each receipt binds the executable and Metal library,
material files, selected implementation sources, base commit and tracked patch.
All completed native invocations recorded bound inputs unchanged during the run.
Earlier receipts omit some individual source files; their patch plus base commit
and metallib hash remain available. Later runners additionally bind the FGMRES
and constitutive evaluator sources and capture the trace environment setting.

Raw experiments remain under `/Users/home/cardboard-closure-evidence-20261006`.
This is deterministic numerical evidence, not newly collected physical data.
The earlier [release/resolution failures](../release-and-resolution/README.md)
remain part of the record.
