# Cardboard implementation and evidence ledger

Current continuation: source067 repairs a measured mismatch between the FEM
and rigid Armijo update fractions; strict full-contact convergence still fails.
Source069 removes unused completed-cycle graph solves with bounded bytewise
parity. Actual selected native graph assembly passes independent checks, while
one local inverse accuracy gate and complete nonlinear coupon convergence
remain open. Source070's compile failure and corrected071 continuation are
retained. See [solver progress](solver-progress-20261014.md) and the versioned
evidence index for source-specific results. No accepted shipping assembly
checkpoint, numerically qualified compression peak, gravity crush or drop
impact is available.

The implementation starts from `a9d801c`. Unrelated upstream work was retained
when the checkout advanced to `bf9e90cb`; the first published solver milestone
is `05fe5430`. The current checkout also retains unrelated upstream `9d0526d2`. The kraft 205 / NSSC 126 / finite elastic starch recipe and the
miniature regression fixture remain the starting physical model.

The source060 continuation is documented in
[the solver ledger](solver-progress-20261014.md). Its latest complete failed
coupon attempts accept three of27 controls, then reject control3 without
relaxing the certificate. Strict diagnostic-on/off histories agree. Independent
same-input assembly now agrees with the captured native operator action;
complete physical convergence and shipping assembly remain open. The256-node
offline Schwarz result is a conditioning experiment only, not a native solver
milestone. No assembled shipping checkpoint or qualified compression peak
exists from which to execute the gravity or drop branches.

| Verdict | Current evidence | Remaining qualification |
| --- | --- | --- |
| Solver correctness | Stable Fischer–Burmeister derivatives; scoped material/element actions; native mixed cohesive frame, regional Galerkin/grip transition and finite-body bridge checks; 80 native CCD checks; source49 combined-contact cases112/113 pass 3 accepted transactions plus rejection, 12 byte-identical on/off payloads, 18-vector rollback and 384 healthy true-residual rows; direct swept validation109 and prescribed BOX translation111 pass | Complete coupon remains unqualified: case108 timed out with 2/27 controls. Initially touching unbonded separating paths are explicitly unsupported by the private sweep gate. Full-cycle and zero-response noise-floor gates remain open |
| Numerical convergence | Sequential source-bound resolution runner and strict full-history comparisons implemented | Actual 8/16/32 × 2/4/8 × 1/2/4 spatial and dt/2/dt/4 histories have not qualified |
| Component calibration | Versioned source identities, condition split, 278 primary-figure marker estimates with pixel uncertainty and unknown specimen identity, published bilinear tensile fits, retained-recipe damage/rate/densification and feasible-set tooling; source47 case095 and source49 case115 verify both 15-state hardening assets on Metal at rest, MD/CD loading, unloading and reloading, including all nine tangent columns; own-grade Robertsson absolute-energy discrepancy is recorded | Point-update correctness and four fitted tensile slopes do not establish a calibrated corrugated failure law; the current Hill tensile onset remains compression based. Complete pin fixture, damage/interface validation, multiaxial transfer and exact-recipe identification remain open |
| Analogue validation | Explicit transfer uncertainty and separate ECT/scored-bend series | Simulated source fixtures and held-out curve comparisons |
| Physical assembly in simulation | Shipping blank and stage/checkpoint helpers pass CPU checks; native initialization uses 10.50 GB Metal allocation with zero physical steps | All twelve score operations, coupon-derived allowance, wall/tab/flap forming, adhesive activation, clamp release, settled assembly and accepted stage checkpoints |
| Crush/impact prediction | Finite-mass rigid/FEM bridge passes 244 native assertions, repeated with source47 in case100; staged box protocol is implemented | Native box compression/first peak, gravity transfer, centered and offset drops, energy/momentum accounting, and impact time refinement |

## Shipping-size preflight

`evidence/shipping-preflight-cpu-20261014-014` is the current CPU
compile-only preflight for a 300×200×200 mm inside FEFCO 0201 blank. It
compiled 453,378 nodes and 1,477,728 tetrahedra at the default scalar
preconditioner, with zero accepted steps and `physical_validation=false`.
The authored shipping parameters remain the explicitly cross-source
`literature2009` recipe, sinusoidal profile fallback, 130 clipped flute
intervals, 48 Y intervals, and one thickness slice. The compiled mesh has no
cohesive faces.

The pre-mesh planner uses those exact compiled cardinalities as its reference,
scales them by pitch, Y-interval, and thickness ratios, then applies a 25%
planning margin. For the current configuration it estimates 566,723 nodes and
1,847,160 tetrahedra, or 821,124,688 bytes in the modeled static FEM/Krylov
arenas. The exact compiled static lower bound is 656,899,296 bytes. Both are
below the configured 16 GiB limit, but are static lower bounds. Native case
`shipping-memory-preflight-032` subsequently initialized this blank: Matter
resident buffers were 10,467,551,760 bytes and Metal current allocation was
10,504,568,832 bytes, below 17,179,869,184 bytes. Host physical footprint was
5,616,488,336 bytes; these counters overlap and must not be added. This case
has no cohesive faces, no physical FEM steps and no runtime peak qualification.
Interface duplication, private trial arenas, checkpoint/staging overlap and
runtime allocation peaks require their own checked admission and measurements.
Shipping execution also checks the required private residual-descent scope
and arenas independently of the selected environment variables. The shipping
mesh exceeds the current bounded full-contact implementation's node, tet and
contact limits. It is rejected before native stepping instead of falling back
to a different solve. Initialization below 16 GiB is not full-stage admission.
The shipping admission helpers now match the effective runtime limit of
8,192 FEM nodes; the previous 16,384-node report was too permissive. The
[runtime scalability audit](shipping-runtime-gaps-20261014.md) identifies the
independent rigid-pair expansion, trial-arena, cohesive and preconditioner
restrictions. Increasing a contact capacity alone cannot enable this scene.

The compact immutable surface-map builder now passes 332 native equivalence
checks. The full-tooling memory planner uses actual source capacities and
rejects default arenas above 16 GiB. An explicitly selected 1,048,576-slot
arena plus compact surfaces fits a conservative 14.55 GB estimate, but its
capacity adequacy and runtime peak remain unqualified.

Every shipping process stage remains unexecuted. The score controller now
selects one depth (default 0.25 caliper) per invocation and derives approach
and withdrawal steps from an explicitly labeled engineering speed; candidate
depth-study runs start from separately constructed flat blanks. The 12 crease
lines remain unscored in this preflight, zero offset is only a lower-bound
allowance (there is no coupon measurement), and no score pass establishes a
qualified crease law. The four walls, manufacturer lap, eight end flaps, nine planned HMA
patches, release, and settled-assembly gates have no accepted endpoints; no
HMA law or starch interface is authored in this preflight. An explicit
engineering allowance and declared analogue HMA/starch parameter files can
enable a measured but unqualified simulation. They do not establish source-
grade calibration. Simulation assembly completion, exact-recipe calibration,
and physical box validation are reported as separate evidence axes. There is
no accepted full-box checkpoint to branch into compression, gravity transfer,
post-peak loading, or centered/offset drop cases. The output records all stage
gates false. Earlier failed or superseded preflight directories remain
preserved beside case 014 rather than being replaced by this compile-only
record.

## Material arithmetic experiments

The scalar Hill initializer supplies an initial guess only. It does not bypass
the authored residuals, generic Newton iterations, constitutive validity or the
production certificate. The scaled-residual policy and initializer remain
opt-in. The scaled captured replay rejected 365 of 14,400 inputs; adding the
initializer alone rejected 392. Both failures are retained.

Ordered two-word FP32 evaluation uses the same authored scalar instructions
for state updates, residuals, Jacobians, stresses and algorithmic derivatives.
On 14,400 exact-widened-FP32 captured inputs, its scaled-plus-initializer replay
still rejected 365. For valid responses with nonzero FP64 reference norms,
all measured stresses and algorithmic tangents meet the 1e-3 material gate:
worst stress error 1.7622e-6 and worst tangent error 1.5223e-4 relative.
The maximum projected-state absolute difference is 1.3960e-8.
Zero-reference quantities remain inconclusive without measured absolute noise
floors; these results are not full solver or convergence qualification.

The restart-16 regional coupon attempt rejects its first control step. The
bounded capture at control step 0 / Newton iteration 3 and its diagnostic-off
control produce byte-identical production files. Exact captured-input FP64
replay gives an assembled analytic action error of 6.049e-6 relative and
production nodal gather error of 5.885e-8. The material algorithmic tangent
error is at most 1.517e-5 relative. The paired finite differences disagree by
about 100%; their realized local deformation directions differ from the
requested direction by a median 24% at epsilon 1/32, worsening at epsilon
1/64. The rounded `I + grad(u)` input is unresolved at that scale. These
differences cannot qualify a smooth derivative.

An opt-in two-word deformation input now carries `I + grad(u)` and material
frame multiplication through the same authored state updates, implicit
projection, stress and derivative programs. Its portable kinematic check
passes 36,000 comparisons against FP64 (maximum absolute error 1.011e-14),
including a sub-ULP strain that the high-word-only control discards. Native
fold and derivative qualification must follow; the CPU check alone does not
admit this repair as a complete solver result.

The unchanged-source-scaling replay `compensated-unscaled-replay-002` completes
all 14,400 projections. All 9,600 nonzero-reference stress responses and 7,200
nonzero-reference algorithmic tangents pass the 1e-3 gate: worst errors are
8.633e-6 and 1.903e-4 relative, respectively. The 4,800 zero-reference stresses
and 7,200 zero-reference tangents remain inconclusive without measured noise
floors. This policy remains opt-in; it does not close element/assembled finite
differences, complete-cycle or resolution qualification.

The rejected scaled candidates include large accumulated shear plastic strain
(approximately 0.014–0.024). Their FP64 source residuals and full partial states
are retained to distinguish arithmetic, stored-state representability and
projection accuracy. No global tolerance was relaxed.

## Retained native records

Full source/executable/shader bindings, commands, predictions, histories,
failures and timeouts are under `/Users/home/cardboard-implementation-20261014`.
The published solver milestone retains an external-artifact manifest in
`evidence/solver-repair-20261014` rather than claiming that local raw records
are included in the repository. Relevant newer cases include:

- `native/safeguarded-material-replay-001`: 392 failed projections.
- `native/compensated-material-replay-001`: 365 failed projections and strict
  exact-input material comparison; no physical FEM steps.
- `solver/compensated-native-001`: complete-cycle attempt with unchanged
  global tolerance stopped at control step 45 / Newton iteration 19, after
  45 accepted steps of the 108-step loading/unloading/relaxation/release cycle.
  Center/trial residuals were 1.244085e-6 / 1.245071e-6, last alpha 0.0078125.
  Its animation is explicitly labeled an incomplete diagnostic and excludes
  the rejected step. An accepted prefix cannot admit a resolution study.
- `native/compensated-unscaled-replay-002`: all 14,400 projections complete;
  strict nonzero material response gates pass, zero-reference comparisons
  remain inconclusive.
- `native/full-contact-abi45-001` and `full-contact-trial-trace-002`: synthetic
  combined-contact fixture rejects before any private residual trial because
  the production line-search bound is zero. Production/private status payloads
  and staged limiter captures are retained; this path is not qualified.
- `solver/compensated-native-regional-003` and
  `regional-failure-capture-004`: restart-16 first-step rejection and complete
  exact-input independent replay; diagnostic production files are identical.
- `native/full-contact-deformable-limit-trace-004`: nonexistent input path
  rejection, retained without a native qualification claim.
- `native/full-contact-deformable-limit-trace-005`: the trace identifies
  duplicated covered mortar endpoints colliding with adjacent triangles.
  Per-feature coverage filtering is repaired without changing CCD thresholds.
- `native/full-contact-covered-endpoint-006`: later contact rejection;
  the source snapshot raced a final contact edit, explicitly marked
  unqualified in `source-binding-limitation.json`. The actual source is
  retained separately and the immutable-source rerun is case 007.

Exact-recipe damage/adhesion calibration and bare-box impact validation remain
open wherever published measurements do not identify them. Feasible analogue
parameter ranges describe model sensitivity, not statistical confidence.

The detailed native component results and preserved failures are in
[`solver-progress-20261014.md`](solver-progress-20261014.md) and the machine-readable
[`native evidence index`](evidence/native-implementation-20261014/summary.json).
The earlier immutable shipping snapshot passes 29 CPU checks; later changes
have their own source-bound checks recorded in the index. No result in a CPU
suite closes a native physical stage or numerical convergence gate.

## Current physical execution repairs

All twelve scoring lines now use three reusable physical tool families,
with one punch and two rectangular supports each. Platen, floor and finite
loading body bring the compiled proxy total to 12. Tools relocate through
accepted fixed-orientation translation steps; no FEM interior rotation sets
panel geometry. This resolves the authored scene's former 39-proxy count mismatch but leaves
5,440,536 dense node/proxy pairs and the other shipping runtime caps open.
Source49 case111 accepts the prescribed BOX translation and verifies its uploaded pose plus rejection rollback. Physical score completion still needs qualification.
The source48 prebuild estimator used the legacy 39-proxy/5-body reserve;
source49 replaces it with shared constants for the actual 12-proxy/9-body scene.
This estimator correction is separate from native physics admission.

The slow-compression scheduler now treats `--dt` as a maximum integration
step and preserves the authored rate, duration and endpoint with a shortened
last step. The portable schedule instrument gives 150/300/600 steps for 8/4/2 ms
limits on the same 0.25 mm, 1.2 s interval. The full-stage adapter verifies actual
accepted compression dt/counts and clock increments, rather than manifest dt
labels alone. It also consumes accepted grip moments and released springback
with endpoint, residual, coverage and shape-fit gates. These new contracts
have CPU evidence, not a completed native refinement series.

Heavy branches require a prior source-bound fixture-free roof measurement.
The compiled FEM-contact BOX and every corresponding rigid-body branch must
share the same explicit footprint, pose, mass and inertia. Missing footprint
or numerically qualified source-bound Fpeak keeps gravity/drop construction
closed without erasing completed compression records. No shipping Fpeak or
heavy-object scenario currently meets those gates.


The ordinary literature2009 material selection now uses separate kraft/NSSC
15-state bilinear tensile transfer assets. Four compiled FP64 stress-free
uniaxial slope checks match the digitized targets within 0.170% and remain
inside their registered digitization intervals. The ideal 13-state assets
remain regression inputs and the base for the separate retained-damage overlay,
which already supplies hardening; combining both hardening paths is rejected.
The directional multiaxial partition remains an uncalibrated transfer assumption.
A separate production Metal material-point checker passes in source47 case095
for both grades, including rest, MD/CD loading, unloading, reloading and all nine
stress-derivative columns. The maximum relative errors are 1.869e-4 for the
algorithmic tangent, 2.896e-6 for stress, 2.982e-6 for evolving state, and
1.498e-4 for branch-aware representable finite differences. Zero responses
remain inconclusive without measured absolute noise floors. These point-update
results do not identify tensile damage, starch separation or hot-melt failure.

Source47 also passes 56 selected CPU checks and 80 native CCD checks. The exact
captured exhausted CCD ray remains uncertified; rerunning the unchanged
algorithm on a 1/64 ray obtains a separately certified miss. The original
capture used an invalid overlapping fixture, so this is a local geometric
instrument, not full nonlinear-contact qualification. The valid replacement
fixture instead stops at 400 candidates against a 384-slot reserve, before a
linear solve or accepted step. Diagnostic on/off raw state and 18 persistent
rollback vectors agree; the full-linear-residual instrument is inconclusive
because no healthy linear solve occurs. Source48 reserves all 18,336 possible
primitive pairs for that small fixture and adds a private base-to-trial swept
gate for new and inactive self-contact candidates. Its native outcomes must
be recorded separately.

## Published source49 native instrument milestone

The integrated source49 build passes all 56 selected CPU checks. Native cases109–115 all pass: the exact production swept-contact kernel distinguishes six crossing/coverage controls (2,014 assertions); the CCD corpus passes80; the prescribed BOX instrument passes2,962 assertions across three native transactions; combined-contact cases112/113 each pass3,725 assertions with three accepted transactions and one deliberate rejection; the finite-body bridge passes244; and the bilinear Metal point update retains its complete stress/state/tangent comparisons. These counts refer to bounded synthetic or component instruments.

The on/off comparator records12 byte-identical production payloads, restoration of18 persistent vectors and both accepted cursors after rejection,384 healthy full-operator true-residual rows, and no excluded failed-production rows. The selected private trial and the next production reassembly have exactly matching residual norms. Source48’s failed namespaced lookup, cursor and uploaded-pose assertions remain retained. Source49 repairs those instruments without changing their physical tolerances. Loaded/rejected fixture traces were checked in memory but not exported in this source; source50 adds explicit trace retention.

Case108 remains a complete-cycle failure: after600 seconds it retained two of27 accepted controls. It does not qualify mesh or time convergence. Configured-work-budget case116 (8 Newton passes/512 restart16 columns) rejects its first control after45.4 seconds at residual2.066e-5, above the unchanged1e-6 gate. The rejection and conditioning capture are retained. Every shipping scoring, folding, closure, settled-checkpoint, compression and heavy-object stage remains unqualified.

## Source50–53 solver and source-data milestone

The original kraft/NSSC/starch recipe, miniature regression and explicit-flute shipping driver remain. New nondefault asymmetry transfer assets use the published tensile strengths and positive compression correction with a C1 numerical transition. Their directional hardening fits include pixel uncertainty and three smoothing-width sensitivity runs, not held-out validation: published marker specimens and conditioning are unavailable. Compression/shear source assumptions, pressure/multiaxial transfer and missing damage calibration remain explicit. The pin extractor separately reports only identifiable visible-curve descriptors; it does not turn truncated traces or whole-fixture response into local starch fracture energies.

The rich solver capture and independent replay now compare accepted history, local projection, source tangent, frame/element action, mass/stiffness, and actual perturbation kinematics. All1,728 nonzero FP64 derivative directions in the captured bent environment pass two adjacent smooth refinements at1e-6. Zero-direction and unresolved Metal noise-floor comparisons remain inconclusive. Production on/off records remain identical. The detailed [solver report](solver-progress-20261014.md) retains the mismatches and original19:3 capture gap.

Source52 repairs a wrong optional-output buffer argument in both tangent-preconditioner builders; it caused a32-entry output write into a nine-entry deformation-low buffer. Source53 adds an opt-in two-word implicit derivative solve/contraction for compensated materials of at most16 states. The CPU cancellation regression passes, but ordinary native C1 case136 does not select this policy and still fails; an explicitly selected native run is required. The native source-paper preconditioner fixture137 also fails its first AP comparison and must be diagnosed before it can support qualification. Repaired coupon135 still times out after two of27 controls, so no mesh/time refinement series qualifies.

Sparse private rigid-contact initialization extends the checked canonical pair index domain without changing production histories or physical contact. Source53 overflow138 passes2,961 assertions with one deliberately rejected native transaction: trial capacity overflow, terminal nonlinear10, history rollback and cleared certificate are retained. High-pair133 has zero physics transactions. Source52 parity134 reveals a late postCommit rejection that leaves the candidate certificate valid; the production fix and fresh native regression are subsequent work. Negative admission139 and material-frame140 fail their instrument setup and remain failures, not physical findings.

## Source54 bounded native progress

The integrated build passes59 selected CPU checks. Native147 verifies the full local tangent cache against direct framed actions (maximum relative difference1.355e-6) and preserves loaded/unloaded15-state paper history and certificates exactly. Native148 verifies the opt-in cohesive workspace alias across six activation/loading/rejection transactions, including the repaired late-rejection certificate invalidation. These are bounded native checks; no shipping transaction is inferred.

Explicitly selecting compensated implicit arithmetic in143 repairs the previously failing C1 center tangent to1.161e-6 worst relative error. Its perturbed F13 projection still fails; the separately captured liner CD unload reaches the16-iteration ceiling. Regional paper AP146 still disagrees with FP64 by2.808664 despite captured deformation/history/timestep inputs. Original-protocol141/142 now fail at control22, while captured control19 passes; their53 production outputs are byte-identical. Full coupon convergence, calibrated damage/debonding, fixture-free shipping assembly and crushing/drop histories remain open. The detailed stage and numerical limitations are recorded in `solver-progress-20261014.md`.

The exact full-world allocation model now includes528,444 nodes,1,477,728 tetrahedra,260,835 cohesive faces and all12 rigid proxies. Compact faces and explicit contact capacity estimate8.59 GB with the tangent cache; several individual runtime arenas still reject the world. This estimate does not qualify the16 GiB simultaneous peak. No fixture-free full-box assembled checkpoint,70% compression curve, qualified Fpeak, gravity crush or100 mm drop has completed. Those six numerical/material/assembly/prediction verdicts remain separate in the source-bound evidence index.

## Source55–57 continuation and corrected allocation accounting

Native157/161 complete the compensated asymmetric C1 material-point suite with an explicitly authored32-iteration local budget and unchanged stress/state/tangent tolerances. Native162 passes the1% assembled reference-action gate after accepted plastic loading; its0.003023 stiffness-only difference still exceeds the tighter element gate. The full coupon remains unresolved: instrument160 reproduces152 bytewise and measures genuine linear residual stagnation through1,024 columns, followed by rejected residual descent. These bounded component passes do not qualify a complete coupon or a refinement series.

The old simultaneous allocation model underestimated separate host storage by using a maximum of modeled domains. It now counts the Metal envelope plus distinct source/compile mesh storage and retained host checkpoints, with shared upload/readback reserves counted once. The unoptimized12-proxy example is21.552 GB and is rejected. The exact-topology sparse, no-free-rigid, aliases-on planning scenario is12.023 GiB including tangent cache after adding the documented coarse host reserve; that reserve is an engineering estimate, and count/arena limits still reject the shipping world. No actual whole-process16 GiB qualification is claimed. Earlier underestimated reports remain retained with explicit supersession notes.

The probe can explicitly compile the compensated generic15-state source bilinear/C1 policy, binds the effective compiled flags and world fingerprint in its manifest, and keeps default material policy unchanged. Resolution admission preserves that policy and constant equivalent force accuracy under timestep refinement; it rejects legacy or mixed-policy manifests it cannot reconstruct. Sparse private contact has a bounded free-BOX implementation, but the first native instruments163/164 stop during test setup and do not qualify that path. Angular BOX trial CCD, gravity/actuator transfer, complete scoring/assembly, calibrated damage/debonding, slow compression and all gravity/drop predictions remain open.
