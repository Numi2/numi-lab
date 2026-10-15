# Cardboard implementation and evidence ledger

## Current milestone: source142 bounded native CGLS

The [source142 report](solver-cgls-source142.md) records the repaired native CGLS helper, closely agreeing selected operator actions and a 73.68% nonlinear residual decrease at control 1/Newton 48. Capture-on/off runs preserve the complete attempted history, but both fail and roll back the control. Independent FP64 intervals resolve every captured paper direction at the 1e-6 gate; the native FP32 derivative comparison remains noise-inconclusive. The negative captured stiffness work is dominated by compressive geometric stiffness. No complete coupon, resolution study, calibrated corrugated failure, released shipping assembly or crush/drop result is qualified. Earlier reports retain their original scope.

## Current milestone: source139 cache admission and attempted-history parity

The [source139 report](solver-cache-parity-source139.md) records the cache admission repair and exact parity across the entire attempted coupon history. Repeated raw Cauchy fallback still exhausts64 Newton iterations and rolls back. Independent FP64 perturbations resolve finite-step curvature and meet the assembled1% gate over a range, while the all-row material derivative gate remains open. Native residual replays of CPU CGLS proposals are separate from integrated solver qualification. No completed coupon, resolution study, calibrated corrugated failure, released shipping assembly or crush/drop result is qualified. Earlier reports retain their original revision scope.

## Current milestone: source135 exact-state descent experiment

The [source135 report](solver-milestone-source135.md) records a native residual-descent repair at control1/Newton48, followed by a later Newton53 rejection and complete control rollback. Independent GPU transpose/forward checks and fallback-disabled history parity pass. The captured matrix has condition number4.59e7; complete independent finite endpoint secants agree closely with native while remaining far from the center tangent. Source137 builds69 selected targets and passes74 CPU tests, but repeated fallback and cache optimization remain experimental. No complete coupon, resolution series, calibrated corrugated failure, released shipping assembly or crush/drop result is qualified. Earlier milestones retain their original scope.

## Current milestone: source123

The [source123 report](solver-milestone-source123.md) records the two-stage local merit repair: native407 clears retained tet858 in14 updates, native408 retains tet674 in5, and native410 returns valid results for3,072 diagnostic samples. Selected independent tangent and endpoint comparisons pass while smaller-step failures remain. Native405/406 locate a separate CCD budget failure with physical-history parity. Full coupon409, mesh/time convergence, calibrated damage, shipping assembly and crush/drop qualification remain separate and open.

## Current milestone: source121

The [source121 report](solver-milestone-source121.md) records the scaled material merit repair, exact674 point regression, complete rejected-contact trial captures, and the new native858 material failure. Source119 passes70 build targets and74 CPU checks; source121 freshly builds three diagnostic targets. The full coupon still rejects control zero. The previous858 CPU replay used the old merit rule and is retained with an erratum. No full resolution study, calibrated corrugated damage, released shipping assembly, capacity or crush/drop prediction is qualified. Earlier milestone headings below preserve their original revision scope.

## Current milestone: source104

The diagnostic-only source104 changes pass68 build targets,73 CPU tests and
native362 physical-history parity against source103. Exact failed-state
captures360/361 identify premature Arnoldi completion and closely reproduced
regional algebra; they do not complete the nonlinear trajectory. Derivative,
mesh/time, calibrated damage, shipping assembly and crush/drop gates remain
open. See [the source104 report](solver-milestone-source104.md), including its
explicit evidence-retention erratum. Earlier entries retain their source scope.

## Current milestone: source103

The integrated source103 contact and restart repairs build68 targets and pass
73 CPU checks. Native353 passes16/16 mortar geometry cases, and combined-contact
356 passes5,683 assertions with exact selected-update and rollback checks.
Complete coupon352 and the free-body355 trajectory still fail; no complete
resolution study, calibrated damage/debonding, fixture-free shipping assembly
or crush/drop is qualified. See [the current report](solver-milestone-source103.md)
and its source-bound evidence. Earlier entries below retain their original
revision-specific scope.

## Selected material derivative refinement: reference011

The published source095 implementation is unchanged. A separately declared
refined perturbation grid now resolves all nine nonzero material directions
at the eight selected points, which contain three unique parameter/F/history
states. FP64 actual-ray errors pass 1e-6 at both 5e-11 and 3.5e-11; the
worst errors across those states are 4.67e-7 and 8.77e-7, respectively.
The native315 stress finite differences pass 1e-3 in every direction at
1e-4: worst errors are 1.03e-4 for the liner and 5.91e-5 / 1.45e-5 for
the two medium states. Both native stress paths agree. The measured rounded
Fdot versus H/dt direction error at this step is 2.53e-8; this source law is
rate independent, so it does not qualify evolving rate state.

Independent 90-digit projections at the actual FP64 and native two-word FP32
endpoints now record componentwise absolute stress disagreement and separate
arithmetic effects from finite-step curvature. These are observed envelopes
for the selected inputs, not universal noise bounds. The smaller-step liner
misses, all endpoint operands, branch indicators and failed008 analysis
consumer are retained. No endpoint branch-label crossing occurs in the
native1e-4 comparison. This closes the bounded nonzero material-direction
checks; full element/contact/residual accuracy and complete nonlinear
trajectories remain separate qualifications.

Experimental source096 builds all 66 targets but its opt-in augmentation
rejects the actual coupon at initialization: the fixture contains two
independent environments and three authored material-table entries. The
feature-off317 one-Newton diagnostic fails its unchanged terminal certificate;
its initial linear solve needs only one iteration and cannot test restart
augmentation. Its complete 34,574,053-byte operator capture is losslessly
restore-tested. No enabled/disabled trajectory parity or augmentation
performance claim follows.

Experimental source097 builds the contact checker and native probe. CCD318
executes all 16 direct production-kernel fixtures: 12 pass and four multi-patch
sweep motions still reject. The corrected positive-gap partial-covered fixture
accepts, while the new positive-gap complement and static4nm off-edge overlap
negatives reject. Original314 inputs and failures remain retained. There are
no accepted FEM transactions in this synthetic checker.

The exact tables and commands are in the source-bound evidence archive under
`analysis/selected-material-direction-adjudication-011`. No complete coupon
resolution series, calibrated damage/debonding, fixture-free shipping box,
qualified capacity, gravity crush or drop impact is established.

## Source095: retained failures and narrower numerical results

Source095 builds all 66 targets and passes 71 selected CPU checks. No global
certificate tolerance, restart ceiling, physical parameter or material history
was relaxed. The scoped positive-epsilon Fischer–Burmeister value repair makes
all 3,456 retained paper stress/tangent projections valid in native300. Default
response words remain unchanged. Native315 completes 1,536 evaluations per
policy, including nine basis actions and ten signed perturbation sizes at eight
points. At two exact medium centers, accurate native matrices agree with the
independent 90-digit matrices within 4.36e-8 in matrix norm and 4.98e-8 for the
worst nonzero column. The exact-pair follow-up also checks all nine actions at the six selected liner
points, which have one identical parameter/F/history state. Across these eight
points (three unique states), the accurate-policy worst column error is5.18e-8.
This remains selected material evidence, not a full solver gate.

The independent 90-digit endpoint study resolves finite-step curvature at those
two centers. Smaller FP64 steps give full-matrix errors 1.94e-7 and 7.22e-8 at
1e-10; tet1634 still misses the every-column 1e-6 finite-difference gate (worst
1.90e-6 there, 1.07e-6 at1e-11). Still smaller steps lose accuracy. All steps,
endpoint branches and input-rounding errors are retained. Near-zero absolute
noise floors remain unresolved; deterministic center repeats are not noise
floor measurements. Native315 independently records substantial rounded-rate
direction loss for the smallest liner perturbations, rather than assuming
that nominal Hdot=H/dt was exactly realized.

Actual coupon303/304 accepts two of27 controls and normally rejects control2.
The selected control2/Newton63 operator agrees with independent assembly
(A-action6.53e-5, captured AP1.00e-7, LU reconstruction1.25e-7), but native
linear residual remains about0.952 of the RHS. The lossless 38,818,506-byte
capture is restore-tested. Same-Newton augmentation is an offline conditioning
experiment until native action, factor, reset and physical-run checks pass;
older accepted-cross-Newton recycling remains disabled after its regression.

The selected private contact trial now has a strict, complete capture: native312
retains129 arrays/19,467,248 bytes, with validated extents and hashes. Corrected
topology/status record types replace the failed307/310 instruments; those
failures remain retained. Capture-on312 and capture-off313 have all11 nonlog
production files and all32 Armijo records identical. The lossless package is
restore-tested. The physical fixture still rejects. Independent replay reproduces the cohesive, rigid-contact FEM and deformable
contact gather results bit-for-bit. This verifies gather assembly of captured
forces. Independent contact constitutive response, full center-to-trial residual
derivative and same-generation Jdelta comparison remain separate gates.

Private CCD314 executes all14 declared cases and retains all12 exact input
buffers per case. Nine pass; five as-authored expected-accept cases reject: partial covered
crossing, three- and four-patch closure, common translation, and separation.
A subsequent source audit identifies invalid initial geometry in the partial
covered positive fixture; its correction is a separate versioned test.
Collision/overlap negative controls still reject. The four-patch bound and
conservative common-displacement rule therefore do not qualify mortar contact.
Shipping CPU audits independently prove that recorded float clipping slivers
have zero exact-rational overlap, but production Metal partition/complement
arithmetic and runtime admission remain unresolved. No CPU audit overrides a
native rejection.

No complete coupon mesh/time/slower-loading series, matched corrugated damage
or starch/hot-melt identification, fixture-free300x200x200mm assembly,
capacity-derived mass, gravity crushing or drop prediction is qualified.

## Earlier retained milestones

Frozen source091 builds all 66 targets and passes 71 selected CPU checks. Its
opt-in bilinear15 repair evaluates the same positive-epsilon Fischer–Burmeister
residual with a rationalized value expression; authored derivative programs,
physical parameters, production acceptance criteria and the default policy
remain unchanged. Native300 accepts all 3,456 captured-point stress/tangent
projections, including the six formerly rejected liners. All 26 prior medium
action-norm nonpasses now meet 1e-3 against the retained FP64 reference. These
are one-direction comparisons; zero directions, all-direction derivatives and
the reference's near-yield root stopping remain separate qualifications.

Source090's bounded mortar union accepts three-patch activation in299. All
legacy/tiled transaction metadata and raw snapshots agree bytewise at each
face count; rejected closure preserves all 18 recorded accepted-history
vectors and an invalid certificate. Closure still fails the strict residual
gate. Source091 repairs a host upload crash and checker output ownership,
then private CCD301 fails matched closure across shared patch boundaries.
This is not a completed contact or assembly result. The complete 27-control
coupon303 normally rejects its third control after two acceptances and
694.80 seconds. Its status10/FG32 failure is retained; no complete cycle or
mesh/time convergence qualifies. The two-point reference audit's analytic
matrices agree with90-digit calculations, but its FP64 finite-difference
errors still exceed1e-6. Stable epsilon differences do not close that gate.

Sources087/088 each build all65 targets and pass all71 selected CPU checks.
Run287 establishes legacy/tiled parity on rejected prefixes only and exposes
the unsupported multiple-patch contact lookup. Run289 captures the full
reserved contact arena but retains an invalid JSON manifest. Source088 fixes
only serialization;291 strictly validates all140 arrays and10,787,124 payload
bytes. Capture-on291 and capture-off292 retain identical physical state and
Armijo histories. Their free-body residual still fails the unchanged tolerance.
Independent FP64 replay of all3,456 run283 paper points fits the16-iteration
local ceiling, so that ceiling does not explain288 on those inputs. Actual
failure capture293 times out. Source089 builds66 targets and passes71 selected
CPU checks. Its separate native point instrument294 reproduces accurate-policy
local exhaustion through projector, stress and tangent on283's retained liner
input. A tiny residual coexists with a stalled scaled correction; comparison
of the exact iteration rows is under way. This does not substitute283's inputs
for288's missing failed candidate or qualify a complete physical trajectory.

The preceding source086 milestone builds all65 targets and passes all71 selected
CPU checks. The scoped, dimensionally scaled15-state projection policy passes
all16 selected captured native material points in285. This is a local numerical
check, not a complete coupon. Default-disabled conditioning run283 preserves
all64 Armijo records and accepted payloads from263 byte for byte. Opt-in
accepted-correction recycling284 regresses that first control and remains
disabled. Free-body278/279/286 and tiled-cohesive280 remain failed; coupled
candidate readback286 is still skipped by its independent arena scope guard.
Complete coupon247/258/266 reject after one accepted control. Local-policy
attempt288 normally rejects its first control after371.71s, with status5 at
tet1528; its first failing material/deformation layer remains under diagnosis. See
[solver progress](solver-progress-20261014.md) for exact scopes and retained
failures. No complete mesh/time study, accepted fixture-free shipping assembly,
qualified capacity, gravity crush or drop impact exists.

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
| Solver correctness | Shared tested fractions and rollback; exact-word material/element/assembled actions; native300 all 3,456 captured-point projections valid after scoped FB value repair; strict capture291 with on/off history parity;299 accepted activation and failed-prefix rollback parity | No completed coupon/contact qualification. Recycling284 stays disabled. One captured direction is not all-direction or near-zero qualification. Actual first-force capture293 times out; private shared-boundary CCD301 still fails |
| Numerical convergence | Complete-history runners preserve duration, paths, rate, tolerances and failed cases | No complete 8/16/32, 2/4/8, 1/2/4 or timestep triplet has qualified. A first-control pass is not a complete coupon |
| Component calibration | Versioned published curves, pixel uncertainty, bilinear tensile fits and explicit compression/shear assumptions; synthetic progressive-damage and interface laws execute transactionally | No matched corrugated damage or local starch/HMA fracture identification; source-grade transfer and Robertsson absolute-energy convention remain open |
| Analogue validation | Separate ECT, scored-bend, pin and BCT series with source identities and transfer rules | Whole native source fixtures and held-out comparisons. Partial pin windows cannot supply full peaks or full work; unrelated grades cannot be labeled same grade |
| Physical assembly in simulation | Explicit shipping blank, actual flap overlap/closure geometry, stage controls, transactional activation and checkpoint helpers | Physical scoring, coupon-derived allowance, all wall/flap operations, bond setting, release and fixture-free settling. Current full mesh remains outside native contact/cohesive admission |
| Crush/impact prediction | Native finite-body coupling fixtures and staged compression/gravity/drop driver | Released shipping checkpoint, complete 70% nominal and half-speed compression, numerically qualified first peak, masses, gravity transfer, impact refinement and energy/momentum accounting |

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

Source145 milestone: see [exact-state, derivative and full-phase evidence](solver-milestone-source145.md). Default-zero native459 attempted-history parity passes; the complete nonlinear coupon, convergence, released shipping assembly and crush/drop remain unqualified.
